#!/usr/bin/env python3
"""
Test script for diagnosing black image generation issues.
This script tests a single batch generation to quickly identify issues.
"""

import torch
import numpy as np
import os
import logging
import argparse
import json
import time
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_device():
    """Setup the device with detailed logs"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        device_cap = torch.cuda.get_device_capability(current_device)
        logger.info(f"CUDA available: {device_count} device(s)")
        logger.info(f"Using device {current_device}: {device_name} (capability {device_cap[0]}.{device_cap[1]})")
        logger.info(f"CUDA version: {torch.version.cuda}")
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("Using Apple Metal (MPS)")
        device = torch.device('mps')
    else:
        logger.warning("CUDA not available! Using CPU - this will be slow!")
        device = torch.device('cpu')
    
    return device

def load_model(device, args):
    """Load the model with detailed debug information"""
    
    # Determine data type based on device and args
    # Force float32 for numerical stability to avoid NaN issues
    if device.type == 'cuda' or device.type == 'mps':
        if args.half_precision:
            logger.warning("Half precision can cause NaN issues - using float32 for stability")
        dtype = torch.float32
        variant = None
        logger.info("Using full precision for numerical stability")
    else:
        dtype = torch.float32
        variant = None
        logger.info("Using full precision for CPU")
    
    try:
        logger.info(f"Loading model from {args.model_path}")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            use_safetensors=True,
            variant=variant
        )
        
        # Move model to device
        logger.info(f"Moving model to {device}")
        pipe = pipe.to(device)
        
        # Optimize scheduler
        if not args.disable_optimization:
            logger.info("Setting up optimized scheduler")
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                use_karras_sigmas=True,
                algorithm_type="dpmsolver++"
            )
            
            # Memory optimizations
            logger.info("Enabling memory optimizations")
            pipe.enable_attention_slicing(1)
            pipe.enable_vae_slicing()
            
            # Enable xformers if available and on CUDA
            if device.type == 'cuda' and args.xformers:
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                    logger.info("xformers memory efficient attention enabled")
                except Exception as e:
                    logger.warning(f"Could not enable xformers: {e}")
        
        # Disable safety checker
        if args.disable_safety_checker:
            logger.info("Disabling safety checker")
            pipe.safety_checker = None
            pipe.requires_safety_checker = False
        
        # Add numerical stability settings
        logger.info("Configuring for numerical stability")
        
        # Ensure VAE is in float32 for stability
        if hasattr(pipe, 'vae'):
            pipe.vae.to(dtype=torch.float32)
            logger.info("VAE set to float32 for stability")
            
        return pipe
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def generate_test_images(pipe, device, args):
    """Generate test images with diagnostics"""
    
    prompts = [
        "A scenic rural road through rolling green hills, professional photography, bright daylight, clear visibility, high resolution",
        "A winding country road through golden wheat fields, sunny day, clear blue sky, award-winning landscape photography",
        "A mountain highway with pine trees, morning sunlight, pristine landscape, professional DSLR photography"
    ]
    
    negative_prompt = """
        low quality, blurry, out of focus, dark image, black image, completely black,
        cartoon, anime, painting, sketch, artificial, fake, unrealistic, oversaturated,
        distorted geometry, impossible perspective, floating objects, unnatural lighting,
        city, urban, buildings, cars, vehicles, people, pedestrians, traffic signs,
        watermark, text, logo, signature, frame, border
    """
    
    # Ensure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    batch_size = min(args.batch_size, len(prompts))
    batch_prompts = prompts[:batch_size]
    
    logger.info(f"Generating {batch_size} test images...")
    logger.info(f"Prompts: {batch_prompts}")
    logger.info(f"Negative prompt: {negative_prompt}")
    logger.info(f"Width: {args.width}, Height: {args.height}")
    logger.info(f"Steps: {args.steps}, Guidance scale: {args.guidance_scale}")
    
    # Set seeds for reproducibility
    seeds = [42, 1337, 8888][:batch_size]
    if device.type == 'cuda':
        generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
    else:
        generators = [torch.Generator().manual_seed(seed) for seed in seeds]
    
    logger.info(f"Using seeds: {seeds}")
    
    # Time the generation
    start_time = time.time()
    
    try:
        # Use appropriate precision with numerical stability fixes
        logger.info("Starting inference...")
        with torch.inference_mode():
            # Disable autocast for numerical stability - this often fixes the NaN issue
            results = pipe(
                prompt=batch_prompts,
                negative_prompt=[negative_prompt] * batch_size,
                width=args.width,
                height=args.height,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generators,
                num_images_per_prompt=1
            )
                
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Generation completed in {duration:.2f}s ({duration/batch_size:.2f}s per image)")
        
        # Analyze and save the images
        for i, image in enumerate(results.images):
            # Analyze image with NaN/inf checking
            img_array = np.array(image)
            
            # Check for invalid values
            has_nan = np.isnan(img_array).any()
            has_inf = np.isinf(img_array).any()
            
            if has_nan or has_inf:
                logger.warning(f"Image {i+1} contains invalid values - NaN: {has_nan}, Inf: {has_inf}")
                # Clean the array
                img_array = np.nan_to_num(img_array, nan=0.0, posinf=255.0, neginf=0.0)
            
            mean_brightness = np.mean(img_array)
            std_brightness = np.std(img_array)
            
            # Calculate quality score
            quality_score = min(1.0, (mean_brightness / 128.0) * (std_brightness / 50.0))
            
            logger.info(f"Image {i+1}:")
            logger.info(f"  - Brightness: {mean_brightness:.2f}")
            logger.info(f"  - Contrast (std dev): {std_brightness:.2f}")
            logger.info(f"  - Quality score: {quality_score:.2f}")
            logger.info(f"  - Histogram min: {img_array.min()}, max: {img_array.max()}")
            logger.info(f"  - Shape: {img_array.shape}")
            
            # Save image
            filename = f"test_image_{i+1}_seed_{seeds[i]}.png"
            filepath = os.path.join(args.output_dir, filename)
            image.save(filepath)
            logger.info(f"  - Saved to: {filepath}")
            
            # Save histogram data
            hist_r, _ = np.histogram(img_array[:,:,0], bins=50)
            hist_g, _ = np.histogram(img_array[:,:,1], bins=50)
            hist_b, _ = np.histogram(img_array[:,:,2], bins=50)
            
            hist_data = {
                "r_histogram": hist_r.tolist(),
                "g_histogram": hist_g.tolist(),
                "b_histogram": hist_b.tolist(),
                "brightness": float(mean_brightness),
                "contrast": float(std_brightness),
                "quality_score": float(quality_score),
                "prompt": batch_prompts[i],
                "seed": seeds[i],
                "steps": args.steps,
                "guidance_scale": args.guidance_scale
            }
            
            hist_filepath = os.path.join(args.output_dir, f"test_image_{i+1}_analysis.json")
            with open(hist_filepath, 'w') as f:
                json.dump(hist_data, f, indent=2)
            
        return True
            
    except Exception as e:
        logger.error(f"Generation error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description="Test SDXL image generation")
    parser.add_argument("--model-path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="Path or HF repo ID for the model")
    parser.add_argument("--output-dir", type=str, default="test_output",
                        help="Directory to save generated images")
    parser.add_argument("--batch-size", type=int, default=3,
                        help="Batch size for generation (max 3)")
    parser.add_argument("--width", type=int, default=1024,
                        help="Image width")
    parser.add_argument("--height", type=int, default=1024,
                        help="Image height")
    parser.add_argument("--steps", type=int, default=30,
                        help="Number of inference steps (try 30-50 for better quality)")
    parser.add_argument("--guidance-scale", type=float, default=7.5,
                        help="Guidance scale (try values between 7-9)")
    parser.add_argument("--half-precision", action="store_true",
                        help="Use half precision (float16)")
    parser.add_argument("--disable-optimization", action="store_true",
                        help="Disable pipeline optimizations")
    parser.add_argument("--disable-safety-checker", action="store_true",
                        help="Disable safety checker")
    parser.add_argument("--xformers", action="store_true",
                        help="Enable xformers memory efficient attention")
                    
    args = parser.parse_args()
    
    # Print test configuration
    logger.info("=" * 50)
    logger.info("SDXL TEST BATCH GENERATION")
    logger.info("=" * 50)
    
    # Print all arguments
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    logger.info("=" * 50)
    
    # Setup device
    device = setup_device()
    
    # Load model
    pipe = load_model(device, args)
    
    # Generate test images
    success = generate_test_images(pipe, device, args)
    
    if success:
        logger.info(" Test completed successfully!")
        return 0
    else:
        logger.error(" Test failed!")
        return 1

if __name__ == "__main__":
    exit(main())
