#!/usr/bin/env python3
"""
Large-Scale SDXL Rural Driving Image Generator
Generates 10,000 high-quality rural driving images with production-grade optimizations
"""

import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import numpy as np
from PIL import Image
import os
import json
import time
import gc
import random
import logging
from datetime import datetime, timedelta
from tqdm import tqdm
import argparse
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import hashlib

# Configure logging with more verbose output for debugging
logging.basicConfig(
    level=logging.DEBUG,  # More verbose for first batch debugging
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('generation_10k.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LargeScaleImageGenerator:
    """Production-grade image generator for 10K+ images"""
    
    def __init__(self, config):
        self.config = config
        self.device = self._setup_device()
        self.pipe = None
        self.generated_count = 0
        self.failed_count = 0
        self.start_time = None
        self.save_queue = queue.Queue(maxsize=100)
        self.stats = {
            'total_generated': 0,
            'total_failed': 0,
            'average_time_per_image': 0,
            'memory_usage': [],
            'quality_scores': []
        }
        
    def _setup_device(self):
        """Setup optimal device configuration"""
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            
            # Optimize CUDA settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(0.95)
            
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Use Apple Metal Performance Shaders (MPS) if available
            device = torch.device('mps')
            logger.warning("Using Apple Metal (MPS) - This will be slower than CUDA")
        else:
            # Fall back to CPU with a warning
            device = torch.device('cpu')
            logger.warning("CUDA not available! Using CPU - this will be VERY slow!")
            logger.warning("10K image generation on CPU will take a very long time")
        
        return device
    
    def _load_pipeline(self):
        """Load and optimize SDXL pipeline for production"""
        logger.info("Loading SDXL pipeline...")
        
        try:
            # Determine optimal data type based on device
            # Use full precision by default for better quality (like test script)
            use_half_precision = self.config.get('use_half_precision', False)
            
            if self.device.type == 'cuda' or self.device.type == 'mps':
                if use_half_precision:
                    # Use half precision for speed (lower quality)
                    dtype = torch.float16
                    variant = "fp16"
                    logger.info("Using half precision (float16) for speed")
                else:
                    # Use full precision for quality (like test script default)
                    dtype = torch.float32
                    variant = None
                    logger.info("Using full precision (float32) for better quality")
            else:
                # Always use full precision for CPU
                dtype = torch.float32
                variant = None
                logger.info("Using full precision (float32) for CPU")
            
            # Load with appropriate optimizations
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=dtype,
                use_safetensors=True,
                variant=variant
            )
            
            self.pipe = self.pipe.to(self.device)
            
            # Optimize scheduler for speed
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config,
                use_karras_sigmas=True,
                algorithm_type="dpmsolver++"
            )
            
            # Memory optimizations
            self.pipe.enable_attention_slicing(1)
            self.pipe.enable_vae_slicing()
            
            # Enable xformers if available and on CUDA
            if self.device.type == 'cuda':
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    logger.info("xformers memory efficient attention enabled")
                except:
                    logger.warning("xformers not available, using default attention")
            
            # Skip compilation for faster startup (can enable later for production)
            # Compilation takes 5-15 minutes initially but speeds up generation
            # Uncomment below for production runs after initial testing
            # if self.device.type != 'cpu':
            #     try:
            #         self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead")
            #         logger.info("UNet compiled for maximum speed")
            #     except:
            #         logger.warning("Model compilation not available")
            logger.info("Skipping UNet compilation for faster startup")
            
            # Disable safety checker for speed
            self.pipe.safety_checker = None
            self.pipe.requires_safety_checker = False
            
            # Ensure VAE is in float32 for numerical stability
            if hasattr(self.pipe, 'vae'):
                self.pipe.vae.to(dtype=torch.float32)
                logger.info("VAE set to float32 for numerical stability")
            
            logger.info(f"Pipeline loaded and optimized successfully on {self.device.type}")
            
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise
    
    def _create_prompts(self):
        """Create diverse, high-quality prompts for rural driving scenes"""
        
        # Base scenarios
        base_scenarios = [
            "winding rural country road through rolling green hills",
            "straight farm road between golden wheat fields",
            "mountain highway through dense pine forest",
            "coastal rural road with ocean glimpses",
            "prairie highway through vast grasslands",
            "country lane with stone walls and hedgerows",
            "forest service road through mixed woodland",
            "rural highway through vineyard country",
            "desert highway through sagebrush landscape",
            "highland road through moorland and heather"
        ]
        
        # Road surface types
        road_surfaces = [
            "asphalt road with yellow center line",
            "concrete highway with lane markings",
            "well-maintained country road",
            "weathered rural highway",
            "freshly paved road surface"
        ]
        
        # Lighting conditions
        lighting_conditions = [
            "golden hour lighting, warm natural light",
            "overcast day, soft diffused lighting",
            "early morning light with long shadows",
            "late afternoon sun, dramatic lighting",
            "bright daylight, clear visibility",
            "partly cloudy sky, dynamic lighting"
        ]
        
        # Weather conditions
        weather_conditions = [
            "clear blue sky with wispy clouds",
            "partly cloudy, dynamic sky",
            "morning mist in the distance",
            "crisp autumn air, perfect visibility",
            "spring day, fresh atmosphere",
            "adverse weather, heavy rain"

        ]
        
        # Quality enhancers
        quality_terms = [
            "professional automotive photography, DSLR quality",
            "ultra-detailed, photorealistic, masterpiece",
            "award-winning photography, crystal clear",
            "high resolution, perfect focus, sharp details",
            "professional grade, ultra-realistic"
        ]
        
        # Generate diverse combinations
        prompts = []
        for i in range(1000):  # Create 1000 unique prompt combinations
            prompt_parts = [
                random.choice(base_scenarios),
                random.choice(road_surfaces),
                random.choice(lighting_conditions),
                random.choice(weather_conditions),
                random.choice(quality_terms),
                "no cars, no people, empty road, clear visibility"
            ]
            
            full_prompt = ", ".join(prompt_parts)
            prompts.append(full_prompt)
        
        logger.info(f"Created {len(prompts)} unique prompt combinations")
        return prompts
    
    def _create_negative_prompt(self):
        """Create comprehensive negative prompt"""
        return """
        low quality, blurry, out of focus, dark image, black image, completely black,
        cartoon, anime, painting, sketch, artificial, fake, unrealistic, oversaturated,
        distorted geometry, impossible perspective, floating objects, unnatural lighting,
        city, urban, buildings, cars, vehicles, people, pedestrians, traffic signs,
        watermark, text, logo, signature, frame, border, multiple exposures,
        bad anatomy, deformed, mutated, noise, grain, artifacts, compression,
        night scene, darkness, overexposed, underexposed, neon colors
        """
    
    def _generate_batch(self, prompts, batch_size, start_idx):
        """Generate a batch of images with error handling"""
        
        batch_prompts = []
        batch_seeds = []
        
        for i in range(batch_size):
            prompt_idx = (start_idx + i) % len(prompts)
            batch_prompts.append(prompts[prompt_idx])
            batch_seeds.append(random.randint(0, 2**32-1))
        
        # Log batch start for debugging
        batch_num = start_idx // batch_size + 1
        logger.debug(f"Starting batch {batch_num} with {batch_size} images...")
        
        try:
            # Use device-appropriate generator
            if self.device.type == 'cuda':
                generators = [torch.Generator(device=self.device).manual_seed(seed) for seed in batch_seeds]
            else:
                generators = [torch.Generator().manual_seed(seed) for seed in batch_seeds]
            
            with torch.inference_mode():
                # Disable autocast for numerical stability - prevents NaN issues
                # The autocast was causing the "invalid value encountered in cast" error
                    # Log inference start for first batch
                    if batch_num == 1:
                        logger.info("Starting first inference - this is typically the slow part...")
                    
                    results = self.pipe(
                        prompt=batch_prompts,
                        negative_prompt=[self._create_negative_prompt()] * len(batch_prompts),
                        width=self.config['width'],
                        height=self.config['height'],
                        num_inference_steps=self.config['inference_steps'],
                        guidance_scale=self.config['guidance_scale'],
                        generator=generators,
                        num_images_per_prompt=1
                    )
                    
                    # Log completion for first batch
                    if batch_num == 1:
                        logger.info("First inference complete - subsequent batches will be faster!")
            
            return results.images, batch_prompts
            
        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            return [], []
    
    def _quality_check(self, image):
        """Quick quality check for generated images"""
        try:
            img_array = np.array(image)
            
            # Check for invalid values (NaN/inf) that cause the cast warning
            has_nan = np.isnan(img_array).any()
            has_inf = np.isinf(img_array).any()
            
            if has_nan or has_inf:
                logger.debug(f"Image rejected: contains invalid values - NaN: {has_nan}, Inf: {has_inf}")
                return False, 0.0
            
            # Check if image is not black/empty
            mean_brightness = np.mean(img_array)
            if mean_brightness < 20:
                logger.debug(f"Image rejected: too dark (brightness: {mean_brightness:.1f})")
                return False, 0.0
            
            # Check for reasonable contrast
            std_brightness = np.std(img_array)
            if std_brightness < 10:
                logger.debug(f"Image rejected: too flat (contrast: {std_brightness:.1f})")
                return False, 0.0
            
            # Simple quality score
            quality_score = min(1.0, (mean_brightness / 128.0) * (std_brightness / 50.0))
            
            # Standard threshold
            threshold = 0.3
            result = quality_score > threshold
            
            if not result:
                logger.debug(f"Image rejected: low quality score ({quality_score:.3f} < {threshold})")
            
            return result, quality_score
            
        except Exception as e:
            logger.error(f"Quality check failed: {e}")
            return False, 0.0
    
    def _save_image_worker(self):
        """Background worker for saving images"""
        while True:
            try:
                item = self.save_queue.get(timeout=30)
                if item is None:  # Shutdown signal
                    break
                
                image, filepath, metadata = item
                
                # Save image
                image.save(filepath, 'PNG', optimize=True)
                
                # Save metadata if provided
                if metadata:
                    metadata_path = filepath.replace('.png', '_metadata.json')
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                
                self.save_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Save worker error: {e}")
                self.save_queue.task_done()
    
    def _monitor_resources(self):
        """Monitor system resources during generation"""
        while self.start_time and time.time() - self.start_time < self.config['max_runtime_hours'] * 3600:
            try:
                # System memory
                system_memory = psutil.virtual_memory()
                
                # Track resources based on available hardware
                if torch.cuda.is_available():
                    # GPU memory for CUDA devices
                    gpu_memory_used = torch.cuda.memory_allocated(0) / 1024**3
                    gpu_memory_cached = torch.cuda.memory_reserved(0) / 1024**3
                    
                    # Log resource usage
                    if self.generated_count % 100 == 0:
                        logger.info(f"Resources - GPU: {gpu_memory_used:.1f}GB used, "
                                  f"{gpu_memory_cached:.1f}GB cached, "
                                  f"RAM: {system_memory.percent:.1f}% used")
                    
                    # Store stats
                    self.stats['memory_usage'].append({
                        'timestamp': time.time(),
                        'gpu_memory_used': gpu_memory_used,
                        'system_memory_percent': system_memory.percent
                    })
                else:
                    # CPU/MPS only stats
                    if self.generated_count % 100 == 0:
                        logger.info(f"Resources - RAM: {system_memory.percent:.1f}% used, "
                                   f"CPU: {psutil.cpu_percent()}%")
                    
                    # Store stats
                    self.stats['memory_usage'].append({
                        'timestamp': time.time(),
                        'system_memory_percent': system_memory.percent,
                        'cpu_percent': psutil.cpu_percent()
                    })
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(60)
    
    def _save_progress(self):
        """Save generation progress and statistics"""
        progress_data = {
            'generated_count': self.generated_count,
            'failed_count': self.failed_count,
            'start_time': self.start_time,
            'current_time': time.time(),
            'config': self.config,
            'stats': self.stats
        }
        
        with open(f"{self.config['output_dir']}/progress.json", 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def generate_10k_images(self):
        """Main function to generate 10K images"""
        
        logger.info("Starting 10K image generation...")
        
        # Setup
        self._load_pipeline()
        prompts = self._create_prompts()
        negative_prompt = self._create_negative_prompt()
        
        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
        os.makedirs(f"{self.config['output_dir']}/images", exist_ok=True)
        
        # Start background workers
        save_thread = threading.Thread(target=self._save_image_worker, daemon=True)
        save_thread.start()
        
        monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        monitor_thread.start()
        
        # Generation parameters
        target_images = self.config['target_images']
        batch_size = self.config['batch_size']
        save_interval = self.config['save_interval']
        
        self.start_time = time.time()
        
        logger.info(f"Target: {target_images} images, Batch size: {batch_size}")
        
        # Main generation loop
        with tqdm(total=target_images, desc="Generating Images") as pbar:
            
            batch_idx = 0
            
            while self.generated_count < target_images:
                
                # Check runtime limit
                elapsed_hours = (time.time() - self.start_time) / 3600
                if elapsed_hours > self.config['max_runtime_hours']:
                    logger.warning(f"Maximum runtime ({self.config['max_runtime_hours']}h) reached")
                    break
                
                # Generate batch
                batch_start_time = time.time()
                
                current_batch_size = min(batch_size, target_images - self.generated_count)
                
                # Add progress info for first batch (which is often very slow)
                if batch_idx == 0:
                    logger.info(f"🎨 Starting first batch generation (batch size: {current_batch_size})")
                    logger.info("   ⚠️ First batch may take 5-15 minutes due to model warmup/compilation")
                    logger.info("   📊 Subsequent batches will be much faster")
                
                images, used_prompts = self._generate_batch(prompts, current_batch_size, batch_idx * batch_size)
                
                batch_time = time.time() - batch_start_time
                
                # Process generated images
                for i, (image, prompt) in enumerate(zip(images, used_prompts)):
                    
                    # Quality check
                    is_good_quality, quality_score = self._quality_check(image)
                    
                    if is_good_quality:
                        # Generate filename
                        image_id = self.generated_count
                        filename = f"rural_driving_{image_id:06d}.png"
                        filepath = os.path.join(self.config['output_dir'], 'images', filename)
                        
                        # Prepare metadata
                        metadata = {
                            'image_id': image_id,
                            'prompt': prompt,
                            'quality_score': quality_score,
                            'generation_time': batch_time / len(images),
                            'batch_idx': batch_idx,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        # Queue for saving
                        self.save_queue.put((image, filepath, metadata))
                        
                        self.generated_count += 1
                        self.stats['quality_scores'].append(quality_score)
                        
                        pbar.update(1)
                        
                    else:
                        self.failed_count += 1
                        logger.debug(f"Rejected low-quality image (score: {quality_score:.3f})")
                
                # Update statistics
                if len(images) > 0:
                    avg_time = batch_time / len(images)
                    self.stats['average_time_per_image'] = (
                        (self.stats['average_time_per_image'] * batch_idx + avg_time) / (batch_idx + 1)
                    )
                
                # Save progress periodically
                if self.generated_count % save_interval == 0:
                    self._save_progress()
                    
                    # Log progress
                    elapsed_time = time.time() - self.start_time
                    if elapsed_time > 0:
                        images_per_hour = self.generated_count / (elapsed_time / 3600)
                        if images_per_hour > 0:
                            eta_hours = (target_images - self.generated_count) / images_per_hour
                            eta_msg = f"ETA: {eta_hours:.1f}h"
                        else:
                            eta_msg = "ETA: calculating..."
                        
                        logger.info(f"Progress: {self.generated_count}/{target_images} "
                                  f"({self.generated_count/target_images*100:.1f}%) - "
                                  f"Speed: {images_per_hour:.1f} img/h - {eta_msg}")
                
                # Memory cleanup
                if batch_idx % 10 == 0:
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    gc.collect()
                
                batch_idx += 1
        
        # Finalization
        logger.info("Generation complete, finalizing...")
        
        # Wait for save queue to empty
        self.save_queue.join()
        
        # Stop workers
        self.save_queue.put(None)  # Shutdown signal
        
        # Final statistics
        total_time = time.time() - self.start_time
        
        final_stats = {
            'total_generated': self.generated_count,
            'total_failed': self.failed_count,
            'total_time_hours': total_time / 3600,
            'average_time_per_image': self.stats['average_time_per_image'],
            'images_per_hour': self.generated_count / (total_time / 3600),
            'success_rate': self.generated_count / (self.generated_count + self.failed_count),
            'average_quality_score': np.mean(self.stats['quality_scores']) if self.stats['quality_scores'] else 0,
            'config': self.config
        }
        
        # Save final statistics
        with open(f"{self.config['output_dir']}/final_stats.json", 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        logger.info(f"🎉 GENERATION COMPLETE!")
        logger.info(f"   Generated: {self.generated_count} images")
        logger.info(f"   Failed: {self.failed_count} images")
        logger.info(f"   Total time: {total_time/3600:.1f} hours")
        logger.info(f"   Speed: {self.generated_count/(total_time/3600):.1f} images/hour")
        logger.info(f"   Success rate: {final_stats['success_rate']*100:.1f}%")
        
        return final_stats

def main():
    """Main function with command line arguments"""
    
    parser = argparse.ArgumentParser(description='Generate 10K rural driving images')
    parser.add_argument('--config', type=str, help='Path to JSON config file')
    parser.add_argument('--target', type=int, default=10000, help='Target number of images')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for generation')
    parser.add_argument('--output-dir', type=str, default='rural_driving_10k', help='Output directory')
    parser.add_argument('--width', type=int, default=1024, help='Image width')
    parser.add_argument('--height', type=int, default=1024, help='Image height')
    parser.add_argument('--inference-steps', type=int, default=20, help='Number of inference steps')
    parser.add_argument('--guidance-scale', type=float, default=6.5, help='Guidance scale')
    parser.add_argument('--max-runtime', type=int, default=72, help='Maximum runtime in hours')
    parser.add_argument('--save-interval', type=int, default=100, help='Save progress every N images')
    parser.add_argument('--chunk-id', type=int, help='Generate specific chunk (1-10)')
    
    args = parser.parse_args()
    
    # Configuration - either from config file or command line args
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config_data = json.load(f)
                
            # Extract generation config from the JSON
            gen_config = config_data.get('generation_config', {})
            output_config = config_data.get('output_config', {})
            perf_config = config_data.get('performance_config', {})
            
            # Create config dictionary from JSON
            config = {
                'target_images': gen_config.get('target_images', 10000),
                'batch_size': gen_config.get('batch_size', 4),
                'output_dir': output_config.get('base_output_dir', 'rural_driving_10k'),
                'width': gen_config.get('width', 1024),
                'height': gen_config.get('height', 1024),
                'inference_steps': gen_config.get('inference_steps', 30),
                'guidance_scale': gen_config.get('guidance_scale', 7.5),
                'max_runtime_hours': gen_config.get('max_runtime_hours', 72),
                'save_interval': gen_config.get('save_interval', 100),
                'use_half_precision': perf_config.get('use_half_precision', False)
            }
            
            # Apply chunk logic if specified
            if args.chunk_id and args.chunk_id > 0:
                chunk_size = config_data.get('recommendations', {}).get('chunk_size', 1000)
                config['target_images'] = min(chunk_size, config['target_images'])
                config['output_dir'] = f"{config['output_dir']}/chunk_{args.chunk_id}"
                
            logger.info(f"Loaded configuration from {args.config}")
                
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            raise
    else:
        # Use command line arguments
        config = {
            'target_images': args.target,
            'batch_size': args.batch_size,
            'output_dir': args.output_dir,
            'width': args.width,
            'height': args.height,
            'inference_steps': args.inference_steps,
            'guidance_scale': args.guidance_scale,
            'max_runtime_hours': args.max_runtime,
            'save_interval': args.save_interval
        }
    
    # Create generator and run
    generator = LargeScaleImageGenerator(config)
    
    try:
        final_stats = generator.generate_10k_images()
        
        print(f"\n SUCCESS! Generated {final_stats['total_generated']} images")
        print(f" Output directory: {config['output_dir']}")
        print(f" Total time: {final_stats['total_time_hours']:.1f} hours")
        print(f" Speed: {final_stats['images_per_hour']:.1f} images/hour")
        
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
        generator._save_progress()
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        generator._save_progress()
        raise

if __name__ == "__main__":
    main()
