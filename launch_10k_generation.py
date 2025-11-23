#!/usr/bin/env python3
"""
Launch Script for 10K Image Generation
Provides easy configuration and monitoring for 10K rural driving image generation
"""

import os
import sys
import json
import argparse
import subprocess
import time
from datetime import datetime, timedelta
import psutil
import torch

def check_system_requirements():
    """Check if system meets requirements for 10K generation"""
    
    print(" SYSTEM REQUIREMENTS CHECK")
    print("=" * 40)
    
    requirements_met = True
    
    # Check GPU
    if not torch.cuda.is_available():
        print(" CUDA not available - GPU required")
        requirements_met = False
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f" GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        
        if gpu_memory < 12:
            print(" Warning: Less than 12GB VRAM may cause issues")
    
    # Check RAM
    ram_gb = psutil.virtual_memory().total / 1024**3
    print(f" RAM: {ram_gb:.1f}GB")
    
    if ram_gb < 16:
        print(" Warning: Less than 16GB RAM may cause issues")
    
    # Check disk space
    disk_usage = psutil.disk_usage('.')
    free_space_gb = disk_usage.free / 1024**3
    print(f" Free disk space: {free_space_gb:.1f}GB")
    
    if free_space_gb < 50:
        print(" Warning: Less than 50GB free space may not be enough")
    
    return requirements_met

def estimate_generation_time():
    """Estimate generation time based on hardware"""
    
    print("\n GENERATION TIME ESTIMATES")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        print(" Cannot estimate - no GPU detected")
        return
    
    gpu_name = torch.cuda.get_device_name(0).lower()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    # Hardware-based estimates (images per hour)
    estimates = {
        'rtx 4090': 1440,
        'rtx 4080': 1200,
        'rtx 3090': 1030,
        'rtx 3080': 800,
        'a5000': 1200,
        'a6000': 1290,
        'a100': 1500
    }
    
    # Find matching GPU
    speed = 600  # default conservative estimate
    for gpu_type, gpu_speed in estimates.items():
        if gpu_type in gpu_name:
            speed = gpu_speed
            break
    
    # Adjust for memory
    if gpu_memory < 16:
        speed = int(speed * 0.7)  # Reduce speed for lower memory
    elif gpu_memory >= 24:
        speed = int(speed * 1.1)  # Boost for high memory
    
    target_images = 10000
    hours = target_images / speed
    
    print(f" Target: {target_images:,} images")
    print(f" Estimated speed: {speed:,} images/hour")
    print(f" Estimated time: {hours:.1f} hours ({hours/24:.1f} days)")
    print(f" Storage needed: ~{target_images * 3 / 1000:.0f}GB")
    
    return hours

def load_config(config_path="config_10k.json"):
    """Load configuration file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f" Config file not found: {config_path}")
        return None
    except json.JSONDecodeError:
        print(f" Invalid JSON in config file: {config_path}")
        return None

def run_generation(chunk_id=None, config_path="config_10k.json"):
    """Run the image generation"""
    
    config = load_config(config_path)
    if not config:
        return False
    
    print(f"\n STARTING 10K IMAGE GENERATION")
    print("=" * 40)
    
    # Build command
    cmd = [sys.executable, "generate_10k_images.py"]
    
    if chunk_id:
        cmd.extend(["--chunk-id", str(chunk_id)])
    
    cmd.extend(["--config", config_path])
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run generation
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output
        for line in process.stdout:
            print(line.rstrip())
        
        process.wait()
        
        if process.returncode == 0:
            print(f"\n Generation completed successfully!")
            return True
        else:
            print(f"\n Generation failed with return code: {process.returncode}")
            return False
            
    except KeyboardInterrupt:
        print(f"\n Generation interrupted by user")
        process.terminate()
        return False
    except Exception as e:
        print(f"\n Error running generation: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Launch 10K Rural Driving Image Generation")
    parser.add_argument("--check-only", action="store_true", 
                       help="Only check system requirements")
    parser.add_argument("--estimate-only", action="store_true",
                       help="Only show time estimates")
    parser.add_argument("--chunk-id", type=int,
                       help="Generate specific chunk (1-10)")
    parser.add_argument("--config", default="config_10k.json",
                       help="Configuration file path")
    
    args = parser.parse_args()
    
    print(" 10K Rural Driving Image Generator")
    print("=" * 50)
    
    # Check system requirements
    if not check_system_requirements():
        print("\n System requirements not met - performance may be severely limited!")
        print("Generation will proceed but may be extremely slow without CUDA GPU.")
        
        response = input("\nProceed anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("Generation cancelled.")
            return 0
    
    if args.check_only:
        print("\n System check complete!")
        return 0
    
    # Show time estimates
    estimate_generation_time()
    
    if args.estimate_only:
        print("\n Time estimation complete!")
        return 0
    
    # Confirm before starting
    if not args.chunk_id:
        print(f"\n About to generate 10,000 images")
        print("This will take several hours and use significant resources.")
        
        response = input("\nProceed? (y/N): ").strip().lower()
        if response != 'y':
            print("Generation cancelled.")
            return 0
    
    # Run generation
    success = run_generation(args.chunk_id, args.config)
    
    if success:
        print(f"\n 10K image generation completed!")
        return 0
    else:
        print(f"\n Generation failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
