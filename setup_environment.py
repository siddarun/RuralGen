#!/usr/bin/env python3
"""
Environment setup script for 10K image generation
Handles PyTorch compatibility and dependency installation
"""

import subprocess
import sys
import platform
import os

def run_command(cmd, description=""):
    """Run a command and handle errors"""
    print(f" {description}")
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        print(" Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f" Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f" Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(" Python 3.8+ required")
        return False
    
    print(" Python version compatible")
    return True

def detect_system():
    """Detect system and recommend PyTorch installation"""
    system = platform.system()
    machine = platform.machine()
    
    print(f" System: {system} {machine}")
    
    # Check for CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f" CUDA available: {torch.version.cuda}")
            print(f" GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(" CUDA not available - will use CPU (very slow)")
    except ImportError:
        print(" PyTorch not installed yet")
    
    return system, machine

def install_pytorch():
    """Install PyTorch with proper CUDA support"""
    system, machine = detect_system()
    
    print("\n Installing PyTorch...")
    
    # For macOS (no CUDA)
    if system == "Darwin":
        cmd = "pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0"
        print(" Installing PyTorch for macOS (CPU/MPS)")
    
    # For Linux/Windows with CUDA
    else:
        # Try CUDA 11.8 first (most compatible)
        cmd = "pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118"
        print(" Installing PyTorch with CUDA 11.8 support")
    
    return run_command(cmd, "Installing PyTorch")

def install_diffusion_libraries():
    """Install diffusion and ML libraries"""
    print("\n Installing Diffusion libraries...")
    
    libraries = [
        "diffusers==0.24.0",
        "transformers==4.35.0", 
        "accelerate==0.24.0",
        "safetensors==0.4.0",
        "huggingface-hub==0.17.0"
    ]
    
    for lib in libraries:
        if not run_command(f"pip install {lib}", f"Installing {lib}"):
            return False
    
    return True

def install_utilities():
    """Install utility libraries"""
    print("\n Installing utilities...")
    
    utilities = [
        "pillow==10.0.1",
        "numpy==1.24.3",
        "psutil==5.9.6",
        "matplotlib==3.7.2",
        "tqdm==4.66.1"
    ]
    
    for util in utilities:
        if not run_command(f"pip install {util}", f"Installing {util}"):
            return False
    
    return True

def try_install_xformers():
    """Try to install xformers for performance boost"""
    print("\n Attempting to install xformers for performance...")
    
    # xformers can be tricky to install, so make it optional
    success = run_command("pip install xformers==0.0.22", "Installing xformers")
    
    if not success:
        print(" xformers installation failed - continuing without it")
        print("   (This is optional and won't prevent generation)")
    
    return True  # Always return True since it's optional

def verify_installation():
    """Verify all components are working"""
    print("\n Verifying installation...")
    
    tests = [
        ("import torch; print(f'PyTorch: {torch.__version__}')", "PyTorch"),
        ("import diffusers; print(f'Diffusers: {diffusers.__version__}')", "Diffusers"),
        ("import transformers; print(f'Transformers: {transformers.__version__}')", "Transformers"),
        ("import PIL; print(f'Pillow: {PIL.__version__}')", "Pillow"),
        ("import numpy; print(f'NumPy: {numpy.__version__}')", "NumPy")
    ]
    
    all_good = True
    for test_code, name in tests:
        try:
            result = subprocess.run([sys.executable, "-c", test_code], 
                                  capture_output=True, text=True, check=True)
            print(f" {result.stdout.strip()}")
        except subprocess.CalledProcessError:
            print(f" {name} not working properly")
            all_good = False
    
    # Test CUDA if available
    try:
        result = subprocess.run([sys.executable, "-c", 
                               "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"], 
                              capture_output=True, text=True, check=True)
        print(f" {result.stdout.strip()}")
    except:
        print(" Could not check CUDA status")
    
    return all_good

def main():
    print(" 10K Image Generation - Environment Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Detect system
    detect_system()
    
    # Install components step by step
    steps = [
        (install_pytorch, "PyTorch installation"),
        (install_diffusion_libraries, "Diffusion libraries"),
        (install_utilities, "Utility libraries"),
        (try_install_xformers, "Performance libraries"),
        (verify_installation, "Installation verification")
    ]
    
    for step_func, step_name in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        if not step_func():
            print(f"\n Failed at: {step_name}")
            print("Please check the error messages above and try again.")
            return 1
    
    print("\n Environment setup complete!")
    print("\nNext steps:")
    print("1. python launch_10k_generation.py --check-only")
    print("2. python launch_10k_generation.py --estimate-only") 
    print("3. python launch_10k_generation.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
