# install_final.py
import os
import subprocess
import sys


def run_command(cmd, description):
    print(f"\n📦 {description}")
    print(f"executed command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print("Successful")
        return True
    else:
        print(f"Failure: {result.stderr[:200]}")
        return False


def main():
    print("=" * 70)
    print("MEDVQA CUDA PyTorch final installation script")
    print("=" * 70)

    # Verify conda environment
    env_path = sys.prefix
    if "conda" in env_path or "envs" in env_path:
        print(f"Current environment:{env_path}")
    else:
        print("⚠ Warning: It is recommended to run this script in a conda environment.")

    # 1. clear current install
    print("\n1. Clear current install...")
    run_command("pip uninstall torch torchvision torchaudio -y", "uninstall current PyTorch")
    run_command("pip cache purge", "clear pip cache")

    # 2. Attempt CUDA PyTorch install
    print("\n2. Attempt install CUDA PyTorch in conda...")
    success = run_command(
        "conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 cudatoolkit=11.8 -c pytorch -c conda-forge -y",
        "conda install PyTorch CUDA"
    )

    if not success:
        print("\n conda install failed, attempt pip installing...")
        # 3. install cudatoolkit
        run_command("conda install cudatoolkit=11.8 -c conda-forge -y", "安装cudatoolkit")

        # 4. pip install CUDA PyTorch
        success = run_command(
            "pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118",
            "pip installing PyTorch CUDA"
        )

    # 5. verify installation
    print("\n3. Verify installation...")
    try:
        import torch
        print(f"Pytorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print("\nCUDA PyTorch successfully installed!")
        else:
            print("\nInstalled CPU-only PyTorch.")
            print("Possible reasons:")
            print("1. without NVIDIA GPU")
            print("2. The graphics card driver is uninstalled or outdated")
            print("3. CUDA Toolkit is not properly installed")

    except Exception as e:
        print(f"Import error: {e}")

    # 6. 安装其他依赖
    print("\n4. Install other dependencies...")
    dependencies = [
        "transformers==4.30.0",
        "sentence-transformers==2.2.2",
        "datasets==2.13.1",
        "accelerate==0.21.0",
        "pytorch-lightning==1.9.5",
        "numpy==1.24.3",
        "pandas==2.0.3",
        "scipy==1.10.1",
        "scikit-learn==1.3.0",
        "tqdm==4.65.0",
        "pyyaml==6.0",
        "matplotlib==3.7.2",
        "seaborn==0.12.2",
        "nltk==3.8.1",
        "pillow==9.5.0"
    ]

    for dep in dependencies:
        run_command(f"pip install {dep}", f"安装 {dep.split('==')[0]}")

    print("\n" + "=" * 70)
    print("Install Successfully！")

    # 最终验证
    print("\nFinal Verification:")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print("Environment supported for CUDA training.")
            print(f"Estimated training time: 12-24 hours")
        else:
            print("Training with CPU, which may be very slow.")
            print(f"Estimated training time: 5-10 days")
    except:
        print("Pytorch import failed during final verification.")

    print("\nNext step:")
    print("1. Start training: python experiments/optimized_train.py")
    print("2. If CUDA unavailable, consider using Google Colab")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()