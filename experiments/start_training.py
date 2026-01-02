# start_training.py
import os
import torch
import subprocess


def setup_environment():
    """设置最优训练环境"""
    # 设置CUDA优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 检查GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}, Memory: {memory_gb:.1f}GB")

        # 设置环境变量
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['OMP_NUM_THREADS'] = '4'  # 匹配num_workers

    print("Environment setup, start training...")


if __name__ == "__main__":
    setup_environment()

    # 启动训练
    subprocess.run(["python", "experiments/train.py"])