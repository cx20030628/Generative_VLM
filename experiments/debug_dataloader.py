"""
Debug script to identify and fix data loading bottlenecks
Run this before training to optimize your setup
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import time
import psutil
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import GPUtil, but don't fail if not available
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("GPUtil not installed. GPU monitoring disabled. Install with: pip install gputil")

# Mock dataset for testing
class MockDataset(Dataset):
    def __init__(self, num_samples=1000, image_shape=(3, 224, 224), seq_length=512):
        self.num_samples = num_samples
        self.image_shape = image_shape
        self.seq_length = seq_length

    def __getitem__(self, idx):
        # Simulate image loading
        image = torch.randn(self.image_shape)

        # Simulate text (token IDs)
        text = torch.randint(0, 10000, (self.seq_length,))

        return {
            'pixel_values': image,
            'input_ids': text,
            'attention_mask': (text != 0).long()
        }

    def __len__(self):
        return self.num_samples

def benchmark_dataloader(config, num_workers_list=None):
    """Benchmark DataLoader with different number of workers"""
    print("\n" + "="*60)
    print("DATALOADER BENCHMARKING")
    print("="*60)

    if num_workers_list is None:
        num_workers_list = [0, 2, 4, 8, 12]

    dataset = MockDataset(num_samples=1000)

    results = {}

    for num_workers in num_workers_list:
        print(f"\nTesting with num_workers={num_workers}")

        dataloader = DataLoader(
            dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=config['data']['pin_memory'],
            prefetch_factor=config['data']['prefetch_factor'] if num_workers > 0 else None,
            persistent_workers=config['data']['persistent_workers'] if num_workers > 0 else False
        )

        # Warmup
        for i, batch in enumerate(dataloader):
            if i >= 5:
                break

        # Benchmark
        start_time = time.time()
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Benchmarking (workers={num_workers})")
        for batch in pbar:
            num_batches += 1
            if num_batches >= 50:  # Benchmark 50 batches
                break

        elapsed = time.time() - start_time
        batches_per_sec = num_batches / elapsed

        results[num_workers] = {
            'time_per_batch': elapsed / num_batches,
            'batches_per_sec': batches_per_sec,
            'samples_per_sec': batches_per_sec * config['training']['batch_size']
        }

        print(f"  Time per batch: {results[num_workers]['time_per_batch']:.4f}s")
        print(f"  Batches per second: {results[num_workers]['batches_per_sec']:.2f}")
        print(f"  Samples per second: {results[num_workers]['samples_per_sec']:.2f}")

    # Find optimal number of workers
    best_workers = max(results.items(), key=lambda x: x[1]['samples_per_sec'])
    print(f"\n✅ Optimal number of workers: {best_workers[0]}")
    print(f"   Samples/sec: {best_workers[1]['samples_per_sec']:.2f}")

    return results, best_workers[0]

def check_system_resources():
    """Check available system resources"""
    print("\n" + "="*60)
    print("SYSTEM RESOURCE CHECK")
    print("="*60)

    # CPU info
    cpu_count = psutil.cpu_count(logical=False)
    cpu_logical = psutil.cpu_count(logical=True)
    cpu_percent = psutil.cpu_percent(interval=1)

    print(f"CPU Cores (physical/logical): {cpu_count}/{cpu_logical}")
    print(f"CPU Usage: {cpu_percent:.1f}%")

    # Memory info
    memory = psutil.virtual_memory()
    print(f"RAM Total: {memory.total / 1e9:.2f} GB")
    print(f"RAM Available: {memory.available / 1e9:.2f} GB")
    print(f"RAM Used: {memory.percent}%")

    # GPU info
    gpu_info = None
    if GPU_AVAILABLE:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_info = {}
                for i, gpu in enumerate(gpus):
                    print(f"\nGPU {i}: {gpu.name}")
                    print(f"  Memory Total: {gpu.memoryTotal} MB")
                    print(f"  Memory Free: {gpu.memoryFree} MB")
                    print(f"  Memory Used: {gpu.memoryUsed} MB ({gpu.memoryUtil*100:.1f}%)")
                    print(f"  Temperature: {gpu.temperature}°C")
                    gpu_info[i] = {
                        'name': gpu.name,
                        'memory_total': gpu.memoryTotal,
                        'memory_used': gpu.memoryUsed
                    }
        except Exception as e:
            print(f"Error getting GPU info: {e}")
            gpu_info = None
    else:
        print("\nGPU monitoring disabled (GPUtil not installed)")
        print("To enable GPU monitoring, run: pip install gputil")

    # Disk speed (simplified)
    print("\nDisk check (writing test file)...")
    test_file = "disk_speed_test.bin"
    size_mb = 10  # Reduced from 100MB to 10MB for faster testing
    data = os.urandom(size_mb * 1024 * 1024)

    try:
        start = time.time()
        with open(test_file, 'wb') as f:
            f.write(data)
        write_time = time.time() - start

        if os.path.exists(test_file):
            os.remove(test_file)

        write_speed = size_mb / write_time
        print(f"Disk write speed: {write_speed:.2f} MB/s")
    except Exception as e:
        print(f"Disk speed test failed: {e}")
        write_speed = 0

    return {
        'cpu_cores': cpu_count,
        'ram_gb': memory.total / 1e9,
        'gpu_info': gpu_info
    }

def test_batch_transfer_speed(batch_size=32):
    """Test data transfer speed to GPU"""
    print("\n" + "="*60)
    print("GPU TRANSFER SPEED TEST")
    print("="*60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU tests")
        return None

    device = torch.device('cuda')

    # Test different tensor sizes
    test_cases = [
        ('Small', (batch_size, 3, 224, 224)),  # Images
        ('Medium', (batch_size, 512, 768)),    # Hidden states
        ('Large', (batch_size, 512, 50272)),   # Logits
    ]

    results = {}

    for name, shape in test_cases:
        # Create tensor on CPU
        tensor_cpu = torch.randn(*shape)

        # Test with pin_memory=True
        if not tensor_cpu.is_pinned():
            tensor_cpu = tensor_cpu.pin_memory()

        # Warmup
        for _ in range(5):
            tensor_gpu = tensor_cpu.to(device, non_blocking=True)
            torch.cuda.synchronize()

        # Benchmark
        torch.cuda.synchronize()
        start = time.time()

        num_transfers = 100
        for _ in range(num_transfers):
            tensor_gpu = tensor_cpu.to(device, non_blocking=True)

        torch.cuda.synchronize()
        elapsed = time.time() - start

        transfer_time = elapsed / num_transfers * 1000  # ms per transfer
        data_size_mb = tensor_cpu.element_size() * tensor_cpu.nelement() / 1e6
        throughput = data_size_mb / (elapsed / num_transfers)  # MB/s

        results[name] = {
            'transfer_time_ms': transfer_time,
            'throughput_mb_s': throughput,
            'data_size_mb': data_size_mb
        }

        print(f"{name} tensor ({data_size_mb:.1f} MB):")
        print(f"  Transfer time: {transfer_time:.2f} ms")
        print(f"  Throughput: {throughput:.1f} MB/s")

    return results

def check_file_system(data_dir="data/processed"):
    """Check file system performance"""
    print("\n" + "="*60)
    print("FILE SYSTEM CHECK")
    print("="*60)

    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist")
        return None

    # Count files
    image_dir = os.path.join(data_dir, "images")
    text_dir = os.path.join(data_dir, "text")

    npy_files = []
    txt_files = []

    if os.path.exists(image_dir):
        npy_files = [f for f in os.listdir(image_dir) if f.endswith('.npy')]
        print(f"Image files (.npy): {len(npy_files)}")

        # Check file sizes
        sizes = []
        for f in npy_files[:10]:  # Check first 10
            path = os.path.join(image_dir, f)
            if os.path.exists(path):
                sizes.append(os.path.getsize(path) / 1e6)  # MB

        if sizes:
            print(f"Sample file sizes: {min(sizes):.1f}-{max(sizes):.1f} MB")

    if os.path.exists(text_dir):
        txt_files = [f for f in os.listdir(text_dir) if f.endswith('.txt')]
        print(f"Text files (.txt): {len(txt_files)}")

    # Test read speed
    if npy_files and len(npy_files) > 0:
        test_file = os.path.join(image_dir, npy_files[0])
        if os.path.exists(test_file):
            print(f"\nTesting read speed for: {test_file}")

            try:
                start = time.time()
                data = np.load(test_file, mmap_mode='r')
                load_time = time.time() - start

                print(f"Memory-mapped load time: {load_time*1000:.2f} ms")
                print(f"Array shape: {data.shape}, dtype: {data.dtype}")

                # Test actual read
                start = time.time()
                _ = data[:10].copy() if data.shape[0] > 10 else data.copy()
                read_time = time.time() - start
                print(f"Small read time: {read_time*1000:.2f} ms")
            except Exception as e:
                print(f"Error reading file: {e}")
        else:
            print(f"Test file not found: {test_file}")

    return {
        'npy_files': len(npy_files),
        'txt_files': len(txt_files)
    }

def main():
    """Main debugging function"""
    print("="*60)
    print("MEDVQA TRAINING DEBUG UTILITY")
    print("="*60)

    # Load config
    config_path = 'experiments/config.yaml'
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print("✓ Config loaded")
        except Exception as e:
            print(f"Error loading config: {e}")
            config = {
                'training': {'batch_size': 32},
                'data': {
                    'pin_memory': True,
                    'persistent_workers': True,
                    'prefetch_factor': 2
                }
            }
    else:
        print("⚠ Config not found, using defaults")
        config = {
            'training': {'batch_size': 32},
            'data': {
                'pin_memory': True,
                'persistent_workers': True,
                'prefetch_factor': 2
            }
        }

    # Run checks
    try:
        resources = check_system_resources()
    except Exception as e:
        print(f"Error checking system resources: {e}")
        resources = {'cpu_cores': 4, 'ram_gb': 16}

    try:
        file_stats = check_file_system()
    except Exception as e:
        print(f"Error checking file system: {e}")
        file_stats = None

    try:
        gpu_results = test_batch_transfer_speed(config['training']['batch_size'])
    except Exception as e:
        print(f"Error testing GPU transfer: {e}")
        gpu_results = None

    # Determine optimal number of workers
    cpu_cores = resources.get('cpu_cores', 4)
    if cpu_cores >= 8:
        num_workers_options = [0, 4, 8, 12, 16]
    elif cpu_cores >= 4:
        num_workers_options = [0, 2, 4, 6, 8]
    else:
        num_workers_options = [0, 1, 2, 3]

    try:
        results, optimal_workers = benchmark_dataloader(config, num_workers_options)
    except Exception as e:
        print(f"Error benchmarking dataloader: {e}")
        results = {}
        optimal_workers = 4  # Default fallback

    # Generate recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    print("\n1. DataLoader Configuration:")
    print(f"   Set num_workers = {optimal_workers}")
    print(f"   Keep pin_memory = {config['data']['pin_memory']}")
    print(f"   Keep prefetch_factor = {config['data']['prefetch_factor']}")

    print("\n2. Batch Size Optimization:")
    gpu_info = resources.get('gpu_info')
    if gpu_info:
        for gpu_id, gpu_data in gpu_info.items():
            gpu_memory_mb = gpu_data['memory_total']
            if gpu_memory_mb >= 32000:  # 32GB GPU
                print(f"   ✓ GPU {gpu_id} ({gpu_data['name']}): 32GB, can handle larger batches")
                print("   → Try increasing batch_size to 64 if model fits")
            elif gpu_memory_mb >= 16000:  # 16GB GPU
                print(f"   ✓ GPU {gpu_id} ({gpu_data['name']}): Current batch_size=32 is appropriate")
            else:
                print(f"   ⚠ GPU {gpu_id} ({gpu_data['name']}): Consider reducing batch_size")
    else:
        print("   ⚠ GPU information not available")

    print("\n3. Performance Tips:")
    print("   • Enable mixed precision training (AMP)")
    print("   • Use gradient accumulation for larger effective batch size")
    print("   • Pre-process and cache your data")
    print("   • Use memory-mapped numpy arrays for large images")

    print("\n4. Expected Performance:")
    if results and optimal_workers in results:
        best_speed = results[optimal_workers]['samples_per_sec']
        # PathVQA has 32,799 samples
        samples_per_epoch = 32799 * 0.8  # Assuming 80% train split
        estimated_hours = samples_per_epoch / best_speed / 3600

        print(f"   Expected samples/sec: {best_speed:.1f}")
        print(f"   Estimated time per epoch: {estimated_hours:.2f} hours")
        print(f"   Estimated total training (10 epochs): {estimated_hours * 10:.2f} hours")
    else:
        print("   ⚠ Performance estimation not available")

    # Save recommendations to file
    try:
        with open('experiments/optimization_report.txt', 'w') as f:
            f.write("MEDVQA Optimization Report\n")
            f.write("="*50 + "\n\n")
            f.write(f"Optimal num_workers: {optimal_workers}\n")
            if results and optimal_workers in results:
                f.write(f"Expected samples/sec: {results[optimal_workers]['samples_per_sec']:.1f}\n")
            f.write(f"CPU Cores: {resources.get('cpu_cores', 'N/A')}\n")
            f.write(f"RAM: {resources.get('ram_gb', 'N/A'):.1f} GB\n")

        print(f"\n✓ Report saved to experiments/optimization_report.txt")
    except Exception as e:
        print(f"Error saving report: {e}")

    print("\nTo apply these optimizations, update your config.yaml:")
    print(f"data:\n  num_workers: {optimal_workers}")
    print("\nHappy training! 🚀")

if __name__ == "__main__":
    main()