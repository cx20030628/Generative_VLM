# verify_config.py
import yaml
import torch


def verify_config():
    with open('experiments/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("Verifying Config...")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Num workers: {config['data']['num_workers']}")
    print(f"Mixed precision: {config['optimization']['use_amp']}")

    # 测试batch能否放入显存
    if torch.cuda.is_available():
        # 模拟一个batch的内存使用
        dummy_image = torch.randn(config['training']['batch_size'], 3, 224, 224).cuda()
        dummy_text = torch.randint(0, 10000, (config['training']['batch_size'], 512)).cuda()

        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_cached = torch.cuda.memory_reserved() / 1e9

        print(f"\nTest batch memory usage:")
        print(f"Memory allocated: {memory_allocated:.2f} GB")
        print(f"Memory cached: {memory_cached:.2f} GB")
        print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        if memory_cached > 28:  # 32GB GPU留4GB余量
            print("Warning: batch_size may be too big, consider reducing batch_size")
        else:
            print("batch_size appropriate for current GPU.")

    return config


if __name__ == "__main__":
    verify_config()