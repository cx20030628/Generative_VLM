import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import yaml
import numpy as np
import torch.nn.functional as F
import sys
import time
from typing import Dict, List, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiments/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Check for empty files (run once)
def check_data_integrity(data_dir: str):
    """Check and report data integrity issues"""
    image_dir = os.path.join(data_dir, "images")
    text_dir = os.path.join(data_dir, "text")

    issues = []

    # Check image files
    for f in os.listdir(image_dir):
        if f.endswith('.npy'):
            full_path = os.path.join(image_dir, f)
            if os.path.getsize(full_path) == 0:
                issues.append(f"Empty image file: {f}")

    # Check text files
    for f in os.listdir(text_dir):
        if f.endswith('.txt'):
            full_path = os.path.join(text_dir, f)
            if os.path.getsize(full_path) == 0:
                issues.append(f"Empty text file: {f}")

    if issues:
        logger.warning(f"Found {len(issues)} data integrity issues")
        for issue in issues[:10]:  # Show first 10
            logger.warning(issue)
    else:
        logger.info("Data integrity check passed")

    return len(issues) == 0


class OptimizedPathVQADataset(Dataset):
    """
    Optimized dataset with:
    - Pre-loaded valid indices
    - Pre-tokenized text (optional)
    - Memory-mapped numpy loading
    """

    def __init__(self,
                 data_dir: str,
                 processor=None,
                 max_length: int = 512,
                 pre_tokenize: bool = True,
                 cache_dir: str = "data/cache"):
        """
        Args:
            data_dir: Directory containing images/ and text/ subdirectories
            processor: HuggingFace processor for tokenization
            max_length: Maximum sequence length
            pre_tokenize: Whether to pre-tokenize texts
            cache_dir: Directory to cache preprocessed data
        """
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "images")
        self.text_dir = os.path.join(data_dir, "text")
        self.max_length = max_length
        self.processor = processor
        self.pre_tokenize = pre_tokenize and processor is not None

        # Create cache directory
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Cache files
        self.cache_file = os.path.join(cache_dir, "dataset_cache.pt")

        # Try to load from cache
        if os.path.exists(self.cache_file):
            logger.info("Loading dataset from cache...")
            cache_data = torch.load(self.cache_file)
            self.valid_indices = cache_data['valid_indices']
            self.image_ids = cache_data['image_ids']
            self.texts = cache_data['texts']
            if self.pre_tokenize:
                self.input_ids = cache_data['input_ids']
                self.attention_masks = cache_data['attention_masks']
            else:
                self.input_ids = None
                self.attention_masks = None
            logger.info(f"Loaded {len(self.valid_indices)} samples from cache")
        else:
            self._build_dataset()
            self._save_to_cache()

    def _build_dataset(self):
        """Build dataset by scanning files and filtering valid ones"""
        logger.info("Building dataset...")

        # Get all .npy files
        all_files = [f for f in os.listdir(self.image_dir) if f.endswith('.npy')]
        logger.info(f"Found {len(all_files)} image files")

        self.valid_indices = []
        self.image_ids = []
        self.texts = []

        # Filter valid samples
        for i, f in enumerate(tqdm(all_files, desc="Scanning files")):
            img_id = f.replace('.npy', '')
            patch_path = os.path.join(self.image_dir, f)
            text_path = os.path.join(self.text_dir, f"{img_id}.txt")

            # Check if both files exist and are not empty
            if (os.path.exists(patch_path) and os.path.exists(text_path) and
                    os.path.getsize(patch_path) > 100 and os.path.getsize(text_path) > 0):

                # Read text
                try:
                    with open(text_path, 'r', encoding='utf-8') as tf:
                        text = tf.read().strip()

                    if text:  # Only add if text is not empty
                        self.valid_indices.append(i)
                        self.image_ids.append(img_id)
                        self.texts.append(text)
                except Exception as e:
                    logger.warning(f"Error reading {text_path}: {e}")
                    continue

        logger.info(f"Loaded {len(self.valid_indices)} valid samples out of {len(all_files)}")

        # Pre-tokenize if enabled
        if self.pre_tokenize:
            logger.info("Pre-tokenizing texts...")
            self.input_ids = []
            self.attention_masks = []

            # Batch tokenization for efficiency
            batch_size = 100
            for i in tqdm(range(0, len(self.texts), batch_size), desc="Tokenizing"):
                batch_texts = self.texts[i:i + batch_size]
                inputs = self.processor(
                    text=batch_texts,
                    return_tensors="pt",
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length
                )

                self.input_ids.extend(inputs['input_ids'])
                self.attention_masks.extend(inputs['attention_mask'])

            # Convert to tensors
            self.input_ids = torch.stack(self.input_ids)
            self.attention_masks = torch.stack(self.attention_masks)

            logger.info(f"Tokenization complete. Shape: {self.input_ids.shape}")
        else:
            self.input_ids = None
            self.attention_masks = None

    def _save_to_cache(self):
        """Save processed dataset to cache"""
        cache_data = {
            'valid_indices': self.valid_indices,
            'image_ids': self.image_ids,
            'texts': self.texts,
        }

        if self.pre_tokenize:
            cache_data['input_ids'] = self.input_ids
            cache_data['attention_masks'] = self.attention_masks

        torch.save(cache_data, self.cache_file)
        logger.info(f"Dataset cached to {self.cache_file}")

    def __getitem__(self, idx: int) -> Optional[Dict]:
        """
        Get a single sample.
        Returns None for corrupted samples to be filtered in collate_fn.
        """
        try:
            img_id = self.image_ids[idx]
            patch_path = os.path.join(self.image_dir, f"{img_id}.npy")

            # Load image with memory mapping for efficiency
            patches = np.load(patch_path, mmap_mode='r')
            pixel_values = torch.from_numpy(patches.copy()).float()

            # Handle different tensor dimensions
            if pixel_values.ndim == 4:
                # Shape: [num_patches, H, W, C] -> [C, H, W]
                pixel_values = pixel_values.permute(0, 3, 1, 2)[0]
            elif pixel_values.ndim == 3:
                # Shape: [H, W, C] -> [C, H, W]
                pixel_values = pixel_values.permute(2, 0, 1)
            else:
                logger.error(f"Invalid dimensions {pixel_values.ndim} for {img_id}")
                return None

            # Ensure proper shape (C, H, W)
            if pixel_values.shape[0] not in [1, 3]:
                if pixel_values.shape[-1] in [1, 3]:
                    pixel_values = pixel_values.permute(2, 0, 1)

            # Normalize to [0, 1] if needed
            if pixel_values.max() > 1.0:
                pixel_values = pixel_values / 255.0

            # Return based on tokenization mode
            if self.pre_tokenize:
                return {
                    'pixel_values': pixel_values,
                    'input_ids': self.input_ids[idx],
                    'attention_mask': self.attention_masks[idx],
                    'image_id': img_id
                }
            else:
                return {
                    'pixel_values': pixel_values,
                    'question': self.texts[idx],
                    'image_id': img_id
                }

        except Exception as e:
            logger.warning(
                f"Error loading sample {idx} ({img_id if 'img_id' in locals() else 'unknown'}): {str(e)[:100]}")
            return None

    def __len__(self) -> int:
        return len(self.valid_indices)


def optimized_collate_fn(batch: List[Optional[Dict]]) -> Dict:
    """
    Collate function that filters out None samples and handles batching.

    Args:
        batch: List of samples, some may be None

    Returns:
        Batched tensors
    """
    # Filter out None samples
    batch = [b for b in batch if b is not None]

    if len(batch) == 0:
        # Return empty batch to be skipped
        return {}

    # Check if using pre-tokenized data
    if 'input_ids' in batch[0]:
        # Pre-tokenized: stack tensors directly
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])

        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image_ids': [item.get('image_id', '') for item in batch]
        }
    else:
        # Not pre-tokenized: collect questions for later processing
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        questions = [item['question'] for item in batch]

        return {
            'pixel_values': pixel_values,
            'question': questions,
            'image_ids': [item.get('image_id', '') for item in batch]
        }


class TrainingMonitor:
    """Monitor training metrics and implement early stopping"""

    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True  # Improved
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False  # Not improved


def create_dataloader(dataset, config: Dict, is_train: bool = True) -> DataLoader:
    """Create optimized DataLoader"""
    return DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=is_train,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        prefetch_factor=config['data']['prefetch_factor'],
        persistent_workers=config['data']['persistent_workers'],
        collate_fn=optimized_collate_fn,
        drop_last=is_train  # Drop last incomplete batch during training
    )


def train_epoch(model, dataloader, optimizer, criterion, scaler, config, epoch: int):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_tokens = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    start_time = time.time()

    for batch_idx, batch in enumerate(pbar):
        # Skip empty batches
        if not batch:
            continue

        # Move to device with non-blocking transfer
        pixel_values = batch['pixel_values'].to('cuda', non_blocking=True)

        # Tokenize if not pre-tokenized
        if 'question' in batch:
            with torch.cuda.amp.autocast(enabled=False):
                inputs = model.processor(
                    text=batch['question'],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=config['model']['max_length']
                )
            input_ids = inputs['input_ids'].to('cuda', non_blocking=True)
            attention_mask = inputs['attention_mask'].to('cuda', non_blocking=True)
        else:
            input_ids = batch['input_ids'].to('cuda', non_blocking=True)
            attention_mask = batch['attention_mask'].to('cuda', non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()

        # Mixed precision forward pass
        with torch.cuda.amp.autocast(enabled=config['optimization']['use_amp']):
            outputs = model(pixel_values, input_ids, attention_mask=attention_mask)

            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            # Shift for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            # Flatten for loss calculation
            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        # Scale loss and backward pass
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config['optimization']['gradient_clip']
        )

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # Update metrics
        total_loss += loss.item()
        total_tokens += shift_labels.numel()

        # Update progress bar
        if batch_idx % 10 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            tokens_per_sec = total_tokens / (time.time() - start_time)
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'tokens/s': f'{tokens_per_sec:.0f}',
                'lr': optimizer.param_groups[0]['lr']
            })

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate_epoch(model, dataloader, criterion, config):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            if not batch:
                continue

            pixel_values = batch['pixel_values'].to('cuda', non_blocking=True)

            if 'question' in batch:
                inputs = model.processor(
                    text=batch['question'],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=config['model']['max_length']
                )
                input_ids = inputs['input_ids'].to('cuda', non_blocking=True)
                attention_mask = inputs['attention_mask'].to('cuda', non_blocking=True)
            else:
                input_ids = batch['input_ids'].to('cuda', non_blocking=True)
                attention_mask = batch['attention_mask'].to('cuda', non_blocking=True)

            with torch.cuda.amp.autocast(enabled=config['optimization']['use_amp']):
                outputs = model(pixel_values, input_ids, attention_mask=attention_mask)

                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()

                loss = criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

            total_loss += loss.item()

    return total_loss / len(dataloader)


def train():
    """Main training function"""
    logger.info("Starting optimized training...")

    start_time = time.time()

    # 1. Load configuration
    config_path = 'experiments/config.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
    else:
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # 2. Check data integrity
    data_dir = 'data/processed'
    if not check_data_integrity(data_dir):
        logger.warning("Data integrity issues found. Training may fail.")

    # 3. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 4. Initialize model
    try:
        from models.rationale_vlm import GenerativeRationaleVLM
        model = GenerativeRationaleVLM(config['model']).to(device)
        logger.info(f"Model initialized: {config['model']['name']}")
    except ImportError as e:
        logger.error(f"Failed to import model: {e}")
        raise

    # 5. Create datasets with pre-tokenization
    logger.info("Creating datasets...")

    # Use processor from model for pre-tokenization
    processor = model.processor if config['model']['use_pretokenize'] else None

    train_dataset = OptimizedPathVQADataset(
        data_dir=data_dir,
        processor=processor,
        max_length=config['model']['max_length'],
        pre_tokenize=config['model']['use_pretokenize']
    )

    # Split for validation (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    logger.info(f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}")

    # 6. Create dataloaders
    train_loader = create_dataloader(train_subset, config, is_train=True)
    val_loader = create_dataloader(val_subset, config, is_train=False)

    # 7. Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=0.01
    )

    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * config['training']['epochs']
    warmup_steps = config['training']['warmup_steps']

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 8. Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=model.processor.tokenizer.pad_token_id)

    # 9. Mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=config['optimization']['use_amp'])

    # 10. Training monitor for early stopping
    monitor = TrainingMonitor(
        patience=config['training'].get('early_stopping_patience', 5),
        min_delta=config['training'].get('early_stopping_min_delta', 0.001)
    )

    # 11. Training loop
    best_val_loss = float('inf')

    for epoch in range(1, config['training']['epochs'] + 1):
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Epoch {epoch}/{config['training']['epochs']}")
        logger.info(f"{'=' * 50}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, scaler, config, epoch
        )
        logger.info(f"Train Loss: {train_loss:.4f}")

        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, config)
        logger.info(f"Validation Loss: {val_loss:.4f}")

        # Learning rate scheduling
        scheduler.step()

        # Check early stopping
        if monitor(val_loss):
            logger.info(f"Validation loss improved to {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'config': config
                }, 'experiments/checkpoints/best_model.pt')
                logger.info("Best model saved")

        if monitor.early_stop:
            logger.info("Early stopping triggered")
            break

        # Save checkpoint every few epochs
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, f'experiments/checkpoints/checkpoint_epoch_{epoch}.pt')
            logger.info(f"Checkpoint saved for epoch {epoch}")

    # 12. Save final model
    torch.save(model.state_dict(), 'experiments/checkpoints/final_model.pt')
    logger.info("Training completed!")

    # 13. Log final stats
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Total training time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    # Set better default for multiprocessing on Windows
    if sys.platform == 'win32':
        import multiprocessing

        multiprocessing.freeze_support()

    # Set CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        train()
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise