from datasets import load_dataset
import os
import pandas as pd
from huggingface_hub import snapshot_download

token = os.getenv("HF_TOKEN", "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")

# 0. downloading dataset

print("Start downloading...")
try:
    # Deleted trust_remote_code，added storage_options for reducing time errors
    dataset = load_dataset(
        "flaviagiammarino/path-vqa",
        token=token
    )
    print("Success!")
    print(dataset)
except Exception as e:
    print(f"Error: {e}")

# 1. creating necessary directories
os.makedirs('data/raw/images', exist_ok=True)

print("Downloading PathVQA from HuggingFace...")
# 2. loading dataset
dataset = load_dataset("flaviagiammarino/path-vqa")

# pre-processing train data
train_data = dataset['train']

metadata_list = []

print("Loading the images and generating indexes ...")
# 3. iterating through the dataset and saving images
for i, item in enumerate(train_data):
    img_id = f"pathvqa_{i}"
    img_path = f"data/raw/images/{img_id}.png"

    # save images
    img = item['image']
    # deal with transparency limit issues
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGBA")
    else:
        img = img.convert("RGB")

    img.save(img_path)

    # record to list
    metadata_list.append({
        'image_id': img_id,
        'question': item['question'],
        'answer': item['answer']
    })

    if i % 50 == 0:
        print(f" {i} images processed...")

# 4. generate metadata.csv required by later steps
df = pd.DataFrame(metadata_list)
df.to_csv('data/raw/metadata.csv', index=False)

print(f"Success！Images are saved to data/raw/images，indexes are saved to data/raw/metadata.csv")