"""This script de-duplicates the data provided by the PathVQA authors,
creates an "imagefolder" dataset and pushes it to the Hugging Face Hub.
"""

import re
import os
import shutil
import pickle
import datasets
import pandas as pd

for split in ["train", "val", "test"]:
    
    os.makedirs(f"data/{split}/", exist_ok=True)
    
    # load the image-question-answer triplets
    data = pd.DataFrame(pickle.load(open(f"pvqa/qas/{split}/{split}_qa.pkl", "rb")))

    # drop the duplicate image-question-answer triplets
    data = data.drop_duplicates(ignore_index=True)

    # perform some basic data cleaning/normalization
    f = lambda x: re.sub(' +', ' ', str(x).lower()).replace(" ?", "?").strip()
    data["question"] = data["question"].apply(f)
    data["answer"] = data["answer"].apply(f)
    
    # copy the images using unique file names
    data.insert(0, "file_name", "")
    for i, row in data.iterrows():
        file_name = f"img_{i}.jpg"
        data["file_name"].iloc[i] = file_name
        shutil.copyfile(src=f"pvqa/images/{split}/{row['image']}.jpg", dst=f"data/{split}/{file_name}")
    _ = data.pop("image")
    
    # save the metadata
    data.to_csv(f"data/{split}/metadata.csv", index=False)

# push the dataset to the hub
dataset = datasets.load_dataset("imagefolder", data_dir="data/")
dataset.push_to_hub("flaviagiammarino/path-vqa")
