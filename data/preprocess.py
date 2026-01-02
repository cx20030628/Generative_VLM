"""
PathVQA data preprocessing module.
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer


class PathVQAPreprocessor:

    def __init__(self, config):

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # UMLS client
        # self.umls_client = umls.UMLSClient(api_key=config.get('umls_api_key'))

    def preprocess_image(self, image_path, output_size=(224, 224)):

        # read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # translate color space BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # half-precision
        img = self.deblur_image(img)

        # multi-scale-patching
        patches = self.multi_scale_patching(img, output_size)

        return patches

    def deblur_image(self, img):

        blurred = cv2.GaussianBlur(img, (5, 5), 0)

        deblurred = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

        return deblurred

    def multi_scale_patching(self, img, output_size=(224, 224)):

        height, width = img.shape[:2]
        patch_size = output_size[0]
        stride = patch_size // 2  # 50%重叠

        patches = []

        for i in range(0, height - patch_size + 1, stride):
            for j in range(0, width - patch_size + 1, stride):
                patch = img[i:i + patch_size, j:j + patch_size]
                patch = cv2.resize(patch, output_size)
                patches.append(patch)

        return np.array(patches)

    def preprocess_text(self, question, answer):

        # UMLS knowledge alignment
        question = self.align_with_umls(question)
        answer = self.align_with_umls(answer)

        # generate rationale
        rationale = self.generate_rationale(question, answer)

        return {
            "question": question,
            "answer": answer,
            "rationale": rationale
        }

    def align_with_umls(self, text):

        # simplified UMLS alignment
        # actual application should utilize UMLS API
        umls_mapping = {
            "cancer cell": "Neoplastic Cell",
            "malignant": "Malignant Neoplasm",
            "benign": "Benign Neoplasm",
            "tumor": "Neoplasm",
            "inflammation": "Inflammatory Process",
            "lymphocyte": "Lymphoid Cell",
            "epithelial": "Epithelial Cell",
            "nucleus": "Cell Nucleus"
        }

        for key, value in umls_mapping.items():
            if key in text.lower():
                text = text.lower().replace(key, value)

        return text

    def generate_rationale(self, question, answer):

        # 6-steps-logic framework
        steps = [
            "Identify key morphological features in images",
            "Locate the regions of interest in pathological slides",
            "Assessing Malignant Risk Based on Visual Features",
            "Consider differential diagnosis",
            "Assess evidence of metastasis or invasion",
            "Draw clinical reasoning conclusions based on standard guidelines"
        ]

        # according to  question type, select relevant steps
        if "location" in question.lower() or "where" in question.lower():
            rationale = "Based on the 6-step structured medical logic: " + ", ".join(steps[1:3])
        elif "malignancy" in question.lower() or "cancer" in question.lower():
            rationale = "Based on the 6-step structured medical logic: " + ", ".join(steps[2:5])
        elif "type" in question.lower() or "what" in question.lower():
            rationale = "Based on the 6-step structured medical logic: " + ", ".join(steps[0:4])
        else:
            rationale = "Based on the 6-step structured medical logic: " + ", ".join(steps)

        return rationale

    def process_dataset(self, input_dir, output_dir):

        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'text'), exist_ok=True)

        metadata_path = os.path.join(input_dir, 'metadata.csv')
        if not os.path.exists(metadata_path):
            print(f"didn't find metadata.csv, skip the processing.")
            return

        metadata = pd.read_csv(metadata_path)

        for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="processing PathVQA dataset"):
            try:

                image_path = os.path.join(input_dir, 'images', row['image_id'] + '.png')
                if os.path.exists(image_path):
                    patches = self.preprocess_image(image_path)

                    np.save(
                        os.path.join(output_dir, 'images', row['image_id'] + '.npy'),
                        patches
                    )

                processed_text = self.preprocess_text(
                    row['question'],
                    row['answer']
                )

                with open(
                        os.path.join(output_dir, 'text', row['image_id'] + '.txt'),
                        'w',
                        encoding='utf-8'
                ) as f:
                    f.write(f"Question: {processed_text['question']}\n")
                    f.write(f"Answer: {processed_text['answer']}\n")
                    f.write(f"Rationale: {processed_text['rationale']}\n")

            except Exception as e:
                print(f"❌ 处理 {row['image_id']} 时出错: {e}")

if __name__ == "__main__":
    config = {
        'umls_api_key': 'your_api_key_here'
    }

    preprocessor = PathVQAPreprocessor(config)

    input_data_dir = 'data/raw'
    output_data_dir = 'data/processed'

    print(f"Start preprocessing, input directory: {input_data_dir}")

    preprocessor.process_dataset(input_data_dir, output_data_dir)

    print("Preprocess complete！Please check data/processed/images has .npy files.")