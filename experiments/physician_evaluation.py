from PIL import Image
import torch
import os
import pandas as pd
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rationale_vlm import GenerativeRationaleVLM
from data.preprocess import PathVQAPreprocessor
from metrics.rqr import RationaleQuestionRelevance
from metrics.clc import ClinicalLogicalConsistency
from experiments.infer import load_trained_model

def show_activation_map(img, student_evidence_map):
    """
    Implementation of the missing function.
    Adjust this based on how your evidence map is structured.
    """
    plt.imshow(img)
    plt.imshow(student_evidence_map, alpha=0.5, cmap='jet')
    plt.show()

def double_blind_eval(sample_id, img, teacher_rationale, student_evidence_map):
    """
    Compact function for double-blind evaluation of teacher and student explanations.
    """
    print(f"Sample ID: {sample_id}")
    print(f"[Teacher Rationale]: {teacher_rationale}")
    # Show student model's activation map (Evidence Tracing)
    show_activation_map(img, student_evidence_map)

    score_t = input("Rate Teacher's Explainability (1-5): ")
    score_s = input("Rate Student's Explainability (1-5): ")
    return score_t, score_s


def generate_physician_report(data_dir, output_report_path='physician_evaluation_report.csv'):

    model, config, device = load_trained_model()

    image_dir = os.path.join(data_dir, 'images')
    text_dir = os.path.join(data_dir, 'text')
    image_ids = [f.replace('.npy', '') for f in os.listdir(image_dir) if f.endswith('.npy')]

    rqr_calculator = RationaleQuestionRelevance()
    clc_calculator = ClinicalLogicalConsistency()
    preprocessor = PathVQAPreprocessor(config)

    report_data = []

    print(f"Start generating physician evaluation report for Pathology VQA dataset...({len(image_ids)} cases)")
    for img_id in tqdm(image_ids, desc="processing cases"):
        try:
            image_path = os.path.join('data/raw/images', f"{img_id}.png")
            text_path = os.path.join(text_dir, f"{img_id}.txt")

            with open(text_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                question = lines[0].replace("Question: ", "").strip()
                true_answer = lines[1].replace("Answer: ", "").strip()

            image_patches = preprocessor.preprocess_image(image_path)
            pixel_values = torch.from_numpy(image_patches).float().permute(0, 3, 1, 2)
            pixel_values = pixel_values[0:1].to(device)

            with torch.no_grad():
                inputs = model.processor(text=question, return_tensors="pt", padding=True)
                input_ids = inputs['input_ids'].to(device)

                # FIXED: Used underscores for unused variables to clear IDE warnings
                _, _ = model(pixel_values, input_ids)

                result = model.generate_answer(Image.open(image_path), question)

            answer_map = {0: "Benign", 1: "Malignant"}
            model_answer = answer_map[result['answer']]
            confidence = round(result['confidence'] * 100, 2)

            rqr_score = rqr_calculator.calculate(result['rationale'], question)
            clc_score = clc_calculator.calculate(result['rationale'], model_answer)

            report_data.append({
                'image_id': img_id,
                'question': question,
                'true_answer': true_answer,
                'model_answer': model_answer,
                'confidence': confidence,
                'rationale': result['rationale'],
                'rqr_score': round(rqr_score, 4),
                'clc_score': round(clc_score, 4),
                'physician_relevance_score': '',
                'physician_logic_score': '',
                'physician_accuracy_score': '',
                'physician_comments': ''
            })
        except Exception as e:
            print(f"Processing {img_id} error: {e}")
            continue

    report_df = pd.DataFrame(report_data)
    report_df.to_csv(output_report_path, index=False, encoding='utf-8-sig')
    print(f"Physician evaluation report saved: {output_report_path}")
    return report_df


def analyze_physician_scores(report_path='physician_evaluation_report.csv', output_analysis_path='score_analysis.csv'):

    report_df = pd.read_csv(report_path, encoding='utf-8-sig')

    score_cols = ['physician_relevance_score', 'physician_logic_score', 'physician_accuracy_score']
    for col in score_cols:
        if report_df[col].isna().all() or (report_df[col] == '').all():
            raise ValueError(f"Please fill in the{col}physician scores, 1-5 scale.")

    for col in score_cols:
        report_df[col] = pd.to_numeric(report_df[col], errors='coerce')

    correlation_data = {
        'Model metrics': ['RQR', 'CLC', 'Model Confidence'],
        'Relevant metrics on Physician Relevance': [
            report_df['rqr_score'].corr(report_df['physician_relevance_score']),
            report_df['clc_score'].corr(report_df['physician_relevance_score']),
            report_df['confidence'].corr(report_df['physician_relevance_score'])
        ],
        'Relevant metrics on Physician Logic': [
            report_df['rqr_score'].corr(report_df['physician_logic_score']),
            report_df['clc_score'].corr(report_df['physician_logic_score']),
            report_df['confidence'].corr(report_df['physician_logic_score'])
        ],
        'Relevant metrics on Physician accuracy': [
            report_df['rqr_score'].corr(report_df['physician_accuracy_score']),
            report_df['clc_score'].corr(report_df['physician_accuracy_score']),
            report_df['confidence'].corr(report_df['physician_accuracy_score'])
        ]
    }

    analysis_df = pd.DataFrame(correlation_data)
    analysis_df = analysis_df.round(4)
    analysis_df.to_csv(output_analysis_path, index=False, encoding='utf-8-sig')

    print(f"Relevance Score correlation matrix: {output_analysis_path}")
    print("\nRelevance Analysis Abstract:")
    print(analysis_df.to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Physician Evaluation for Pathology VQA')
    parser.add_argument('--mode', required=True, choices=['generate_report', 'analyze_scores'],
                        help='Running mode: generate_report/ analyze_scores')
    parser.add_argument('--data_dir', default='data/processed', help='Preprocess data directory')
    parser.add_argument('--report_path', default='physician_evaluation_report.csv', help='Path to the physician evaluation report CSV file')
    args = parser.parse_args()

    if args.mode == 'generate_report':
        generate_physician_report(args.data_dir, args.report_path)
    elif args.mode == 'analyze_scores':
        analyze_physician_scores(args.report_path)