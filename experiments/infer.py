import torch
import os
import yaml
import numpy as np
from PIL import Image
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rationale_vlm import GenerativeRationaleVLM
from data.preprocess import PathVQAPreprocessor
from metrics.rio import RationaleImageOverlap
from metrics.rqr import RationaleQuestionRelevance
from metrics.clc import ClinicalLogicalConsistency
from metrics.robustness import RobustnessEvaluator


def load_trained_model(config_path='experiments/config.yaml', checkpoint_path=None):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'model_name': 'Salesforce/blip2-opt-2.7b',
            'num_steps': 6,
            'hidden_dim': 768,
            'num_classes': 2
        }

    model = GenerativeRationaleVLM(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if checkpoint_path is None:
        checkpoint_dir = 'experiments/checkpoints'
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError("Trained checkpoint not found. Please provide a valid checkpoint path.")
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        checkpoints.sort(key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
        checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"successfully loaded model from checkpoint: {checkpoint_path}")
    return model, config, device


def infer_single_case(image_path, question, model, config, device):
    preprocessor = PathVQAPreprocessor(config)
    image_patches = preprocessor.preprocess_image(image_path)
    pixel_values = torch.from_numpy(image_patches).float().permute(0, 3, 1, 2)
    pixel_values = pixel_values[0:1].to(device)  # [1, C, H, W]

    with torch.no_grad():
        inputs = model.processor(text=question, return_tensors="pt", padding=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs.get('attention_mask', None).to(device)

        logits, rationale = model(pixel_values, input_ids, attention_mask)
        result = model.generate_answer(Image.open(image_path), question)

    answer_map = {0: "Benign", 1: "Malignant"}
    result['answer_text'] = answer_map[result['answer']]
    result['confidence'] = round(result['confidence'] * 100, 2)

    rqr_calculator = RationaleQuestionRelevance()
    rqr_score = rqr_calculator.calculate(result['rationale'], question)

    clc_calculator = ClinicalLogicalConsistency()
    clc_score = clc_calculator.calculate(result['rationale'], result['answer_text'])

    result.update({
        'rqr_score': round(rqr_score, 4),
        'clc_score': round(clc_score, 4),
        'question': question
    })

    return result


def print_result(result):
    print("\n" + "=" * 50)
    print("Medical Visual Question Answering Inference Result")
    print("=" * 50)
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer_text']} (Confidence: {result['confidence']}%)")
    print(f"Rationale: {result['rationale']}")
    print(f"Metrics:")
    print(f"Reason Question Relevance（RQR）: {result['rqr_score']}")
    print(f"Clinical Logical Consistency（CLC）: {result['clc_score']}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='MedVQA inference for single case')
    parser.add_argument('--image_path', required=True, help='Pathology image format(.png)')
    parser.add_argument('--question', required=True, help='Clinical question about the image')
    parser.add_argument('--checkpoint', default=None, help='Model weights checkpoint (selective)')
    args = parser.parse_args()

    model, config, device = load_trained_model(checkpoint_path=args.checkpoint)

    result = infer_single_case(args.image_path, args.question, model, config, device)

    print_result(result)