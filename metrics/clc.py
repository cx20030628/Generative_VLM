import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel


class ClinicalLogicalConsistency:

    def __init__(self, model_name='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.logic_rules = {
            'morphology_before_diagnosis': ['Form', 'Diagnosis'],
            'location_before_malignancy': ['Position', 'Assessment'],
            'evidence_before_conclusion': ['Reasoning', 'Conclusion']
        }

    def calculate(self, rationale_text, answer_text):

        order_score = self._check_logical_order(rationale_text)

        consistency_score = self._check_consistency(rationale_text, answer_text)

        terminology_score = self._check_terminology(rationale_text)

        clc_score = (order_score + consistency_score + terminology_score) / 3

        return clc_score

    def _check_logical_order(self, text):
        keywords = ['Form', 'Position', 'Assessment', 'Reasoning', 'Conclusion']
        matches = sum(1 for kw in keywords if kw in text)
        return matches / len(keywords)

    def _check_consistency(self, rationale, answer):
        rationale_words = set(rationale.lower().split())
        answer_words = set(answer.lower().split())

        if len(answer_words) == 0:
            return 0.0

        overlap = len(rationale_words.intersection(answer_words))
        return min(overlap / len(answer_words), 1.0)

    def _check_terminology(self, text):
        medical_terms = [
            'neoplasm', 'malignant', 'benign', 'cell', 'tissue',
            'morphology', 'pathology', 'diagnosis'
        ]
        matches = sum(1 for term in medical_terms if term.lower() in text.lower())
        return min(matches / 3, 1.0)

    def calculate_batch(self, rationale_texts, answer_texts):
        scores = []
        for rationale, answer in zip(rationale_texts, answer_texts):
            score = self.calculate(rationale, answer)
            scores.append(score)

        return np.mean(scores)

if __name__ == "__main__":
    clc = ClinicalLogicalConsistency()

    rationale = "Based on morphological analysis, identify location characteristics, assess malignancy risk, and reach a diagnostic conclusion."
    answer = "Malignant tumor"

    score = clc.calculate(rationale, answer)
    print(f"CLC Score: {score:.4f}")
    print(f"ClinicalLogicalConsistency successfully initialized！")