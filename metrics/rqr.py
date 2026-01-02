import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import numpy as np


class RationaleQuestionRelevance:

    def __init__(self, model_name='all-MiniLM-L6-v2'):

        self.model = SentenceTransformer(model_name)

    def calculate(self, rationale_text, question_text):

        rationale_emb = self.model.encode(rationale_text, convert_to_tensor=True)
        question_emb = self.model.encode(question_text, convert_to_tensor=True)

        similarity = F.cosine_similarity(
            rationale_emb.unsqueeze(0),
            question_emb.unsqueeze(0)
        )

        rqr_score = (similarity.item() + 1) / 2

        return rqr_score

    def calculate_batch(self, rationale_texts, question_texts):
        scores = []
        for rationale, question in zip(rationale_texts, question_texts):
            score = self.calculate(rationale, question)
            scores.append(score)

        return np.mean(scores)


# 示例使用
if __name__ == "__main__":
    print("Loading Sentence Transformer...")
    rqr = RationaleQuestionRelevance()

    rationale = "Based on a 6-step structured medical logic, identify morphological features and locate areas of focus"
    question = "Where is the tumor located in the image?"

    score = rqr.calculate(rationale, question)
    print(f"RQR Score: {score:.4f}")
    print(f"RationaleQuestionRelevance succesfully loaded！")