import torch
import torch.nn as nn
import torch.nn.functional as F


class HallucinationCorrectionModule(nn.Module):

    def __init__(self, rationale_dim, image_dim, hidden_dim=256, dropout=0.1):
        super(HallucinationCorrectionModule, self).__init__()

        self.rationale_dim = rationale_dim
        self.image_dim = image_dim

        self.confidence_scorer = nn.Sequential(
            nn.Linear(rationale_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        combined_dim = rationale_dim + image_dim
        self.correction_gate = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rationale_dim),
            nn.Sigmoid()
        )

        if image_dim != rationale_dim:
            self.image_projector = nn.Linear(image_dim, rationale_dim)
        else:
            self.image_projector = nn.Identity()

        self.knowledge_alignment = nn.Linear(rationale_dim, rationale_dim)

    def forward(self, rationale, image_features=None):

        confidence = self.confidence_scorer(rationale)

        if image_features is not None:
            combined = torch.cat([rationale, image_features], dim=-1)
            gate = self.correction_gate(combined)

            projected_image = self.image_projector(image_features)

            corrected_rationale = gate * rationale + (1 - gate) * projected_image
        else:
            knowledge_aligned = self.knowledge_alignment(rationale)
            corrected_rationale = confidence * rationale + (1 - confidence) * knowledge_aligned

        return corrected_rationale

    def detect_hallucination(self, rationale, threshold=0.5):
        self.eval()
        with torch.no_grad():
            confidence = self.confidence_scorer(rationale)
            is_hallucination = (confidence < threshold).any()
        return is_hallucination.item()


if __name__ == "__main__":
    config = {'hidden_dim': 768}
    module = HallucinationCorrectionModule(config)

    rationale = torch.randn(2, 768)
    image_features = torch.randn(2, 768)

    corrected = module(rationale, image_features)
    print(f"corrected shape of vector: {corrected.shape}")

    is_hallucination = module.detect_hallucination(rationale)
    print(f"hallucination check result: {is_hallucination}")
    print(f"HallucinationCorrectionModule is successfully loaded！")