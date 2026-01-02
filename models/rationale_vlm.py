import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Blip2Processor, Blip2ForConditionalGeneration

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.modules.dynamic_weight import DynamicWeightingModule
from models.modules.hallucination import HallucinationCorrectionModule


class GenerativeRationaleVLM(nn.Module):

    def __init__(self, config):
        super(GenerativeRationaleVLM, self).__init__()
        self.config = config

        model_name = config.get('model_name', 'Salesforce/blip2-opt-2.7b')
        self.processor = Blip2Processor.from_pretrained(model_name, use_fast=False)
        self.base_model = Blip2ForConditionalGeneration.from_pretrained(model_name)

        # Freeze vision model
        for param in self.base_model.vision_model.parameters():
            param.requires_grad = False

        # Architecture dimensions
        # BLIP-2 OPT-2.7b uses 2560 for text hidden states and 768 for Q-Former
        self.text_hidden_dim = 2560
        self.rationale_dim = 768

        self.question_type_classifier = nn.Linear(self.rationale_dim, 3)
        self.dynamic_reasoner = DynamicWeightingModule(config)

        self.hallucination_correction = HallucinationCorrectionModule(
            rationale_dim=self.rationale_dim,
            image_dim=1408,  # BLIP-2 vision tower output dim
            hidden_dim=256
        )

        # NEW: Projector to bridge 768 (Rationale) -> 2560 (Text Embeddings)
        self.rationale_projector = nn.Linear(self.rationale_dim, self.text_hidden_dim)

        # Updated Output Layer for Generative VQA
        # Maps text hidden states to the full vocabulary size
        self.output_layer = nn.Linear(self.text_hidden_dim, self.base_model.config.text_config.vocab_size)

    # Inside models/rationale_vlm.py -> forward()
    def forward(self, pixel_values, input_ids, attention_mask=None):
        # 1. Vision Encoder: Define image_embeds
        vision_outputs = self.base_model.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs.last_hidden_state  # [Batch, Patches, 1408]

        # 2. Q-Former: Bridge vision and language
        qformer_output = self.base_model.qformer(
            query_embeds=self.base_model.query_tokens.expand(pixel_values.shape[0], -1, -1),
            encoder_hidden_states=image_embeds
        )
        qformer_embeds = qformer_output.last_hidden_state  # [Batch, Queries, 768]

        # 3. Rationale Generation
        question_type_logits = self.question_type_classifier(qformer_embeds[:, 0, :])
        rationale = self.dynamic_reasoner(qformer_embeds, question_type_logits)  # [Batch, 768]

        # 4. Hallucination Correction
        image_features = image_embeds.mean(dim=1)  # [Batch, 1408]
        rationale = self.hallucination_correction(rationale, image_features)  # [Batch, 768]

        # 5. Dimension Alignment (NEW)
        # Project rationale [Batch, 768] -> [Batch, 2560]
        projected_rationale = self.rationale_projector(rationale)

        # 6. Language Modeling & Multimodal Fusion
        # text_embeds shape: [Batch, Seq, 2560]
        text_embeds = self.base_model.language_model.get_input_embeddings()(input_ids)

        # Fusion via broadcasting: [Batch, Seq, 2560] + [Batch, 1, 2560]
        combined_features = text_embeds + projected_rationale.unsqueeze(1)

        # 7. Final Projection: [Batch, Seq, Vocab_Size]
        logits = self.output_layer(combined_features)

        return logits, rationale

    def generate_answer(self, image, question, max_length=50):

        inputs = self.processor(
            images=image,
            text=question,
            return_tensors="pt",
            padding=True
        )

        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits, rationale = self.forward(
                inputs['pixel_values'],
                inputs['input_ids'],
                inputs.get('attention_mask')
            )

            predicted_class = torch.argmax(logits, dim=1)

            rationale_text = self.dynamic_reasoner.generate_rationale_text(
                F.softmax(logits[0], dim=0)
            )

        return {
            'answer': predicted_class.item(),
            'rationale': rationale_text,
            'confidence': F.softmax(logits, dim=1).max().item()
        }



if __name__ == "__main__":
    config = {
        'model_name': 'Salesforce/blip2-opt-2.7b',
        'num_steps': 6,
        'hidden_dim': 768,
        'num_classes': 2
    }

    print("Initialling the GenerativeRationaleVLM model...")
    model = GenerativeRationaleVLM(config)
    print("GenerativeRationaleVLM initialized！")

    # Test forward pass
    batch_size = 1
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (batch_size, 20))

    logits, rationale = model(pixel_values, input_ids)
    print(f"Output the shape of logits: {logits.shape}")
    print(f"Rationale vector shape: {rationale.shape}")