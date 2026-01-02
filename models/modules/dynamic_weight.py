import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicWeightingModule(nn.Module):

    def __init__(self, config):

        super(DynamicWeightingModule, self).__init__()
        self.config = config
        self.num_steps = config.get('num_steps', 6)

        self.step_names = [
            "morphology",
            "location",
            "size",
            "density",
            "infiltration",
            "malignancy"
        ]

        self.weight_generator = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.num_steps)
        )

        self.step_heads = nn.ModuleList([
            nn.Linear(768, 768) for _ in range(self.num_steps)
        ])

        self.fusion_layer = nn.Linear(768, 768)

    def forward(self, hidden_states, question_type_logits):

        # attain question type
        question_type = torch.argmax(question_type_logits, dim=1)

        # use the hint of [CLS] token to generate dynamic weights
        cls_hidden = hidden_states[:, 0, :]  # [batch_size, hidden_dim]

        # generate dynamic weights for each reasoning step
        weights = F.softmax(
            self.weight_generator(cls_hidden),
            dim=1
        )  # [batch_size, num_steps]

        # generate rationale for each step
        rationale_steps = []
        for i in range(self.num_steps):
            # get step hidden state from step head
            step_hidden = self.step_heads[i](cls_hidden)  # [batch_size, hidden_dim]

            # apply dynamic weight
            weighted_step = weights[:, i].unsqueeze(1) * step_hidden

            rationale_steps.append(weighted_step)

        # stack all steps
        stacked_steps = torch.stack(rationale_steps, dim=1)  # [batch_size, num_steps, hidden_dim]

        # sum over steps to get final rationale
        rationale = torch.sum(stacked_steps, dim=1)  # [batch_size, hidden_dim]

        # fusion layer
        rationale = self.fusion_layer(rationale)

        return rationale

    def generate_rationale_text(self, weights):

        # get step with the highest weights
        top_steps_idx = torch.topk(weights, k=3).indices.tolist()
        top_steps = [self.step_names[i] for i in top_steps_idx]

        rationale = f"Based on the 6-step-logic reasoning, we can conclude that, the most important steps are: {', '.join(top_steps)}"
        return rationale


# 示例使用
if __name__ == "__main__":
    config = {'num_steps': 6}
    module = DynamicWeightingModule(config)

    # 测试前向传播
    batch_size = 2
    hidden_states = torch.randn(batch_size, 10, 768)
    question_type_logits = torch.randn(batch_size, 3)

    rationale = module(hidden_states, question_type_logits)
    print(f"input rationale vector shape: {rationale.shape}")
    print(f"DynamicWeightingModule initialized successfully！")