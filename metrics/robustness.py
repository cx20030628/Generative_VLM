import torch
import torch.nn.functional as F
import numpy as np


class RobustnessEvaluator:

    def __init__(self, model):
        self.model = model
        self.model.eval()

    def calculate_ldr(self, original_logits, perturbed_logits):
        p = F.softmax(original_logits, dim=-1)
        p_prime = F.softmax(perturbed_logits, dim=-1)

        # 计算 KL Divergence
        ldr = F.kl_div(
            p_prime.log(),
            p,
            reduction='batchmean'
        )

        return ldr.item()

    def calculate_sa(self, input_tensor, target_label, rqr_variance):
        input_tensor.requires_grad = True

        output, _ = self.model(
            input_tensor,
            torch.zeros(input_tensor.shape[0], 20, dtype=torch.long)
        )

        loss = F.cross_entropy(output, target_label)

        loss.backward()

        grad_norm = input_tensor.grad.norm().item()

        sa_score = grad_norm * rqr_variance

        return sa_score

    def add_gaussian_noise(self, tensor, noise_level=0.2):
        noise = torch.randn_like(tensor) * noise_level
        return tensor + noise

    def evaluate_robustness(self, dataloader, noise_level=0.2):
        total_ldr = 0.0
        total_sa = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                images, questions, labels = batch

                original_logits, _ = self.model(images, questions)

                perturbed_images = self.add_gaussian_noise(images, noise_level)

                perturbed_logits, _ = self.model(perturbed_images, questions)

                ldr = self.calculate_ldr(original_logits, perturbed_logits)
                total_ldr += ldr

                images_for_sa = images.clone().detach()
                rqr_variance = 0.05
                sa = self.calculate_sa(images_for_sa, labels, rqr_variance)
                total_sa += sa

                num_batches += 1

        return {
            'average_ldr': total_ldr / num_batches if num_batches > 0 else 0.0,
            'average_sa': total_sa / num_batches if num_batches > 0 else 0.0
        }


if __name__ == "__main__":
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(768, 2)

        def forward(self, x, q=None):
            if len(x.shape) == 4:
                x = x.mean(dim=[2, 3])
            elif len(x.shape) == 3:
                x = x.mean(dim=1)
            return self.fc(x), x


    model = DummyModel()
    evaluator = RobustnessEvaluator(model)

    original_logits = torch.randn(4, 2)
    perturbed_logits = torch.randn(4, 2)
    ldr_score = evaluator.calculate_ldr(original_logits, perturbed_logits)
    print(f"Logic Drift Rate (LDR): {ldr_score:.4f}")

    input_tensor = torch.randn(4, 768)
    target_label = torch.tensor([0, 1, 0, 1])
    rqr_variance = 0.05
    sa_score = evaluator.calculate_sa(input_tensor, target_label, rqr_variance)
    print(f"Sensitivity Analysis (SA): {sa_score:.4f}")

    print("RobustnessEvaluator successfully initialized！")