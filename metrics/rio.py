import numpy as np
from sklearn.metrics import jaccard_score


class RationaleImageOverlap:

    def __init__(self):
        pass

    def calculate(self, rationale_mask, image_mask):

        rationale_mask = np.where(rationale_mask > 0.5, 1, 0).astype(np.uint8)
        image_mask = np.where(image_mask > 0.5, 1, 0).astype(np.uint8)

        intersection = np.logical_and(rationale_mask, image_mask).sum()
        union = np.logical_or(rationale_mask, image_mask).sum()

        if union == 0:
            return 0.0

        rio_score = intersection / union

        return float(rio_score)

    def calculate_batch(self, rationale_masks, image_masks):

        scores = []
        for r_mask, i_mask in zip(rationale_masks, image_masks):
            score = self.calculate(r_mask, i_mask)
            scores.append(score)

        return np.mean(scores)


if __name__ == "__main__":
    rio = RationaleImageOverlap()

    rationale_mask = np.random.rand(224, 224)
    image_mask = np.random.rand(224, 224)

    score = rio.calculate(rationale_mask, image_mask)
    print(f"RIO Score: {score:.4f}")
    print(f"RationaleImageOverlap successfully initialized！")