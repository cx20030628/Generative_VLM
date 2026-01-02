import torch
import torch.nn as nn
import torchvision.models as models


class CNNAttentionStudent(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(CNNAttentionStudent, self).__init__()
        # 1. Image Stream: ResNet50
        resnet = models.resnet50(pretrained=True)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-2])  # 提取特征图

        # 2. Text Stream: Bi-LSTM
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.text_encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

        # 3. Symmetric Co-Attention
        self.attention = nn.MultiheadAttention(embed_dim=2048, num_heads=8)  # 简化示意

        # 4. Heads
        self.classifier = nn.Sequential(
            nn.Linear(2048 + hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, questions):
        # Image feature: [Batch, 2048, 7, 7]
        img_feats = self.image_encoder(images)
        # Text feature: [Batch, Seq, Hidden*2]
        text_output, _ = self.text_encoder(self.embedding(questions))

        # 1. Average Pooling to all image features [Batch, 2048, 7, 7] -> [Batch, 2048]
        img_avg = img_feats.mean(dim=[2, 3])

        # 2. Text Average Pooling [Batch, Seq, Hidden*2] -> [Batch, Hidden*2]
        text_avg = text_output.mean(dim=1)

        # 3. Fuse features classifier's input dimensions (2048 + hidden_dim * 2)
        fused_feats = torch.cat([img_avg, text_avg], dim=1)

        logits = self.classifier(fused_feats)

        return logits, fused_feats  # return logits and rationale features