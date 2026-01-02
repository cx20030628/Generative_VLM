import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import json

IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_TOKEN = 0
MAX_TEXT_LEN = 64

def get_image_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def process_single_input(image_path, clinical_query, word_embedding, image_transform):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"病理图像不存在：{image_path}")
    image = Image.open(image_path).convert('RGB')
    image_tensor = image_transform(image).unsqueeze(0)  # 增加batch维度 [1, 3, 224, 224]

    query_tensor = torch.tensor(clinical_query, dtype=torch.long)
    if len(query_tensor) > MAX_TEXT_LEN:
        query_tensor = query_tensor[:MAX_TEXT_LEN]

    pad_len = MAX_TEXT_LEN - len(query_tensor)
    if pad_len > 0:
        query_tensor = torch.cat([
            query_tensor,
            torch.tensor([PAD_TOKEN] * pad_len, dtype=torch.long)
        ])
    query_tensor = query_tensor.unsqueeze(0)

    text_embedding = word_embedding(query_tensor)

    image_tensor = image_tensor.to(DEVICE)
    text_embedding = text_embedding.to(DEVICE)

    return image_tensor, text_embedding

class ResNet50FeatureExtractor(nn.Module):
    """ResNet50 特征提取器（提取多尺度空间特征，文档指定）"""
    def __init__(self):
        super().__init__()
        resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        self.features = nn.Sequential(*list(resnet50.children())[:-2])

    def forward(self, x):
        return self.features(x)

class BiLSTMTextEncoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.hidden_dim = hidden_dim
        self.num_directions = 2

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        return lstm_out

class SymmetricCoAttention(nn.Module):
    def __init__(self, image_feature_dim, text_feature_dim, hidden_dim=512):
        super().__init__()
        self.image_proj = nn.Linear(image_feature_dim, hidden_dim)
        self.text_proj = nn.Linear(text_feature_dim, hidden_dim)
        self.attention_score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, image_features, text_features):
        b, c, h, w = image_features.shape
        image_features_flat = image_features.permute(0, 2, 3, 1).reshape(b, h*w, c)

        img_proj = self.image_proj(image_features_flat)
        txt_proj = self.text_proj(text_features)

        img2txt_att = torch.bmm(img_proj, txt_proj.transpose(1, 2))
        img2txt_att = torch.softmax(img2txt_att, dim=-1)
        img_guided_text = torch.bmm(img2txt_att, text_features)

        txt2img_att = torch.bmm(txt_proj, img_proj.transpose(1, 2))
        txt2img_att = torch.softmax(txt2img_att, dim=-1)
        txt_guided_img = torch.bmm(txt2img_att, image_features_flat)

        image_fused = torch.cat([image_features_flat, img_guided_text], dim=-1)
        text_fused = torch.cat([text_features, txt_guided_img], dim=-1)

        global_image_feat = torch.mean(image_fused, dim=1)  # [B, c+text_feature_dim]
        global_text_feat = torch.mean(text_fused, dim=1)    # [B, text_feature_dim+c]
        fused_feature = torch.cat([global_image_feat, global_text_feat], dim=-1)  # [B, 2*(c+text_feature_dim)]

        return fused_feature, image_features_flat

class CNNAttentionBaseline(nn.Module):
    def __init__(self, embed_dim=300, text_hidden_dim=256, num_classes=2):
        super().__init__()

        self.image_extractor = ResNet50FeatureExtractor()
        self.image_feature_dim = 2048

        self.text_encoder = BiLSTMTextEncoder(
            embed_dim=embed_dim,
            hidden_dim=text_hidden_dim
        )
        self.text_feature_dim = text_hidden_dim * 2

        self.co_attention = SymmetricCoAttention(
            image_feature_dim=self.image_feature_dim,
            text_feature_dim=self.text_feature_dim
        )

        fused_feature_dim = 2 * (self.image_feature_dim + self.text_feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fused_feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

        self.evidence_tracing = nn.Linear(self.image_feature_dim, 1)

    def forward(self, image_tensor, text_embedding):

        image_features = self.image_extractor(image_tensor)  # [B, 2048, 7, 7]

        text_features = self.text_encoder(text_embedding)    # [B, MAX_TEXT_LEN, 512]

        fused_feature, image_features_flat = self.co_attention(image_features, text_features)

        logits = self.classifier(fused_feature)  # [B, num_classes]
        pred_probs = torch.softmax(logits, dim=-1)

        activation_scores = self.evidence_tracing(image_features_flat)  # [B, 49, 1]
        activation_map = activation_scores.reshape(-1, 7, 7)
        activation_map = torch.sigmoid(activation_map)

        return pred_probs, activation_map

def baseline_infer(model_path, image_path, clinical_query, word_embedding_path, num_classes=2, save_result=True):

    image_transform = get_image_transform()
    embed_dim = 300

    word_embedding = nn.Embedding.from_pretrained(
        torch.load(word_embedding_path, map_location=DEVICE),
        freeze=True
    ).to(DEVICE)

    model = CNNAttentionBaseline(
        embed_dim=embed_dim,
        num_classes=num_classes
    ).to(DEVICE)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found：{model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Successfully loaded model from checkpoint：{model_path}")

    image_tensor, text_embedding = process_single_input(
        image_path=image_path,
        clinical_query=clinical_query,
        word_embedding=word_embedding,
        image_transform=image_transform
    )

    with torch.no_grad():
        pred_probs, activation_map = model(image_tensor, text_embedding)

    pred_probs_np = pred_probs.cpu().numpy().squeeze()
    pred_class = np.argmax(pred_probs_np)
    activation_map_np = activation_map.cpu().numpy().squeeze()

    result = {
        "image_path": image_path,
        "clinical_query_len": len(clinical_query),
        "pred_class": int(pred_class),
        "pred_probs": {f"class_{i}": float(prob) for i, prob in enumerate(pred_probs_np)},
        "activation_map_shape": activation_map_np.shape,
        "activation_map_max": float(np.max(activation_map_np)),
        "activation_map_min": float(np.min(activation_map_np))
    }

    if save_result:
        output_dir = "infer_results/baseline"
        os.makedirs(output_dir, exist_ok=True)
        image_name = os.path.basename(image_path).split('.')[0]
        result_path = os.path.join(output_dir, f"{image_name}_infer_result.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        np.save(os.path.join(output_dir, f"{image_name}_activation_map.npy"), activation_map_np)
        print(f"Inference result saved to:{output_dir}")

    print("\n========== Inference report ==========")
    print(f"Predicted class：class_{pred_class}")
    print(f"Predicted probabilities：{result['pred_probs']}")
    print(f"Shape of activation map：{activation_map_np.shape}(localization heatmap)")
    print("==================================")

    return result, pred_probs_np, activation_map_np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN-Attention baseline_infer")
    parser.add_argument("--model_path", type=str, required=True, help="Trained model weights path(e.g. ./checkpoints/baseline_best.pth)")
    parser.add_argument("--image_path", type=str, required=True, help="Path to pathological images pending analysis(e.g ./data/test/001.png)")
    parser.add_argument("--word_embedding_path", type=str, required=True, help="Word Embedding Weight Path（e.g. ./embeddings/word_embedding.pth）")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of categories (default is binary classification)")
    parser.add_argument("--clinical_query", type=str, default="[10,23,45,67]", help="Clinical Query Text Term Index Sequence (JSON Format List)")

    args = parser.parse_args()

    try:
        clinical_query = json.loads(args.clinical_query)
        assert isinstance(clinical_query, list), "Clinical queries must be in a list format of term indexes"
    except:
        raise ValueError("clinical_query format error, should be a JSON list (e.g., '[10,23,45]')")

    baseline_infer(
        model_path=args.model_path,
        image_path=args.image_path,
        clinical_query=clinical_query,
        word_embedding_path=args.word_embedding_path,
        num_classes=args.num_classes,
        save_result=True
    )