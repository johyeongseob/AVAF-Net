import torch
import torch.nn as nn
from model import SqueezeNet
from torchsummary import summary


class EmbeddingInput(nn.Module):
    def __init__(self):
        super().__init__()
        self.squeezenet = SqueezeNet()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, images):
        """
        images: [B, 4, 3, 200, 200] (Batch, Views, Channels, Height, Width)
        """
        B, V, C, H, W = images.size()  # B: Batch, V: Views
        images = images.view(B * V, C, H, W)  # [B*V, 3, 200, 200]
        features = self.squeezenet(images)  # [B*V, 512, H, W]
        features = self.gap(features).squeeze(-1).squeeze(-1)  # [B*V, 512]
        embeddings = features.view(B, V, -1)  # [B, V, 512]

        return embeddings  # [B, V, 512]


class TransformerClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.EmbeddingInput = EmbeddingInput()
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.fc = nn.Sequential(
            nn.Linear(512 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, images):
        """
        images: [B, 4, C, H, W] - 네 가지 조명 조건의 이미지
        """
        embeddings = self.EmbeddingInput(images)  # [B, V, 512]
        Batch_size = embeddings.size(0)

        # Transformer expects [V, B, E]
        transformer_out = self.transformer(embeddings.permute(1, 0, 2))  # [V, B, 512]

        outputs = transformer_out.permute(1, 0, 2)  # [B, V, 512]
        return self.fc(outputs.reshape(Batch_size, -1))  # [B, num_classes]


if __name__ == '__main__':
    # 샘플 데이터 생성
    B, C, H, W = 2, 3, 200, 200  # 배치 크기와 이미지 크기
    images = torch.randn(B, 4, C, H, W)  # [B, 4, 3, 200, 200]

    # 모델 초기화
    num_classes = 5
    model = TransformerClassifier(num_classes=num_classes)

    # 출력 확인
    output = model(images)
    print(output.shape)  # torch.Size([B, num_classes])