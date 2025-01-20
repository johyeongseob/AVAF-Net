import torch
import torch.nn as nn
from model import SqueezeNet
from torchsummary import summary


class EmbeddingInput(nn.Module):
    def __init__(self):
        super().__init__()
        self.squeezenet = SqueezeNet()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_token = nn.Parameter(torch.randn(2 ** 6, 1, 512))  # [B, 1, C]

    def forward(self, images):
        """
        images: [B, 4, 3, 200, 200] (Batch, Views, Channels, Height, Width)
        """
        B, V, C, H, W = images.size()  # B: Batch, V: Views
        images = images.view(B * V, C, H, W)  # [B*V, 3, 200, 200]
        features = self.squeezenet(images)  # [B*V, 512, H, W]
        features = self.gap(features).squeeze(-1).squeeze(-1)  # [B*V, 512]
        embeddings = features.view(B, V, -1)  # [B, V, 512]
        transformer_input = torch.cat([self.cls_token[:B], embeddings], dim=1)  # [B, V+1, C]

        return transformer_input  # [B, V+1, 512]


class CLSClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.EmbeddingInput = EmbeddingInput()
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, images):
        """
        images: [B, 4, C, H, W] - 네 가지 조명 조건의 이미지
        """
        transformer_input = self.EmbeddingInput(images)  # [B, V, 512]
        transformer_output = self.transformer(transformer_input.permute(1, 0, 2))  # [V+1, B, 512]
        cls_out = transformer_output[0]

        return self.fc(cls_out)  # [B, num_classes]


if __name__ == '__main__':
    # 샘플 데이터 생성
    B, C, H, W = 2, 3, 200, 200  # 배치 크기와 이미지 크기
    images = torch.randn(B, 4, C, H, W)  # [B, 4, 3, 200, 200]

    # 모델 초기화
    num_classes = 5
    model = CLSClassifier(num_classes=num_classes)

    # 출력 확인
    output = model(images)
    print(output.shape)  # torch.Size([B, num_classes])
