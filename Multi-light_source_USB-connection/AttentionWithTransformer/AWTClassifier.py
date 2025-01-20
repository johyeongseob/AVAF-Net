import torch
import torch.nn as nn
from model import SqueezeNet
from torchsummary import summary


class EmbeddingInput(nn.Module):
    def __init__(self):
        super().__init__()
        self.squeezenet_views = nn.ModuleList([SqueezeNet() for _ in range(4)])
        self.conv = nn.Conv2d(in_channels=512 * 4, out_channels=512, kernel_size=3, padding=1)
        self.gap_views = nn.ModuleList([nn.AdaptiveAvgPool2d((1, 1)) for _ in range(4)])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, images):
        """
        images: [B, 4, 3, 200, 200] (Batch, Views, Channels, Height, Width)
        """
        B, V, C, H, W = images.size()  # B: Batch, V: Views
        assert V == 4, "This model is designed for 4 views only."

        features_list, embedded_list = [], []
        for v in range(V):
            view_images = images[:, v, :, :, :]  # [B, 3, H, W]
            features = self.squeezenet_views[v](view_images)  # [B, C, H', W']
            embedded = self.gap_views[v](features).view(B, 1, -1)  # [B, 1, C]
            features_list.append(features)
            embedded_list.append(embedded)

        keys = torch.cat(embedded_list, dim=1)  # [B, 4, C]
        fusion = torch.cat(features_list, dim=1)  # [B, C*4, H', W']
        global_features = self.conv(fusion)  # [B, C, H', W']
        query = self.gap(global_features).view(B, 1, -1)  # [B, 1, C]

        return keys, query


class CLSClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(2 ** 6, 1, 512))  # [B, 1, C]
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

    def forward(self, embeddings):
        """
        images: [B, 4, C, H, W] - 네 가지 조명 조건의 이미지
        """
        B = embeddings.size(0)  # B: Batch

        transformer_input = torch.cat([self.cls_token[:B], embeddings], dim=1)  # [B, V+1, C]
        transformer_output = self.transformer(transformer_input)  # [V+1, B, 512]
        cls_out = transformer_output[:, 0, :]  # [B, 512]

        return cls_out


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size=512, output_size=5):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, keys, query):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys))).squeeze(2)  # [B, 4]
        scores = scores

        weights = self.softmax(scores).unsqueeze(1)  # [B, 1, 4]
        context = torch.bmm(weights, keys)  # [B,1,4] x [B,4,512] = [B,1,512]
        attn_output = context.squeeze(1)

        return attn_output  # [B, 512]

class AWTClassifier(nn.Module):
    def __init__(self, hidden_size=512, output_size=5):
        super().__init__()

        self.EmbeddingInput = EmbeddingInput()
        self.CLSClassifier = CLSClassifier()
        self.Attention = BahdanauAttention()
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, views):
        keys, query = self.EmbeddingInput(views)  # [B, 4, C],  [B, 1, C]
        cls_out = self.CLSClassifier(keys)
        attn_output = self.Attention(keys, query)

        return self.fc1(cls_out), self.fc2(attn_output)


if __name__ == '__main__':
    # 샘플 데이터 생성
    B, C, H, W = 2, 3, 200, 200  # 배치 크기와 이미지 크기
    images = torch.randn(B, 4, C, H, W)  # [B, 4, 3, 200, 200]

    # 모델 초기화
    num_classes = 5
    model = AWTClassifier(output_size=num_classes)

    # 출력 확인
    output = model(images)
    print(output[0], output[1])  # torch.Size([B, num_classes])