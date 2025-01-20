import torch
import torch.nn as nn
from model import SqueezeNet


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


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size=512, output_size=5):
        super(BahdanauAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=hidden_size*4, out_channels=hidden_size, kernel_size=3, padding=1)
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.out = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, keys, query):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)  # Softmax 적용 값은 마지막 차원: [batch_size, 4, 1] -> [batch_size, 1, 4]

        weights = self.softmax(scores)
        context = torch.bmm(weights, keys)  # [B,1,4] x [B,4,512] = [B,1,512]
        output = self.out(context.squeeze(1))
        # combined = torch.cat((context.squeeze(1), query.squeeze(1)), dim=1)  # [B, 512] + [B, 512] -> [B, 1024]
        # output = self.out(combined)

        return output


class AttentionClassifier(nn.Module):
    def __init__(self, output_size=5):
        super(AttentionClassifier, self).__init__()
        self.EmbeddingInput = EmbeddingInput()
        self.attention = BahdanauAttention(output_size=output_size)

    def forward(self, views):
        keys, query = self.EmbeddingInput(views)  # [B, 4, C]. [B, 1, C]
        outputs = self.attention(keys, query)  # [B, 5]

        return outputs


if __name__ == '__main__':
    x = torch.rand(16, 4, 3, 200, 200)
    Classifier = AttentionClassifier()

    out = Classifier(x)

