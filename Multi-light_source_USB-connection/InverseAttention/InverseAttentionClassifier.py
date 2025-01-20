import torch
import torch.nn as nn
import torch.nn.functional as F
from model import SqueezeNet


class EmbeddingInput(nn.Module):
    def __init__(self):
        super().__init__()
        self.squeezenet_views = nn.ModuleList([SqueezeNet() for _ in range(5)])
        self.conv = nn.Conv2d(in_channels=512 * 4, out_channels=512, kernel_size=3, padding=1)
        self.gap_views = nn.ModuleList([nn.AdaptiveAvgPool2d((1, 1)) for _ in range(5)])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, images):
        """
        images: [B, 5, 3, 200, 200] (Batch, Views, Channels, Height, Width)
        """
        B, V, C, H, W = images.size()  # B: Batch, V: Views
        assert V == 5, "This model is designed for 5 views only. 4 are defect view, 1 is normal view"

        features_list, embedded_list = [], []
        for v in range(V):
            view_images = images[:, v, :, :, :]  # [B, 5, H, W]
            features = self.squeezenet_views[v](view_images)  # [B, C, H', W']
            if v < V - 1: features_list.append(features)
            embedded = self.gap_views[v](features).view(B, 1, -1)  # [B, 1, C]
            embedded_list.append(embedded)

        normal_embedding = embedded_list[-1]  # [B, 1, C]
        defect_embeddings = torch.cat(embedded_list[:-1], dim=1)  # [B, 4, C]

        fusion = torch.cat(features_list, dim=1)  # [B, C*4, H', W']
        global_features = self.gap(self.conv(fusion))  # [B, C, 1, 1]

        return normal_embedding, defect_embeddings, global_features.squeeze(3).squeeze(2)


class Attention(nn.Module):
    def __init__(self, hidden_size=512, output_size=5, temperature=1.0):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(hidden_size, int(hidden_size/1))
        self.Ua = nn.Linear(hidden_size, int(hidden_size/1))
        self.temperature = temperature

        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            # nn.dropout(0.1)
            nn.Linear(128, output_size)
        )

    def forward(self, normal_embedding, defect_embeddings, global_features):
        distances = torch.norm(self.Wa(defect_embeddings) - self.Ua(normal_embedding), dim=2)  # [B, 4] Euclidean distance

        weights = F.softmax(distances / self.temperature, dim=1)
        context = torch.bmm(weights.unsqueeze(1), defect_embeddings)  # [B,1,4] x [B,4,512] = [B,1,512]
        combined = torch.cat((context.squeeze(1), global_features), dim=1)  # [B, 512] + [B, 512] -> [B, 1024]
        output = self.out(combined)

        return output


class InverseAttentionClassifier(nn.Module):
    def __init__(self, output_size=5, temperature=1.0):
        super(InverseAttentionClassifier, self).__init__()
        self.EmbeddingInput = EmbeddingInput()
        self.attention = Attention(output_size=output_size, temperature=temperature)

    def forward(self, views):
        query, keys, global_features = self.EmbeddingInput(views)  # [B, 4, C]. [B, 1, C]
        outputs = self.attention(query, keys, global_features)  # [B, 5]

        return outputs


if __name__ == '__main__':
    x = torch.rand(16, 5, 3, 200, 200)
    Classifier = InverseAttentionClassifier(temperature=1.0)

    out = Classifier(x)
    print(f"out: {out.shape}")
