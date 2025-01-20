
import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary


class SENet(nn.Module):
    def __init__(self, c, r=2):
        super(SENet, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input_filter):
        batch, channel, _, _ = input_filter.size()
        se = (self.squeeze(input_filter)).view(batch, channel)
        ex = (self.excitation(se))
        alpha = ex.view(batch, channel, 1, 1)
        return alpha * input_filter


class SqueezeNet(nn.Module):
    def __init__(self, dropout_prob=0.2):
        super(SqueezeNet, self).__init__()

        # squeezenet1_0 model and pre-trained weights with ImageNet
        squeezenet1_0 = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.DEFAULT)
        # features: no use fully connected layer of SqueezeNet
        self.features = nn.Sequential(*list(squeezenet1_0.features.children()))

    def forward(self, x):
        x = self.features(x) # Start: (batch_size, 3, 100, 100)
        return x

if __name__ == "__main__":
    model = SqueezeNet().cuda()
    summary(model, input_size=(3, 200, 200))
