from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(model)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
