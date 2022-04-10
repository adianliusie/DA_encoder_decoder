import torch.nn as nn

class LinearHead(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.classifier = nn.Linear(768, num_class)

    def forward(self, encoder_outputs, encoder_mask=None, labels=None):
        """teacher forcing training"""
        y = self.classifier(encoder_outputs)
        return y