import torch
from torch import nn

class classification_loss(nn.Module):
    """计算分类损失
    使用交叉熵损失函数计算损失
    """
    def __init__(self):
        super(classification_loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, output, target):
        loss = self.cross_entropy(output, target)

        return loss
        
