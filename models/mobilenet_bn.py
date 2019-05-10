from torch import nn
from torch.autograd import variable
import torch

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, onnx_compatible=False):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        nn.BatchNorm2d(in_channels),
        ReLU(),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )


class InvertedResidual(nn.Module):
    '''ＭobileNetv2特有模块，采用两边通道少，中间通道多的结构
    '''
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.in_channels = inp
        self.out_channels = oup
        hidden_dim = int(round(inp * expand_ratio))
        # 判断是否使用shortcut连接
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear，在第二个pw使用线性激活
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ExtraLyers(nn.Module):
    '''extra层使用的操作
    '''
    def __init__(self, inp, oup, stride, expand_ratio, BN=True):
        super(ExtraLyers, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.in_channels = inp
        self.out_channels = oup

        self.hidden_dim = int(round(inp*expand_ratio))
        self.use_res_connect = self.stride == 1 and self.in_channels == self.out_channels

        # 该参数决定是否添加BN层
        self.BN = BN

        layers = []
        if expand_ratio != 1:
            # pw 点卷积改变通道
            layers.append(nn.Conv2d(self.in_channels, self.hidden_dim, 1, 1))
            if self.BN:
                layers.append(nn.BatchNorm2d(self.hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        
        # dw 深度卷积改变大小
        layers.append(nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, stride, padding=1, groups=self.hidden_dim, bias=False))
        if self.BN:
            layers.append(nn.BatchNorm2d(self.hidden_dim))
        layers.append(nn.ReLU6(inplace=True))
        
        # pw 点卷积改变深度,使用线性激活
        layers.append(nn.Conv2d(self.hidden_dim, self.out_channels, 1, 1, 0, bias=False))
        if self.BN:
            layers.append(nn.BatchNorm2d(self.out_channels))
        
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x+self.features(x)
        else:
            return self.features(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        extra_block = ExtraLyers
        input_channel = 32
        last_channel = 1280

        self.inverted_residual_setting = [
            # t, c, n, s
            # t：通道扩展比率，c：输出通道数，当前模块重复次数，s：stride
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.extra_layers_setting = [
            # c, s, e_r
            # c:输出通道, s:stride, e_r:通道扩展比例
            [512, 2, 0.2],
            [256, 2, 0.25],
            [256, 2, 0.5],
            [64, 2, 0.25],
        ]

        # 需要输出的中间特征的通道大小
        self.out_channels_setting = [576, 1280, 512, 256, 256, 64]
        # 需要输出的中间特征的序号，以整个网络的第一模块为0
        self.out_features_index_base = {14:0, 18:0}
        self.out_features_index_extras = [19, 20, 21, 22]
        
        # 第一层
        input_channel = int(input_channel * width_mult)
        inverted_residual_blocks_features = []
        
        inverted_residual_blocks_features.append(ConvBNReLU(3, input_channel, stride=2))
        # 创建基础inverted residual blocks
        for t, c, n, s in self.inverted_residual_setting:
            
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                inverted_residual_blocks_features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # 加入一层带BN的1x1卷积进行通道变换
        inverted_residual_blocks_features.append(ConvBNReLU(input_channel, last_channel, 1, 1))
        input_channel = last_channel
        self.features = nn.Sequential(*inverted_residual_blocks_features)
        
        self.classifier = nn.Sequential(
                            nn.Dropout(0.2),
                            nn.Linear(input_channel, num_classes),
                        )


    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3]) 

        x = self.classifier(x)

        return x

    def init_weights(self):
        # weight initialization　
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


def mobilenet_v2(pretrained=False, **kwargs):
    return MobileNetV2(**kwargs)

if __name__ == "__main__":
    x = variable(torch.randn(10, 3, 300, 300))
    net = MobileNetV2().cuda()
    net.init_weights()
    
    output = net(x.cuda())

    print("-----------------------------------")
    for output_ in output:
        print(output_.size())

    pass