from functools import partial
from torch import nn
from src.init import GeneralRelu, conv
from src.utils import noop



act_gr = partial(GeneralRelu, leak=0.1, sub=0.4)


def _conv_block(ni, nf, stride, act=act_gr, norm=None, ks=3):
    conv1 = conv(ni, nf, stride=1, act=act, norm=norm, ks=ks)
    conv2 = conv(nf, nf, stride=stride, act=None, norm=norm, ks=ks)
    if norm: nn.init.constant_(conv2[1].weight, 0.)
    return nn.Sequential(
        conv1,
        conv2
    )

class ResBlock(nn.Module):
    def __init__(self, ni, nf, stride=1, ks=3, act=act_gr, norm=None):
        super().__init__()
        self.convs = _conv_block(ni, nf, stride, act=act, ks=ks, norm=norm)
        self.idconv = noop if ni==nf else conv(ni, nf, ks=1, stride=1, act=None)
        self.pool = noop if stride==1 else nn.AvgPool2d(2, ceil_mode=True)
        self.act = act()

    def forward(self, x): return self.act(self.convs(x) + self.idconv(self.pool(x)))