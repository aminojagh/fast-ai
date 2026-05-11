import os, torch, math
from torch.utils.data import DataLoader,default_collate
from torch.nn import init
from functools import partial


def abar(t): return (t*math.pi/2).cos()**2
def inv_abar(x): return x.sqrt().acos()*2/math.pi

def noisify(x0):
    device = x0.device
    n = len(x0)
    t = torch.rand(n,).to(x0).clamp(0,0.999)
    ε = torch.randn(x0.shape, device=device)
    abar_t = abar(t).reshape(-1, 1, 1, 1).to(device)
    xt = abar_t.sqrt()*x0 + (1-abar_t).sqrt()*ε
    return (xt, t.to(device)), ε

def collate_ddpm(b, xl): return noisify(default_collate(b)[xl])
def dl_ddpm(ds, ds_xl, bs): return DataLoader(
            ds, batch_size=bs,
            collate_fn=partial(collate_ddpm, xl=ds_xl),
            num_workers=os.cpu_count()
)

def init_ddpm(model):
    for o in model.down_blocks:
        for p in o.resnets:
            p.conv2.weight.data.zero_()
            if o.downsamplers:
                for p in list(o.downsamplers): init.orthogonal_(p.conv.weight)
    for o in model.up_blocks:
        for p in o.resnets: p.conv2.weight.data.zero_()
    model.conv_out.weight.data.zero_()