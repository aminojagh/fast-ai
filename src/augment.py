import torch,random
from torch import nn
from torch.nn import init

from src.utils import in_notebook
from src.learner import SingleBatchCB, Learner, Callback, to_cpu
from src.activations import Hooks
from torch import distributions, nn
from src.resnet import ResBlock

def _flops(x, h, w):
    if x.dim()<3: return x.numel()
    if x.dim()==4: return x.numel()*h*w

def summary(self:Learner):
    res = '|Module|Input|Output|Num params|MFLOPS|\n|--|--|--|--|--|\n'
    totp,totf = 0,0
    def _f(hook, mod, inp, outp):
        nonlocal res,totp,totf
        nparms = sum(o.numel() for o in mod.parameters())
        totp += nparms
        *_,h,w = outp.shape
        flops = sum(_flops(o, h, w) for o in mod.parameters())/1e6
        totf += flops
        res += f'|{type(mod).__name__}|{tuple(inp[0].shape)}|{tuple(outp.shape)}|{nparms}|{flops:.1f}|\n'
    with Hooks(self.model, _f) as hooks: self.fit(1, lr=1, train=False, cbs=[SingleBatchCB()])
    print(f"Tot params: {totp}; MFLOPS: {totf:.1f}")
    if in_notebook():
        from IPython.display import Markdown
        return Markdown(res)
    else: print(res)

# TODO: refactor this code ASAP !!!
Learner.summary = summary


class CapturePreds(Callback):
    def __init__(self, keep_loss):
        super().__init__()
        self.keep_loss = keep_loss
    def before_fit(self, learn):
        self.all_inps,self.all_preds,self.all_targs = [],[],[]
        # we don't need the loss. we just need to capture predictions
        if not self.keep_loss: learn.get_loss = lambda *args, **kwargs: ...
    def after_batch(self, learn):
        self.all_inps.append(to_cpu(learn.batch[0]))
        self.all_preds.append(to_cpu(learn.preds))
        self.all_targs.append(to_cpu(learn.batch[1]))
    def after_fit(self, learn):
        self.all_preds,self.all_targs,self.all_inps = map(torch.cat, [self.all_preds,self.all_targs,self.all_inps])

def capture_preds(self: Learner, cbs=[], inps=False, keep_loss=False):
    cp = CapturePreds(keep_loss)
    self.fit(1, train=False, cbs=[cp]+cbs)
    res = cp.all_preds,cp.all_targs
    if inps: res = res+(cp.all_inps,)
    return res
Learner.capture_preds = capture_preds


def _rand_erase1(x, pct, xm, xs, mn, mx):
    szx = int(pct*x.shape[-2])
    szy = int(pct*x.shape[-1])
    stx = int(random.random()*(1-pct)*x.shape[-2])
    sty = int(random.random()*(1-pct)*x.shape[-1])
    init.normal_(x[:,:,stx:stx+szx,sty:sty+szy], mean=xm, std=xs)
    x.clamp_(mn, mx)


def rand_erase(x, pct=0.2, max_num = 4):
    xm,xs,mn,mx = x.mean(),x.std(),x.min(),x.max()
    num = random.randint(0, max_num)
    # print(f"inside rand_erase: num = {num}")
    for i in range(num): _rand_erase1(x, pct, xm, xs, mn, mx)
    return x


class RandErase(nn.Module):
    def __init__(self, pct=0.2, max_num=4):
        super().__init__()
        self.pct,self.max_num = pct,max_num
    def forward(self, x): return rand_erase(x, self.pct, self.max_num)


def _rand_copy1(x, pct):
    szx = int(pct*x.shape[-2])
    szy = int(pct*x.shape[-1])
    stx1 = int(random.random()*(1-pct)*x.shape[-2])
    sty1 = int(random.random()*(1-pct)*x.shape[-1])
    stx2 = int(random.random()*(1-pct)*x.shape[-2])
    sty2 = int(random.random()*(1-pct)*x.shape[-1])
    x[:,:,stx1:stx1+szx,sty1:sty1+szy] = x[:,:,stx2:stx2+szx,sty2:sty2+szy]


def rand_copy(x, pct=0.2, max_num = 4):
    num = random.randint(0, max_num)
    for i in range(num): _rand_copy1(x, pct)
    return x

class RandCopy(nn.Module):
    def __init__(self, pct=0.2, max_num=4):
        super().__init__()
        self.pct,self.max_num = pct,max_num
    def forward(self, x): return rand_copy(x, self.pct, self.max_num)



#|export
class Dropout(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training: return x
        dist = distributions.binomial.Binomial(tensor(1.0).to(x.device), probs=1-self.p)
        return x * dist.sample(x.size()) * 1/(1-self.p)
    

#|export
def get_dropmodel(
    act=nn.ReLU,
    nfs=(16,32,64,128,256,512),
    norm=nn.BatchNorm2d,
    drop=0.0
):
    layers = [
        ResBlock(1, 16, ks=5, stride=1, act=act, norm=norm),
        nn.Dropout2d(drop)
    ]
    layers += [
        ResBlock(nfs[i], nfs[i+1], act=act, norm=norm, stride=2)
        for i in range(len(nfs)-1)
    ]
    layers += [
        nn.Flatten(),
        Dropout(drop),
        nn.Linear(nfs[-1], 10, bias=False),
        nn.BatchNorm1d(10)
    ]
    return nn.Sequential(*layers)