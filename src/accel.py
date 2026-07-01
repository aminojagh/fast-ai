import torch
from accelerate import Accelerator
from src.learner import DeviceCB, TrainCB
from src.conv import def_device



class MixedPrecision(TrainCB):
    order = DeviceCB.order+10
    def __init__(
            self,
            n_inp=1,
            autocast_dtype = torch.float16,
            max_grad_norm=100.0
    ):
        super().__init__(n_inp=n_inp)
        self.ac_dtype = autocast_dtype
        self.max_grad_norm = max_grad_norm
    
    def before_fit(self, learn): self.scaler = torch.amp.GradScaler(def_device)

    def before_batch(self, learn):
        self.autocast = torch.autocast("cuda", self.ac_dtype)
        self.autocast.__enter__()

    def after_loss(self, learn):
        self.autocast.__exit__(None, None, None)

    def backward(self, learn):
        self.scaler.scale(learn.loss).backward()
        torch.nn.utils.clip_grad_norm_(
            learn.model.parameters(),
            max_norm=self.max_grad_norm
        )

    def step(self, learn):
        old_scale = self.scaler.get_scale()
        self.scaler.step(learn.opt)
        self.scaler.update()
        learn.did_opt_step = self.scaler.get_scale() >= old_scale
        # new_scale < old_scale means
        # AMP found inf/nan gradients, skipped optimizer.step(),
        # and reduced the scale.


class AccelerateCB(TrainCB):
    order = DeviceCB.order+10
    def __init__(self, n_inp=1, mixed_precision="fp16"):
        super().__init__(n_inp=n_inp)
        self.acc = Accelerator(mixed_precision=mixed_precision)
        
    def before_fit(self, learn):
        learn.model,learn.opt,learn.dls.train,learn.dls.valid = self.acc.prepare(
            learn.model, learn.opt, learn.dls.train, learn.dls.valid)

    def backward(self, learn): self.acc.backward(learn.loss)


class MultDL:
    def __init__(self, dl, mult=2): self.dl,self.mult = dl,mult
    def __len__(self): return len(self.dl)*self.mult
    def __iter__(self):
        for o in self.dl:
            for i in range(self.mult): yield o