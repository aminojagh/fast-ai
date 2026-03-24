import math,torch,matplotlib.pyplot as plt
from collections.abc import Mapping
from operator import attrgetter
from functools import partial
from copy import copy
from torch import optim
import torch.nn.functional as F
from torcheval.metrics import Mean
from fastprogress import progress_bar,master_bar


from src.conv import def_device, to_device
from src.utils import store_attr


class CancelFitException(Exception): pass
class CancelBatchException(Exception): pass
class CancelEpochException(Exception): pass

class Callback: order = 0

def run_cbs(cbs, method_nm, learn=None):
    for cb in sorted(cbs, key=attrgetter('order')):
        method = getattr(cb, method_nm, None)
        if method is not None: method(learn)

class SingleBatchCB(Callback):
    order = 1
    def after_batch(self, learn):
      raise CancelEpochException()


def to_cpu(x):
    if isinstance(x, Mapping): return {k:to_cpu(v) for k,v in x.items()}
    if isinstance(x, list): return [to_cpu(o) for o in x]
    if isinstance(x, tuple): return tuple(to_cpu(list(x)))
    res = x.detach().cpu()
    return res.float() if res.dtype==torch.float16 else res

class MetricsCB(Callback):
    def __init__(self, *ms, **metrics):
        for o in ms: metrics[type(o).__name__] = o
        self.metrics = metrics
        self.all_metrics = copy(metrics)
        self.all_metrics['loss'] = self.loss = Mean()

    def _log(self, d): print(d)
    def before_fit(self, learn): learn.metrics = self
    def before_epoch(self, learn): [o.reset() for o in self.all_metrics.values()]

    def after_epoch(self, learn):
        log = {k:f'{v.compute():.3f}' for k,v in self.all_metrics.items()}
        log['epoch'] = learn.epoch
        log['train'] = 'train' if learn.model.training else 'eval'
        self._log(log)

    def after_batch(self, learn):
        x,y,*_ = learn.batch
        x,y = to_cpu((x,y))
        for m in self.metrics.values(): m.update(to_cpu(learn.preds), y)
        self.loss.update(to_cpu(learn.loss), weight=len(x))

class DeviceCB(Callback):
    def __init__(self, device=def_device): store_attr()
    def before_fit(self, learn):
        if hasattr(learn.model, 'to'): learn.model.to(self.device)
    def before_batch(self, learn): learn.batch = to_device(learn.batch, device=self.device)


class TrainCB(Callback):
  def __init__(self, n_inp=1): self.n_inp = n_inp
  def predict(self, learn): learn.preds = learn.model(*learn.batch[:self.n_inp])
  def get_loss(self, learn): learn.loss = learn.loss_func(learn.preds, *learn.batch[self.n_inp:])
  def backward(self, learn): learn.loss.backward()
  def step(self, learn): learn.opt.step()
  def zero_grad(self, learn): learn.opt.zero_grad()


class ProgressCB(Callback):
    order = MetricsCB.order+1
    def __init__(self, plot=False): self.plot = plot
    def before_fit(self, learn):
        learn.epochs = self.mbar = master_bar(learn.epochs)
        self.first = True
        if hasattr(learn, 'metrics'): learn.metrics._log = self._log
        self.losses = []
        self.val_losses = []

    def _log(self, d):
        if self.first:
            self.mbar.write(list(d), table=True)
            self.first = False
        self.mbar.write(list(d.values()), table=True)

    def before_epoch(self, learn):
      learn.dl = progress_bar(learn.dl, leave=False, parent=self.mbar)
    def after_batch(self, learn):
        learn.dl.comment = f'{learn.loss:.3f}'
        if self.plot and hasattr(learn, 'metrics') and learn.training:
            self.losses.append(learn.loss.item())
            if self.val_losses:
              self.mbar.update_graph(
                [
                    [range(len(self.losses)), self.losses],
                    [list(map(lambda x: (x+1)*len(learn.dls.train), range(learn.epoch))), self.val_losses]
                ]
            )
            else: self.mbar.update_graph([[range(len(self.losses)), self.losses]])

    def after_epoch(self, learn):
        if not learn.training:
            if self.plot and hasattr(learn, 'metrics'):
                self.val_losses.append(learn.metrics.all_metrics['loss'].compute())
                self.mbar.update_graph(
                    [
                        [range(len(self.losses)), self.losses],
                        [list(map(lambda x: (x+1)*len(learn.dls.train), range(learn.epoch+1))), self.val_losses]
                    ]
                )


class with_cbs:
    def __init__(self, nm): self.nm = nm
    def __call__(self, f):
        def _f(o, *args, **kwargs):
            try:
                o.callback(f'before_{self.nm}')
                f(o, *args, **kwargs)
                o.callback(f'after_{self.nm}')
            except globals()[f'Cancel{self.nm.title()}Exception']: pass
            finally: o.callback(f'cleanup_{self.nm}')
        return _f
    

class Learner():
  def __init__(
      self, model, dls,
      loss_func=F.mse_loss,
      lr=0.1,
      cbs=[],
      opt_func=optim.SGD
  ):store_attr()

  @with_cbs('batch')
  def _one_batch(self):
    self.predict()
    self.callback('after_predict')
    self.get_loss()
    self.callback('after_loss')
    if self.training:
      self.backward()
      self.callback('after_backward')
      self.step()
      self.callback('after_step')
      self.zero_grad()
  @with_cbs('epoch')
  def _one_epoch(self):
    for self.iter,self.batch in enumerate(self.dl): self._one_batch()
  def one_epoch(self, training):
    self.model.train(training)
    self.dl = self.dls.train if training else self.dls.valid
    self._one_epoch()
  @with_cbs('fit')
  def _fit(self, train, valid):
    for self.epoch in self.epochs:
      if train: self.one_epoch(True)
      if valid: torch.no_grad()(self.one_epoch)(False)

  def fit(self, n_epochs=1, train=True, valid=True, cbs=[], lr=None):
    for cb in cbs: self.cbs.append(cb)
    try:
      self.n_epochs = n_epochs
      self.epochs = range(n_epochs)
      self.opt = self.opt_func(self.model.parameters(), self.lr if lr is None else lr)
      self._fit(train, valid)
    finally:
      for cb in cbs: self.cbs.remove(cb)

  def __getattr__(self, name):
    if name in ('predict','get_loss','backward','step','zero_grad'):
      return partial(self.callback, name)
    raise AttributeError(name)

  def callback(self, method_nm): run_cbs(self.cbs, method_nm, self)

  @property
  def training(self): return self.model.training


class TrainLearner(Learner):
  def predict(self): self.preds = self.model(self.batch[0])
  def get_loss(self): self.loss = self.loss_func(self.preds, self.batch[1])
  def backward(self): self.loss.backward()
  def step(self): self.opt.step()
  def zero_grad(self): self.opt.zero_grad()

class MomentumLearner(TrainLearner):
  def __init__(self, model, dls, loss_func, lr, cbs=[], opt_func=optim.SGD, mom=0.85):
    self.mom = mom
    super().__init__(model, dls, loss_func, lr, cbs, opt_func)

  def zero_grad(self):
    with torch.no_grad():
      for p in self.model.parameters(): p.grad *= self.mom


class MomentumTrainCB(TrainCB):
  def __init__(self, mom=0.85):
    self.mom = mom
    super().__init__()
  def zero_grad(self, learn):
    with torch.no_grad():
      for p in learn.model.parameters(): p.grad *= self.mom


from torch.optim.lr_scheduler import ExponentialLR

class LRFinderCB(Callback):
  def __init__(self, gamma=1.3, max_mult=3): store_attr()
  def before_fit(self, learn):
    self.sched = ExponentialLR(learn.opt, self.gamma)
    self.lrs,self.losses = [],[]
    self.min = math.inf

  def after_batch(self, learn):
    if not learn.training: raise CancelEpochException()
    self.lrs.append(learn.opt.param_groups[0]['lr'])
    loss = to_cpu(learn.loss)
    self.losses.append(loss)
    if loss < self.min: self.min = loss
    if math.isnan(loss) or (loss > self.min*self.max_mult):
      raise CancelFitException()
    self.sched.step()

  def cleanup_fit(self, learn):
    plt.plot(self.lrs, self.losses)
    plt.xscale('log')


def lr_find(self:Learner, gamma=1.3, max_mult=3, start_lr=1e-5, max_epochs=10):
    self.fit(max_epochs, lr=start_lr, cbs=[LRFinderCB(gamma=gamma, max_mult=max_mult)])

Learner.lr_find = lr_find