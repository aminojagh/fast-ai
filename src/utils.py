# https://github.com/AnswerDotAI/fastcore/blob/main/fastcore/test.py
from functools import partial
from typing import Iterable,Generator
def is_close(a,b,eps=1e-5):
    "Is `a` within `eps` of `b`"
    if hasattr(a, '__array__') or hasattr(b,'__array__'):
        return (abs(a-b)<eps).all()
    if isinstance(a, (Iterable,Generator)) or isinstance(b, (Iterable,Generator)):
        return all(abs(a_-b_)<eps for a_,b_ in zip(a,b))
    return abs(a-b)<eps

def test(a, b, cmp, cname=None):
    "`assert` that `cmp(a,b)`; display inputs and `cname or cmp.__name__` if it fails"
    if cname is None: cname=cmp.__name__
    assert cmp(a,b),f"{cname}:\n{a}\n{b}"

def test_close(a,b,eps=1e-5):
    "`test` that `a` is within `eps` of `b`"
    test(a,b,partial(is_close,eps=eps),'close')

from typing import Iterable,Generator, Iterator
import sys, math
from itertools import islice
# https://github.com/AnswerDotAI/fastcore/blob/main/fastcore/basics.py#L437
def store_attr():
    fr = sys._getframe(1)
    code = getattr(fr, 'f_code')
    args = code.co_varnames[:code.co_argcount+code.co_kwonlyargcount]
    self = fr.f_locals[args[0]]
    ns = getattr(self, '__slots__', args[1:])
    attrs = {n:fr.f_locals[n] for n in ns}
    for n,v in attrs.items():
        setattr(self, n, v)

# https://github.com/AnswerDotAI/fastcore/blob/main/fastcore/basics.py#250
def chunked(it: Iterator, chunk_sz=None, drop_last=False, n_chunks=None, pad=False, pad_val=None):
    "Return batches from iterator `it` of size `chunk_sz` (or return `n_chunks` total)"
    assert bool(chunk_sz) ^ bool(n_chunks)
    if n_chunks: chunk_sz = max(math.ceil(len(it)/n_chunks), 1)
    while True:
        res = list(islice(it, chunk_sz))
        if res and (len(res)==chunk_sz or not drop_last):
            if pad: yield res + [pad_val]*(chunk_sz-len(res))
            else: yield res
        else: return