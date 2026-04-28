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

import torch, random, numpy as np
def set_seed(seed, deterministic=False):
    torch.use_deterministic_algorithms(deterministic)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def noop (x=None, *args, **kwargs):
    "Do nothing"
    return x


from IPython import get_ipython
import sys

def in_colab():
    "Check if the code is running in Google Colaboratory"
    return 'google.colab' in sys.modules

def ipython_shell():
    "Same as `get_ipython` but returns `False` if not in IPython"
    try: return get_ipython()
    except NameError: return False

def in_ipython():
    "Check if code is running in some kind of IPython environment"
    return bool(ipython_shell())

def in_jupyter():
    "Check if the code is running in a jupyter notebook"
    if not in_ipython(): return False
    return 'InteractiveShell' in ipython_shell().__class__.__name__

def in_notebook():
    "Check if the code is running in a jupyter notebook"
    return in_colab() or in_jupyter()


#|export --> utils.net
import urllib, json
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse, urlunparse
from urllib.request import Request


url_default_headers = {
    "Accept":
    "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "max-age=0",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
}

class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if code in (307, 308):
            new_req = Request(newurl, data=req.data, method=req.get_method())
            for k,v in req.headers.items(): new_req.add_header(k, v)
        else: new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req and urlparse(newurl).netloc != urlparse(req.full_url).netloc: new_req.remove_header('Authorization')
        return new_req

def urlopener():
    _opener = urllib.request.build_opener(_SafeRedirectHandler)
    _opener.addheaders = list(url_default_headers.items())
    return _opener


def urlquote(url):
    "Update url's path with `urllib.parse.quote`"
    subdelims = "!$&'()*+,;="
    gendelims = ":?#[]@"
    safe = subdelims+gendelims+"%/"
    p = list(urlparse(url))
    p[2] = urllib.parse.quote(p[2], safe=safe)
    for i in range(3,6): p[i] = urllib.parse.quote(p[i], safe=safe)
    return urlunparse(p)

def urlwrap(url, data=None, headers=None):
    "Wrap `url` in a urllib `Request` with `urlquote`"
    return url if isinstance(url,Request) else Request(urlquote(url), data=data, headers=headers or {})

def urlopen(url, data=None, headers=None, timeout=None, **kwargs):
    "Like `urllib.request.urlopen`, but first `urlwrap` the `url`, and encode `data`"
    if kwargs and not data: data=kwargs
    if data is not None:
        if not isinstance(data, (str,bytes)): data = urlencode(data)
        if not isinstance(data, bytes): data = data.encode('ascii')
    try: return urlopener().open(urlwrap(url, data=data, headers=headers), timeout=timeout)
    except HTTPError as e: 
        e.msg += f"\n====Error Body====\n{e.read().decode(errors='ignore')}"
        raise


def urlread(url, data=None, headers=None, decode=True, return_json=False, return_headers=False, timeout=None, **kwargs):
    "Retrieve `url`, using `data` dict or `kwargs` to `POST` if present"
    try:
        with urlopen(url, data=data, headers=headers, timeout=timeout, **kwargs) as u: res,hdrs = u.read(),u.headers
    except HTTPError as e:
        e.msg += f"\n====Error Body====\n{e.read().decode(errors='ignore')}"
        raise

    if decode: res = res.decode()
    if return_json: res = json.loads(res)
    return (res,dict(hdrs)) if return_headers else res