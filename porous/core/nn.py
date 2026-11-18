import math
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax


class KeyGen:

    def __init__(self, key):
        self.key = key

    def __call__(self):
        self.key, sub = jax.random.split(self.key)
        return sub


class Ctx:

    def __init__(self, training=False, amp_dtype=None, key=None):
        self.training = training
        self.amp_dtype = amp_dtype
        self._key = key
        self._n = 0

    def next_key(self):
        assert self._key is not None, "a PRNG key is required when training (dropout)"
        k = jax.random.fold_in(self._key, self._n)
        self._n += 1
        return k

    def child(self, key):
        return Ctx(self.training, self.amp_dtype, key)


def _uniform(key, shape, bound):
    return jax.random.uniform(key, shape, jnp.float32, -bound, bound)


class SiLU:
    def init(self, kg):
        return {}

    def apply(self, p, x, ctx):
        return x * jax.nn.sigmoid(x)


class Identity:
    def init(self, kg):
        return {}

    def apply(self, p, x, ctx):
        return x


class GroupNorm32:

    def __init__(self, num_groups, channels, eps=1e-5):
        assert channels % num_groups == 0, f"GroupNorm: {channels} channels not divisible by {num_groups} groups"
        self.num_groups, self.channels, self.eps = num_groups, channels, eps

    def init(self, kg):
        return {'weight': jnp.ones((self.channels,), jnp.float32), 'bias': jnp.zeros((self.channels,), jnp.float32)}

    def apply(self, p, x, ctx):
        dt = x.dtype
        xf = x.astype(jnp.float32)
        N, C = xf.shape[0], xf.shape[1]
        assert C == self.channels, f"GroupNorm expected {self.channels} channels, got {C}"
        g = xf.reshape(N, self.num_groups, -1)
        mean = g.mean(axis=2, keepdims=True)
        var = g.var(axis=2, keepdims=True)
        g = (g - mean) / jnp.sqrt(var + self.eps)
        y = g.reshape(xf.shape)
        shape = (1, C) + (1,) * (xf.ndim - 2)
        y = y * p['weight'].reshape(shape) + p['bias'].reshape(shape)
        return y.astype(dt)


def normalization(channels):
    return GroupNorm32(32, channels)


class Conv:

    def __init__(self, dims, in_ch, out_ch, kernel_size, stride=1, padding=0, zero=False):
        if dims not in (1, 2, 3):
            raise ValueError(f"unsupported dimensions: {dims}")
        self.dims, self.in_ch, self.out_ch, self.zero = dims, in_ch, out_ch, zero
        self.kernel = (kernel_size,) * dims if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = (stride,) * dims if isinstance(stride, int) else tuple(stride)
        self.padding = (padding,) * dims if isinstance(padding, int) else tuple(padding)

    def init(self, kg):
        wshape = (self.out_ch, self.in_ch) + self.kernel
        if self.zero:
            return {'weight': jnp.zeros(wshape, jnp.float32), 'bias': jnp.zeros((self.out_ch,), jnp.float32)}
        fan_in = self.in_ch * int(np.prod(self.kernel))
        bound = 1.0 / math.sqrt(fan_in)
        return {'weight': _uniform(kg(), wshape, bound), 'bias': _uniform(kg(), (self.out_ch,), bound)}

    def apply(self, p, x, ctx):
        assert x.shape[1] == self.in_ch, f"Conv expected {self.in_ch} input channels, got {x.shape[1]}"
        w, b = p['weight'], p['bias']
        if ctx.amp_dtype is not None:
            x, w, b = x.astype(ctx.amp_dtype), w.astype(ctx.amp_dtype), b.astype(ctx.amp_dtype)
        sp = 'DHW'[3 - self.dims :]
        dn = lax.conv_dimension_numbers(x.shape, w.shape, ('NC' + sp, 'OI' + sp, 'NC' + sp))
        y = lax.conv_general_dilated(x, w, self.stride, [(q, q) for q in self.padding], dimension_numbers=dn)
        return y + b.reshape((1, -1) + (1,) * self.dims)


def conv_nd(dims, *args, **kwargs):
    return Conv(dims, *args, **kwargs)


class Linear:

    def __init__(self, in_features, out_features, bias=True):
        self.in_features, self.out_features, self.bias = in_features, out_features, bias

    def init(self, kg):
        bound = 1.0 / math.sqrt(self.in_features)
        p = {'weight': _uniform(kg(), (self.out_features, self.in_features), bound)}
        if self.bias:
            p['bias'] = _uniform(kg(), (self.out_features,), bound)
        return p

    def apply(self, p, x, ctx):
        w = p['weight']
        b = p['bias'] if self.bias else None
        if ctx.amp_dtype is not None:
            x, w = x.astype(ctx.amp_dtype), w.astype(ctx.amp_dtype)
            b = b.astype(ctx.amp_dtype) if b is not None else None
        y = jnp.matmul(x, w.T)
        return y + b if b is not None else y


def linear(*args, **kwargs):
    return Linear(*args, **kwargs)


class Dropout:

    def __init__(self, p):
        self.p = p

    def init(self, kg):
        return {}

    def apply(self, p_, x, ctx):
        if not ctx.training or self.p == 0:
            return x
        keep = jax.random.bernoulli(ctx.next_key(), 1.0 - self.p, x.shape)
        return jnp.where(keep, x / (1.0 - self.p), jnp.zeros_like(x))


class AvgPool:

    def __init__(self, dims, kernel_size, stride):
        if dims not in (1, 2, 3):
            raise ValueError(f"unsupported dimensions: {dims}")
        self.dims = dims
        self.kernel = (kernel_size,) * dims if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = (stride,) * dims if isinstance(stride, int) else tuple(stride)

    def init(self, kg):
        return {}

    def apply(self, p, x, ctx):
        window, strides = (1, 1) + self.kernel, (1, 1) + self.stride
        s = lax.reduce_window(x.astype(jnp.float32), 0.0, lax.add, window, strides, 'VALID')
        return (s / float(np.prod(self.kernel))).astype(x.dtype)


def avg_pool_nd(dims, *args, **kwargs):
    return AvgPool(dims, *args, **kwargs)


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half)
    args = timesteps[:, None].astype(jnp.float32) * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    return embedding


def flatten_tree(tree, prefix=''):
    out = {}
    for k, v in tree.items():
        name = f'{prefix}.{k}' if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_tree(v, name))
        else:
            out[name] = np.asarray(v)
    return out


def checkpoint(func, inputs, params, flag):
    if flag:
        return jax.checkpoint(func)(*inputs)
    else:
        return func(*inputs)
