import math
import jax
import jax.numpy as jnp
from .nn import Dropout, Identity, avg_pool_nd, conv_nd, linear, normalization


class ConstantEmbedding:
    def __init__(self, in_channels, out_channels):
        self.in_channels, self.out_channels = in_channels, out_channels

    def init(self, kg):
        b = self.in_channels**0.5
        return {'embedding_table': jax.random.uniform(kg(), (1, self.out_channels), jnp.float32, -b, b)}

    def apply(self, p, emb, ctx):
        return jnp.tile(p['embedding_table'], (emb.shape[0], 1))


class QKVAttentionLegacy:

    def __init__(self, n_heads):
        self.n_heads = n_heads

    def __call__(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = jnp.split(qkv.reshape(bs * self.n_heads, ch * 3, length), 3, axis=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = jnp.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = jax.nn.softmax(weight.astype(jnp.float32), axis=-1).astype(weight.dtype)
        a = jnp.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class QKVAttention:

    def __init__(self, n_heads):
        self.n_heads = n_heads

    def __call__(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = jnp.split(qkv, 3, axis=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = jnp.einsum(
            "bct,bcs->bts",
            (q * scale).reshape(bs * self.n_heads, ch, length),
            (k * scale).reshape(bs * self.n_heads, ch, length),
        )
        weight = jax.nn.softmax(weight.astype(jnp.float32), axis=-1).astype(weight.dtype)
        a = jnp.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)


class TimestepBlock:
    pass


class Sequential:

    def __init__(self, *layers):
        self.layers = list(layers)

    def init(self, kg):
        return {str(i): layer.init(kg) for i, layer in enumerate(self.layers)}

    def apply(self, p, x, ctx):
        for i, layer in enumerate(self.layers):
            x = layer.apply(p[str(i)], x, ctx)
        return x


class TimestepEmbedSequential(Sequential):

    def apply(self, p, x, emb, ctx):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, TimestepBlock):
                x = layer.apply(p[str(i)], x, emb, ctx)
            else:
                x = layer.apply(p[str(i)], x, ctx)
        return x


class Upsample:

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def init(self, kg):
        return {'conv': self.conv.init(kg)} if self.use_conv else {}

    def apply(self, p, x, ctx):
        assert x.shape[1] == self.channels
        axes = tuple(range(2, x.ndim))
        assert len(axes) == self.dims, f'Upsample(dims={self.dims}) got a {x.ndim}-D tensor'
        for a in axes:
            x = jnp.repeat(x, 2, axis=a)
        if self.use_conv:
            x = self.conv.apply(p['conv'], x, ctx)
        return x


class Downsample:

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def init(self, kg):
        return {'op': self.op.init(kg)} if self.use_conv else {}

    def apply(self, p, x, ctx):
        assert x.shape[1] == self.channels
        return self.op.apply(p['op'] if self.use_conv else {}, x, ctx)


class ResBlock(TimestepBlock):

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        emb_off=False,
    ):
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_norm = normalization(channels)
        self.in_conv = conv_nd(dims, channels, self.out_channels, 3, padding=1)

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = Identity()

        self.emb_off = emb_off
        emb_out_ch = 2 * self.out_channels if use_scale_shift_norm else self.out_channels
        if emb_off:
            self.emb_const = ConstantEmbedding(emb_channels, emb_out_ch)
        else:
            self.emb_linear = linear(emb_channels, emb_out_ch)

        self.out_norm = normalization(self.out_channels)
        self.out_dropout = Dropout(p=dropout)
        self.out_conv = conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1, zero=True)

        if self.out_channels == channels:
            self.skip_connection = Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def init(self, kg):
        p = {
            'in_layers': {'0': self.in_norm.init(kg), '2': self.in_conv.init(kg)},
            'emb_layers': self.emb_const.init(kg) if self.emb_off else {'1': self.emb_linear.init(kg)},
            'out_layers': {'0': self.out_norm.init(kg), '3': self.out_conv.init(kg)},
        }
        if not isinstance(self.skip_connection, Identity):
            p['skip_connection'] = self.skip_connection.init(kg)
        return p

    def apply(self, p, x, emb, ctx):
        if self.use_checkpoint and ctx.training:
            key = ctx.next_key()
            f = lambda p_, x_, e_, k_: self._forward(p_, x_, e_, ctx.child(k_))
            return jax.checkpoint(f)(p, x, emb, key)
        return self._forward(p, x, emb, ctx)

    def _forward(self, p, x, emb, ctx):
        silu = lambda v: v * jax.nn.sigmoid(v)
        if self.updown:
            h = silu(self.in_norm.apply(p['in_layers']['0'], x, ctx))
            h = self.h_upd.apply({}, h, ctx)
            x = self.x_upd.apply({}, x, ctx)
            h = self.in_conv.apply(p['in_layers']['2'], h, ctx)
        else:
            h = self.in_conv.apply(p['in_layers']['2'], silu(self.in_norm.apply(p['in_layers']['0'], x, ctx)), ctx)
        if self.emb_off:
            emb_out = self.emb_const.apply(p['emb_layers'], emb, ctx)
        else:
            emb_out = self.emb_linear.apply(p['emb_layers']['1'], silu(emb), ctx)
        emb_out = emb_out.astype(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            scale, shift = jnp.split(emb_out, 2, axis=1)
            h = self.out_norm.apply(p['out_layers']['0'], h, ctx) * (1 + scale) + shift
            h = self.out_conv.apply(p['out_layers']['3'], self.out_dropout.apply({}, silu(h), ctx), ctx)
        else:
            h = h + emb_out
            h = self.out_norm.apply(p['out_layers']['0'], h, ctx)
            h = self.out_conv.apply(p['out_layers']['3'], self.out_dropout.apply({}, silu(h), ctx), ctx)
        skip = x if isinstance(self.skip_connection, Identity) else self.skip_connection.apply(p['skip_connection'], x, ctx)
        return skip + h


class AttentionBlock:

    def __init__(self, channels, num_heads=1, num_head_channels=-1, use_checkpoint=False, use_new_attention_order=False):
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            self.attention = QKVAttention(self.num_heads)
        else:
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = conv_nd(1, channels, channels, 1, zero=True)

    def init(self, kg):
        return {'norm': self.norm.init(kg), 'qkv': self.qkv.init(kg), 'proj_out': self.proj_out.init(kg)}

    def apply(self, p, x, ctx):
        if self.use_checkpoint and ctx.training:
            return jax.checkpoint(lambda p_, x_: self._forward(p_, x_, ctx))(p, x)
        return self._forward(p, x, ctx)

    def _forward(self, p, x, ctx):
        b, c = x.shape[:2]
        spatial = x.shape[2:]
        x = x.reshape(b, c, -1)
        qkv = self.qkv.apply(p['qkv'], self.norm.apply(p['norm'], x, ctx), ctx)
        h = self.attention(qkv)
        h = self.proj_out.apply(p['proj_out'], h, ctx)
        return (x + h).reshape((b, c) + tuple(spatial))
