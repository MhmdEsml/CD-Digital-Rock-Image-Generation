import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from .nn import Identity, SiLU, conv_nd, linear, normalization, timestep_embedding
from .unet_layers import AttentionBlock, Downsample, ResBlock, Sequential, TimestepEmbedSequential, Upsample


def unet_level_widths(model_channels, channel_mult):
    return [int(m * model_channels) for m in channel_mult]


def check_unet_widths(model_channels, channel_mult, groups=32):
    widths = unet_level_widths(model_channels, channel_mult)
    bad = [w for w in widths if w % groups]
    if bad:
        raise ValueError(
            f'GroupNorm32 needs channel widths divisible by {groups}, but model_channels={model_channels} with channel_mult={list(channel_mult)} '
            f'gives widths {widths}; offending: {bad}. Use e.g. model_channels=32 or 64 (model_channels=48 fails: 48 % 32 = 16).'
        )
    return widths


def porosity_fourier_features(p, levels):
    p = p.astype(jnp.float32)
    freqs = (2.0 ** jnp.arange(levels, dtype=jnp.float32)) * jnp.pi
    a = p[:, :, None] * freqs[None, None, :]
    return jnp.concatenate([jnp.sin(a), jnp.cos(a)], axis=-1).reshape(p.shape[0], -1)


class UNetModel:

    def __init__(
        self,
        in_channels,
        model_channels=32,
        out_channels=1,
        num_res_blocks=2,
        attention_resolutions=(),
        dropout=0.0,
        channel_mult=(1, 2, 2, 4),
        conv_resample=True,
        dims=3,
        num_classes=None,
        fraction_dim=0,
        porosity_embedding='fourier',
        porosity_fourier_levels=8,
        porosity_dropout=0.0,
        use_checkpoint=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        with_fourier_features=False,
        ignore_time=False,
        input_projection=True,
        image_size=-1,
        max_attention_tokens=4096,
        _target_="lib.models.gd_unet.UNetModel",
    ):
        if with_fourier_features:
            raise NotImplementedError(
                'with_fourier_features is not supported in this 3-D U-Net (the +12 input channels assume 3 RGB-like channels); keep it False'
            )
        if porosity_embedding not in ('linear', 'fourier'):
            raise ValueError(f"porosity_embedding must be 'linear' or 'fourier', got {porosity_embedding!r}")
        if not 0.0 <= porosity_dropout <= 1.0:
            raise ValueError(f'porosity_dropout must be in [0, 1], got {porosity_dropout}')
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = tuple(attention_resolutions)
        self.dropout = dropout
        self.channel_mult = tuple(channel_mult)
        self.conv_resample = conv_resample
        self.dims = dims
        self.num_classes = num_classes
        self.fraction_dim = fraction_dim
        self.porosity_embedding = porosity_embedding
        self.porosity_fourier_levels = porosity_fourier_levels
        self.porosity_dropout = porosity_dropout
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.use_scale_shift_norm = use_scale_shift_norm
        self.resblock_updown = resblock_updown
        self.use_new_attention_order = use_new_attention_order
        self.with_fourier_features = with_fourier_features
        self.ignore_time = ignore_time
        self.input_projection = input_projection
        self.image_size = image_size
        self.max_attention_tokens = max_attention_tokens
        self._target_ = _target_
        self._attn_sites = []

        check_unet_widths(self.model_channels, self.channel_mult)

        if self.num_heads_upsample == -1:
            self.num_heads_upsample = self.num_heads

        self.time_embed_dim = self.model_channels * 4
        if self.ignore_time:
            self.time_embed = None
        else:
            self.time_embed = Sequential(
                linear(self.model_channels, self.time_embed_dim),
                SiLU(),
                linear(self.time_embed_dim, self.time_embed_dim),
            )

        self.fraction_emb = None
        self.porosity_mlp = None
        if self.fraction_dim:
            if porosity_embedding == 'linear':
                self.fraction_emb = linear(self.fraction_dim, self.time_embed_dim, bias=False)
            else:
                f_in = self.fraction_dim * 2 * self.porosity_fourier_levels
                self.porosity_mlp = Sequential(linear(f_in, self.time_embed_dim), SiLU(), linear(self.time_embed_dim, self.time_embed_dim))

        ch = input_ch = int(self.channel_mult[0] * self.model_channels)
        if self.input_projection:
            self.input_blocks = [TimestepEmbedSequential(conv_nd(self.dims, self.in_channels, ch, 3, padding=1))]
        else:
            self.input_blocks = [TimestepEmbedSequential(Identity())]
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        self.time_embed_dim,
                        self.dropout,
                        out_channels=int(mult * self.model_channels),
                        dims=self.dims,
                        use_checkpoint=self.use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                        emb_off=self.ignore_time and self.num_classes is None,
                    )
                ]
                ch = int(mult * self.model_channels)
                if ds in self.attention_resolutions:
                    self._attn_sites.append((f'input_blocks.{len(self.input_blocks)}.{len(layers)}', ds))
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=self.use_checkpoint,
                            num_heads=self.num_heads,
                            num_head_channels=self.num_head_channels,
                            use_new_attention_order=self.use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(self.channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            self.time_embed_dim,
                            self.dropout,
                            out_channels=out_ch,
                            dims=self.dims,
                            use_checkpoint=self.use_checkpoint,
                            use_scale_shift_norm=self.use_scale_shift_norm,
                            down=True,
                            emb_off=self.ignore_time and self.num_classes is None,
                        )
                        if self.resblock_updown
                        else Downsample(ch, self.conv_resample, dims=self.dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self._attn_sites.append(('middle_block.1', ds))
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                self.time_embed_dim,
                self.dropout,
                dims=self.dims,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm,
                emb_off=self.ignore_time and self.num_classes is None,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=self.use_checkpoint,
                num_heads=self.num_heads,
                num_head_channels=self.num_head_channels,
                use_new_attention_order=self.use_new_attention_order,
            ),
            ResBlock(
                ch,
                self.time_embed_dim,
                self.dropout,
                dims=self.dims,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm,
                emb_off=self.ignore_time and self.num_classes is None,
            ),
        )
        self._feature_size += ch

        self.output_blocks = []
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for i in range(self.num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        self.time_embed_dim,
                        self.dropout,
                        out_channels=int(self.model_channels * mult),
                        dims=self.dims,
                        use_checkpoint=self.use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                        emb_off=self.ignore_time and self.num_classes is None,
                    )
                ]
                ch = int(self.model_channels * mult)
                if ds in self.attention_resolutions:
                    self._attn_sites.append((f'output_blocks.{len(self.output_blocks)}.{len(layers)}', ds))
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=self.use_checkpoint,
                            num_heads=self.num_heads_upsample,
                            num_head_channels=self.num_head_channels,
                            use_new_attention_order=self.use_new_attention_order,
                        )
                    )
                if level and i == self.num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            self.time_embed_dim,
                            self.dropout,
                            out_channels=out_ch,
                            dims=self.dims,
                            use_checkpoint=self.use_checkpoint,
                            use_scale_shift_norm=self.use_scale_shift_norm,
                            up=True,
                            emb_off=self.ignore_time and self.num_classes is None,
                        )
                        if self.resblock_updown
                        else Upsample(ch, self.conv_resample, dims=self.dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = Sequential(
            normalization(ch),
            SiLU(),
            conv_nd(self.dims, input_ch, self.out_channels, 3, padding=1, zero=True),
        )

        if self.image_size and self.image_size > 0:
            self.check_attention_budget((self.image_size,) * self.dims)

    def attention_report(self, spatial):
        down = 2 ** (len(self.channel_mult) - 1)
        if any(s % down for s in spatial):
            raise ValueError(f'input size {tuple(spatial)} must be divisible by {down} (= 2^(levels-1)) in every dimension')
        out = []
        for name, ds in self._attn_sites:
            res = tuple(s // ds for s in spatial)
            out.append((name, ds, res, int(np.prod(res))))
        return out

    def check_attention_budget(self, spatial):
        rep = self.attention_report(spatial)
        worst = max(rep, key=lambda r: r[3])
        if worst[3] > self.max_attention_tokens:
            raise ValueError(
                f'attention layer {worst[0]} (downsample x{worst[1]}) would attend over {worst[3]} tokens ({"x".join(map(str, worst[2]))}) at input size '
                f'{tuple(spatial)}, above architecture.max_attention_tokens={self.max_attention_tokens}. Remove {worst[1]} from attention_resolutions, '
                f'add a downsampling level (channel_mult), or raise the limit deliberately.'
            )
        return rep

    def init(self, kg):
        p = {}
        if self.time_embed is not None:
            p['time_embed'] = self.time_embed.init(kg)
        if self.fraction_emb is not None:
            p['fraction_emb'] = self.fraction_emb.init(kg)
        if self.porosity_mlp is not None:
            p['porosity_emb'] = self.porosity_mlp.init(kg)
        if self.fraction_dim:
            p['porosity_null_emb'] = {'weight': jnp.zeros((self.time_embed_dim,), jnp.float32)}
        if self.num_classes is not None:
            w = jax.random.normal(kg(), (self.num_classes + 1, self.time_embed_dim), jnp.float32)
            w = w.at[self.num_classes].set(0.0)
            p['label_emb'] = {'weight': w}
        p['input_blocks'] = {str(i): m.init(kg) for i, m in enumerate(self.input_blocks)}
        p['middle_block'] = self.middle_block.init(kg)
        p['output_blocks'] = {str(i): m.init(kg) for i, m in enumerate(self.output_blocks)}
        p['out'] = self.out.init(kg)
        return p

    def porosity_term(self, params, frac, n, dtype, ctx):
        null = params['porosity_null_emb']['weight'][None, :]
        if frac is None:
            return jnp.broadcast_to(null, (n, self.time_embed_dim)).astype(dtype)
        frac = jnp.asarray(frac).astype(jnp.float32).reshape(n, self.fraction_dim)
        if self.porosity_embedding == 'linear':
            pe = self.fraction_emb.apply(params['fraction_emb'], frac, ctx)
        else:
            pe = self.porosity_mlp.apply(params['porosity_emb'], porosity_fourier_features(frac, self.porosity_fourier_levels), ctx)
        pe = pe.astype(dtype)
        if ctx.training and self.porosity_dropout > 0:
            drop = jax.random.uniform(ctx.next_key(), (n, 1)) < self.porosity_dropout
            pe = jnp.where(drop, null.astype(dtype), pe)
        return pe

    def apply(self, params, x, timesteps, extra, ctx, trace=None):
        extra = dict(extra)
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels).astype(x.dtype)
        if self.ignore_time:
            emb = jnp.zeros((x.shape[0], self.time_embed_dim), dtype=x.dtype)
        else:
            emb = self.time_embed.apply(params['time_embed'], t_emb, ctx)
        if self.ignore_time:
            emb = emb * 0.0

        if self.num_classes and "label" not in extra:
            extra["label"] = jnp.full((x.shape[0],), self.num_classes, dtype=jnp.int32)
        if self.num_classes is not None and "label" in extra:
            y = extra["label"]
            assert y.shape == x.shape[:1], f"Labels have shape {y.shape}, which does not match the batch dimension of the input {x.shape}"
            table = params['label_emb']['weight']
            e = jnp.take(table, y, axis=0)
            e = jnp.where((y == self.num_classes)[:, None], lax.stop_gradient(e), e)
            emb = emb + e

        if self.fraction_dim:
            emb = emb + self.porosity_term(params, extra.get('fraction'), x.shape[0], emb.dtype, ctx)

        h = x
        for i, module in enumerate(self.input_blocks):
            h = module.apply(params['input_blocks'][str(i)], h, emb, ctx)
            hs.append(h)
            if trace is not None:
                trace.append((f'input_blocks.{i}', tuple(h.shape)))
        h = self.middle_block.apply(params['middle_block'], h, emb, ctx)
        if trace is not None:
            trace.append(('middle_block', tuple(h.shape)))
        for i, module in enumerate(self.output_blocks):
            h = jnp.concatenate([h, hs.pop()], axis=1)
            h = module.apply(params['output_blocks'][str(i)], h, emb, ctx)
            if trace is not None:
                trace.append((f'output_blocks.{i}', tuple(h.shape)))
        h = h.astype(x.dtype)
        result = self.out.apply(params['out'], h, ctx)
        return result


def make_unet(arch):
    return UNetModel(**arch)


def count_params(tree):
    return int(sum(int(np.prod(np.shape(l))) for l in jax.tree_util.tree_leaves(tree)))
