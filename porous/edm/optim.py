import jax
import jax.numpy as jnp


def resolve_precision(train_cfg):
    cd = str(train_cfg.get('compute_dtype', 'float32')).lower()
    if cd not in ('float32', 'bfloat16'):
        raise ValueError(f"training.compute_dtype must be 'float32' or 'bfloat16', got {cd!r}")
    if cd == 'bfloat16':
        return {
            'compute_dtype': cd,
            'amp_dtype': jnp.bfloat16,
            'use_scaler': False,
            'label': 'bfloat16 compute (float32 params / EMA / optimizer / norms / loss), no loss scaling',
        }
    use = jax.default_backend() == 'gpu' and bool(train_cfg.get('amp_on_gpu', True))
    return {
        'compute_dtype': cd,
        'amp_dtype': jnp.float16 if use else None,
        'use_scaler': use,
        'label': 'fp16 autocast + dynamic loss scaling (GPU, as in A)' if use else 'float32',
    }


class AdamWHost:

    def __init__(self, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        self.param_groups = [
            {
                'lr': lr,
                'betas': tuple(betas),
                'eps': eps,
                'weight_decay': weight_decay,
                'amsgrad': False,
                'maximize': False,
                'foreach': None,
                'capturable': False,
                'differentiable': False,
                'fused': None,
            }
        ]


class _LRScheduler:
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
        self.last_epoch = last_epoch
        self._initial_step()

    def _initial_step(self):
        self._step_count = 0
        self.step()

    def get_lr(self):
        raise NotImplementedError

    def step(self):
        self._step_count += 1
        self.last_epoch += 1
        values = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, values):
            group['lr'] = lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


class ConstantLR(_LRScheduler):
    def __init__(self, optimizer, factor=1.0 / 3, total_iters=5, last_epoch=-1):
        if factor > 1.0 or factor < 0:
            raise ValueError("Constant multiplicative factor expected to be between 0 and 1.")
        self.factor = factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return [group["lr"] * self.factor for group in self.optimizer.param_groups]
        if self.last_epoch != self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [group["lr"] * (1.0 / self.factor) for group in self.optimizer.param_groups]


class LinearLR(_LRScheduler):
    def __init__(self, optimizer, start_factor=1.0 / 3, end_factor=1.0, total_iters=5, last_epoch=-1):
        if start_factor > 1.0 or start_factor <= 0:
            raise ValueError("Starting multiplicative factor expected to be greater than 0 and less or equal to 1.")
        if end_factor > 1.0 or end_factor < 0:
            raise ValueError("Ending multiplicative factor expected to be between 0 and 1.")
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return [group["lr"] * self.start_factor for group in self.optimizer.param_groups]
        if self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [
            group["lr"]
            * (
                1.0
                + (self.end_factor - self.start_factor)
                / (self.total_iters * self.start_factor + (self.last_epoch - 1) * (self.end_factor - self.start_factor))
            )
            for group in self.optimizer.param_groups
        ]


def build_lr_schedule(optimizer, train_cfg):
    if train_cfg['decay_lr']:
        return LinearLR(optimizer, total_iters=train_cfg['total_epochs'], start_factor=1.0, end_factor=1e-8 / train_cfg['lr'])
    return ConstantLR(optimizer, total_iters=train_cfg['total_epochs'], factor=1.0)


def clip_grad_norm(grads, max_norm):
    leaves = jax.tree_util.tree_leaves(grads)
    total_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g.astype(jnp.float32))) for g in leaves))
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = jnp.minimum(clip_coef, 1.0)
    return jax.tree_util.tree_map(lambda g: g * clip_coef_clamped, grads)


def grad_scaler_update(scale, growth_tracker, found_inf, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000):
    tracker_inc = growth_tracker + 1
    grow = jnp.logical_and(jnp.logical_not(found_inf), tracker_inc == growth_interval)
    grown = scale * growth_factor
    grown = jnp.where(jnp.isfinite(grown), grown, scale)
    new_scale = jnp.where(found_inf, scale * backoff_factor, jnp.where(grow, grown, scale))
    new_tracker = jnp.where(jnp.logical_or(found_inf, grow), 0, tracker_inc).astype(growth_tracker.dtype)
    return new_scale, new_tracker


class NativeScalerWithGradNormCount:
    state_dict_key = 'amp_scaler'

    def __init__(self, enabled, init_scale=2.0**16, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000):
        self._enabled = bool(enabled)
        self._init_scale = init_scale
        self._scale = init_scale if self._enabled else None
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._growth_tracker = 0 if self._enabled else None

    def state_dict(self):
        if not self._enabled:
            return {}
        return {
            "scale": self._scale,
            "growth_factor": self._growth_factor,
            "backoff_factor": self._backoff_factor,
            "growth_interval": self._growth_interval,
            "_growth_tracker": self._growth_tracker,
        }

    def load_state_dict(self, state_dict):
        if not self._enabled:
            return
        if len(state_dict) == 0:
            raise RuntimeError("The source state dict is empty, possibly because it was saved from a disabled " "instance of GradScaler.")
        self._scale = state_dict["scale"]
        self._growth_factor = state_dict["growth_factor"]
        self._backoff_factor = state_dict["backoff_factor"]
        self._growth_interval = state_dict["growth_interval"]
        self._growth_tracker = state_dict["_growth_tracker"]


def lr_at_step(step, base_lr, warmup_steps):
    if warmup_steps <= 0:
        return base_lr
    return base_lr * min(1.0, (step + 1) / warmup_steps)


def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr
