import numpy as np
import jax
import jax.numpy as jnp


def adamw_init(params):
    zeros = lambda: jax.tree_util.tree_map(lambda p: np.zeros(np.shape(p), np.float32), params)
    return {'step': np.int32(0), 'exp_avg': zeros(), 'exp_avg_sq': zeros()}


def adamw_update(params, grads, opt_state, lr, betas, eps=1e-8, weight_decay=1e-2):
    b1, b2 = betas
    step = opt_state['step'] + 1
    stepf = step.astype(jnp.float32)
    bias_correction1 = 1.0 - b1**stepf
    bias_correction2 = 1.0 - b2**stepf
    step_size = lr / bias_correction1
    bias_correction2_sqrt = jnp.sqrt(bias_correction2)

    def upd(p, g, m, v):
        p = p * (1.0 - lr * weight_decay)
        m = m + (g - m) * (1.0 - b1)
        v = v * b2 + (1.0 - b2) * g * g
        denom = jnp.sqrt(v) / bias_correction2_sqrt + eps
        p = p - step_size * (m / denom)
        return p, m, v

    leaves_p, treedef = jax.tree_util.tree_flatten(params)
    leaves_g = treedef.flatten_up_to(grads)
    leaves_m = treedef.flatten_up_to(opt_state['exp_avg'])
    leaves_v = treedef.flatten_up_to(opt_state['exp_avg_sq'])
    out = [upd(p, g, m, v) for p, g, m, v in zip(leaves_p, leaves_g, leaves_m, leaves_v)]
    new_params = treedef.unflatten([o[0] for o in out])
    new_m = treedef.unflatten([o[1] for o in out])
    new_v = treedef.unflatten([o[2] for o in out])
    return new_params, {'step': step, 'exp_avg': new_m, 'exp_avg_sq': new_v}
