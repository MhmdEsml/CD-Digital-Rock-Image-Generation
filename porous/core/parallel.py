import numpy as np
import jax


def _sum_local(x):
    try:
        return sum(np.asarray(s.data, np.float64) for s in x.addressable_shards).reshape(-1)
    except AttributeError:
        return np.asarray(x, np.float64).sum(0)


def _replicate_for_pmap(tree, n):
    stacked = jax.tree_util.tree_map(lambda x: np.stack([np.asarray(x)] * n), tree)
    return jax.pmap(lambda s: s, axis_name='batch')(stacked)


def _replica0(tree):
    def first(x):
        try:
            return np.asarray(x.addressable_shards[0].data)[0]
        except AttributeError:
            return np.asarray(x)[0]

    return jax.tree_util.tree_map(first, tree)
