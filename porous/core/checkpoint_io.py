import copy
import hashlib
import os
import pickle
import shutil
import threading
import time

import numpy as np

from porous.core.nn import flatten_tree


def hash_state(sd):
    h = hashlib.blake2b(digest_size=16)
    for k in sorted(sd):
        v = np.asarray(sd[k])
        h.update(k.encode())
        h.update(("torch." + v.dtype.name).encode())
        h.update(np.ascontiguousarray(v).tobytes())
    return h.hexdigest()


def load_state_dict_into(template, flat, what):
    unexpected = sorted(set(flat) - set(flatten_tree(template)))
    if unexpected:
        raise KeyError(f'unexpected keys in {what}: {unexpected[:5]}{" ..." if len(unexpected) > 5 else ""}')
    return fill_tree(template, flat)


class _TolerantUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == '_frombuffer' and module.startswith('numpy'):

            def _fb(buf, dtype, shape, order, axis_order=None):
                a = np.frombuffer(buf, dtype=dtype)
                if order == 'K' and axis_order is not None:
                    return a.reshape(shape, order='C').transpose(axis_order)
                return a.reshape(shape, order=order)

            return _fb
        return super().find_class(module, name)


def load_pickle_file(f):
    return _TolerantUnpickler(f).load()


def save_checkpoint_file(path_, payload):
    os.makedirs(os.path.dirname(path_), exist_ok=True)
    tmp_path = path_ + '.tmp'
    with open(tmp_path, 'wb') as f:
        pickle.dump(payload, f, protocol=4)
    os.replace(tmp_path, path_)


def _write_checkpoint_files(paths, payload):
    save_checkpoint_file(paths[0], payload)
    for extra in paths[1:]:
        os.makedirs(os.path.dirname(extra) or '.', exist_ok=True)
        shutil.copyfile(paths[0], extra + '.tmp')
        os.replace(extra + '.tmp', extra)


def _snapshot_payload(payload):
    memo = {}

    def reg(o):
        if isinstance(o, np.ndarray):
            memo[id(o)] = o
        elif isinstance(o, dict):
            for v in o.values():
                reg(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                reg(v)

    reg(payload)
    return copy.deepcopy(payload, memo)


class _CheckpointWriter:
    def __init__(self):
        self._thread, self._error = None, None
        self.n_skipped = 0
        self.n_written = 0
        self.write_seconds = []

    def busy(self):
        return self._thread is not None and self._thread.is_alive()

    def _check(self):
        if self._error is not None:
            err, self._error = self._error, None
            raise RuntimeError(f'a background checkpoint write failed: {err!r}') from err

    def submit(self, paths, payload, info, t_device, label='checkpoint', must=False):
        self._check()
        if self.busy():
            assert must, 'submit() while a write is running is only allowed for must-saves (the caller skips routine saves)'
            self._thread.join()
            self._check()

        def work():
            t0 = time.perf_counter()
            try:
                _write_checkpoint_files(paths, payload)
                dt = time.perf_counter() - t0
                self.n_written += 1
                self.write_seconds.append(dt)
                copies = f' (+{len(paths) - 1} snapshot copy)' if len(paths) > 1 else ''
                print(
                    f'Saved {label} -> {paths[0]}{copies} ({info}; device->host {t_device:.1f} s, pickle+write {dt:.1f} s, on a background thread)'
                )
            except BaseException as e:
                self._error = e

        self._thread = threading.Thread(target=work, name='checkpoint-writer')
        self._thread.start()

    def wait(self):
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._check()


def load_checkpoint_file(path_):
    with open(path_, 'rb') as f:
        return load_pickle_file(f)


def fill_tree(template, flat, prefix=''):
    if isinstance(template, dict):
        return {k: fill_tree(v, flat, f'{prefix}.{k}' if prefix else str(k)) for k, v in template.items()}
    if prefix not in flat:
        raise KeyError(f'missing key {prefix!r} in checkpoint state dict')
    arr = np.asarray(flat[prefix])
    if arr.shape != tuple(np.shape(template)):
        raise ValueError(f'shape mismatch for {prefix!r}: checkpoint {arr.shape} vs model {tuple(np.shape(template))}')
    return arr
