import re

from porous.core.consistency_math import parse_schedule, snap_taus_to_grid


def ordinary_taus(n_steps, cd):
    n_steps = int(n_steps)
    if n_steps <= 1:
        return []
    smin, smax, rho = float(cd['sigma_min']), float(cd['sigma_max']), float(cd['rho'])
    raw = [(smax ** (1 / rho) + (i / n_steps) * (smin ** (1 / rho) - smax ** (1 / rho))) ** rho for i in range(1, n_steps)]
    taus = list(snap_taus_to_grid(raw, cd))
    if len(taus) != n_steps - 1:
        print(f'  WARNING: {n_steps}-step schedule collapsed to {len(taus)} intermediate sigma(s) after snapping to the training grid')
    return taus


def is_excluded_schedule(name, spec, patterns, gamma_excluded=(0.3,)):
    if any(p in str(name) for p in patterns):
        return True
    try:
        return any(abs(parse_schedule(spec)[1] - g) < 1e-9 for g in gamma_excluded)
    except Exception:
        return False


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")


def edm_schedules(cfg, info):
    sp = info["sampling"]
    sampler = cfg.edm_sampler or sp.get("sampler", "heun")
    if sampler not in ("heun", "euler"):
        raise ValueError(f"EDM_SAMPLER must be 'heun' or 'euler', got {sampler!r}")
    fs = float(cfg.edm_cfg_scale if cfg.edm_cfg_scale is not None else sp.get("porosity_cfg_scale", 1.0))
    churn = float(cfg.edm_s_churn if cfg.edm_s_churn is not None else sp.get("S_churn", 0.0))
    out = {}
    for n in cfg.edm_steps:
        n = int(n)
        name = f"{sampler}-{n}" + (f"_cfg{fs:g}" if fs != 1.0 else "") + (f"_churn{churn:g}" if churn > 0 else "")
        out[name] = dict(
            steps=n,
            sampler=sampler,
            cfg_scale=fs,
            S_churn=churn,
            S_tmin=float(sp.get("S_tmin", 0.0)),
            S_tmax=float(sp.get("S_tmax", 1e9)),
            S_noise=float(sp.get("S_noise", 1.0)),
        )
    return out
