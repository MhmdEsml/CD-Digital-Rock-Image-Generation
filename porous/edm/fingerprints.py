from porous.core.fingerprint import fp_compare, network_fingerprint, porosity_fingerprint


def check_fingerprints(info, unet, net_params, allow_missing=False):
    need = ['network_fingerprint'] + (['porosity_fingerprint'] if getattr(unet, 'fraction_dim', 0) else [])
    missing = [k for k in need if info.get(k) is None]
    if missing:
        msg = (
            f'teacher_info.json has no {missing}: the U-Net of this code cannot be compared numerically with the teacher\'s, so a same-shape / different-function '
            'mismatch (another Fourier frequency base, the embedding injected elsewhere, anisotropic sampling, ...) would load silently. Re-export the teacher with '
            'this code (it writes the fingerprint fields into teacher_info.json), or set teacher.allow_missing_fingerprint = true in the CD config '
            'to continue without this safeguard.'
        )
        if not allow_missing:
            raise RuntimeError(msg)
        print('[WARN] ' + msg)
    res = {}
    fp = info.get('porosity_fingerprint')
    if fp is not None and need[-1] == 'porosity_fingerprint':
        got = porosity_fingerprint(unet, net_params, fp['porosities'])
        c1, c2 = fp_compare(fp['embedding'], got['embedding']), fp_compare(fp['null'], got['null'])
        res['porosity'] = (c1, c2)
        print(
            f"porosity fingerprint: embedding rel. error {c1['norm_rel']:.2e} (max |diff| {c1['max_abs']:.2e}), null rel. error {c2['norm_rel']:.2e} (max |diff| {c2['max_abs']:.2e})"
        )
        if not (c1['ok'] and c2['ok']):
            raise RuntimeError(
                'the porosity branch of this code computes something different from the teacher\'s although the parameter names and shapes match: the embedding code '
                f'(Fourier frequency base / levels / layout / null embedding) differs. embedding {c1}, null {c2}. Use the same U-Net implementation as the teacher.'
            )
    nf = info.get('network_fingerprint')
    if nf is not None:
        got = network_fingerprint(unet, net_params, nf['shape'], nf['c_noise'], nf['porosities'], nf['seed'], nf['n_values'])
        c = fp_compare(nf['forward'], got['forward'])
        res['network'] = c
        print(
            f"network fingerprint: relative error {c['norm_rel']:.2e} (max |diff| {c['max_abs']:.2e}) over {len(nf['forward'])} forward passes"
        )
        if not c['ok']:
            raise RuntimeError(
                'the FULL network output of this code differs from the teacher\'s on the fixed fingerprint input although the parameter names and shapes match '
                f'(and the porosity branch itself may agree): {c}. The way the embedding is combined, the down/up-sampling layers or the normalisation differ. '
                'Use the same U-Net implementation as the teacher.'
            )
    return res
