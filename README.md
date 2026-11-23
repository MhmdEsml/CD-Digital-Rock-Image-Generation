# EDM teacher and consistency-distilled (CD) student for 3-D porous media (JAX)

Code for *Distilling Diffusion Models for Efficient Generation of 3D Digital Rock Microstructures*:
a porosity-conditioned EDM teacher (Karras et al.) is trained on 64³ binary patches of a micro-CT volume,
then distilled into a consistency-distillation student that generates volumes in 1–8 network evaluations.

## Layout

```
train_edm.py        train the EDM teacher (single or multi accelerator) and export the teacher
train_cd.py         distil the teacher into the CD student and export the student
sample.py           generate volumes from the teacher and/or the student
configs/            edm.json, cd.json, sample.json (all hyper-parameters; --set key.sub=value overrides)
porous/core/        shared code: layers, 3-D U-Net, EDM preconditioning, checkpoint I/O, EMA, AdamW, fingerprints, consistency maths
porous/edm/         teacher: config, data, model, optimiser, EMA, checkpoints, training, export
porous/cd/          student: config, data, teacher loading, consistency maths, training, export
porous/sampling/    schedules, checkpoint loading, samplers, outputs
```

## Install

```
pip install -U "jax[cuda12]"      # or plain `pip install -U jax` for CPU
pip install -r requirements.txt
```

## Data

A single binary volume as a raw `uint8` file (`0` = pore, `1` = grain; a `.npy` array also works for the
teacher), e.g. `Berea.raw` with dimensions 1000 1000 1000.

## 1. Train the EDM teacher

```
python train_edm.py --volume /data/Berea.raw --volume-dims 1000 1000 1000 --work-dir runs/edm
# resume after an interruption: add  --resume from_disk
```
At the end the teacher is exported to `runs/edm/teacher/` (`edm.pkl`, `teacher_info.json`, `eval_bundle.zip`).
Export from an existing checkpoint: `python train_edm.py --export-only --checkpoint <ckpt.pkl> --work-dir runs/edm`.

## 2. Distil the CD student

```
python train_cd.py --teacher-dir runs/edm/teacher --volume /data/Berea.raw --work-dir runs/cd
```
The student is exported to `runs/cd/student/` (`cd_student.pkl`, `cd_train_state.pkl`, `cd_config.json`).

## 3. Generate samples

```
python sample.py --model edm  --teacher-dir runs/edm/teacher --volume /data/Berea.raw --out-dir samples --edm-steps 8,16,32
python sample.py --model cd   --student runs/cd/student/cd_student.pkl --volume /data/Berea.raw --out-dir samples --cd-steps 1,2,4,8
python sample.py --model both --teacher-dir runs/edm/teacher --student runs/cd/student/cd_student.pkl --volume /data/Berea.raw
```
Target porosities are those of `--n-samples` (default 3000) real patches drawn from the volume with `--seed`;
sample *i* is conditioned on the porosity of real patch *i*. Per schedule, in
`<out-dir>/<dataset>/<model>/<checkpoint tag>/<schedule>/`:

| file | content |
|---|---|
| `generated_samples.npy` | float32 `(N, D, H, W, 1)`, pore = −1, grain = +1 |
| `generated_binary.npz` | the same, bit-packed (1 = pore) |
| `target_porosity.npy` | conditioning porosities |
| `sampling_info.json` | checkpoint, schedule, seed, timing |

plus `real_samples.csv` (patch coordinates) in `<out-dir>/<dataset>/shared/`.

All scripts use every local accelerator (data-parallel).

## Citation

If you use this code in your research, please cite the accompanying paper:

> Esmaeili, M. (2026). *Distilling Diffusion Models for Efficient Generation of 3D Digital Rock Microstructures*.
> SSRN. https://doi.org/10.2139/ssrn.7695998

```bibtex
@article{esmaeili2026distilling,
  title   = {Distilling Diffusion Models for Efficient Generation of 3{D} Digital Rock Microstructures},
  author  = {Esmaeili, Mohammad},
  year    = {2026},
  month   = nov,
  journal = {SSRN Electronic Journal},
  doi     = {10.2139/ssrn.7695998},
  url     = {https://ssrn.com/abstract=7695998}
}
```

The EDM teacher builds on Karras et al., *Elucidating the Design Space of Diffusion-Based Generative Models* (NeurIPS 2022);
please also cite that work where appropriate.

## License

Released under the [MIT License](LICENSE). You may use, modify and redistribute the code, including in your own research and
publications, provided that the copyright notice and the licence text are kept in copies or substantial portions of the code.
