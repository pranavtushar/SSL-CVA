# SSL-CVA (Child-centric Voice Anonymization)

This repo contains the code used for **selective (reference/pool-based) anonymization** experiments from the paper, including **Base/Base (SSL-B)** and **FT/FT (SSL-FT)** variants.

## What you need on disk

1. **This repository** — your working copy of `SSL-CVA/`.
2. **Checkpoints** — under `all_checkpoints/`:
   - `all_checkpoints/HiFi-GAN_B_Soft_B` (Base/Base)
   - `all_checkpoints/HIFI-GAN_FT_Soft_FT` (FT/FT)
3. **Inputs** — for a transparent demo, this repo includes:
   - **Single-speaker**: `demo/input/audio/` + `demo/input/lists/demo_inputs_mps_5x5s.lst`
   - **Multi-speaker**: `demo/input/multispeaker/mixture.wav` + `demo/input/multispeaker/reference.wav`

Optional but convenient:

4. **Reference audio pool** (paper reference speakers):
   - Reference directory: `demo/reference_audio/`
   - List (for transparency): `demo/reference_audio/reference_audio.lst` (44 utterances)

## Path rules (important)

- **`--input_test_file`**: each line is an audio path. Relative paths are resolved relative to the list file’s directory. Lines starting with `#` are ignored.
- **`--reference_dir`**: must be a directory that contains reference `.wav`/`.flac` files (recursively). For paper-style reference speakers, use `demo/reference_audio/` (documented by `demo/reference_audio/reference_audio.lst`).

## 1. Copy

After `git clone` (or unpacking a tarball):

- Copy **checkpoints** into `all_checkpoints/` if they are not already present.

## 2. (Optional) Regenerate the demo input list (5 utterances ≥ 5s)

If you have the sibling dataset repo at `../latest-child-speech-dataset/`, you can (re)generate the demo set:

```bash
cd /path/to/SSL-CVA
python3 make_demo_mps_list.py --n 5 --min_seconds 5.0
```

This writes/refreshes the *source* demo artifacts (`demo_inputs/`, `demo_inputs_mps_5x5s.lst`). The repo also includes a cleaned, self-contained demo folder under `demo/` used in the commands below.

## 3. Run inference (BB or FT/FT)

Both runs below use:

- **Input list**: `demo_inputs_mps_5x5s.lst`
- **Reference dir**: `demo/reference_audio/` (documented by `demo/reference_audio/reference_audio.lst`)

### FT/FT (paper default, SSL-FT)

```bash
cd /path/to/SSL-CVA
python3 inference.py --ft \
  --input_test_file demo/input/lists/demo_inputs_mps_5x5s.lst \
  --output_dir output/demo_selective_ft \
  --reference_dir demo/reference_audio \
  --min_duration 1.0 \
  --debug_stats
```

### Base/Base (SSL-B)

```bash
cd /path/to/SSL-CVA
python3 inference.py --base \
  --input_test_file demo/input/lists/demo_inputs_mps_5x5s.lst \
  --output_dir output/demo_selective_bb \
  --reference_dir demo/reference_audio \
  --min_duration 1.0 \
  --debug_stats
```

Successful runs write anonymized wavs under `output_dir` and a list at `output_dir/filtered_list.txt`.

## 5. Multi-speaker demo (TSE → anonymize target → recombine)

This uses the demo mixture + enrollment wavs under `demo/input/multispeaker/` and outputs into `demo/output/`.

```bash
cd /path/to/SSL-CVA
python3 inference_multispeaker.py \
  --mixture_wav demo/input/multispeaker/mixture.wav \
  --enroll_wav demo/input/multispeaker/reference.wav \
  --target_age child \
  --reference_dir demo/reference_audio \
  --out_dir demo/output/myst999466_on_tsa000000_child_ftft \
  --duration 5 \
  --min_duration 1.0
```

### Multi-speaker v2 (time-splice into full mixture)

`inference_tse_v2.py` runs the same TSE → anonymize steps, but writes a **full-length** `mixture_anonymized.wav` and only modifies a time window of the mixture.

- **`--target_age child`**: uses FT/FT (`inference.py --ft`)
- **`--target_age adult`**: uses Base/Base (`inference.py --base`)

Recombination modes:

- **`--recombine replace`** (default): overwrite the window with the anonymized target only  
  (this removes the non-target speaker in overlapped regions of that window).
- **`--recombine residual_add`**: window-only version of the v1 algebra  
  \( \text{anon} + (\text{mix} - \text{target}) \), which better preserves non-target energy during overlap.

Example (process 0–5s, keep the rest untouched):

```bash
cd /path/to/SSL-CVA
python3 inference_tse_v2.py \
  --mixture_wav demo/input/multispeaker/mixture.wav \
  --enroll_wav demo/input/multispeaker/reference.wav \
  --target_age adult \
  --reference_dir demo/reference_audio \
  --out_dir demo/output/multispeaker_v2_adult_bb \
  --t_start_sec 0 \
  --duration 5 \
  --recombine residual_add \
  --min_duration 1.0
```

## 4. If it works, you are done with this step

Check:

- Log shows a positive **reference speaker** count and processes ≥ 1 input.
- `output/.../filtered_list.txt` is created and the wavs exist.

## Installation / dependencies

Python **3.9+** is typical for this stack.

Core packages you will need:

- **PyTorch** (+ CUDA if you want GPU)
- **fairseq** (HuBERT loading)
- **speechbrain** (ECAPA speaker embedding used in TSE + some model utilities)
- **soundfile**, **librosa**
- **numpy**
- Optional: **resampy** (cleaner resampling), **tqdm**

One workable install pattern (CPU example; adjust to your CUDA / PyTorch install):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchaudio
pip install fairseq speechbrain soundfile librosa numpy resampy tqdm
```

**PyTorch 2.6+** changed `torch.load` defaults; this repo imports `adapted_from_facebookresearch/torch_load_compat.py` from `models.py` so fairseq Hubert checkpoints load without changing site-packages.

## Troubleshooting

| Symptom | Likely cause |
|--------|----------------|
| `No reference audio >= …s` | Empty or too-short pool; lower `--min_duration` or fix `--reference_dir` / `reference.lst`. |
| `No valid inputs after filtering` | Input paths missing on disk or wrong remap; use paths under `latest-child-speech-dataset` or absolute paths. |
| `Weights only load failed` / unpickling Hubert | Ensure you are on current `models.py` + `torch_load_compat.py`; use a PyTorch/fairseq combo that matches your checkpoints. |

---

When open-sourcing, next cleanups might include: configurable dataset root instead of hard-coded `/app`, pinning dependencies, and moving large weights to Hugging Face with download scripts.
