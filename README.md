# Child-Centric Voice Anonymization in Single and Multi-Speaker Speech via Domain-Adapted SSL Models

This repository contains code for **selective (reference/pool-based) voice anonymization** in:

- **Single-speaker speech**: Base/Base (SSL-B) and FT/FT (SSL-FT)
- **Multi-speaker mixtures**: TSE → anonymize target → recombine

This codebase is adapted from and inspired by the SSL-SAS family of systems.

## Papers

This repository follows the core ideas and components introduced in:

1. [Language-independent speaker anonymization approach using self-supervised pre-trained models](https://arxiv.org/abs/2202.13097)
2. [Analyzing Language-Independent Speaker Anonymization Framework under Unseen Conditions](https://arxiv.org/abs/2203.14834)

Please cite these papers if you use this code.

## Dependencies

```bash
git clone https://github.com/pranavtushar/SSL-CVA.git
cd SSL-CVA
bash scripts/install.sh
source env.sh
```

## Checkpoints (download from Hugging Face)

Model weights are **not committed** to this repository. Download them from a Hugging Face model repo into `all_checkpoints/`:

```bash
cd /path/to/SSL-CVA
bash scripts/download_checkpoints.sh --repo pranavtushar/ssl-cva-checkpoints
```

If your Hugging Face repo is private, export `HF_TOKEN` in your shell before downloading:

```bash
export HF_TOKEN=...
```

Optional: keep checkpoints outside the repo folder:

```bash
export SSL_CVA_CHECKPOINTS_DIR=/path/to/all_checkpoints
```

Both `inference.py` and `inference_tse_v2.py` also accept `--checkpoints_dir /path/to/all_checkpoints`.

## Single-speaker anonymization (FT/FT and Base/Base)

### FT/FT (paper default, SSL-FT)

```bash
cd /path/to/SSL-CVA
python3 inference.py --ft \
  --input_test_file demo/input/lists/inputs_mps.lst \
  --reference_dir demo/input/reference_audio \
  --output_dir demo/output/selective_ft \
  --min_duration 1.0 \
  --quiet
```

### Base/Base (SSL-B)

```bash
cd /path/to/SSL-CVA
python3 inference.py --base \
  --input_test_file demo/input/lists/inputs_mps.lst \
  --reference_dir demo/input/reference_audio \
  --output_dir demo/output/selective_base \
  --min_duration 1.0 \
  --quiet
```

Successful runs write anonymized wavs under `output_dir` and a list at `output_dir/filtered_list.txt`.

## Multi-speaker demo (TSE → anonymize target → recombine)

`inference_tse_v2.py` runs the TSE → anonymize steps, writes a **full-length** `mixture_anonymized.wav`, and only modifies a time window of the mixture.

<<<<<<< HEAD

`inference_tse_v2.py` runs the TSE → anonymize steps, writes a **full-length** `mixture_anonymized.wav`. 

- **`--target_age child`**: uses FT/FT (`inference.py --ft`)
- **`--target_age adult`**: uses Base/Base (`inference.py --base`)
=======
- `--target_age child`: uses FT/FT (`inference.py --ft`)
- `--target_age adult`: uses Base/Base (`inference.py --base`)
>>>>>>> 09a7960 (installation added)

Recombination modes:

- `--recombine replace` (default): overwrite the window with anonymized target only
- `--recombine residual_add`: window-only version of the v1 algebra \( \text{anon} + (\text{mix} - \text{target}) \)

Example (process 0–5s, keep the rest untouched):

```bash
cd /path/to/SSL-CVA
python3 inference_tse_v2.py \
  --mixture_wav demo/input/multispeaker/000001_s1-myst_999455_s2-libri_7729_ov40.mix.wav \
  --enroll_wav demo/input/multispeaker/myst_999455_2009-07-12_00-00-00_MS_2.1_002.wav \
  --target_age child \
  --reference_dir demo/input/reference_audio \
  --out_dir demo/output/multispeaker \
  --duration 5 \
  --recombine residual_add \
  --quiet
```

## License

This repository contains code adapted from upstream projects. Please see the license files in:

- `adapted_from_facebookresearch/`
- `adapted_from_speechbrain/`

<<<<<<< HEAD
## Installation / dependencies
 to be added. 



=======
## Acknowledgments

This work builds on prior open-source releases from the SSL-SAS ecosystem, including components originally adapted from Facebook Research and SpeechBrain.

That's all and good luck!
>>>>>>> 09a7960 (installation added)
