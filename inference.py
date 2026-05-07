#!/usr/bin/env python3
"""
Single entrypoint for selective anonymization (no subprocess wrappers).

This script implements the full pipeline:

1) Load YAML config (selects checkpoint dir for BB vs FT/FT)
2) Load config.json + generator weights (HuBERT content encoder + HiFi-GAN vocoder)
3) Scan a reference pool directory and build a speaker->files map
4) For each input utterance:
   - extract HuBERT content features (and optional F0 if enabled by the checkpoint)
   - pick a random reference file and extract its ECAPA x-vector embedding
   - synthesize anonymized waveform with the generator conditioned on that embedding
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.io.wavfile import write


SCRIPT_DIR = Path(__file__).resolve().parent
AFR_DIR = SCRIPT_DIR / "adapted_from_facebookresearch"
if str(AFR_DIR) not in sys.path:
    sys.path.insert(0, str(AFR_DIR))

# Imported from adapted_from_facebookresearch/*
from dataset import MAX_WAV_VALUE, latentDataset  # noqa: E402
from models import Generator, latentGenerator  # noqa: E402
from utils import AttrDict  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REFERENCE_EXTS = (".wav", ".flac")


def _scan_checkpoint(cp_dir: str, prefix: str) -> str:
    pattern = os.path.join(cp_dir, prefix + "*")
    files = glob.glob(pattern)
    return "" if not files else sorted(files)[-1]


def _load_checkpoint(filepath: str) -> dict:
    assert os.path.isfile(filepath), f"Checkpoint not found: {filepath}"
    print(f"Loading checkpoint: {filepath}", flush=True)
    return torch.load(filepath, map_location="cpu")


def _audio_duration_seconds(path: Path) -> float:
    try:
        info = sf.info(str(path))
        if info.samplerate and info.frames:
            return info.frames / float(info.samplerate)
    except Exception:
        return -1.0
    return -1.0


def _speaker_id_for_path(root_dir: Path, file_path: Path) -> str:
    rel_path = file_path.relative_to(root_dir)
    parts = rel_path.parts
    if len(parts) > 1:
        return parts[0]
    return file_path.stem


def _build_reference_map(reference_root: Path, min_duration_sec: float) -> dict[str, list[Path]]:
    speaker_to_files: dict[str, list[Path]] = {}
    for root, _, files in os.walk(reference_root):
        for name in files:
            if not name.lower().endswith(REFERENCE_EXTS):
                continue
            file_path = Path(root) / name
            duration = _audio_duration_seconds(file_path)
            if duration >= 0 and duration < min_duration_sec:
                continue
            speaker_id = _speaker_id_for_path(reference_root, file_path)
            speaker_to_files.setdefault(speaker_id, []).append(file_path)
    if not speaker_to_files:
        raise RuntimeError(f"No reference audio >= {min_duration_sec:.2f}s found under: {reference_root}")
    return speaker_to_files


def _load_audio_for_ecapa(path: Path, target_sr: int) -> torch.Tensor:
    audio, sampling_rate = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sampling_rate != target_sr:
        try:
            import resampy

            audio = resampy.resample(audio, sampling_rate, target_sr)
        except Exception:
            duration = len(audio) / float(sampling_rate)
            target_len = int(round(duration * target_sr))
            x_old = np.linspace(0.0, duration, num=len(audio), endpoint=False)
            x_new = np.linspace(0.0, duration, num=target_len, endpoint=False)
            audio = np.interp(x_new, x_old, audio).astype(np.float32)
    audio = audio / MAX_WAV_VALUE
    audio = librosa.util.normalize(audio) * 0.95
    return torch.FloatTensor(audio).unsqueeze(0)


def _extract_reference_xv(generator: latentGenerator, audio: torch.Tensor) -> torch.Tensor:
    if not hasattr(generator, "xv_model"):
        raise RuntimeError("Generator does not have an ECAPA model configured.")
    with torch.no_grad():
        xv_input = generator.fbank(audio.squeeze(1))
        xv_input = generator.mean_var_norm(
            xv_input, torch.ones(xv_input.shape[0]).to(xv_input.device)
        )
        xv, _ = generator.xv_model(xv_input)
    xv = F.layer_norm(xv, xv.shape)
    xv = xv.transpose(2, 1)
    return xv


def _extract_content_features(generator: latentGenerator, audio: torch.Tensor) -> torch.Tensor:
    if getattr(generator, "ssl_type", None) != "hubert_soft":
        raise RuntimeError(f"Selective inference expects ssl_type=hubert_soft; got {getattr(generator, 'ssl_type', None)}")
    x = generator.latent_encoder(audio)
    x = F.layer_norm(x, x.shape)
    x = x.transpose(2, 1)
    x = torch.nn.functional.pad(x, (0, 1), "replicate")
    return x


def _generate_audio_relative(h, generator: latentGenerator, inputs: dict, reference_xv: torch.Tensor):
    start = time.time()
    audio_data = inputs["audio"]
    x = _extract_content_features(generator, audio_data)

    if getattr(generator, "f0", False):
        f0 = inputs["f0"]
        if x.shape[-1] < f0.shape[-1]:
            x = generator._upsample(x, f0.shape[-1])
        else:
            f0 = generator._upsample(f0, x.shape[-1])
        x = torch.cat([x, f0], dim=1)

    xv = generator._upsample(reference_xv, x.shape[-1])
    x = torch.cat([x, xv], dim=1)

    output = Generator.forward(generator, x).to(DEVICE)
    audio_tensor = output[0] if isinstance(output, tuple) else output
    rtf = (time.time() - start) / (audio_tensor.shape[-1] / h.sampling_rate)

    audio = audio_tensor.detach().squeeze().cpu().numpy() * MAX_WAV_VALUE
    audio = audio.astype(np.int16)
    return audio, rtf


def _resolve_input_path(raw: str, list_parent: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return (list_parent / p).resolve()


def _output_path_for_input(input_path: Path, input_root: Path | None, output_root: Path) -> Path:
    if input_root is not None:
        try:
            rel = input_path.relative_to(input_root)
        except ValueError:
            rel = Path(input_path.name)
    else:
        rel = Path(input_path.name)
    return (output_root / rel).with_suffix(".wav")


def main() -> None:
    p = argparse.ArgumentParser(description="Run selective anonymization (BB or FT/FT).")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--base", action="store_true", help="Use Base/Base (SSL-B) checkpoint")
    mode.add_argument("--ft", action="store_true", help="Use FT/FT (SSL-FT) checkpoint")
    p.add_argument(
        "--checkpoint_file",
        default=None,
        help="Optional override checkpoint directory/file (otherwise uses a default for --base/--ft)",
    )

    p.add_argument("--input_test_file", required=True, help="Text file listing input audio paths (one per line)")
    p.add_argument("--output_dir", required=True, help="Output root directory")
    p.add_argument("--reference_dir", required=True, help="Reference pool root directory to scan for wav/flac")
    p.add_argument("--min_duration", type=float, default=3.0, help="Minimum reference clip duration (seconds)")
    p.add_argument("--input_root", default=None, help="Optional input root used for mirrored output layout")
    p.add_argument("--output_list", default=None, help="Optional output list path")
    p.add_argument("--relative_list", action="store_true", help="Write list paths relative to output_dir")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reference sampling")
    p.add_argument("--skip_existing", action="store_true", help="Skip items whose output wav already exists")
    p.add_argument("--debug_stats", action="store_true", help="Print per-utterance RTF after each item")

    args = p.parse_args()

    script_dir = SCRIPT_DIR
    checkpoint_path = Path(args.checkpoint_file) if args.checkpoint_file else None
    if checkpoint_path is None:
        checkpoint_path = script_dir / (
            "all_checkpoints/HiFi-GAN_B_Soft_B" if args.base else "all_checkpoints/HIFI-GAN_FT_Soft_FT"
        )
    if not checkpoint_path.is_absolute():
        checkpoint_path = (script_dir / checkpoint_path).resolve()

    print("[mode]", "base" if args.base else "ft", flush=True)
    print("[checkpoint]", str(checkpoint_path), flush=True)
    print("[reference_dir]", str(args.reference_dir), flush=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load config.json from checkpoint directory (or file parent)
    cp_path = str(checkpoint_path)
    config_json = (
        os.path.join(cp_path, "config.json")
        if os.path.isdir(cp_path)
        else os.path.join(os.path.dirname(cp_path), "config.json")
    )
    with open(config_json, "r", encoding="utf-8") as f:
        h = AttrDict(json.load(f))

    # Instantiate generator and load weights
    generator = latentGenerator(h).to(DEVICE)
    cp_gen = _scan_checkpoint(cp_path, "g_") if os.path.isdir(cp_path) else cp_path
    ckpt = _load_checkpoint(cp_gen)

    generator_state_dict = ckpt["generator"]
    filtered_state_dict = {}
    model_state_dict = generator.state_dict()
    for k, v in generator_state_dict.items():
        if k in model_state_dict:
            if model_state_dict[k].shape == v.shape:
                filtered_state_dict[k] = v
            else:
                print(
                    f"Skipping size-mismatched key in generator: {k} "
                    f"(checkpoint: {tuple(v.shape)}, model: {tuple(model_state_dict[k].shape)})",
                    flush=True,
                )
        else:
            filtered_state_dict[k] = v
    generator.load_state_dict(filtered_state_dict, strict=False)
    generator.remove_weight_norm()
    generator.eval()

    # Resolve and load input list
    input_list_path = Path(args.input_test_file)
    if not input_list_path.is_absolute():
        input_list_path = (script_dir / input_list_path).resolve()
    list_parent = input_list_path.parent

    file_list: list[str] = []
    with input_list_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            pth = _resolve_input_path(raw, list_parent)
            if not pth.exists():
                print(f"Warning: input not found: {pth}", flush=True)
                continue
            file_list.append(str(pth))

    dataset = latentDataset(
        file_list,
        -1,
        h.n_fft,
        h.num_mels,
        h.hop_size,
        h.win_size,
        h.sampling_rate,
        h.fmin,
        h.fmax,
        n_cache_reuse=0,
        fmax_loss=h.fmax_for_loss,
        device=DEVICE,
    )

    n = len(dataset)
    if n == 0:
        raise SystemExit("No valid inputs after filtering; check input_list paths.")

    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = (script_dir / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    input_root = None
    if args.input_root:
        input_root = Path(args.input_root)
        if not input_root.is_absolute():
            input_root = (script_dir / input_root).resolve()

    output_list_path = Path(args.output_list) if args.output_list else (output_root / "filtered_list.txt")
    if not output_list_path.is_absolute():
        output_list_path = (script_dir / output_list_path).resolve()
    output_list_path.parent.mkdir(parents=True, exist_ok=True)

    # Reference pool
    reference_root = Path(args.reference_dir)
    if not reference_root.is_absolute():
        reference_root = (script_dir / reference_root).resolve()
    speaker_to_files = _build_reference_map(reference_root, args.min_duration)
    speaker_ids = list(speaker_to_files.keys())
    reference_xv_cache: dict[Path, torch.Tensor] = {}
    print(f"Reference speakers: {len(speaker_ids)} (>= {args.min_duration:.1f}s)", flush=True)

    def pick_reference() -> tuple[str, Path]:
        spk = random.choice(speaker_ids)
        ref = random.choice(speaker_to_files[spk])
        return spk, ref

    def get_reference_xv(ref_path: Path) -> torch.Tensor:
        cached = reference_xv_cache.get(ref_path)
        if cached is not None:
            return cached.to(DEVICE)
        ref_audio = _load_audio_for_ecapa(ref_path, h.sampling_rate).to(DEVICE)
        xv = _extract_reference_xv(generator, ref_audio)
        reference_xv_cache[ref_path] = xv.detach().cpu()
        return xv

    with output_list_path.open("w", encoding="utf-8") as out_f:
        skipped = 0
        with torch.no_grad():
            for i in range(n):
                x, _, _, _ = dataset[i]
                inputs = {k: v.to(DEVICE) for k, v in x.items()}

                input_path = Path(dataset.audio_files[i]).resolve()
                out_path = _output_path_for_input(input_path, input_root, output_root)
                out_path.parent.mkdir(parents=True, exist_ok=True)

                if args.skip_existing and out_path.is_file():
                    skipped += 1
                    continue

                ref_speaker_id, ref_path = pick_reference()
                reference_xv = get_reference_xv(ref_path)
                audio, rtf = _generate_audio_relative(h, generator, inputs, reference_xv)

                norm_audio = librosa.util.normalize(audio.astype(np.float32))
                write(str(out_path), h.sampling_rate, norm_audio)

                out_list_item = out_path.relative_to(output_root) if args.relative_list else out_path
                out_f.write(f"{out_list_item}\n")

                bar = f"[{i+1}/{n}] RTF={rtf:.3f}"
                sys.stdout.write("\r" + bar)
                sys.stdout.flush()
                if args.debug_stats:
                    sys.stdout.write("\n")
                    print(
                        f"  debug | in={input_path} | out={out_path} | ref={ref_speaker_id}:{ref_path}",
                        flush=True,
                    )

    print("\nInference complete.", flush=True)
    if skipped:
        print(f"Skipped (existing outputs): {skipped}", flush=True)


if __name__ == "__main__":
    main()

