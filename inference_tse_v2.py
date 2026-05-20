
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf


def _configure_quiet() -> None:
    # Reduce third-party log spam (fairseq/speechbrain) and hide common deprecations.
    logging.getLogger("fairseq").setLevel(logging.ERROR)
    logging.getLogger("fairseq.tasks.text_to_speech").setLevel(logging.ERROR)
    logging.getLogger("speechbrain").setLevel(logging.ERROR)
    logging.getLogger("speechbrain.utils.train_logger").setLevel(logging.ERROR)

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)


def _resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio.astype(np.float32)
    try:
        import resampy

        return resampy.resample(audio, sr_in, sr_out).astype(np.float32)
    except Exception:
        duration = len(audio) / float(sr_in)
        n_out = int(round(duration * sr_out))
        x_old = np.linspace(0.0, duration, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, duration, num=n_out, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32)


def _load_mono(path: Path, sr: int) -> np.ndarray:
    audio, sr_in = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return _resample(audio.astype(np.float32), int(sr_in), int(sr))


def _write(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)


def _align_to_min(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    L = min(len(a) for a in arrays)
    return tuple(np.asarray(a[:L], dtype=np.float32) for a in arrays)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="TSE → anonymize → splice anonymized target into full mixture (v2)"
    )
    ap.add_argument("--mixture_wav", type=Path, required=True)
    ap.add_argument("--enroll_wav", type=Path, required=True)
    ap.add_argument("--target_age", choices=("child", "adult"), required=True)
    ap.add_argument("--reference_dir", type=Path, required=True)
    ap.add_argument("--min_duration", type=float, default=1.0)
    ap.add_argument(
        "--t_start_sec",
        type=float,
        default=0.0,
        help="Start time (seconds) in the full mixture for the TSE/anonymize window",
    )
    ap.add_argument(
        "--duration",
        type=float,
        default=None,
        help="If set, window length (s) from t_start; if unset, window is [t_start, EOF)",
    )
    ap.add_argument(
        "--recombine",
        choices=("replace", "residual_add"),
        default="replace",
        help="replace: overwrite window with anonymized target; "
        "residual_add: anon + (mix - target) inside window only",
    )
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--tse_checkpoint", type=Path, default=Path("all_checkpoints/TSE/libri2talker_libri2vox"))
    ap.add_argument("--ecapa_checkpoint", type=Path, default=Path("all_checkpoints/TSE/embedding_model.ckpt"))
    ap.add_argument(
        "--checkpoints_dir",
        type=Path,
        default=None,
        help="Root directory containing checkpoints (e.g. /path/to/all_checkpoints). "
        "Overrides SSL_CVA_CHECKPOINTS_DIR.",
    )
    ap.add_argument("--anon_checkpoint_file", type=Path, default=None)
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress common warnings and third-party logs (fairseq/speechbrain/torch).",
    )
    ap.add_argument(
        "--write_residual",
        action="store_true",
        help="Also write nontarget_residual.wav for the processed window (mix_seg - target)",
    )
    ap.add_argument(
        "--write_tse_interference",
        action="store_true",
        help="Also write tse_interference.wav for the processed window",
    )

    args = ap.parse_args()
    quiet_enabled = args.quiet or os.environ.get("SSL_CVA_QUIET", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if quiet_enabled:
        _configure_quiet()

    root = Path(__file__).resolve().parent
    ckpt_root_raw = (
        str(args.checkpoints_dir)
        if args.checkpoints_dir is not None
        else os.environ.get("SSL_CVA_CHECKPOINTS_DIR", "")
    )
    ckpt_root = (Path(ckpt_root_raw).expanduser().resolve() if ckpt_root_raw else (root / "all_checkpoints"))

    def _resolve_ckpt(p: Path) -> Path:
        if p.is_absolute():
            return p
        parts = p.parts
        if parts and parts[0] == "all_checkpoints":
            p = Path(*parts[1:])
        return (ckpt_root / p).resolve()

    mix = args.mixture_wav if args.mixture_wav.is_absolute() else (root / args.mixture_wav).resolve()
    enr = args.enroll_wav if args.enroll_wav.is_absolute() else (root / args.enroll_wav).resolve()
    ref_dir = args.reference_dir if args.reference_dir.is_absolute() else (root / args.reference_dir).resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else (root / args.out_dir).resolve()
    tse_ckpt = _resolve_ckpt(args.tse_checkpoint)
    ecapa_ckpt = _resolve_ckpt(args.ecapa_checkpoint)

    if not mix.is_file():
        raise SystemExit(f"Missing mixture wav: {mix}")
    if not enr.is_file():
        raise SystemExit(f"Missing enroll wav: {enr}")
    if not ref_dir.is_dir():
        raise SystemExit(f"Missing reference dir: {ref_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    mix_full = _load_mono(mix, args.sr)
    _write(out_dir / "mixture_input.wav", mix_full, args.sr)
    n_full = len(mix_full)
    start = int(round(args.t_start_sec * args.sr))
    start = max(0, min(start, n_full))
    if args.duration is not None:
        end = start + int(round(args.duration * args.sr))
        end = max(start, min(end, n_full))
    else:
        end = n_full

    if end <= start:
        raise SystemExit(f"Empty splice window: start_sample={start} end_sample={end}")

    mix_segment = np.asarray(mix_full[start:end], dtype=np.float32)
    print(
        f"[v2] Full mix {n_full} samples ({n_full/args.sr:.3f}s); "
        f"window [{start}:{end}) = {end-start} samples ({(end-start)/args.sr:.3f}s) "
        f"[t={start/args.sr:.3f}s .. {end/args.sr:.3f}s]; recombine={args.recombine}",
        flush=True,
    )
    if args.recombine == "replace":
        print(
            "[v2] NOTE: replace mode overwrites the mixture in that window with the "
            "anonymized target only. Overlapping non-target energy in that window is removed.",
            flush=True,
        )

    sys.path.insert(0, str(root / "TSE"))
    from tse_inference import TSEInference  # type: ignore

    tse = TSEInference(tse_model_path=str(tse_ckpt), ecapa_model_path=str(ecapa_ckpt))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        seg_path = Path(tmp.name)
    try:
        _write(seg_path, mix_segment, args.sr)
        d = tse.separate_speech(str(seg_path), str(enr), target_duration=None, return_diagnostics=True)
    finally:
        seg_path.unlink(missing_ok=True)

    target_extracted = np.asarray(d["target"], dtype=np.float32)
    mix_used = np.asarray(d["mix_used"], dtype=np.float32)
    tse_interference = np.asarray(d["tse_interference"], dtype=np.float32)

    if len(target_extracted) != len(mix_used):
        print(
            f"[warn] TSE target length {len(target_extracted)} != mix_used {len(mix_used)}; aligning.",
            flush=True,
        )
    target_extracted, mix_used, tse_interference = _align_to_min(
        target_extracted, mix_used, tse_interference
    )
    L = len(mix_used)
    region_len = end - start
    if L != region_len:
        print(
            f"[warn] TSE output length {L} != window length {region_len}; "
            "splicing will use min length.",
            flush=True,
        )
    Lsplice = min(L, region_len)
    target_extracted = target_extracted[:Lsplice]
    mix_used = mix_used[:Lsplice]
    tse_interference = tse_interference[:Lsplice]

    residual = mix_used - target_extracted
    max_err = float(np.max(np.abs(mix_used - target_extracted - residual)))
    print(f"[v2] Window algebra check max|mix_used-target-residual|={max_err:.6e}", flush=True)

    _write(out_dir / "target_extracted.wav", target_extracted, args.sr)
    if args.write_residual:
        _write(out_dir / "nontarget_residual.wav", residual, args.sr)
    if args.write_tse_interference:
        _write(out_dir / "tse_interference.wav", tse_interference, args.sr)

    # Run anonymization in a temporary directory (keeps out_dir clean).
    with tempfile.TemporaryDirectory(prefix="ssl_cva_anon_") as tmpdir:
        tmpdir_p = Path(tmpdir)
        anon_in_wav = tmpdir_p / "target_extracted.wav"
        _write(anon_in_wav, target_extracted, args.sr)

        input_list = tmpdir_p / "input.lst"
        input_list.write_text(anon_in_wav.name + "\n", encoding="utf-8")

        anon_out_dir = tmpdir_p / "anon_out"
        anon_out_dir.mkdir(parents=True, exist_ok=True)

        mode_flag = "--ft" if args.target_age == "child" else "--base"
        cmd = [
            sys.executable,
            str(root / "inference.py"),
            mode_flag,
            "--input_test_file",
            str(input_list),
            "--output_dir",
            str(anon_out_dir),
            "--reference_dir",
            str(ref_dir),
            "--min_duration",
            str(args.min_duration),
            "--skip_existing",
        ]
        if args.anon_checkpoint_file is not None:
            ck = (
                args.anon_checkpoint_file
                if args.anon_checkpoint_file.is_absolute()
                else (root / args.anon_checkpoint_file)
            )
            cmd += ["--checkpoint_file", str(ck)]

        print("[anon]", " ".join(cmd), flush=True)
        anon_env = os.environ.copy()
        if quiet_enabled:
            anon_env["SSL_CVA_QUIET"] = "1"
            anon_env["PYTHONWARNINGS"] = "ignore"
        r = subprocess.run(cmd, cwd=str(root), env=anon_env)
        if r.returncode != 0:
            raise SystemExit(r.returncode)

        anon_target = anon_out_dir / "target_extracted.wav"
        if not anon_target.is_file():
            fl = anon_out_dir / "filtered_list.txt"
            if fl.is_file():
                lines = [x.strip() for x in fl.read_text(encoding="utf-8").splitlines() if x.strip()]
                if lines:
                    anon_target = (anon_out_dir / lines[0]).resolve()
        if not anon_target.is_file():
            raise SystemExit("Could not find anonymized target output under temporary anon_out/")

        anon_target_audio = _load_mono(anon_target, args.sr)[:Lsplice]

    mix_out = np.asarray(mix_full, dtype=np.float32).copy()
    Lt = min(Lsplice, len(anon_target_audio), len(mix_used), len(target_extracted))
    if Lt < Lsplice:
        print(f"[warn] anonymized/TSE length shorter than window; splicing {Lt} samples.", flush=True)
    if args.recombine == "replace":
        mix_out[start : start + Lt] = anon_target_audio[:Lt]
    else:
        # Use mix_used (librosa/TSE input), not soundfile mix_segment, for consistent algebra.
        mix_out[start : start + Lt] = (
            anon_target_audio[:Lt] + mix_used[:Lt] - target_extracted[:Lt]
        )

    _write(out_dir / "target_anonymized.wav", anon_target_audio, args.sr)
    _write(out_dir / "mixture_anonymized.wav", mix_out, args.sr)

    
    print(
        f"Wrote: {out_dir / 'mixture_anonymized.wav'} "
        f"(full length {len(mix_out)} samples = {len(mix_out)/args.sr:.3f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
