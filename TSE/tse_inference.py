"""
TSE (Target Speaker Extraction) inference.

This module provides a lightweight class wrapper (`TSEInference`) that:
- loads the ECAPA embedding model (speaker embedding)
- loads the Conformer TSE model (target extraction)
- exposes `separate_speech(mix.wav, reference.wav) -> waveform`

It is used by `SSL-CVA/inference_tse_v2.py`.
"""

import os

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from speechbrain.lobes.features import Fbank
from tqdm import tqdm

from conformer_tse import Model
from ecapa import ECAPA_TDNN


class TSEInference:
    def __init__(self, tse_model_path, ecapa_model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ECAPA model path: use provided path or fallback to same dir as this script / default
        if ecapa_model_path is None:
            ecapa_model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "embedding_model.ckpt"
            )
        ecapa_path = ecapa_model_path

        # Initialize ECAPA model for speaker embedding
        self.ecapa_model = ECAPA_TDNN(80, lin_neurons=192, out_neurons=5994)
        ecapa_checkpoint = torch.load(ecapa_path, map_location="cpu")
        self.ecapa_model.load_state_dict(ecapa_checkpoint)
        self.ecapa_model.to(self.device)
        self.ecapa_model.eval()

        # Load TSE model directly
        print(f"Loading TSE model from: {tse_model_path}")
        self.model = Model()
        tse_checkpoint = torch.load(tse_model_path, map_location="cpu")
        self.model.load_state_dict(tse_checkpoint)

        # Ensure model is in eval mode and on correct device
        self.model.to(self.device)
        self.model.eval()

        # Initialize feature extractor
        self.fbank = Fbank(n_mels=80)

        print("TSE Inference initialized successfully!")

    def preprocess_audio(self, audio_path, target_sr=16000, target_duration=None):
        """Load and preprocess audio file"""
        audio, sr = librosa.load(audio_path, sr=target_sr)

        # Adjust length if specified
        if target_duration is not None:
            target_samples = int(target_duration * target_sr)
            if len(audio) > target_samples:
                audio = audio[:target_samples]
            elif len(audio) < target_samples:
                padding = np.zeros(target_samples - len(audio))
                audio = np.concatenate([audio, padding])

        return torch.FloatTensor(audio).unsqueeze(0).to(self.device)

    def extract_speaker_embedding(self, audio_tensor):
        """Extract speaker embedding using ECAPA-TDNN"""
        with torch.no_grad():
            if audio_tensor.dim() == 3:
                audio_tensor = audio_tensor.squeeze(2)
            elif audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            ecapa_input = self.fbank(audio_tensor)
            xvector, _ = self.ecapa_model.feature_forward(ecapa_input)

        return xvector

    def separate_speech(
        self,
        mixed_audio_path,
        reference_audio_path,
        target_duration=6,
        return_diagnostics=False,
    ):
        """
        Perform target speaker extraction

        Args:
            mixed_audio_path: Path to mixed audio file
            reference_audio_path: Path to reference audio for target speaker
            target_duration: Duration in seconds to process (default: 6s)
            return_diagnostics: If True, return a dict with target, mix_used (exact waveform
                fed to the model after librosa load + crop/pad), and tse_interference (model's
                second waveform output). If False, return only the target numpy array.

        Returns:
            separated_audio, or dict with keys target, mix_used, tse_interference
        """
        # Load and preprocess audio files
        mixed_audio = self.preprocess_audio(
            mixed_audio_path, target_duration=target_duration
        )
        reference_audio = self.preprocess_audio(reference_audio_path)

        # Extract speaker embedding from reference
        speaker_embedding = self.extract_speaker_embedding(reference_audio)

        # Prepare inputs for the model
        mixed_input = mixed_audio.unsqueeze(2)  # Add channel dimension
        reference_input = reference_audio.unsqueeze(2)  # Add channel dimension

        inputs = [mixed_input, reference_input]

        # Perform separation
        with torch.no_grad():
            outputs = self.model(inputs, speaker_embedding, speaker_embedding)
            separated_audio = outputs[0]  # target estimate
            interference_wav = outputs[2] if return_diagnostics else None

        # Convert to numpy and remove batch dimension
        separated_audio = separated_audio.squeeze().cpu().numpy()
        mix_used = mixed_audio.squeeze().cpu().numpy()

        if return_diagnostics:
            interf = interference_wav.squeeze().cpu().numpy()
            return {
                "target": separated_audio.astype(np.float32, copy=False),
                "mix_used": mix_used.astype(np.float32, copy=False),
                "tse_interference": interf.astype(np.float32, copy=False),
            }

        return separated_audio


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TSE inference: extract target speaker from mix using reference."
    )
    parser.add_argument(
        "--tse-checkpoint",
        default="checkpoint_model_2000",
        help="Path to TSE Conformer checkpoint",
    )
    parser.add_argument(
        "--ecapa-checkpoint",
        default=None,
        help="Path to ECAPA embedding_model.ckpt (default: ./embedding_model.ckpt)",
    )
    parser.add_argument(
        "--mix", type=str, help="Mixed audio path (optional; if set, run inference)"
    )
    parser.add_argument(
        "--reference", type=str, help="Reference audio path (target speaker)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="target_extracted.wav",
        help="Output waveform path",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=6.0,
        help="Max duration in seconds to process from mix",
    )
    args = parser.parse_args()

    tse_checkpoint = args.tse_checkpoint
    ecapa_path = args.ecapa_checkpoint

    if not os.path.isfile(tse_checkpoint):
        print("Missing TSE checkpoint. Set --tse-checkpoint to your Conformer checkpoint path.")
        raise SystemExit(1)

    tse = TSEInference(tse_checkpoint, ecapa_model_path=ecapa_path)
    target = tse.separate_speech(args.mix, args.reference, target_duration=args.duration)
    sf.write(args.output, target, 16000)
    print(f"Saved: {args.output}")

