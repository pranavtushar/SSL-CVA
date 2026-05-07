"""
TSE (Target Speaker Extraction) inference.

Inputs:
  - reference_audio: short clip of the TARGET speaker (identifies whose voice to extract)
  - mixed_audio: recording where target + other speakers/noise are mixed

Output:
  - extracted target speaker waveform (not embeddings).

Required files:
  - embedding_model.ckpt (ECAPA speaker embedding model) — in script dir or pass ecapa_model_path=
  - TSE checkpoint (Conformer model) — pass as tse_model_path=
"""
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
import numpy as np
from ecapa import ECAPA_TDNN
from speechbrain.lobes.features import Fbank
from conformer_tse import Model
import os
from tqdm import tqdm


class TSEInference:
    def __init__(self, tse_model_path, ecapa_model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ECAPA model path: use provided path or fallback to same dir as script / default
        if ecapa_model_path is None:
            ecapa_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embedding_model.ckpt")
        ecapa_path = ecapa_model_path

        # Initialize ECAPA model for speaker embedding
        self.ecapa_model = ECAPA_TDNN(80, lin_neurons=192, out_neurons=5994)
        ecapa_checkpoint = torch.load(ecapa_path, map_location='cpu')
        self.ecapa_model.load_state_dict(ecapa_checkpoint)
        self.ecapa_model.to(self.device)
        self.ecapa_model.eval()

        # Load TSE model directly
        print(f"Loading TSE model from: {tse_model_path}")
        self.model=Model()
        tse_checkpoint = torch.load(tse_model_path, map_location='cpu')
        self.model.load_state_dict(tse_checkpoint)

        # Ensure model is in eval mode and on correct device
        self.model.to(self.device)
        self.model.eval()

        # Initialize feature extractor
        self.fbank = Fbank(n_mels=80)

        print("TSE Inference initialized successfully!")

    def load_libri2vox_dataset_list(self):
        """Load libri2vox dataset folder list"""
        libri2vox_folder_base_name = '/home/smg/liu/data/syntse_sv56/trainset/salt_k4p5_snr55_gb/'

        with open('/home/smg/liu/data/syntse/train_set_syn/train_syn_base_list.txt', 'r') as file:
            folder_names = [libri2vox_folder_base_name + line.strip() for line in file]

        return folder_names

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

    def separate_speech(self, mixed_audio_path, reference_audio_path, target_duration=6, return_diagnostics=False):
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
        mixed_audio = self.preprocess_audio(mixed_audio_path, target_duration=target_duration)
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

    def test_libri2vox_dataset_samples(self, num_samples=10, output_dir="./libri2vox_test_outputs/", target_duration=None):
        """
        Test TSE model on random samples from libri2vox dataset
        Args:
            num_samples: Number of random samples to test
            output_dir: Directory to save separated audio files
            target_duration: Duration in seconds to process
        """
        # Load dataset folder list
        folder_list = self.load_libri2vox_dataset_list()

        # Randomly sample folders
        import random
        random.shuffle(folder_list)
        test_folders = folder_list[:num_samples]

        os.makedirs(output_dir, exist_ok=True)

        print(f"Testing TSE model on {num_samples} libri2vox dataset samples...")

        for i, folder_path in enumerate(tqdm(test_folders, desc="Processing")):
            folder_name = os.path.basename(folder_path)

            # Check for required files
            mix_path = os.path.join(folder_path, 'mix.wav')
            reference_path = os.path.join(folder_path, 'reference.wav')
            target_path = os.path.join(folder_path, 'target.wav')

            if all(os.path.exists(p) for p in [mix_path, reference_path, target_path]):
                try:
                    # Perform separation
                    separated = self.separate_speech(mix_path, reference_path, target_duration)

                    # Save outputs
                    output_separated = os.path.join(output_dir, f"{folder_name}_separated.wav")
                    sf.write(output_separated, separated, 16000)

                    # Also copy original files for comparison
                    target_audio, _ = librosa.load(target_path, sr=16000)
                    mix_audio, _ = librosa.load(mix_path, sr=16000)
                    ref_audio, _ = librosa.load(reference_path, sr=16000)

                    # Adjust length to match separated audio
                    if target_duration:
                        target_samples = int(target_duration * 16000)
                        if len(target_audio) > target_samples:
                            target_audio = target_audio[:target_samples]
                        if len(mix_audio) > target_samples:
                            mix_audio = mix_audio[:target_samples]

                    sf.write(os.path.join(output_dir, f"{folder_name}_target.wav"), target_audio, 16000)
                    sf.write(os.path.join(output_dir, f"{folder_name}_mix.wav"), mix_audio, 16000)
                    sf.write(os.path.join(output_dir, f"{folder_name}_reference.wav"), ref_audio, 16000)

                    print(f"[{i + 1}/{num_samples}] Processed: {folder_name}")

                except Exception as e:
                    print(f"[{i + 1}/{num_samples}] Error processing {folder_name}: {str(e)}")
            else:
                print(f"[{i + 1}/{num_samples}] Missing files in {folder_name}, skipping...")

        print(f"Testing complete! Results saved to: {output_dir}")




# ---------------------------------------------------------------------------
# How to run inference
# ---------------------------------------------------------------------------
# 1. Put these files in place (or set paths when creating TSEInference):
#    - embedding_model.ckpt  (ECAPA; default: same folder as this script)
#    - Your TSE checkpoint   (e.g. checkpoint_model_2000)
#
# 2. Simple one-shot inference (reference + mix -> target waveform):
#
#    from inference_server import TSEInference
#    import soundfile as sf
#
#    tse = TSEInference(tse_model_path="path/to/checkpoint_model_2000",
#                       ecapa_model_path="path/to/embedding_model.ckpt")  # optional
#    target_audio = tse.separate_speech("mix.wav", "reference.wav", target_duration=6)
#    sf.write("target_extracted.wav", target_audio, 16000)
#
# 3. Libri2vox batch test: run this script and set paths below.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TSE inference: extract target speaker from mix using reference.")
    parser.add_argument("--tse-checkpoint", default="checkpoint_model_2000", help="Path to TSE Conformer checkpoint")
    parser.add_argument("--ecapa-checkpoint", default=None, help="Path to ECAPA embedding_model.ckpt (default: ./embedding_model.ckpt)")
    parser.add_argument("--mix", type=str, help="Mixed audio path (optional; if set, run single-file inference)")
    parser.add_argument("--reference", type=str, help="Reference audio path (target speaker)")
    parser.add_argument("--output", type=str, default="target_extracted.wav", help="Output waveform path")
    parser.add_argument("--duration", type=float, default=6.0, help="Max duration in seconds to process from mix")
    args = parser.parse_args()

    tse_checkpoint = args.tse_checkpoint
    ecapa_path = args.ecapa_checkpoint

    if args.mix and args.reference:
        # Single-file inference
        if not os.path.isfile(tse_checkpoint):
            print("Missing TSE checkpoint. Set --tse-checkpoint to your Conformer checkpoint path.")
            exit(1)
        tse = TSEInference(tse_checkpoint, ecapa_model_path=ecapa_path)
        target = tse.separate_speech(args.mix, args.reference, target_duration=args.duration)
        sf.write(args.output, target, 16000)
        print(f"Saved: {args.output}")
    else:
        # Batch test on Libri2vox-style folders (original behavior)
        tse = TSEInference(tse_checkpoint, ecapa_model_path=ecapa_path)
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libri2vox_test_outputs")
        tse.test_libri2vox_dataset_samples(num_samples=1, output_dir=out_dir)
        print("If you see path errors, run single-file inference: --mix mix.wav --reference ref.wav --output out.wav")

