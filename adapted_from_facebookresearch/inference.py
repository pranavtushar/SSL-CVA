# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# This source code was adapted from https://github.com/facebookresearch/speech-resynthesis by Xiaoxiao Miao (NII, Japan).

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
import torch
from scipy.io.wavfile import write

from dataset import latentDataset, mel_spectrogram, MAX_WAV_VALUE
from utils import AttrDict
from models import latentGenerator

# Compute on GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def scan_checkpoint(cp_dir: str, prefix: str) -> str:
    """
    Get latest checkpoint file matching prefix in cp_dir.
    """
    pattern = os.path.join(cp_dir, prefix + '*')
    files = glob.glob(pattern)
    return '' if not files else sorted(files)[-1]


def load_checkpoint(filepath: str) -> dict:
    """
    Load a PyTorch checkpoint dict from filepath.
    """
    assert os.path.isfile(filepath), f"Checkpoint not found: {filepath}"
    print(f"Loading checkpoint: {filepath}")
    return torch.load(filepath, map_location='cpu')


def generate_audio(h, generator, inputs: dict) -> (np.ndarray, float):
    """
    Run generator on inputs and return waveform + real-time factor.
    """
    start = time.time()
    output = generator.forward(**inputs).to(DEVICE)
    # Some generators return a tuple
    audio_tensor = output[0] if isinstance(output, tuple) else output
    rtf = (time.time() - start) / (audio_tensor.shape[-1] / h.sampling_rate)

    # Denormalize and convert to int16
    audio = audio_tensor.squeeze().cpu().numpy() * MAX_WAV_VALUE
    audio = audio.astype(np.int16)
    return audio, rtf


def init_worker(args):
    """
    Initialize global objects: configuration, generator, dataset.
    """
    global h, generator, dataset
    # Load config.json from checkpoint directory or file parent
    cp_path = args.checkpoint_file
    config_path = os.path.join(cp_path, 'config.json') if os.path.isdir(cp_path) else os.path.join(os.path.dirname(cp_path), 'config.json')
    with open(config_path) as f:
        h = AttrDict(json.load(f))

    # Instantiate latentGenerator and load weights
    generator = latentGenerator(h).to(DEVICE)
    cp_gen = scan_checkpoint(cp_path, 'g_') if os.path.isdir(cp_path) else cp_path
    ckpt = load_checkpoint(cp_gen)
    
    # Filter out size-mismatched keys (e.g., label_embs_concat from different SSL models)
    generator_state_dict = ckpt['generator']
    filtered_state_dict = {}
    model_state_dict = generator.state_dict()
    
    for k, v in generator_state_dict.items():
        if k in model_state_dict:
            model_shape = model_state_dict[k].shape
            if model_shape == v.shape:
                filtered_state_dict[k] = v
            else:
                print(f"Skipping size-mismatched key in generator: {k} (checkpoint: {v.shape}, model: {model_shape})")
        else:
            # Keep keys that might be in submodules
            filtered_state_dict[k] = v
    
    generator.load_state_dict(filtered_state_dict, strict=False)
    generator.remove_weight_norm()
    generator.eval()

    # Prepare dataset from input list file
    file_list = []
    for line in open(args.input_test_file):
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        path = line.split()[-1]
        file_list.append(path)
    dataset = latentDataset(file_list,
                             -1, h.n_fft, h.num_mels,
                             h.hop_size, h.win_size,
                             h.sampling_rate, h.fmin, h.fmax,
                             n_cache_reuse=0,
                             fmax_loss=h.fmax_for_loss,
                             device=DEVICE)
    os.makedirs(args.output_dir, exist_ok=True)


@torch.no_grad()
def inference_item(idx: int, args):
    """
    Process a single dataset item: generate and save audio.
    """
    x, _, _, filename = dataset[idx]
    inputs = {k: v.to(DEVICE) for k, v in x.items()}
    audio, rtf = generate_audio(h, generator, inputs)

    out_name = Path(filename).stem + '.wav'
    out_path = os.path.join(args.output_dir, out_name)

    # Normalize and write wav
    norm_audio = librosa.util.normalize(audio.astype(np.float32))
    write(out_path, h.sampling_rate, norm_audio)

    return idx, rtf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_test_file', required=True,
                        help='Path to text file listing test audio paths')
    parser.add_argument('--checkpoint_file', required=True,
                        help='Directory or file path for generator checkpoint')
    parser.add_argument('--output_dir', default='generated_files',
                        help='Directory to save generated audio')
    args = parser.parse_args()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Initialize global generator and dataset
    init_worker(args)

    # Run inference sequentially or in parallel
    for i in range(len(dataset)):
        idx, rtf = inference_item(i, args)
        bar = f"[{i+1}/{len(dataset)}] RTF={rtf:.3f}"
        sys.stdout.write('\r' + bar)
        sys.stdout.flush()
    print('\nInference complete.')


if __name__ == '__main__':
    main()
