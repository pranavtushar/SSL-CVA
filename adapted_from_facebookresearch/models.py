# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This source code was adapted from https://github.com/facebookresearch/speech-resynthesis by Xiaoxiao Miao (NII, Japan).

import sys, os
import torch

import torch_load_compat  # noqa: F401  # PyTorch>=2.6: allow fairseq Hubert ckpt unpickle

import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from collections import OrderedDict
import joblib
import numpy as np

# Add local directories to the system path for custom module imports.
sys.path.append(os.getcwd())
facebook_root = os.path.abspath("adapted_from_facebookresearch")
if facebook_root not in sys.path:
    sys.path.insert(0, facebook_root)
fairseq_root = os.path.abspath("fairseq")
if fairseq_root not in sys.path:
    sys.path.insert(0, fairseq_root)

# Custom and third-party imports
from adapted_from_facebookresearch.utils import init_weights, get_padding
from fairseq import checkpoint_utils
from speechbrain.lobes.features import Fbank
from speechbrain.processing.features import InputNormalization
from adapted_from_speechbrain.ecapa_tdnn_sb import ECAPA_TDNN


LRELU_SLOPE = 0.1


def state_dict_wrapper_for_ecapa(stat_dict, new_model):
    """
    Loads a pretrained state dictionary into an ECAPA_TDNN model.

    This function handles key mismatches between the pretrained model and the new model
    by renaming keys (e.g., removing prefixes like '0.') and filtering out
    any layers that do not match in name or number of elements.

    Args:
        stat_dict (OrderedDict): The state dictionary from the pretrained model.
        new_model (nn.Module): The new model instance to load the weights into.

    Returns:
        nn.Module: The model with the loaded weights.
    """
    pretrained_state_dict = OrderedDict()
    # Remap keys from the pretrained model to match the new model's key format.
    for k, v in stat_dict.items():
        if k.startswith("0"):
            if "blocks" in k:
                pretrain_key = k.replace("0.blocks", "blocks")
            else:
                pretrain_key = k.replace("0.", "")
        elif k.startswith("1."):
            pretrain_key = k.replace("1.", "")
        else:
            pretrain_key = k
        pretrained_state_dict[pretrain_key] = v

    model_dict = new_model.state_dict()

    # Filter the pretrained dictionary to include only keys that exist in the new model
    # and have matching numbers of elements.
    pre_dict_tmp = {
        k: v for k, v in pretrained_state_dict.items()
        if k in model_dict and model_dict[k].numel() == pretrained_state_dict[k].numel()
    }

    # Identify and report any keys that were not loaded.
    mismatch_keys = [k for k in model_dict.keys() if k not in pre_dict_tmp]
    if mismatch_keys:
        print("Partially loading model, ignoring buffers: {:s}".format(' '.join(mismatch_keys)))

    # Load the filtered state dictionary into the new model.
    model_dict.update(pre_dict_tmp)
    new_model.load_state_dict(model_dict)

    return new_model


def set_parameter_requires_grad(model, feature_extracting):
    """
    Freezes or unfreezes the parameters of a model.

    Args:
        model (nn.Module): The model whose parameters will be set.
        feature_extracting (bool): If True, freeze the parameters. If False, leave them trainable.
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class SoftPredictor(nn.Module):
    """
    A predictor that computes logits for "soft" cluster assignments from SSL features.
    It takes a pretrained SSL model and learns a linear projection to predict cluster probabilities.
    """

    def __init__(self, ssl_model, km_class, ssl_features=768):
        """
        Initializes the SoftPredictor.

        Args:
            ssl_model (nn.Module): The pretrained self-supervised learning model (e.g., HuBERT).
            km_class (int): The number of target clusters (K-means classes).
            ssl_features (int): The feature dimension of the SSL model output.
        """
        super(SoftPredictor, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_features
        # self.km_class = 1000
        self.km_class = 200
        # The weight matrix for the linear projection.
        self.W = torch.nn.Parameter(torch.randn(self.km_class, self.ssl_features), requires_grad=True)

    def forward(self, wav):
        """
        Forward pass to compute soft predictions.

        Args:
            wav (Tensor): Input waveform tensor of shape [batch, 1, audio_len].

        Returns:
            Tensor: Logits of shape [batch, time_steps, km_class].
        """
        wav = wav.squeeze(1)  # Shape: [batch, audio_len]
        # Extract features from the SSL model.
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res['x']  # Shape: [batch, time_steps, ssl_features]

        # Normalize features and weights before computing the linear projection.
        x = F.normalize(x)
        W = F.normalize(self.W)
        logits = F.linear(x, W)

        return logits


class ResBlock1(torch.nn.Module):
    """Residual block with three dilated convolutions, as used in HiFi-GAN."""

    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    """Simpler residual block with two dilated convolutions."""

    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    """The HiFi-GAN Generator model."""

    def __init__(self, h):
        """
        Initializes the Generator.

        Args:
            h (AttrDict): A dictionary-like object containing hyperparameters.
        """
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        # Pre-convolution layer
        self.conv_pre = weight_norm(
            Conv1d(getattr(h, "model_in_dim", 128), h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel // (2 ** i), h.upsample_initial_channel // (2 ** (i + 1)),
                                k, u, padding=(k - u) // 2)))

        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        # Post-convolution layer
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        """
        Forward pass for the generator.

        Args:
            x (Tensor): Input tensor of shape [batch, channels, time_steps].

        Returns:
            Tensor: Generated waveform of shape [batch, 1, audio_len].
        """
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        """Removes weight normalization from all convolutional layers."""
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


def load_checkpoint(model_path, model, *, verbose: bool = True):
    """
    Loads a checkpoint into a model, handling different checkpoint formats.

    Args:
        model_path (str or Path): Path to the checkpoint file.
        model (nn.Module): The model to load the state dictionary into.

    Returns:
        nn.Module: The model with loaded weights.
    """
    checkpoint = torch.load(str(model_path), map_location='cpu')

    if str(model_path).endswith('.ckpt'):
        # PyTorch Lightning checkpoint
        state_dict_full = checkpoint.get('state_dict', checkpoint)
        # Filter out keys from the feature extractor model, if any.
        state_dict = {k: v for k, v in state_dict_full.items() if not k.startswith('feat_model')}
    else:
        # Regular PyTorch checkpoint
        state_dict = checkpoint.get('model', checkpoint)

    # Filter out size-mismatched keys (like label_embs_concat from SSL model)
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model.state_dict():
            model_shape = model.state_dict()[k].shape
            if model_shape == v.shape:
                filtered_state_dict[k] = v
            else:
                print(f"Skipping size-mismatched key: {k} (checkpoint: {v.shape}, model: {model_shape})")
        else:
            # Keep keys that might be in submodules
            filtered_state_dict[k] = v
    
    # Load the state dict, ignoring missing or unexpected keys.
    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)

    if verbose:
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")
        print(f"Successfully loaded checkpoint from: {str(model_path)}")

    return model


class latentGenerator(Generator):
    """
    A generator that uses latent features from SSL models and speaker embeddings.

    This class extends the base Generator by first extracting content features
    (e.g., from HuBERT) and speaker features (e.g., from ECAPA-TDNN),
    concatenating them, and then feeding them into the generator network.
    """

    def __init__(self, h):
        """
        Initializes the latentGenerator.

        Args:
            h (AttrDict): Hyperparameters containing model paths and configurations.
        """
        super().__init__(h)

        # --- Configure Content Feature Extractor (SSL Model) ---
        if h.get('soft_model_path') and h.get('hubert_model_path'):
            # Use HuBERT with a soft predictor for content features.
            feat_model, _, _ = checkpoint_utils.load_model_ensemble_and_task([str(h.hubert_model_path)])
            feat_model = feat_model[0]
            feat_model.remove_pretraining_modules()
            self.latent_encoder = SoftPredictor(feat_model, h.km_class)
            self.latent_encoder = load_checkpoint(h.soft_model_path, self.latent_encoder, verbose=False)
            set_parameter_requires_grad(self.latent_encoder, h.get('ssl_freeze', True))
            self.ssl_type = 'hubert_soft'

        elif h.get('km_model_path') and h.get('hubert_model_path'):
            # Use HuBERT with a K-Means model for discrete content features.
            feat_model, _, _ = checkpoint_utils.load_model_ensemble_and_task([str(h.hubert_model_path)])
            self.feat_model = feat_model[0]
            self.feat_model.remove_pretraining_modules()
            set_parameter_requires_grad(self.feat_model, h.get('ssl_freeze', True))
            self.kmeans_model = joblib.load(open(h.get('km_model_path'), "rb"))
            self.kmeans_model.verbose = False
            self.dict = nn.Embedding(200, 200)  # Embedding for K-means clusters
            self.ssl_type = 'hubert_km'

        else:
            # Use a generic SSL model (e.g., HuBERT or Wav2Vec) directly for content features.
            feat_model, _, _ = checkpoint_utils.load_model_ensemble_and_task([str(h.ssl_model_path)])
            self.feat_model = feat_model[0]
            self.feat_model.remove_pretraining_modules()
            set_parameter_requires_grad(self.feat_model, h.get('ssl_freeze', True))
            if 'wav2vec' in str(h.ssl_model_path):
                self.ssl_type = 'w2v_ssl'
            elif 'hubert' in str(h.ssl_model_path):
                self.ssl_type = 'hubert_ssl'

        # --- Configure Speaker Feature Extractor (x-vector model) ---
        if h.get('ecapa_fbank_model_path'):
            self.xv_feature = 'fbank'
            self.fbank = Fbank(n_mels=80)
            self.mean_var_norm = InputNormalization(norm_type='sentence', std_norm=False)
            self.xv_model = ECAPA_TDNN(80, lin_neurons=192) # Assumes 192 output dimension for ECAPA
            pre_train_model = torch.load(str(h.ecapa_fbank_model_path), map_location="cpu")
            # Custom loader may be needed if state_dict keys don't match perfectly
            self.xv_model.load_state_dict(pre_train_model)
            set_parameter_requires_grad(self.xv_model, h.get('xv_freeze', True))

        # --- Configure input features to the generator ---
        self.f0 = h.get('f0', None)
        self.latent = h.get('latent', None)
        self.xv = h.get('xv', None)

    @staticmethod
    def _upsample(signal, max_frames):
        """
        Upsamples a conditioning signal to match the number of frames.

        Args:
            signal (Tensor): The signal to upsample, of shape [B, C, L].
            max_frames (int): The target length to upsample to.

        Returns:
            Tensor: The upsampled signal of shape [B, C, max_frames].
        """
        if signal.dim() == 3:
            bsz, channels, cond_length = signal.size()
        elif signal.dim() == 2:
            signal = signal.unsqueeze(2)
            bsz, channels, cond_length = signal.size()
        else:
            signal = signal.view(-1, 1, 1)
            bsz, channels, cond_length = signal.size()

        signal = signal.unsqueeze(3).repeat(1, 1, 1, max_frames // cond_length)

        # pad zeros as needed (if signal's shape does not divide completely with max_frames)
        reminder = (max_frames - signal.shape[2] * signal.shape[3]) // signal.shape[3]
        if reminder > 0:
            raise NotImplementedError('Padding condition signal - misalignment between condition features.')

        signal = signal.view(bsz, channels, max_frames)
        return signal


    def forward(self, **kwargs):
        """
        The main forward pass for training and inference.

        Args:
            kwargs (dict): A dictionary of inputs, must include 'audio'.
                           Can also include 'f0'.

        Returns:
            Tensor: The generated waveform from the base Generator.
        """
        audio_data = kwargs['audio']
        if self.latent and self.xv:
            if self.ssl_type == 'hubert_soft':
                # audio-(batchsize,1,len)
                # x-(batchsize,frames,200)
                x = self.latent_encoder(audio_data)
            if self.xv_feature == 'fbank':
                # fbank input (batchsize,wav_len)
                with torch.no_grad():
                    xv_input = self.fbank(audio_data.squeeze(1))
                    xv_input = self.mean_var_norm(xv_input, torch.ones(xv_input.shape[0]).to(xv_input.device))
                xv, _ = self.xv_model(xv_input)

            x = F.layer_norm(x, x.shape)
            xv = F.layer_norm(xv, xv.shape)
            x = x.transpose(2, 1)
            xv = xv.transpose(2, 1)

            #ssl model hop_size=320, but losss 1 dim, need to add
            x = torch.nn.functional.pad(x,(0,1),'replicate')
        if self.f0:
            if x.shape[-1] < kwargs['f0'].shape[-1]:
                x = self._upsample(x, kwargs['f0'].shape[-1])
            else:
                kwargs['f0'] = self._upsample(kwargs['f0'], x.shape[-1])
            x = torch.cat([x, kwargs['f0']], dim=1)

        xv = self._upsample(xv, x.shape[-1])
        x = torch.cat([x, xv], dim=1)

        return super().forward(x)


class DiscriminatorP(torch.nn.Module):
    """Period-based discriminator (from HiFi-GAN's MPD)."""

    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        # Reshape 1D audio into 2D for processing
        b, c, t = x.shape
        if t % self.period != 0:  # Pad to be divisible by period
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    """Multi-Period Discriminator (MPD), a collection of DiscriminatorP modules."""

    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    """Scale-based discriminator (from HiFi-GAN's MSD)."""

    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    """Multi-Scale Discriminator (MSD), a collection of DiscriminatorS modules on different scales."""

    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []

        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


# --- GAN Loss Functions ---

def feature_loss(fmap_r, fmap_g):
    """
    Computes the feature matching loss between real and generated feature maps.
    """
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """
    Computes the least-squares GAN loss for the discriminator.
    """
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    """
    Computes the least-squares GAN loss for the generator.
    """
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l
    return loss, gen_losses