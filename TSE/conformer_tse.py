import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def swish(x):
    return x * torch.sigmoid(x)


def l2_norm(s1, s2):
    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm


def snr(s1, s2, eps=1e-8):
    s_target = s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
    return torch.mean(snr)


def spsnr(s1, s2, eps=1e-8):
    s_target = s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
    return snr


def si_snr(s1, s2, eps=1e-8):
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
    return torch.mean(snr)


def mse(s1, s2):
    return torch.mean((s1 - s2) ** 2)


class Residual(nn.Module):
    def __init__(self, module, half=False, rezero=False):
        super(Residual, self).__init__()
        self.net = module
        self.half = half
        if rezero:
            self.resweight = nn.Parameter(torch.Tensor([0]))
        self.rezero = rezero
        if self.half and self.rezero:
            raise NotImplementedError('half and rezero cannot be True at the same time')

    def forward(self, inputs, **kwargs):
        x = self.net(inputs, **kwargs)
        if self.half:
            return (x * 0.5) + inputs
        elif self.rezero:
            return (x * self.resweight) + inputs
        else:
            return x + inputs


class FFModule2(nn.Module):
    def __init__(self, d_model, outdim, h_size, dropout=0.2):
        super(FFModule2, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer1 = nn.Linear(d_model, h_size)
        self.swish_activation = swish
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(h_size, outdim)

    def forward(self, inputs):
        x = self.layer_norm(inputs)
        x = self.layer1(x)
        x = self.swish_activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, in_channels, kernel_size=31, dropout=0.2, bias=True):
        super(ConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0

        self.layer_norm = nn.LayerNorm(in_channels)
        self.pos_conv1 = nn.Conv1d(
            in_channels, 2 * in_channels, kernel_size=1, stride=1, padding=0, bias=bias,
        )
        self.glu_activation = F.glu
        self.depthwise_conv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=in_channels,
            bias=bias
        )
        self.batch_norm = nn.BatchNorm1d(in_channels)
        self.swish_activation = swish
        self.pointwise_conv2 = nn.Conv1d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=bias,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.pos_conv1(x.transpose(1, 2))
        x = self.glu_activation(x, dim=1)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish_activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()
        self.demb = demb
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1).transpose(0, 1)
        else:
            return pos_emb[:, None, :].transpose(0, 1)


class RelPositionMultiHeadedAttention(nn.Module):
    def __init__(self, d_model, n_head=4, dropout=0.2):
        super(RelPositionMultiHeadedAttention, self).__init__()
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

        assert d_model % n_head == 0
        self.d_k = d_model // n_head
        self.n_head = n_head

        self.qkv_net = nn.Linear(d_model, d_model * 3, bias=False)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

        self.pos_bias_u = nn.Parameter(torch.Tensor(self.n_head, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.n_head, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)
        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]
        return x

    def forward_qkv(self, w, mem):
        bsz = w.size(0)
        qlen = w.size(1)
        if mem is not None:
            cat = torch.cat([mem, w], 0)
            w_heads = self.qkv_net(self.layer_norm(cat))
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            w_heads = self.qkv_net(self.layer_norm(w))
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(1)
        w_head_q = w_head_q.view(bsz, qlen, self.n_head, self.d_k)
        w_head_k = w_head_k.view(bsz, klen, self.n_head, self.d_k)
        w_head_v = w_head_v.view(bsz, klen, self.n_head, self.d_k)

        return w_head_q, w_head_k.transpose(1, 2), w_head_v.transpose(1, 2)

    def forward_attention(self, value, scores, mask):
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)
            min_value = float(
                np.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min
            )
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            self.attn = torch.softmax(scores, dim=-1)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.n_head * self.d_k)
        return self.linear_out(x)

    def forward(self, query, pos_emb, mem=None, mask=None):
        q, k, v = self.forward_qkv(query, mem)
        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.n_head, self.d_k)
        p = p.transpose(1, 2)

        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)


class MHAModule(nn.Module):
    def __init__(self, d_model, dropout=0.2, **kwargs):
        super(MHAModule, self).__init__()
        self.pe = PositionalEmbedding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.mha_RPE = RelPositionMultiHeadedAttention(d_model, dropout=dropout, **kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, attn_mask=None, mems=None, head_mask=None):
        pos_emb = self.pe(inputs[0, :, 0])
        x = self.layer_norm(inputs)
        x = self.mha_RPE(x, pos_emb, mems, attn_mask)
        x = self.dropout(x)
        return x


class Conformer(nn.Module):
    def __init__(
            self,
            d_model=512 + 192,
            ff1_hsize=1024,
            ff1_dropout=0.2,
            n_head=4,
            mha_dropout=0.2,
            kernel_size=3,
            conv_dropout=0.2,
            ff2_hsize=1024,
            ff2_dropout=0.2,
            half=True,
            rezero=False,
            output_reshape=True,
            outdim=512,
    ):
        super(Conformer, self).__init__()

        self.ff_module1 = Residual(
            module=FFModule2(
                d_model=d_model,
                outdim=d_model,
                h_size=ff1_hsize,
                dropout=ff1_dropout
            ),
            half=half, rezero=rezero
        )
        self.mha_module = Residual(
            module=MHAModule(
                d_model=d_model,
                n_head=n_head,
                dropout=mha_dropout
            ), rezero=rezero
        )
        self.conv_module = Residual(
            module=ConvModule(
                in_channels=d_model,
                kernel_size=kernel_size,
                dropout=conv_dropout
            ), rezero=rezero
        )
        self.ff_module2 = Residual(
            FFModule2(
                d_model=d_model,
                outdim=d_model,
                h_size=ff2_hsize,
                dropout=ff2_dropout
            ),
            half=half, rezero=rezero
        )

        if output_reshape:
            self.out = nn.Linear(d_model, outdim)

    def forward(self, inputs, **kwargs):
        x = self.ff_module1(inputs)
        x = self.mha_module(x, **kwargs)
        x = self.conv_module(x)
        x = self.ff_module2(x)
        x = self.out(x)
        return x


class Model(nn.Module):
    def __init__(self, stack_num=4, fft_size=512):
        super(Model, self).__init__()
        self.conformer_layers = nn.ModuleList([
            nn.Sequential(Conformer(fft_size + 192)) for _ in range(stack_num)
        ])
        self.fft_size = fft_size
        self.window = torch.hann_window(fft_size).to(device='cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, xvector, _):
        x, ref = x
        mixout = x
        x = x.squeeze(2)
        # istft defaults to the "minimal" reconstruction length, which can be
        # shorter than the mixture (e.g. half the samples). Force match to input
        # so downstream mix = target + residual aligns sample-for-sample.
        wav_len = int(x.shape[-1])

        with torch.no_grad():
            stft_complex = torch.stft(
                x, 512, hop_length=128, window=self.window,
                win_length=512, return_complex=True
            ).permute(0, 2, 1)[:, :, 1:]
            xr = torch.real(stft_complex)
            xi = torch.imag(stft_complex)
            mr, mi = xr, xi

        _, T, F = xr.size()

        xvector = xvector / torch.max(xvector)
        xvector = xvector.repeat(1, T, 1)
        x = torch.cat([xr, xi, xvector], dim=-1)

        for idx, layer in enumerate(self.conformer_layers):
            if idx != 0:
                x = torch.cat([x, xvector], dim=-1)
            x = layer(x)

        # Get estimated complex ratio mask
        s1_r, s1_i = x[:, :, :F], x[:, :, F:]

        # Calculate target speaker spectrum using estimated mask
        target_real = mr * s1_r - mi * s1_i
        target_imag = mr * s1_i + mi * s1_r
        target_stft_complex = torch.complex(target_real, target_imag)

        # Calculate interference speaker spectrum using (1 - estimated mask)
        # For complex mask: (1 - mask) = (1 - s1_r) + j*(-s1_i)
        interference_mask_r = 1.0 - s1_r
        interference_mask_i = -s1_i
        interference_real = mr * interference_mask_r - mi * interference_mask_i
        interference_imag = mr * interference_mask_i + mi * interference_mask_r
        interference_stft_complex = torch.complex(interference_real, interference_imag)

        # Convert to waveforms
        target_stft_padded = torch.nn.functional.pad(
            target_stft_complex, (1, 0), "constant", 0
        ).permute(0, 2, 1)
        target_wav = torch.istft(
            target_stft_padded,
            512,
            hop_length=128,
            win_length=512,
            window=self.window,
            return_complex=False,
            length=wav_len,
        )
        target_wav = torch.clamp(target_wav, -1, 1)

        interference_stft_padded = torch.nn.functional.pad(
            interference_stft_complex, (1, 0), "constant", 0
        ).permute(0, 2, 1)
        interference_wav = torch.istft(
            interference_stft_padded,
            512,
            hop_length=128,
            win_length=512,
            window=self.window,
            return_complex=False,
            length=wav_len,
        )
        interference_wav = torch.clamp(interference_wav, -1, 1)

        # Estimated soft mask (complex)
        estimated_mask = torch.complex(s1_r, s1_i)

        return  target_wav,estimated_mask, interference_wav

    def total_param(self):
        return sum([p.numel() for p in self.parameters()]) / 1000.0 / 1000.0

    def compute_loss(self, out, label, epoch=50, **kwargs):
        pwav, mix_out = out
        mix_out = mix_out.squeeze(2)

        if isinstance(label, tuple):
            twav = label[0].squeeze(2)
        else:
            twav = label.squeeze(2)

        loss_type = kwargs['loss_type']

        if loss_type == 'spsnr':
            est_wav = pwav
            label = twav
            minlen = min(label.shape[1], est_wav.shape[1])
            label = label[:, :minlen]
            est_wav = est_wav[:, :minlen]

            snr_sc = spsnr(est_wav, label)
            snr_th = kwargs['snr_th']
            ep = kwargs['ep']

            if ep[1] > epoch > ep[0]:
                if (snr_sc > snr_th[0]).numel() == 0:
                    sc = snr_sc.mean()
                else:
                    sc = snr_sc[snr_sc > snr_th[0]].mean()
            elif ep[2] > epoch >= ep[1]:
                if (snr_sc > snr_th[1]).numel() == 0:
                    sc = snr_sc.mean()
                else:
                    sc = snr_sc[snr_sc > snr_th[1]].mean()
            elif ep[3] > epoch >= ep[2]:
                if (snr_sc > snr_th[2]).numel() == 0:
                    sc = snr_sc.mean()
                else:
                    sc = snr_sc[snr_sc > snr_th[2]].mean()
            elif ep[4] > epoch >= ep[3]:
                if (snr_sc > snr_th[3]).numel() == 0:
                    sc = snr_sc.mean()
                else:
                    sc = snr_sc[snr_sc > snr_th[3]].mean()
            else:
                sc = snr_sc.mean()

            loss = {'total_loss': -sc}
            return loss

        elif loss_type == 'SNR_datamap':
            est_wav = pwav
            label = twav
            minlen = min(label.shape[1], est_wav.shape[1])
            label = label[:, :minlen]
            mix_out = mix_out[:, :minlen]
            est_wav = est_wav[:, :minlen]
            snr_est = spsnr(est_wav, label)
            snr_mix = spsnr(mix_out, label)
            total_loss = -torch.mean(snr_est)
            loss = {'total_loss': total_loss, 'snr_est': snr_est, 'snr_mix': snr_mix}
            return loss

        elif loss_type == 'SNR':
            est_wav = pwav
            label = twav
            minlen = min(label.shape[1], est_wav.shape[1])
            label = label[:, :minlen]
            mix_out = mix_out[:, :minlen]
            est_wav = est_wav[:, :minlen]
            snr_est = spsnr(est_wav, label)
            snr_mix = spsnr(mix_out, label)
            total_loss = -torch.mean(snr_est)
            loss = {'total_loss': total_loss, 'snr_est': snr_est, 'snr_mix': snr_mix}
            return loss

        elif loss_type == 'SISNR':
            est_wav = pwav
            label = twav
            minlen = min(label.shape[1], est_wav.shape[1])
            label = label[:, :minlen]
            est_wav = est_wav[:, :minlen]
            loss = {'total_loss': -(si_snr(est_wav, label))}
            return loss


if __name__ == '__main__':
    torch.manual_seed(10)
    inputs = torch.randn([2, 16000, 1]).clamp_(-1, 1)
    xvector = torch.randn([2, 1, 192]).clamp_(-1, 1)

    net = Model(stack_num=4)
    target_wav, estimated_mask,  interference_wav = net((inputs, inputs), xvector, inputs)
    """
    1.target-speaker only waveform
    2.estimated soft mask (complex ratio mask)
    3.inference-speaker (plus noise) waveform (by computing “1 - estimated soft mask” and applying it for the input spectrum).
    
    """

    print(f"Estimated mask shape: {estimated_mask.shape}")
    print(f"Target speaker waveform shape: {target_wav.shape}")
    print(f"Interference speaker waveform shape: {interference_wav.shape}")
