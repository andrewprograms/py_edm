#!/usr/bin/env python
"""
Hybrid latent-space distortion + reverb demo
-------------------------------------------
• Encoder  : nonlinear (1×1 Conv + tanh) lifts audio into a latent space
• Reverb   : depth-wise 1-D convolution (linear, fixed kernel) in that space
• Decoder  : linear (1×1 Conv) maps back to a single audio channel
• Plot     : log-frequency spectrograms
             – **high time resolution, low frequency resolution**
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================
fs           = 48_000      # sample rate (Hz) – only for the y-axis labels
signal_len   = 2048        # samples in the test signal
latent_dim   = 16          # width of the latent space
kernel_size  = 63          # length of the reverb kernel (odd ⇒ “same” length)

# ============================================================================
# Synthetic input signal (unit impulse)
# ============================================================================
x = torch.zeros(1, 1, signal_len)   # [batch, channels, time]
x[0, 0, 0] = 1.0                    # Dirac impulse at t = 0

# ============================================================================
# Model components
# ============================================================================
class Encoder(nn.Module):
    """Non-linear encoder: 1×1 Conv followed by tanh."""
    def __init__(self, latent=latent_dim):
        super().__init__()
        self.proj = nn.Conv1d(1, latent, kernel_size=1)
    def forward(self, z):
        return torch.tanh(self.proj(z))

class LatentReverb(nn.Module):
    """Depth-wise 1-D convolution – one independent IR per channel."""
    def __init__(self, latent=latent_dim, ksize=kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(latent, latent, ksize,
                              padding=ksize // 2,
                              groups=latent, bias=False)
        # exponential-decay impulse response, identical per channel
        t  = torch.arange(ksize, dtype=torch.float32)
        ir = torch.exp(-t / (ksize / 6))
        ir = (ir / ir.sum()).flip(0)                      # causal → conv kernel
        with torch.no_grad():
            self.conv.weight.copy_(ir.repeat(latent, 1).unsqueeze(1))
        for p in self.parameters():                      # fixed kernel
            p.requires_grad = False
    def forward(self, z):
        return self.conv(z)

class Decoder(nn.Module):
    """Linear decoder: 1×1 Conv collapses latent_dim → 1 channel."""
    def __init__(self, latent=latent_dim):
        super().__init__()
        self.linear = nn.Conv1d(latent, 1, kernel_size=1)
    def forward(self, z):
        return self.linear(z)

# ============================================================================
# Forward pass (inference only)
# ============================================================================
encoder = Encoder()
reverb  = LatentReverb()
decoder = Decoder()

with torch.no_grad():
    y = decoder(reverb(encoder(x)))      # [1, 1, T]

# ============================================================================
# Helper: STFT → magnitude in dB
#   – **64-point FFT ⇒ coarse frequency,
#     hop=1 ⇒ single-sample time resolution**
# ============================================================================
n_fft = 2048
hop   = n_fft 
win   = torch.hann_window(n_fft)

def stft_mag_db(sig):
    """Return |STFT| in dB plus frequency & frame indices."""
    S = torch.stft(sig, n_fft, hop_length=hop, window=win,
                   center=True, pad_mode='reflect',
                   return_complex=True)                # [freq, time]
    mag_db = 20 * torch.log10(S.abs().clamp(min=1e-10))
    freqs  = torch.linspace(0, fs / 2, mag_db.size(0)) # Hz
    frames = torch.arange(mag_db.size(1)) * hop        # samples
    return mag_db, freqs, frames

S_in,  f, t = stft_mag_db(x.squeeze())
S_out, _, _ = stft_mag_db(y.squeeze())

# ============================================================================
# Log-frequency spectrogram figure
# ============================================================================
vmax, vmin = 0.0, -80.0            # centred, shared colour scale

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

def draw(ax, spec, title):
    pcm = ax.pcolormesh(t, f, spec, shading='nearest',
                        vmin=vmin, vmax=vmax)
    ax.set_yscale('log')
    ax.set_ylim(20, 20_000)
    ax.set_xlabel('Sample')
    ax.set_title(title)
    return pcm

draw(axes[0], S_in,  'Input (Impulse)')
pcm = draw(axes[1], S_out, 'Output (Processed)')
axes[0].set_ylabel('Frequency (Hz)')

cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])   # [left, bottom, width, height]
fig.colorbar(pcm, cax=cbar_ax, label='Magnitude (dB)')

fig.suptitle('Log-Frequency Spectrograms\n(high time-resolution, low freq-resolution)')
fig.tight_layout(rect=[0, 0, 0.90, 1])  # leave space for the standalone colourbar
plt.show()
