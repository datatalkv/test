import torch
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import dataclasses
import json
import math
import random
import copy


def prettify(z: Union[complex, List[complex], np.array]) -> str:
    def fix_float(f: float) -> Union[float, int]:
        if abs(round(f) - f) < 1e-6:
            return round(f)
        return f

    if isinstance(z, complex) or isinstance(z, float) or isinstance(z, int):
        re = fix_float(z.real)
        im = fix_float(z.imag)
        if im == 1:
            im_str = "i"
        elif im == -1:
            im_str = "-i"
        else:
            im_str = f"{im}i" if isinstance(im, int) else f"{im:.3f}i"

        re_str = f"{re}" if isinstance(re, int) else f"{re:.3f}"

        if not re:
            if not im:
                return "0"
            return im_str
        elif not im:
            return re_str
        else:
            return f"{re_str} + {im_str}"
    elif isinstance(z, list):
        return prettify(np.array(z))
    else:
        if len(z.shape) == 1:
            return "  ".join([prettify(a) for a in z.tolist()])
        elif len(z.shape) == 2:
            return "\n".join(
                ["\t".join([prettify(a) for a in row]) for row in z.tolist()]
            )
        else:
            raise NotImplementedError("3rd and higher order tensors unsupported")


@dataclasses.dataclass
class Config:
    learning_rate: float
    epochs: int
    betas: Tuple[float, float]
    weight_decay: float
    device_type: str
    num_workers: int


@dataclasses.dataclass
class HyenaConfig(Config):
    d_model: int
    n_layers: int
    vocab_size: int
    d_embed: int
    d_filter_mlp: int
    n_filter_layers: int
    context_length: int
    short_conv_size: int
    order: int
    pdrop_hyena: float
    pdrop_embed: float
    omega: Optional[int]


@dataclasses.dataclass
class AttentionConfig(Config):
    d_model: int
    n_layers: int
    vocab_size: int
    d_embed: int
    n_head: int
    context_length: int
    pdrop_attn: float
    pdrop_embed: float


class Projection(torch.nn.Module):
    def __init__(self, d_model: int, N: int, conv_len: int):
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.linear = torch.nn.Linear(d_model, d_model * (N + 1))
        self.conv = torch.nn.Conv1d(
            in_channels=d_model * (N + 1),
            out_channels=d_model * (N + 1),
            kernel_size=conv_len,
            groups=d_model * (N + 1),  # Depthwise convolution
            padding=conv_len - 1,
        )

    def forward(self, u: torch.Tensor) -> List[torch.Tensor]:
        # u.shape == (batch, len, d_model)
        z = self.linear(u)
        z = z.transpose(1, 2)  # Channels (embedding dim) needs to come first

        L = z.shape[2]
        z = self.conv(z)[..., :L]

        x = torch.split(z, self.d_model, dim=1)
        # len(x) == (N+1)
        # x[0].shape == (batch, d_model, len)
        return x


class FFTConv(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, h: torch.Tensor, x: torch.Tensor, B: torch.Tensor
    ) -> torch.Tensor:
        L = max(x.shape[-1], h.shape[-1])
        h_f = torch.fft.rfft(h, n=2 * L, norm="forward")
        x_f = torch.fft.rfft(x.to(dtype=h.dtype), n=2 * L)
        y = torch.fft.irfft(h_f * x_f, n=2 * L, norm="forward")[..., :L]
        y = y + x * B
        y = y.to(dtype=h.dtype)  # y is ComplexFloat but we need it to be float
        return y


class HyenaBlock(torch.nn.Module):
    def __init__(self, config: HyenaConfig):
        super().__init__()
        self.proj_input = Projection(
            config.d_model, config.order, config.short_conv_size
        )
        self.proj_output = torch.nn.Linear(config.d_model, config.d_model)
        self.filter = HyenaFilter(
            d_model=config.d_model,
            d_mlp=config.d_filter_mlp,
            d_embed=config.d_embed,
            N=config.order,
            n_layers=config.n_filter_layers,
            max_seq_len=config.context_length,
            omega=config.omega,
        )
        self.dropout = torch.nn.Dropout(config.pdrop_hyena)
        self.fft_conv = FFTConv()
        self.B = torch.nn.Parameter(torch.randn((config.order, 1, config.d_model, 1)))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        L = u.shape[1]

        *x, v = self.proj_input(u)
        v = v + u.transpose(1, 2)  # skip connection

        h = self.filter(L)

        for i, x_i in enumerate(x):
            h_i = h[i].unsqueeze(0)
            v = v + torch.softmax(x_i, dim=1) * self.fft_conv(
                h_i, v, self.B[i]
            )  # skip connection

        v = v.transpose(1, 2)
        y = v + self.proj_output(v)  # skip connection

        return y


class Window(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        fast_decay_pct: float = 0.3,
        slow_decay_pct: float = 1.5,
        target: float = 1e-2,
    ):
        super().__init__()
        self.b = torch.nn.Parameter(torch.zeros((1, d_model, 1)))
        min_decay = math.log(target) / slow_decay_pct
        max_decay = math.log(target) / fast_decay_pct
        self.alphas = torch.nn.Parameter(
            torch.linspace(start=min_decay, end=max_decay, steps=d_model)[None, :, None]
        )
        self.t = torch.nn.Parameter(
            torch.linspace(start=0, end=1, steps=max_seq_len)[None, None, :],
            requires_grad=False,
        )

    def forward(self, x):
        L = x.shape[2]
        c = torch.exp(self.alphas * self.t)[:, :, :L]
        x = x * (c + self.b)
        return x


class HyenaFilter(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_mlp: int,
        d_embed: int,
        N: int,
        n_layers: int = 4,
        max_seq_len: int = 128,
        omega: int = 8,
    ):
        assert n_layers >= 2, "n_layers must be at least 2"
        super().__init__()

        self.N = N
        self.d_model = d_model

        # Making this a parameter, even though it is not trained, ensures
        # it will be moved to the gpu with the rest of the model
        self.h = torch.nn.Parameter(torch.randn((N, d_model, max_seq_len)))

        self.window = Window(d_model, max_seq_len)

    def forward(self, L: int) -> torch.Tensor:
        h = self.h[:, :, :L]
        h = self.window(h)

        h = h / torch.norm(h, dim=-2, p=1, keepdim=True)

        return h


if __name__ == "__main__":

    hyena_config = HyenaConfig(
        d_model=4,
        n_layers=12,
        vocab_size=1e5,
        d_embed=33,
        d_filter_mlp=64,
        n_filter_layers=4,
        context_length=33,
        short_conv_size=3,
        order=2,
        pdrop_hyena=0.0,
        pdrop_embed=0.2,
        omega=12,
        epochs=40,
        learning_rate=6e-4,
        betas=(0.9, 0.98),
        weight_decay=0.4,
        #device_type="gpu",  # cpu, gpu
        device_type="cpu",  # cpu, gpu
        num_workers=4,
    )
    x = torch.randn(33, 100, 4)
    m = Projection(d_model=4, N=100, conv_len=9)
    yhat = m(x)

    m_hyena = HyenaBlock(hyena_config)
    yhat = m_hyena(x)
    x2 = copy.deepcopy(x)
    x2[:, 3, :] = 1e10 * torch.sign(x2[:, 3, :])
    yhat2 = m_hyena(x2)
    print(yhat[0,:20,0])
    print(yhat2[0,:20,0])
    import sys, IPython

    IPython.embed()
    sys.exit(0)
