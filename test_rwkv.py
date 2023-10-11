import torch
from torch import nn, Tensor
from typing import Optional, Tuple, List
import torch.nn.functional as F

def get_mask(tsz: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> Tensor:
    """Returns the forward mask, used for training.

    Args:
        tsz: The number of timesteps in the mask
        device: The mask device
        dtype: The mask dtype

    Returns:
        The forward mask, with shape (T, T)
    """
    mask = torch.empty(tsz, tsz, device=device, dtype=dtype)
    mask.fill_(float("-inf"))
    # mask.triu_(1)
    mask.tril_(-1)
    return mask


def run_wkv(
    tsz: int,
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    last_num: Tensor,
    last_den: Tensor,
    mask: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Runs the core WKV computation.

    Args;
        tsz: The number of timesteps
        w: The decay tensor, with shape (D)
        u: The output multiplier tensor, with shape (D)
        k: The K tensor, with shape (B, T, D)
        v: The V tensor, with shape (B, T, D)
        last_num: The last numerator, with shape (B, 1, D)
        last_den: The last denominator, with shape (B, 1, D)
        mask: The attention mask, with shape (T, T)

    Returns:
        The WKV tensor, with shape (B, T, D), and the next numerator and
        denominator tensors, each with shape (B, T, D)
    """
    assert w.dim() == u.dim() == 1
    assert mask is None or mask.dim() == 2
    assert k.dim() == v.dim() == last_num.dim() == last_den.dim() == 3

    t = torch.arange(tsz + 1, device=w.device)[None, :, None]
    wt = t[:, None, :-1, :] - t[:, :-1, None, :]
    w = -torch.exp(w)
    tw = w * t[:, 1:]
    twt = w * wt
    ktw = twt + k[:, :, None]
    if mask is not None:
        ktw = ktw + mask[None, :tsz, :tsz, None]

    etw, ektw = torch.exp(tw), torch.exp(ktw)
    num = etw * last_num + (ektw * v[:, :, None]).sum(1)
    den = etw * last_den + ektw.sum(1)

    last_num = torch.cat((last_num, num[..., :-1, :]), dim=-2)
    last_den = torch.cat((last_den, den[..., :-1, :]), dim=-2)

    out = (last_num + torch.exp(u + k) * v) / (last_den + torch.exp(u + k))

    return out, num, den


class Attention(nn.Module):
    init_x: Tensor
    init_num: Tensor
    init_den: Tensor
    mask: Tensor

    def __init__(self, emb_dim: int, max_tsz: int = 1024) -> None:
        super().__init__()

        self.time_decay = nn.Parameter(torch.empty(emb_dim))
        self.time_first = nn.Parameter(torch.empty(emb_dim))

        self.time_mix_k = nn.Parameter(torch.empty(1, 1, emb_dim))
        self.time_mix_v = nn.Parameter(torch.empty(1, 1, emb_dim))
        self.time_mix_r = nn.Parameter(torch.empty(1, 1, emb_dim))

        self.key = nn.Linear(emb_dim, emb_dim, bias=False)
        self.value = nn.Linear(emb_dim, emb_dim, bias=False)
        self.receptance = nn.Linear(emb_dim, emb_dim, bias=False)
        self.output = nn.Linear(emb_dim, emb_dim, bias=False)

        self.register_buffer("init_x", torch.zeros(1, 1, emb_dim), persistent=False)
        self.register_buffer("init_num", torch.zeros(1, 1, emb_dim), persistent=False)
        self.register_buffer("init_den", torch.zeros(1, 1, emb_dim), persistent=False)
        self.register_buffer("mask", get_mask(max_tsz), persistent=False)

    def time_shift(self, last_x: Tensor, x: Tensor) -> Tensor:
        _, tsz, _ = x.shape
        if tsz > 1:
            last_x = torch.cat((last_x, x[..., :-1, :]), dim=-2)
        return last_x

    def forward(self, x: Tensor, state: Tuple) -> tuple[Tensor, Tuple]:
        bsz, tsz, _ = x.shape

        # last_x, last_num, last_den = (self.init_x, self.init_num, self.init_den) if state is None else state
        last_x, last_num, last_den = (
            torch.tile(self.init_x, [bsz, 1, 1]),
            torch.tile(self.init_num, [bsz, 1, 1]), 
            torch.tile(self.init_den, [bsz, 1, 1])
        ) if state is None else state
        last_x = self.time_shift(last_x, x)

        k = self.key(x * self.time_mix_k + last_x * (1 - self.time_mix_k))
        v = self.value(x * self.time_mix_v + last_x * (1 - self.time_mix_v))
        r = self.receptance(x * self.time_mix_r + last_x * (1 - self.time_mix_r))
        sr = torch.sigmoid(r)

        w, u = self.time_decay, self.time_first
        wkv, num, den = run_wkv(tsz, w, u, k, v, last_num, last_den, self.mask)
        rwkv = wkv * sr

        return self.output(rwkv), (x[..., -1:, :], num[..., -1:, :], den[..., -1:, :])


class FeedForward(nn.Module):
    init_state: Tensor

    def __init__(self, emb_dim: int, ffn_dim: int) -> None:
        super().__init__()

        self.time_mix_k = nn.Parameter(torch.empty(1, 1, emb_dim))
        self.time_mix_r = nn.Parameter(torch.empty(1, 1, emb_dim))

        self.key = nn.Linear(emb_dim, ffn_dim, bias=False)
        self.receptance = nn.Linear(emb_dim, emb_dim, bias=False)
        self.value = nn.Linear(ffn_dim, emb_dim, bias=False)

        self.register_buffer("init_state", torch.zeros(1, 1, emb_dim), persistent=False)

    def time_shift(self, last_x: Tensor, x: Tensor) -> Tensor:
        _, tsz, _ = x.shape
        if tsz > 1:
            last_x = torch.cat((last_x, x[..., :-1, :]), dim=-2)
        return last_x

    def forward(self, x: Tensor, state: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        last_x = self.time_shift(torch.tile(self.init_state, [x.shape[0],1,1]) if state is None else state, x)

        k = self.key(x * self.time_mix_k + last_x * (1 - self.time_mix_k))
        r = self.receptance(x * self.time_mix_r + last_x * (1 - self.time_mix_r))
        vk = self.value(F.relu(k) ** 2)

        return torch.sigmoid(r) * vk, x[..., -1:, :]


class Block(nn.Module):
    def __init__(self, emb_dim: int, pre_norm: bool) -> None:
        super().__init__()

        self.ln0 = nn.LayerNorm(emb_dim) if pre_norm else None
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)

        self.att = Attention(emb_dim)
        self.ffn = FeedForward(emb_dim, emb_dim * 4)

    def forward(self, x: Tensor, state: Optional[Tuple] = None) -> tuple[Tensor, Tuple]:
        if self.ln0 is not None:
            x = self.ln0(x)
        dx, att_state_out = self.att(self.ln1(x), None if state is None else state[0])
        x = x + dx
        dx, ffn_state_out = self.ffn(self.ln2(x), None if state is None else state[1])
        x = x + dx
        return x, (att_state_out, ffn_state_out)


class Rwkv(nn.Module):
    def __init__(self, emb_dim: int, num_layers: int, out_dim: int) -> None:
        super().__init__()

        self.blocks = nn.ModuleList([Block(emb_dim, i == 0) for i in range(num_layers)])
        self.ln_out = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, out_dim, bias=False)

    def forward(self, x: Tensor, states_in: Optional[List[Tuple]] = None) -> tuple[Tensor, List[Tuple]]:
        # x.shape = (batch, len_seq, emb_dim)
        states_out: list[Tuple] = []
        for i, block in enumerate(self.blocks):
            x, state_out = block(x, None if states_in is None else states_in[i])
            states_out.append(state_out)
        yhat = self.head(self.ln_out(x))
        return yhat, states_out


if __name__ == "__main__":
    import copy
    x = torch.randn(33, 100, 4)
    m = Rwkv(emb_dim=4, out_dim=5, num_layers=3)
    yhat, states = m(x)
    x2 = copy.deepcopy(x)
    x2[:, 3, :] = 1e10 * torch.sign(x2[:, 3, :])
    yhat2, states2 = m(x2)
    print(yhat[0,:20,0])
    print(yhat2[0,:20,0])
    import sys,IPython; IPython.embed(); sys.exit(0)
