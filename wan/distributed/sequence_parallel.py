# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

try:
    import torch_musa
except ModuleNotFoundError:
    torch_musa = None

from wan.modules.model import sinusoidal_embedding_1d
from wan.distributed.ulysses import distributed_attention
from wan.distributed.util import gather_forward, get_rank, get_world_size
from wan.utils.platform import get_device_type


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


def pad_tensor(original_tensor, target_len, pad_value=0.0):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.full(
        (pad_size, s1, s2),
        pad_value,
        dtype=original_tensor.dtype,
        device=original_tensor.device,
    )
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


@torch.amp.autocast(get_device_type(), enabled=False)
def rope_apply(x, grid_sizes, freqs):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
            s, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        sp_size = get_world_size()
        sp_rank = get_rank()
        freqs_i = pad_freqs(freqs_i, s * sp_size)
        s_per_rank = s
        freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) *
                                                       s_per_rank), :, :]
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


@torch.amp.autocast(get_device_type(), enabled=False)
def rope_apply_musa(x, grid_sizes, freqs):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    c0 = c - 2 * (c // 3)
    c1 = c // 3
    c2 = c // 3

    # split freqs
    freqs_real = freqs[0].split([c0, c1, c2], dim=1)
    freqs_imag = freqs[-1].split([c0, c1, c2], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = x[i, :seq_len].reshape(s, n, -1, 2)
        x_real = x_i[..., 0]
        x_imag = x_i[..., 1]
        freqs_real = torch.cat(
            [
                freqs_real[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs_real[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs_real[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        freqs_imag = torch.cat(
            [
                freqs_imag[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs_imag[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs_imag[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        sp_size = torch.distributed.get_world_size()
        sp_rank = torch.distributed.get_rank()

        freqs_real = pad_tensor(freqs_real, s * sp_size, 1.0)
        freqs_imag = pad_tensor(freqs_imag, s * sp_size, 0.0)

        freqs_real_rank = freqs_real[(sp_rank * s) : ((sp_rank + 1) * s), :, :]
        freqs_imag_rank = freqs_imag[(sp_rank * s) : ((sp_rank + 1) * s), :, :]

        out_real = x_real * freqs_real_rank - x_imag * freqs_imag_rank
        out_imag = x_real * freqs_imag_rank + x_imag * freqs_real_rank

        x_out = torch.stack([out_real, out_imag], dim=-1).flatten(2)
        x_out = torch.cat([x_out, x[i, seq_len:]], dim=0)

        # append to collection
        output.append(x_out)
    return torch.stack(output)


def sp_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    y=None,
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    if self.model_type == 'i2v':
        assert y is not None
    # params
    dtype = self.patch_embedding.weight.dtype
    device = self.patch_embedding.weight.device
    if torch_musa is None:
        if self.freqs.dtype != dtype or self.freqs.device != device:
            self.freqs = self.freqs.to(dtype=dtype, device=device)
    else:
        if self.freqs[0].dtype != dtype or self.freqs[0].device != device:
            self.freqs = (
                self.freqs[0].to(dtype=dtype, device=device),
                self.freqs[-1].to(dtype=dtype, device=device)
            )

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    # time embeddings
    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    with torch.amp.autocast(get_device_type(), dtype=torch.float32):
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim,
                                    t).unflatten(0, (bt, seq_len)).float())
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    # Context Parallel
    x = torch.chunk(x, get_world_size(), dim=1)[get_rank()]
    e = torch.chunk(e, get_world_size(), dim=1)[get_rank()]
    e0 = torch.chunk(e0, get_world_size(), dim=1)[get_rank()]

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens)

    for block in self.blocks:
        x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # Context Parallel
    x = gather_forward(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]


def sp_attn_forward(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    if torch_musa is None:
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
    else:
        q = rope_apply_musa(q, grid_sizes, freqs)
        k = rope_apply_musa(k, grid_sizes, freqs)

    x = distributed_attention(
        half(q),
        half(k),
        half(v),
        seq_lens,
        window_size=self.window_size,
    )

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x
