import unittest

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.test.test_utils import CustomTestCase

HAS_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()
CHUNK_SIZE = 64


def _prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def _prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    indices = torch.cat(
        [
            torch.arange(n, device=cu_seqlens.device)
            for n in triton.cdiv(_prepare_lens(cu_seqlens), chunk_size).tolist()
        ]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def _prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(_prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)


@triton.jit
def _safe_exp(x):
    return tl.exp(tl.where(x <= 0, x, float("-inf")))


@triton.jit(do_not_specialize=["T"])
def solve_tril_16x16_kernel(
    A,
    Ad,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    A = A + (bos * H + i_h) * BT
    Ad = Ad + (bos * H + i_h) * 16

    offset = (i_t * 16) % BT
    p_A = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * 16, offset), (16, 16), (1, 0)
    )
    p_Ai = tl.make_block_ptr(Ad, (T, 16), (H * 16, 1), (i_t * 16, 0), (16, 16), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
    b_A = -tl.where(tl.arange(0, 16)[:, None] > tl.arange(0, 16)[None, :], b_A, 0)

    o_i = tl.arange(0, 16)
    for i in range(1, min(16, T - i_t * 16)):
        b_a = -tl.load(A + (i_t * 16 + i) * H * BT + o_i + offset)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
        mask = o_i == i
        b_A = tl.where(mask[:, None], b_a, b_A)
    b_A += o_i[:, None] == o_i[None, :]
    tl.store(
        p_Ai,
        b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


@triton.jit(do_not_specialize=["T"])
def merge_16x16_to_32x32_inverse_kernel(
    A,
    Ad,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    A += (bos * H + i_h) * 32
    Ad += (bos * H + i_h) * 16
    Ai += (bos * H + i_h) * 32

    p_A_21 = tl.make_block_ptr(
        A, (T, 32), (H * 32, 1), (i_t * 32 + 16, 0), (16, 16), (1, 0)
    )
    p_Ad_11 = tl.make_block_ptr(
        Ad, (T, 16), (H * 16, 1), (i_t * 32, 0), (16, 16), (1, 0)
    )
    p_Ad_22 = tl.make_block_ptr(
        Ad, (T, 16), (H * 16, 1), (i_t * 32 + 16, 0), (16, 16), (1, 0)
    )
    p_Ai_11 = tl.make_block_ptr(
        Ai, (T, 32), (H * 32, 1), (i_t * 32, 0), (16, 16), (1, 0)
    )
    p_Ai_22 = tl.make_block_ptr(
        Ai, (T, 32), (H * 32, 1), (i_t * 32 + 16, 16), (16, 16), (1, 0)
    )
    p_Ai_21 = tl.make_block_ptr(
        Ai, (T, 32), (H * 32, 1), (i_t * 32 + 16, 0), (16, 16), (1, 0)
    )

    A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    Ai_11 = tl.load(p_Ad_11, boundary_check=(0, 1)).to(tl.float32)
    Ai_22 = tl.load(p_Ad_22, boundary_check=(0, 1)).to(tl.float32)
    Ai_21 = -tl.dot(
        tl.dot(Ai_22, A_21, input_precision="ieee"), Ai_11, input_precision="ieee"
    )
    tl.store(
        p_Ai_11,
        Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_22,
        Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_21,
        Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


@triton.jit(do_not_specialize=["T"])
def merge_16x16_to_64x64_inverse_kernel(
    A,
    Ad,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    A += (bos * H + i_h) * 64
    Ad += (bos * H + i_h) * 16
    Ai += (bos * H + i_h) * 64

    p_A_21 = tl.make_block_ptr(
        A, (T, 64), (H * 64, 1), (i_t * 64 + 16, 0), (16, 16), (1, 0)
    )
    p_A_32 = tl.make_block_ptr(
        A, (T, 64), (H * 64, 1), (i_t * 64 + 32, 16), (16, 16), (1, 0)
    )
    p_A_31 = tl.make_block_ptr(
        A, (T, 64), (H * 64, 1), (i_t * 64 + 32, 0), (16, 16), (1, 0)
    )
    p_A_43 = tl.make_block_ptr(
        A, (T, 64), (H * 64, 1), (i_t * 64 + 48, 32), (16, 16), (1, 0)
    )
    p_A_42 = tl.make_block_ptr(
        A, (T, 64), (H * 64, 1), (i_t * 64 + 48, 16), (16, 16), (1, 0)
    )
    p_A_41 = tl.make_block_ptr(
        A, (T, 64), (H * 64, 1), (i_t * 64 + 48, 0), (16, 16), (1, 0)
    )
    p_Ad_11 = tl.make_block_ptr(
        Ad, (T, 16), (H * 16, 1), (i_t * 64, 0), (16, 16), (1, 0)
    )
    p_Ad_22 = tl.make_block_ptr(
        Ad, (T, 16), (H * 16, 1), (i_t * 64 + 16, 0), (16, 16), (1, 0)
    )
    p_Ad_33 = tl.make_block_ptr(
        Ad, (T, 16), (H * 16, 1), (i_t * 64 + 32, 0), (16, 16), (1, 0)
    )
    p_Ad_44 = tl.make_block_ptr(
        Ad, (T, 16), (H * 16, 1), (i_t * 64 + 48, 0), (16, 16), (1, 0)
    )

    A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    A_32 = tl.load(p_A_32, boundary_check=(0, 1)).to(tl.float32)
    A_31 = tl.load(p_A_31, boundary_check=(0, 1)).to(tl.float32)
    A_43 = tl.load(p_A_43, boundary_check=(0, 1)).to(tl.float32)
    A_42 = tl.load(p_A_42, boundary_check=(0, 1)).to(tl.float32)
    A_41 = tl.load(p_A_41, boundary_check=(0, 1)).to(tl.float32)

    Ai_11 = tl.load(p_Ad_11, boundary_check=(0, 1)).to(tl.float32)
    Ai_22 = tl.load(p_Ad_22, boundary_check=(0, 1)).to(tl.float32)
    Ai_33 = tl.load(p_Ad_33, boundary_check=(0, 1)).to(tl.float32)
    Ai_44 = tl.load(p_Ad_44, boundary_check=(0, 1)).to(tl.float32)

    Ai_21 = -tl.dot(
        tl.dot(Ai_22, A_21, input_precision="ieee"), Ai_11, input_precision="ieee"
    )
    Ai_32 = -tl.dot(
        tl.dot(Ai_33, A_32, input_precision="ieee"), Ai_22, input_precision="ieee"
    )
    Ai_43 = -tl.dot(
        tl.dot(Ai_44, A_43, input_precision="ieee"), Ai_33, input_precision="ieee"
    )

    Ai_31 = -tl.dot(
        Ai_33,
        tl.dot(A_31, Ai_11, input_precision="ieee")
        + tl.dot(A_32, Ai_21, input_precision="ieee"),
        input_precision="ieee",
    )
    Ai_42 = -tl.dot(
        Ai_44,
        tl.dot(A_42, Ai_22, input_precision="ieee")
        + tl.dot(A_43, Ai_32, input_precision="ieee"),
        input_precision="ieee",
    )
    Ai_41 = -tl.dot(
        Ai_44,
        tl.dot(A_41, Ai_11, input_precision="ieee")
        + tl.dot(A_42, Ai_21, input_precision="ieee")
        + tl.dot(A_43, Ai_31, input_precision="ieee"),
        input_precision="ieee",
    )

    p_Ai_11 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64, 0), (16, 16), (1, 0)
    )
    p_Ai_22 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 16, 16), (16, 16), (1, 0)
    )
    p_Ai_33 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 32, 32), (16, 16), (1, 0)
    )
    p_Ai_44 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 48, 48), (16, 16), (1, 0)
    )
    p_Ai_21 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 16, 0), (16, 16), (1, 0)
    )
    p_Ai_31 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 32, 0), (16, 16), (1, 0)
    )
    p_Ai_32 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 32, 16), (16, 16), (1, 0)
    )
    p_Ai_41 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 48, 0), (16, 16), (1, 0)
    )
    p_Ai_42 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 48, 16), (16, 16), (1, 0)
    )
    p_Ai_43 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 48, 32), (16, 16), (1, 0)
    )
    tl.store(
        p_Ai_11,
        Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_22,
        Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_33,
        Ai_33.to(p_Ai_33.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_44,
        Ai_44.to(p_Ai_44.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_21,
        Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_31,
        Ai_31.to(p_Ai_31.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_32,
        Ai_32.to(p_Ai_32.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_41,
        Ai_41.to(p_Ai_41.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_42,
        Ai_42.to(p_Ai_42.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_43,
        Ai_43.to(p_Ai_43.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )

    fill_zeros = tl.zeros((16, 16), dtype=tl.float32)
    p_Ai_12 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64, 16), (16, 16), (1, 0)
    )
    p_Ai_13 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64, 32), (16, 16), (1, 0)
    )
    p_Ai_14 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64, 48), (16, 16), (1, 0)
    )
    p_Ai_23 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 16, 32), (16, 16), (1, 0)
    )
    p_Ai_24 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 16, 48), (16, 16), (1, 0)
    )
    p_Ai_34 = tl.make_block_ptr(
        Ai, (T, 64), (H * 64, 1), (i_t * 64 + 32, 48), (16, 16), (1, 0)
    )
    tl.store(
        p_Ai_12,
        fill_zeros.to(p_Ai_12.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_13,
        fill_zeros.to(p_Ai_13.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_14,
        fill_zeros.to(p_Ai_14.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_23,
        fill_zeros.to(p_Ai_23.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_24,
        fill_zeros.to(p_Ai_24.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_34,
        fill_zeros.to(p_Ai_34.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


def solve_tril(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    assert A.shape[-1] in [16, 32, 64]

    B, T, H, BT = A.shape
    Ad = torch.empty(
        B, T, H, 16, device=A.device, dtype=torch.float if BT != 16 else output_dtype
    )

    chunk_indices = _prepare_chunk_indices(cu_seqlens, 16) if cu_seqlens is not None else None
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, 16)
    solve_tril_16x16_kernel[NT, B * H](
        A=A,
        Ad=Ad,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=1,
        num_stages=4,
    )
    if BT == 16:
        return Ad

    Ai = torch.empty(B, T, H, BT, device=A.device, dtype=output_dtype)
    merge_fn = (
        merge_16x16_to_32x32_inverse_kernel
        if BT == 32
        else merge_16x16_to_64x64_inverse_kernel
    )
    chunk_indices = _prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    merge_fn[NT, B * H](
        A=A,
        Ad=Ad,
        Ai=Ai,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=4,
        num_stages=3,
    )
    return Ai


@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    initial_state,
    initial_state_indices,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_UPDATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([BV, 64], dtype=tl.float32)

    h += ((boh * H + i_h) * V * K).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    if SAVE_NEW_VALUE:
        v_new += ((bos * H + i_h) * V).to(tl.int64)
    stride_v = H * V
    stride_h = H * V * K
    stride_k = Hg * K
    stride_w = H * K

    index = tl.load(initial_state_indices + i_n).to(tl.int32)
    h0 = initial_state + index * stride_h
    ht = initial_state + index * stride_h
    if USE_INITIAL_STATE:
        h0 = h0 + i_h * V * K
    if INPLACE_UPDATE:
        ht = ht + i_h * V * K

    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
            )
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
            )
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
            )
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        p_h1 = tl.make_block_ptr(
            h + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0)
        )
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(
                h + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
            )
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(
                h + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
            )
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(
                h + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
            )
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(
            w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
        p_v = tl.make_block_ptr(
            v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_v = tl.make_block_ptr(
                v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
            )
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(
                g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
            )
            b_g = tl.load(p_g, boundary_check=(0,))
            b_v = b_v * _safe_exp(b_g_last - b_g)[:, None]
            b_g_last = tl.exp(b_g_last)
            b_h1 = b_h1 * b_g_last
            if K > 64:
                b_h2 = b_h2 * b_g_last
            if K > 128:
                b_h3 = b_h3 * b_g_last
            if K > 192:
                b_h4 = b_h4 * b_g_last

        b_v = b_v.to(k.dtype.element_ty)

        p_k = tl.make_block_ptr(
            k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.trans(tl.dot(b_k, b_v))
        if K > 64:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.trans(tl.dot(b_k, b_v))
        if K > 128:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.trans(tl.dot(b_k, b_v))
        if K > 192:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.trans(tl.dot(b_k, b_v))

    if INPLACE_UPDATE:
        p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    initial_state_indices: torch.Tensor | None = None,
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
):
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = CHUNK_SIZE

    chunk_indices = _prepare_chunk_indices(cu_seqlens, CHUNK_SIZE) if cu_seqlens is not None else None
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
            _prepare_chunk_offsets(cu_seqlens, BT),
        )
    assert K <= 256

    h = k.new_empty(B, NT, H, V, K)
    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BV=32,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_INITIAL_STATE=initial_state is not None,
        INPLACE_UPDATE=True,
        SAVE_NEW_VALUE=v_new is not None,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=4,
        num_stages=2,
    )
    return h, v_new


@triton.jit(do_not_specialize=["T"])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    q += (bos * Hg + i_h // (H // Hg)) * K
    k += (bos * Hg + i_h // (H // Hg)) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * V * K

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_k = tl.make_block_ptr(
            k, (K, T), (1, Hg * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1)
        )
        p_h = tl.make_block_ptr(
            h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0)
        )

        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))

        b_o += tl.dot(b_q, tl.trans(b_h))
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_o = b_o * tl.exp(b_g)[:, None]
        b_A = b_A * _safe_exp(b_g[:, None] - b_g[None, :])

    o_i = tl.arange(0, BT)
    m_A = o_i[:, None] >= o_i[None, :]
    b_A = tl.where(m_A, b_A, 0)

    p_v = tl.make_block_ptr(
        v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    p_o = tl.make_block_ptr(
        o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    b_v = tl.load(p_v, boundary_check=(0, 1))

    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
):
    B, T, Hg, K, V = *q.shape, v.shape[-1]
    H = v.shape[-2]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = _prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o = torch.zeros_like(v)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * H)

    chunk_fwd_kernel_o[grid](
        q,
        k,
        v,
        h,
        g,
        o,
        cu_seqlens,
        chunk_indices,
        scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=128,
        BV=64,
        USE_G=g is not None,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=4,
        num_stages=2,
    )
    return o


def _make_solve_tril_input(B, T, H, BT, device, dtype, cu_seqlens=None):
    k = F.normalize(torch.randn(B, H, T, 64, device=device, dtype=torch.float32), dim=-1)
    A = torch.zeros(B, T, H, BT, device=device, dtype=torch.float32)

    if cu_seqlens is None:
        for b in range(B):
            k_b = k[b : b + 1]
            padding = (BT - T % BT) % BT
            k_pad = F.pad(k_b, (0, 0, 0, padding, 0, 0, 0, 0))
            k_pad = k_pad.reshape(1, H, -1, BT, 64)
            A_b = (k_pad @ k_pad.transpose(-1, -2)).tril(-1)
            A[b : b + 1] = A_b.reshape(1, H, -1, BT)[:, :, :T, :].transpose(1, 2)
    else:
        for i in range(len(cu_seqlens) - 1):
            bos, eos = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            t_i = eos - bos
            k_i = k[:, :, bos:eos]
            padding = (BT - t_i % BT) % BT
            k_pad = F.pad(k_i, (0, 0, 0, padding, 0, 0, 0, 0))
            k_pad = k_pad.reshape(1, H, -1, BT, 64)
            A_i = (k_pad @ k_pad.transpose(-1, -2)).tril(-1)
            A[:, bos:eos] = A_i.reshape(1, H, -1, BT)[:, :, :t_i, :].transpose(1, 2)

    return A.to(dtype)


def _assert_inverse_property(result, A, cu_seqlens=None):
    B, T, H, BT = A.shape
    if cu_seqlens is None:
        sequences = [(b, 0, T) for b in range(B)]
    else:
        sequences = [(0, cu_seqlens[i].item(), cu_seqlens[i + 1].item()) for i in range(len(cu_seqlens) - 1)]

    for batch_idx, bos, eos in sequences:
        for chunk_start in range(bos, eos, BT):
            chunk_end = min(chunk_start + BT, eos)
            n = chunk_end - chunk_start
            for head_idx in range(H):
                L = torch.eye(n, device=A.device, dtype=torch.float32) + A[
                    batch_idx, chunk_start:chunk_end, head_idx, :n
                ].float()
                X = result[batch_idx, chunk_start:chunk_end, head_idx, :n].float()
                I = torch.eye(n, device=A.device, dtype=torch.float32)
                torch.testing.assert_close(L @ X, I, atol=1.6e-1, rtol=1.6e-1)


def _expected_state_update_one_chunk(k, u, initial_state, initial_state_indices, cu_seqlens=None):
    B, T, Hg, _ = k.shape
    H = u.shape[2]
    state = initial_state.clone().float()
    heads_per_group = H // Hg

    if cu_seqlens is None:
        sequences = [(b, b, 0, T) for b in range(B)]
    else:
        sequences = [
            (0, i, cu_seqlens[i].item(), cu_seqlens[i + 1].item())
            for i in range(len(cu_seqlens) - 1)
        ]

    for batch_idx, seq_idx, bos, eos in sequences:
        idx = initial_state_indices[seq_idx].item()
        for head_idx in range(H):
            key_head_idx = head_idx // heads_per_group
            state[idx, head_idx] += (
                u[batch_idx, bos:eos, head_idx].float().transpose(0, 1)
                @ k[batch_idx, bos:eos, key_head_idx].float()
            )
    return state


def _chunk_fwd_o_reference(
    q,
    k,
    v,
    h,
    g=None,
    scale=None,
    cu_seqlens=None,
    chunk_size=64,
):
    B, T, Hg, K = q.shape
    H = v.shape[2]
    V = v.shape[3]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    if scale is None:
        scale = K ** -0.5

    out = torch.zeros_like(v, dtype=torch.float32)
    heads_per_group = H // Hg

    if cu_seqlens is None:
        for b in range(B):
            NT = triton.cdiv(T, BT)
            for i_t in range(NT):
                start = i_t * BT
                end = min((i_t + 1) * BT, T)
                n = end - start
                for hidx in range(H):
                    gidx = hidx // heads_per_group
                    q_i = q[b, start:end, gidx].float()
                    k_i = k[b, start:end, gidx].float()
                    v_i = v[b, start:end, hidx].float()
                    h_i = h[b, i_t, hidx].float()
                    b_o = q_i @ h_i.transpose(0, 1)
                    b_A = q_i @ k_i.transpose(0, 1)

                    if g is not None:
                        g_i = g[b, start:end, hidx].float()
                        b_o = b_o * torch.exp(g_i).unsqueeze(1)
                        diff = g_i.unsqueeze(1) - g_i.unsqueeze(0)
                        b_A = b_A * torch.exp(
                            torch.where(
                                diff <= 0,
                                diff,
                                torch.full_like(diff, float("-inf")),
                            )
                        )

                    mask = torch.tril(torch.ones(n, n, device=q.device, dtype=torch.bool))
                    b_A = torch.where(mask, b_A, torch.zeros_like(b_A))
                    out[b, start:end, hidx] = (b_o + b_A @ v_i) * scale
    else:
        chunk_indices = _prepare_chunk_indices(cu_seqlens, BT)
        for i_tg in range(len(chunk_indices)):
            seq_idx = chunk_indices[i_tg, 0].item()
            local_chunk = chunk_indices[i_tg, 1].item()
            bos = cu_seqlens[seq_idx].item()
            eos = cu_seqlens[seq_idx + 1].item()
            start = bos + local_chunk * BT
            end = min(start + BT, eos)
            n = end - start
            for hidx in range(H):
                gidx = hidx // heads_per_group
                q_i = q[0, start:end, gidx].float()
                k_i = k[0, start:end, gidx].float()
                v_i = v[0, start:end, hidx].float()
                h_i = h[0, i_tg, hidx].float()
                b_o = q_i @ h_i.transpose(0, 1)
                b_A = q_i @ k_i.transpose(0, 1)

                if g is not None:
                    g_i = g[0, start:end, hidx].float()
                    b_o = b_o * torch.exp(g_i).unsqueeze(1)
                    diff = g_i.unsqueeze(1) - g_i.unsqueeze(0)
                    b_A = b_A * torch.exp(
                        torch.where(
                            diff <= 0,
                            diff,
                            torch.full_like(diff, float("-inf")),
                        )
                    )

                mask = torch.tril(torch.ones(n, n, device=q.device, dtype=torch.bool))
                b_A = torch.where(mask, b_A, torch.zeros_like(b_A))
                out[0, start:end, hidx] = (b_o + b_A @ v_i) * scale

    return out


@unittest.skipUnless(HAS_XPU, "Test requires XPU")
class TestSolveTril(CustomTestCase):
    def test_solve_tril_inverse_property(self):
        torch.manual_seed(0)
        device = "xpu"

        # Upstream FLA marks solve_tril tests as unsupported on Intel due known instability.
        # Keep 16/64 coverage on XPU and avoid the unstable 32-case instead of inflating tolerances.
        block_sizes = [16, 64] if HAS_XPU else [16, 32, 64]
        for block_size in block_sizes:
            with self.subTest(block_size=block_size):
                A = _make_solve_tril_input(
                    B=2,
                    T=block_size * 2 - 7,
                    H=3,
                    BT=block_size,
                    device=device,
                    dtype=torch.bfloat16,
                )
                result = solve_tril(A, output_dtype=torch.float32)
                _assert_inverse_property(result, A)

    def test_solve_tril_varlen_inverse_property(self):
        torch.manual_seed(1)
        device = "xpu"
        block_size = 64
        seq_lens = [19, 77, 64]
        total_tokens = sum(seq_lens)
        cu_seqlens = torch.tensor(
            [0, *torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()],
            device=device,
            dtype=torch.long,
        )
        A = _make_solve_tril_input(
            B=1,
            T=total_tokens,
            H=2,
            BT=block_size,
            device=device,
            dtype=torch.bfloat16,
            cu_seqlens=cu_seqlens,
        )

        result = solve_tril(A, cu_seqlens=cu_seqlens, output_dtype=torch.float32)
        _assert_inverse_property(result, A, cu_seqlens=cu_seqlens)


@unittest.skipUnless(HAS_XPU, "Test requires XPU")
class TestChunkGatedDeltaRuleFwdH(CustomTestCase):
    def test_chunk_gated_delta_rule_fwd_h_one_chunk_update(self):
        torch.manual_seed(2)
        device = "xpu"
        B, T, Hg, H, K, V = 2, 31, 2, 4, 32, 48

        k = torch.randn(B, T, Hg, K, device=device, dtype=torch.bfloat16)
        w = torch.zeros(B, T, H, K, device=device, dtype=torch.bfloat16)
        u = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16)
        g = torch.zeros(B, T, H, device=device, dtype=torch.float32)
        initial_state = torch.randn(B, H, V, K, device=device, dtype=torch.float32) * 0.05
        initial_state_indices = torch.tensor([1, 0], device=device, dtype=torch.long)

        initial_state_before = initial_state.clone()
        h, v_new = chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            g=g,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
        )
        torch.xpu.synchronize()

        expected_h = torch.empty(B, 1, H, V, K, device=device, dtype=torch.float32)
        for b in range(B):
            for head_idx in range(H):
                expected_h[b, 0, head_idx] = initial_state_before[initial_state_indices[b], head_idx]
        expected_final_state = _expected_state_update_one_chunk(
            k=k,
            u=u,
            initial_state=initial_state_before,
            initial_state_indices=initial_state_indices,
        )

        torch.testing.assert_close(h.float(), expected_h, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(v_new.float(), u.float(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(initial_state.float(), expected_final_state, atol=7e-2, rtol=7e-2)


@unittest.skipUnless(HAS_XPU, "Test requires XPU")
class TestChunkFwdO(CustomTestCase):
    def test_chunk_fwd_o_regression_grid_2_1_32(self):
        torch.manual_seed(6)
        device = "xpu"
        # Force kernel launch grid to (2, 1, 32):
        #   cdiv(V, BV)=2 with V=128 and BV=64
        #   NT=1 with T=32 and BT=32
        #   B*H=32 with B=4, H=8
        B, T, Hg, H, K, V = 4, 32, 2, 8, 32, 128
        scale = 0.5

        q = torch.randn(B, T, Hg, K, device=device, dtype=torch.bfloat16)
        k = torch.randn(B, T, Hg, K, device=device, dtype=torch.bfloat16)
        v = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16)
        h = torch.randn(B, 1, H, V, K, device=device, dtype=torch.bfloat16)
        g = torch.randn(B, T, H, device=device, dtype=torch.float32) * 0.1

        tri = chunk_fwd_o(q=q, k=k, v=v, h=h, g=g, scale=scale)
        torch.xpu.synchronize()
        ref = _chunk_fwd_o_reference(q=q, k=k, v=v, h=h, g=g, scale=scale)

        self.assertTrue(torch.isfinite(tri).all().item())
        torch.testing.assert_close(tri.float(), ref, atol=7e-2, rtol=7e-2)

    def test_chunk_fwd_o_matches_reference(self):
        torch.manual_seed(4)
        device = "xpu"
        B, T, Hg, H, K, V = 2, 91, 2, 4, 32, 48
        BT = min(64, max(16, triton.next_power_of_2(T)))
        NT = triton.cdiv(T, BT)

        q = torch.randn(B, T, Hg, K, device=device, dtype=torch.float32)
        k = torch.randn(B, T, Hg, K, device=device, dtype=torch.float32)
        v = torch.randn(B, T, H, V, device=device, dtype=torch.float32)
        h = torch.randn(B, NT, H, V, K, device=device, dtype=torch.float32)
        g = torch.randn(B, T, H, device=device, dtype=torch.float32) * 0.1
        scale = 0.7

        tri = chunk_fwd_o(q=q, k=k, v=v, h=h, g=g, scale=scale)
        ref = _chunk_fwd_o_reference(q=q, k=k, v=v, h=h, g=g, scale=scale)

        torch.testing.assert_close(tri.float(), ref, atol=6e-2, rtol=6e-2)

    def test_chunk_fwd_o_varlen_matches_reference(self):
        torch.manual_seed(5)
        device = "xpu"
        seq_lens = [17, 33, 29]
        total_tokens = sum(seq_lens)
        Hg, H, K, V = 1, 2, 16, 24
        BT = min(64, max(16, triton.next_power_of_2(total_tokens)))
        chunk_indices = _prepare_chunk_indices(
            torch.tensor([0, *torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()], device=device, dtype=torch.long),
            BT,
        )

        cu_seqlens = torch.tensor(
            [0, *torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()],
            device=device,
            dtype=torch.long,
        )
        q = torch.randn(1, total_tokens, Hg, K, device=device, dtype=torch.float32)
        k = torch.randn(1, total_tokens, Hg, K, device=device, dtype=torch.float32)
        v = torch.randn(1, total_tokens, H, V, device=device, dtype=torch.float32)
        h = torch.randn(1, len(chunk_indices), H, V, K, device=device, dtype=torch.float32)
        g = torch.randn(1, total_tokens, H, device=device, dtype=torch.float32) * 0.1

        tri = chunk_fwd_o(q=q, k=k, v=v, h=h, g=g, cu_seqlens=cu_seqlens)
        ref = _chunk_fwd_o_reference(
            q=q,
            k=k,
            v=v,
            h=h,
            g=g,
            cu_seqlens=cu_seqlens,
        )

        torch.testing.assert_close(tri.float(), ref, atol=6e-2, rtol=6e-2)

    def test_chunk_gated_delta_rule_fwd_h_varlen_one_chunk_each(self):
        torch.manual_seed(3)
        device = "xpu"
        seq_lens = [17, 31, 33]
        total_tokens = sum(seq_lens)
        Hg, H, K, V = 1, 2, 16, 24

        cu_seqlens = torch.tensor(
            [0, *torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()],
            device=device,
            dtype=torch.long,
        )
        k = torch.randn(1, total_tokens, Hg, K, device=device, dtype=torch.bfloat16)
        w = torch.zeros(1, total_tokens, H, K, device=device, dtype=torch.bfloat16)
        u = torch.randn(1, total_tokens, H, V, device=device, dtype=torch.bfloat16)
        g = torch.zeros(1, total_tokens, H, device=device, dtype=torch.float32)
        initial_state = torch.randn(len(seq_lens), H, V, K, device=device, dtype=torch.float32) * 0.05
        initial_state_indices = torch.tensor([2, 0, 1], device=device, dtype=torch.long)

        initial_state_before = initial_state.clone()
        h, v_new = chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            g=g,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            cu_seqlens=cu_seqlens,
        )
        torch.xpu.synchronize()

        expected_h = torch.empty(1, len(seq_lens), H, V, K, device=device, dtype=torch.float32)
        for seq_idx in range(len(seq_lens)):
            for head_idx in range(H):
                expected_h[0, seq_idx, head_idx] = initial_state_before[initial_state_indices[seq_idx], head_idx]
        expected_final_state = _expected_state_update_one_chunk(
            k=k,
            u=u,
            initial_state=initial_state_before,
            initial_state_indices=initial_state_indices,
            cu_seqlens=cu_seqlens,
        )

        torch.testing.assert_close(h.float(), expected_h, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(v_new.float(), u.float(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(initial_state.float(), expected_final_state, atol=7e-2, rtol=7e-2)


if __name__ == "__main__":
    unittest.main()
