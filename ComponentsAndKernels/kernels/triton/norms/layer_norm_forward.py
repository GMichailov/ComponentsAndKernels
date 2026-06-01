import torch
import triton
import triton.language as tl

@triton.jit
def layer_norm_fwd_bias_train(
    X, Y, W, B, eps, M, INV_STD, DIM_X, BLOCK_SIZE: tl.constexpr
):
    """Assumes contiguity."""
    row = tl.program_id(0)
    X += row * DIM_X
    Y += row * DIM_X
    # Compute mean
    mean = 0
    _buf = tl.zeros([BLOCK_SIZE], dtype=tl.bfloat16)
    for off in range(0, DIM_X, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < DIM_X, other=0.).to(tl.bfloat16)
        _buf += a
    mean = tl.sum(_buf, axis=0) / DIM_X
    # Compute variance
    _buf = tl.zeros([BLOCK_SIZE], dtype=tl.bfloat16)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols, other=0.).to(tl.bfloat16)
        x = tl.where(cols < N, x - mean, 0.)
        _buf += x * x
    var = tl.sum(_var, axis=0) / DIM_X
    inv_std = 1 / tl.sqrt(var + eps)
    tl.store(M + row, mean)
    tl.store(INV_STD + row, inv_std)
    # Perform the calculation
    for off in range(0, DIM_X, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < DIM_X
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.bfloat16)
        x_hat = (x - mean) * inv_std
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)

@triton.jit
def layer_norm_fwd_nobias_train(
    X, Y, W, eps, M, INV_STD, DIM_X, BLOCK_SIZE: tl.constexpr
):
    """Assumes contiguity."""
    row = tl.program_id(0)
    X += row * DIM_X
    Y += row * DIM_X
    # Compute mean
    mean = 0
    _buf = tl.zeros([BLOCK_SIZE], dtype=tl.bfloat16)
    for off in range(0, DIM_X, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < DIM_X, other=0.).to(tl.bfloat16)
        _buf += a
    mean = tl.sum(_buf, axis=0) / DIM_X
    # Compute variance
    _buf = tl.zeros([BLOCK_SIZE], dtype=tl.bfloat16)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols, other=0.).to(tl.bfloat16)
        x = tl.where(cols < N, x - mean, 0.)
        _buf += x * x
    var = tl.sum(_var, axis=0) / DIM_X
    inv_std = 1 / tl.sqrt(var + eps)
    tl.store(M + row, mean)
    tl.store(INV_STD + row, inv_std)
    # Perform the calculation
    for off in range(0, DIM_X, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < DIM_X
        w = tl.load(W + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.bfloat16)
        x_hat = (x - mean) * inv_std
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)

@triton.jit
def layer_norm_fwd_bias_inf(
    X, Y, W, B, eps, DIM_X, BLOCK_SIZE: tl.constexpr
):
    """Assumes contiguity."""
    row = tl.program_id(0)
    X += row * DIM_X
    Y += row * DIM_X
    # Compute mean
    mean = 0
    _buf = tl.zeros([BLOCK_SIZE], dtype=tl.bfloat16)
    for off in range(0, DIM_X, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < DIM_X, other=0.).to(tl.bfloat16)
        _buf += a
    mean = tl.sum(_buf, axis=0) / DIM_X
    # Compute variance
    _buf = tl.zeros([BLOCK_SIZE], dtype=tl.bfloat16)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols, other=0.).to(tl.bfloat16)
        x = tl.where(cols < N, x - mean, 0.)
        _buf += x * x
    var = tl.sum(_var, axis=0) / DIM_X
    inv_std = 1 / tl.sqrt(var + eps)
    # Perform the calculation
    for off in range(0, DIM_X, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < DIM_X
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.bfloat16)
        x_hat = (x - mean) * inv_std
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)

@triton.jit
def layer_norm_fwd_nobias_inf(
    X, Y, W, B, eps, DIM_X, BLOCK_SIZE: tl.constexpr
):
    """Assumes contiguity."""
    row = tl.program_id(0)
    X += row * DIM_X
    Y += row * DIM_X
    # Compute mean
    mean = 0
    _buf = tl.zeros([BLOCK_SIZE], dtype=tl.bfloat16)
    for off in range(0, DIM_X, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < DIM_X, other=0.).to(tl.bfloat16)
        _buf += a
    mean = tl.sum(_buf, axis=0) / DIM_X
    # Compute variance
    _buf = tl.zeros([BLOCK_SIZE], dtype=tl.bfloat16)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols, other=0.).to(tl.bfloat16)
        x = tl.where(cols < N, x - mean, 0.)
        _buf += x * x
    var = tl.sum(_var, axis=0) / DIM_X
    inv_std = 1 / tl.sqrt(var + eps)
    # Perform the calculation
    for off in range(0, DIM_X, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < DIM_X
        w = tl.load(W + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.bfloat16)
        x_hat = (x - mean) * inv_std
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)