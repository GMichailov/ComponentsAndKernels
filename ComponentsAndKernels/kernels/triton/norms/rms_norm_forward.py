import triton
import triton.language as tl

@triton.jit
def rms_norm_fwd_bias_train(
    X, Y, W, B, eps, RMS, DIM_X, BLOCK_SIZE: tl.constexpr
):
    """Assumes contiguity."""
    row = tl.program_id(0)
    X += row * DIM_X
    Y += row * DIM_X
    # Compute mean
    _buf = tl.zeros([BLOCK_SIZE], dtype=tl.bfloat16)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols, other=0.).to(tl.bfloat16)
        _buf += x * x
    inv_rms = tl.sqrt(tl.sum(_buf, axis=0) + eps)
    tl.store(RMS + row, inv_rms)
    inv_rms = 1 / inv_rms
    # Perform the calculation
    for off in range(0, DIM_X, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < DIM_X
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.bfloat16)
        y = (x * inv_rms) * w + b
        tl.store(Y + cols, y, mask=mask)


@triton.jit
def rms_norm_fwd_nobias_train(
    X, Y, W, eps, RMS, DIM_X, BLOCK_SIZE: tl.constexpr
):
    """Assumes contiguity."""
    row = tl.program_id(0)
    X += row * DIM_X
    Y += row * DIM_X
    # Compute mean
    _buf = tl.zeros([BLOCK_SIZE], dtype=tl.bfloat16)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols, other=0.).to(tl.bfloat16)
        _buf += x * x
    inv_rms = tl.sqrt(tl.sum(_buf, axis=0) + eps)
    tl.store(RMS + row, inv_rms)
    inv_rms = 1 / inv_rms
    # Perform the calculation
    for off in range(0, DIM_X, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < DIM_X
        w = tl.load(W + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.bfloat16)
        y = (x * rms) * w
        tl.store(Y + cols, y, mask=mask)


@triton.jit
def rms_norm_fwd_bias_inf(
    X, Y, W, B, eps, DIM_X, BLOCK_SIZE: tl.constexpr
):
    """Assumes contiguity."""
    row = tl.program_id(0)
    X += row * DIM_X
    Y += row * DIM_X
    # Compute mean
    _buf = tl.zeros([BLOCK_SIZE], dtype=tl.bfloat16)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols, other=0.).to(tl.bfloat16)
        _buf += x * x
    inv_rms = 1 / tl.sqrt(tl.sum(_buf, axis=0) + eps)
    # Perform the calculation
    for off in range(0, DIM_X, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < DIM_X
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.bfloat16)
        y = (x * inv_rms) * w + b
        tl.store(Y + cols, y, mask=mask)

@triton.jit
def rms_norm_fwd_nobias_inf(
    X, Y, W, eps, DIM_X, BLOCK_SIZE: tl.constexpr
):
    """Assumes contiguity."""
    row = tl.program_id(0)
    X += row * DIM_X
    Y += row * DIM_X
    # Compute mean
    _buf = tl.zeros([BLOCK_SIZE], dtype=tl.bfloat16)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols, other=0.).to(tl.bfloat16)
        _buf += x * x
    inv_rms = 1 / tl.sqrt(tl.sum(_buf, axis=0) + eps)
    # Perform the calculation
    for off in range(0, DIM_X, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < DIM_X
        w = tl.load(W + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.bfloat16)
        y = (x * inv_rms) * w
        tl.store(Y + cols, y, mask=mask)
