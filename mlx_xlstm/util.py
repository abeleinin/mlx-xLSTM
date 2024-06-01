import mlx.core as mx
import mlx.nn as nn

def init_orthogonal(shape, gain=1.0):
    a = mx.random.normal(shape)
    q, r = mx.linalg.qr(a, stream=mx.cpu)
    if q.shape != shape:
        q = q.T
    q *= gain
    return q

def enlarge_as(src, other):
    new_dims = other.ndim - src.ndim
    if new_dims > 0:
        src = src.reshape(list(src.shape) + [1] * new_dims)
    return src

def clamp(x, min_value=None, max_value=None):
    if min_value is not None:
        x = mx.maximum(x, mx.array(min_value, dtype=x.dtype))
    if max_value is not None:
        x = mx.minimum(x, mx.array(max_value, dtype=x.dtype))
    return x

def repeat(value, times=None):
    if times is None:
        while True:
            yield value
    else:
        for _ in range(times):
            yield value

class CausalConv1d(nn.Conv1d):
    def __init__(
        self, 
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
    ):
        self._padding = (kernel_size - 1) * dilation
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=self._padding,
            dilation=dilation,
        )

    def __call__(self, x):
        x = super().__call__(x)
        return x[:, :-self._padding, ...] 

def block_diag(*matrices):
    rows = sum(mat.shape[0] for mat in matrices)
    cols = sum(mat.shape[1] for mat in matrices)

    result = mx.zeros((rows, cols))

    current_row = 0
    current_col = 0
    for mat in matrices:
        r, c = mat.shape
        result[current_row:current_row + r, current_col:current_col + c] = mat
        current_row += r
        current_col += c

    return result

class BlockLinear(nn.Module):
    def __init__(self, block_dims, bias=False):
        super().__init__()

        self._blocks = [
            mx.random.normal(size)
            for size in block_dims
        ]

        self._bias = mx.zeros(mx.sum(block_dims)) if bias else None
    
    def __call__(self, x):
        full = block_diag(*self._blocks)

        out = mx.matmul(full, x)

        if self._bias is not None:
            out += self._bias
        
        return out