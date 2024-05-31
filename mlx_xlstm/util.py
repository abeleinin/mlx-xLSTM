import mlx.core as mx
import mlx.nn as nn

def init_orthogonal(shape, gain=1.0):
    a = mx.random.normal(shape)
    q, r = mx.linalg.qr(a, stream=mx.cpu)
    if q.shape != shape:
        q = q.T
    q *= gain
    return q

def unsqueeze(arr, dim):
    shape = list(arr.shape)
    if dim < 0:
        shape.insert(len(shape)+dim+1, 1)
    else:
        shape.insert(dim, 1)
    return arr.reshape(shape)

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
        return x[:, :-self._padding, :]