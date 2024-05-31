import mlx.core as mx

def init_orthogonal(shape, gain=1.0):
    a = mx.random.normal(shape)
    q, r = mx.linalg.qr(a, stream=mx.cpu)
    if q.shape != shape:
        q = q.T
    q *= gain
    return q

def unsqueeze(arr, dim):
    shape = list(arr.shape)
    shape.insert(dim, 1)
    return arr.reshape(shape)