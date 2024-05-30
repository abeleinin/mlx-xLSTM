import mlx.core as mx
import mlx.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Xavier Uniform
        # k = mx.sqrt(0.6 / (input_size + hidden_size))
        
        # PyTorch LSTMCell U(sqrt(-k), sqrt(k)), k=1/hidden_size
        k = mx.sqrt(1 / hidden_size)

        self.weight_ih = mx.random.uniform(-k, k, shape=(4*hidden_size, input_size))
        self.weight_hh = mx.random.uniform(-k, k, shape=(4*hidden_size, hidden_size))
        self.bias_ih = mx.random.uniform(-k, k, shape=(4*hidden_size,))
        self.bias_hh = mx.random.uniform(-k, k, shape=(4*hidden_size,))

    def __call__(self, input_t, hidden_state):
        h_prev, c_prev = hidden_state

        gates = (
            mx.matmul(input_t, self.weight_ih.T) + self.bias_ih +
            mx.matmul(h_prev, self.weight_hh.T) + self.bias_hh
        )
        
        i_gate, f_gate, g_gate, o_gate = mx.split(gates, 4, axis=1)
        # v = mx.split(gates, self.hidden_size, axis=1)
        
        i_t = mx.sigmoid(i_gate)
        f_t = mx.sigmoid(f_gate)
        g_t = mx.tanh(g_gate)
        o_t = mx.sigmoid(o_gate)
        
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * mx.tanh(c_t)
        
        return h_t, (h_t, c_t)

if __name__ == '__main__':
    # Example usage:
    input_size = 10
    hidden_size = 20
    rrn = LSTMCell(input_size, hidden_size)
    x = mx.random.uniform(0, 1, shape=(1, input_size))
    h_prev = mx.random.uniform(0, 1, shape=(1, hidden_size))
    c_prev = mx.random.uniform(0, 1, shape=(1, hidden_size))
    h_t, (h_t, c_t) = rrn(x, (h_prev, c_prev))
    print("Hidden state:", h_t)
    print("Cell state:", c_t)