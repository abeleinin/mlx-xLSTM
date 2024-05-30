import mlx.core as mx
import mlx.nn as nn

class mLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.W_q = nn.Linear(input_size, hidden_size)
        self.W_k = nn.Linear(input_size, num_layers)
        self.W_v = nn.Linear(input_size, num_layers)

        self.input_gates = nn.Linear(input_size, 1)
        self.forget_gates = nn.Linear(input_size, 1)
        self.output_gates = nn.Linear(input_size, 1)

        self.reset_parameters()
    
    def reset_parameters(self):
        k = mx.sqrt(0.6 / (self.input_size + self.hidden_size))
        
        self.W_q.weight = mx.random.uniform(-k, k, shape=self.W_q.weight.shape)
        self.W_k.weight = mx.random.uniform(-k, k, shape=self.W_k.weight.shape)
        self.W_v.weight = mx.random.uniform(-k, k, shape=self.W_v.weight.shape)
        self.W_q.bias = mx.zeros((self.input_size, 1))
        self.W_k.bias = mx.zeros((self.input_size, 1))
        self.W_v.bias = mx.zeros((self.input_size, 1))

        for gate in [self.input_gates, self.forget_gates, self.output_gates]:
            gate.weight = mx.random.uniform(-k, k, shape=gate.weight.shape)
            gate.bias = mx.zeros(gate.bias.shape)

    def __call__(self, x, hidden_state=None):
        if hidden_state is None:
            hidden_state = self.init_hidden()
        
        C_prev, n_prev = hidden_state
        qt = mx.matmul(self.W_q.weight, x) + self.W_q.bias
        kt = (1 / mx.sqrt(self.num_layers)) * (mx.matmul(self.W_k.weight, x) + self.W_k.bias.T)
        vt = mx.matmul(self.W_v.weight, x) + self.W_v.bias.T

        it = mx.exp(mx.matmul(self.input_gates.weight, x) + self.input_gates.bias)
        ft = mx.sigmoid(mx.matmul(self.forget_gates.weight, x) + self.forget_gates.bias)

        vt = mx.squeeze(vt)
        kt = mx.squeeze(kt)

        C = ft * C_prev + it * mx.outer(vt, kt)
        n = ft * n_prev + it * kt.reshape((kt.shape[0], 1))

        max_nqt = mx.abs(mx.matmul(n.T, qt)).max()
        max_nqt = 1.0 if 1.0 > max_nqt else max_nqt
        # if 1.0 > max_nqt:
        #     max_nqt = 1.0

        h_tilde = mx.matmul(C, qt) / max_nqt
        ot = mx.sigmoid(mx.matmul(self.output_gates.weight, x) + self.output_gates.bias)
        ht = ot * h_tilde

        return ht, (C, n)        

    def init_hidden(self):
        C = mx.zeros((self.num_layers, self.num_layers))
        h = mx.zeros((self.num_layers, 1))
        return C, h
