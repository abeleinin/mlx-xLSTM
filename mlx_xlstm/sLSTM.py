import mlx.core as mx
import mlx.nn as nn

from .util import init_orthogonal, unsqueeze

class sLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # initialize xavier uniform
        k = mx.sqrt(0.6 / (self.input_size + self.hidden_size))
        self.w_i = mx.random.uniform(-k, k, shape=(hidden_size, input_size))
        self.w_f = mx.random.uniform(-k, k, shape=(hidden_size, input_size))
        self.w_o = mx.random.uniform(-k, k, shape=(hidden_size, input_size))
        self.w_z = mx.random.uniform(-k, k, shape=(hidden_size, input_size))

        # initialize orthogonal
        self.r_i = init_orthogonal((hidden_size, hidden_size))
        self.r_f = init_orthogonal((hidden_size, hidden_size))
        self.r_o = init_orthogonal((hidden_size, hidden_size))
        self.r_z = init_orthogonal((hidden_size, hidden_size))

        # initialize zeros
        self.b_i = mx.zeros((hidden_size))
        self.b_f = mx.zeros((hidden_size))
        self.b_o = mx.zeros((hidden_size))
        self.b_z = mx.zeros((hidden_size))

        self.sigmoid = nn.Sigmoid()
    
    def __call__(self, x, states):
        h_prev, c_prev, n_prev, m_prev = states

        h_prev = h_prev.T
        x = x.T

        i_tilda = (
            mx.matmul(self.w_i, x)
            + mx.matmul(self.r_i, h_prev)
            + self.b_i
        )
        f_tilda = (
            mx.matmul(self.w_f, x)
            + mx.matmul(self.r_f, h_prev)
            + self.b_f
        )
        o_tilda = (
            mx.matmul(self.w_o, x)
            + mx.matmul(self.r_o, h_prev)
            + self.b_o
        )
        z_tilda = (
            mx.matmul(self.w_z, x)
            + mx.matmul(self.r_z, h_prev)
            + self.b_z
        )

        i_t = mx.exp(i_tilda)
        # choose either sigmoid or exp based on context
        f_t = self.sigmoid(
            f_tilda
        )

        t_1 = mx.max(mx.log(f_t) + m_prev)
        t_2 = mx.max(mx.log(i_t))
        if t_1 > t_2:
            m_t = t_1
        else:
            m_t = t_2
        
        i_prime = mx.exp(mx.log(i_t) - m_t)
        f_prime = mx.exp(mx.log(f_t) + m_prev - m_t)

        c_t = f_prime * c_prev + i_prime * mx.tanh(z_tilda)
        n_t = f_prime * n_prev + i_prime

        c_hat = c_t / n_t
        h_t = self.sigmoid(o_tilda) * c_hat

        return h_t, (h_t, c_t, n_t, m_t)

class sLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.layers = [
            sLSTMCell(
                input_size if i == 0 else hidden_size, hidden_size
            )
            for i in range(num_layers)
        ]

    def __call__(self, x, initial_states=None):
        batch_size, seq_len, _ = x.shape
        if initial_states is None:
            initial_states = self.init_hidden(batch_size)
        
        outputs = []
        current_states = initial_states

        for t in range(seq_len):
            x_t = x[:, t, :]
            new_states = []
            for layer, state in zip(self.layers, current_states):
                h_t, new_state = layer(x_t, state)
                new_states.append(new_state)
                x_t = h_t
            outputs.append(unsqueeze(h_t, 1))
            current_states = new_states
        
        outputs = mx.concatenate(outputs, axis=1)
        return outputs, current_states
    
    def init_hidden(self, batch_size):
        initial_states = [
            (
                mx.zeros((batch_size, self.layers[0].hidden_size)),
                mx.zeros((batch_size, self.layers[0].hidden_size)),
                mx.zeros((batch_size, self.layers[0].hidden_size)),
                mx.zeros((batch_size, self.layers[0].hidden_size)),
            )
            for _ in self.layers
        ]
        return initial_states
 