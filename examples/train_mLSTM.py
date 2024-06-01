import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx_xlstm import mLSTM

import matplotlib.pyplot as plt

def generate_sine_wave(seq_len, num_sequences):
    x = mx.linspace(0, 2*3.14, seq_len)
    y = mx.sin(x)
    
    res = mx.array(y, dtype=mx.float32).reshape(-1, 1)
    res = mx.tile(res, (1, 1, num_sequences))
    return res

def loss_fn(model, X, states, y):
    return nn.losses.mse_loss(model(X, states)[0], y)

input_size = 1
hidden_size = 10
num_layers = 10
seq_len = 100
num_sequences = 1

model = mLSTM(input_size, hidden_size, num_layers)
mx.eval(model.parameters())

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

optimizer = optim.Adam(learning_rate=0.01)

data = generate_sine_wave(seq_len, num_sequences)

for epoch in range(250):
    states = model.init_hidden()
    loss = 0
    for t in range(seq_len - 1):
        x = data[:, t]
        y_true = mx.broadcast_to(data[:, t+1], (hidden_size, input_size))
        l, grads = loss_and_grad_fn(model, x, states, y_true)
        loss += l

        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

    if epoch % 10 == 0:
        print(f'Epoch: {epoch:>3}, Loss: {loss.item()}')

test_output = []
states = model.init_hidden()
for t in range(seq_len-1):
    x = data[:, t]
    y_pred, states = model(x, states)
    test_output.append(y_pred.flatten().tolist()[0])

plt.figure(figsize=(10, 4))
plt.title('Learning: y = sin(x), [0, 2π]')
plt.plot(data.flatten().tolist(), label='Function')
plt.plot(test_output, label='mLSTM Prediction')
plt.legend()
plt.show()