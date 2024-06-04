import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx_xlstm import mLSTMBlock

import matplotlib.pyplot as plt

def generate_sine_wave(seq_len, num_sequences, batch_size):
    x = mx.linspace(0, 4*3.14, seq_len)
    y = mx.sin(x)
    
    res = mx.array(y, dtype=mx.float32).reshape(-1, 1)
    res = mx.tile(res, (batch_size, 1, num_sequences))
    return res

def loss_fn(model, X, hid, y):
    global states
    pred, states = model(X, hid)
    return nn.losses.mse_loss(pred, y)

input_size = 1
head_dim = 4
head_num = 8
seq_len = 500
num_sequences = 1

batch_size = 1

model = mLSTMBlock(input_size, head_dim, head_num)
mx.eval(model.parameters())

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

optimizer = optim.Adam(learning_rate=0.01)

data = generate_sine_wave(seq_len, num_sequences, batch_size)

states = model.init_hidden(batch_size)
for epoch in range(30):
    loss = 0
    for t in range(seq_len-1):
        X = data[:, t, :]
        y = data[:, t+1, :]
        l, grads = loss_and_grad_fn(model, X, states, y)
        loss += l

        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

    if epoch % 10 == 0:
        print(f'Epoch: {epoch:>3}, Loss: {loss.item()}')

test_output = []
states = model.init_hidden(batch_size)
for t in range(seq_len-1):
    x = data[:, t, :]
    y_pred, states = model(x, states)
    test_output.append(y_pred.squeeze(0).flatten().tolist()[0])

plt.figure(figsize=(10, 4))
plt.title('Learning: y = sin(x), [0, 4π]')
plt.plot(data.squeeze(0).flatten().tolist(), label='Function')
plt.plot(test_output, label='mLSTMBlock Prediction')
plt.legend()
plt.show()