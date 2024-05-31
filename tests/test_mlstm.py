import unittest
import mlx.core as mx
import mlx.nn as nn

from mlx_xlstm import mLSTMBlock

class TestMLSTM(unittest.TestCase):
    def setUp(self):        
        self.inp_dim = 10
        self.head_dim = 8
        self.head_num = 4
        self.hid_dim = self.head_num * self.head_dim
        
        self.batch_size = 5
        
        self.model = mLSTMBlock(self.inp_dim, self.head_num, self.head_dim)
        self.input = mx.random.normal((self.batch_size, self.inp_dim))
        
        self.hid_0 = self.model.init_hidden(self.batch_size)
    
    def test_forward(self):
        output, next_hid = self.model(self.input, self.hid_0)

        self.assertEqual(output.shape, (self.batch_size, self.inp_dim))

        self.assertEqual(next_hid[0].shape, (self.batch_size, self.head_num, self.head_dim, self.head_dim))
        self.assertEqual(next_hid[1].shape, (self.batch_size, self.head_num, self.head_dim))
        self.assertEqual(next_hid[2].shape, (self.batch_size, self.head_num))

    def test_backward(self):
        mx.eval(self.model.parameters())

        def loss_fn(model, X, states, y):
            return nn.losses.mse_loss(model(X, states)[0], y)

        loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)

        target = mx.random.normal((self.batch_size, self.inp_dim))

        l, grads = loss_and_grad_fn(self.model, self.input, self.hid_0, target)

        for grad in grads:
            self.assertIsNotNone(grad)

if __name__ == '__main__':
    unittest.main()