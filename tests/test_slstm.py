import unittest
import mlx.core as mx
import mlx.nn as nn

from mlx_xlstm import sLSTMBlock

class TestSLSTM(unittest.TestCase):
    def setUp(self):        
        self.inp_dim = 10
        self.head_dim = 8
        self.head_num = 4
        self.hid_dim = self.head_num * self.head_dim
        
        self.batch_size = 5
        
        self.model = sLSTMBlock(self.inp_dim, self.head_dim, self.head_num)
        self.input = mx.random.normal((self.batch_size, self.inp_dim))
        
        self.hid_0 = self.model.init_hidden()

    def test_output_shape(self):
        output, _ = self.model(self.input, self.hid_0)
        
        self.assertEqual(output.shape, (self.batch_size, self.inp_dim))

    def test_hidden_shape(self):
        hid = self.model.init_hidden()
        self.assertEqual(len(hid), 4) 
        
        self.assertEqual(hid[0].shape, (self.hid_dim,))
        self.assertEqual(hid[1].shape, (self.hid_dim,))
        self.assertEqual(hid[2].shape, (self.hid_dim,))
        self.assertEqual(hid[3].shape, (self.hid_dim,))

    def test_forward_no_conv(self):
        output, _ = self.model(self.input, self.hid_0)
        self.assertEqual(output.shape, (self.batch_size, self.inp_dim))
        
    def test_forward_with_conv(self):
        output, _ = self.model(self.input, self.hid_0, use_conv=True)
        self.assertEqual(output.shape, (self.batch_size, self.inp_dim))

    def test_backward(self):
        mx.eval(self.model.parameters())

        def loss_fn(model, X, states, y):
            return nn.losses.mse_loss(model(X, states)[0], y)

        loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)

        target = mx.random.normal((self.batch_size, self.inp_dim))

        l, grads = loss_and_grad_fn(self.model, self.input, self.hid_0, target)

        for name, grads in zip(self.model.parameters(), grads):
            if 'causal_conv' in name: continue
            self.assertIsNotNone(grads)

if __name__ == '__main__':
    unittest.main()