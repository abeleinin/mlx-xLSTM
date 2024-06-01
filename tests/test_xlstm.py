import unittest
import mlx.core as mx
import mlx.nn as nn

from mlx_xlstm import xLSTM

class TestXLSTM(unittest.TestCase):
    def setUp(self):
        self.num_layers = 8
        self.signature = (7, 1)
        self.inp_dim = 16
        self.head_dim = 8
        self.head_num = 4
        self.ker_size = 4
        self.p_factor = (2, 4/3)

        self.seq_len = 32
        self.batch_size = 4
        self.vocab_size = 24
        
        self.seq = mx.random.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
    def test_llm_forward(self):
        model = xLSTM(
            vocab_size = self.vocab_size,
            num_layers = self.num_layers,
            signature = self.signature,
            inp_dim= self.inp_dim,
            head_dim= self.head_dim,
            head_num= self.head_num,
            p_factor= self.p_factor,
            ker_size = self.ker_size,
        )
        
        out, _ = model(self.seq, batch_first=True)
        
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.vocab_size))

if __name__ == '__main__':
    unittest.main()