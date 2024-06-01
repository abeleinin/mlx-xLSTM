import mlx.core as mx
import mlx.nn as nn

from .mLSTM import mLSTMBlock
from .sLSTM import sLSTMBlock

from .util import repeat

class xLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_layers,
        signature,
        inp_dim,
        head_dim,
        head_num,
        p_factor,
        ker_size,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hid_dim = head_dim * head_num

        self.embedding = nn.Embedding(
            vocab_size, 
            inp_dim, 
        )

        m_factor, s_factor = p_factor
        m_num, s_num = signature
        which = [True] * m_num + [False] * s_num

        self.blocks = [
            mLSTMBlock(inp_dim, head_dim, head_num, m_factor, ker_size) if w else 
            sLSTMBlock(inp_dim, head_dim, head_num, s_factor, ker_size)
            for w, _ in zip(repeat(which), range(num_layers))
        ]

        self.head = nn.Linear(inp_dim, vocab_size)
        
    def __call__(self, tok, hid=None, batch_first = False):
        tok = mx.atleast_2d(tok)
        seq = self.embedding(tok)

        if batch_first: 
            bs, s, i = seq.shape
            seq = seq.reshape(s, bs, i)
        if hid is None: hid = [l.init_hidden(seq.shape[1]) for l in self.blocks]

        out = []
        for inp in seq:
            for i, lstm in enumerate(self.blocks):
                inp, hid[i] = lstm(inp, hid[i])
            
            out.append(inp)
        
        out = mx.stack(out, axis=1 if batch_first else 0)
        out = self.head(out)

        return out, hid
