import torch
from ampn.encoder import AttentionModelEncoder


def test_encoder_runs() -> None:
    enc = AttentionModelEncoder(in_dim=2, d=64, h=4, n_layers=2)
    x = torch.randn(4, 20, 2)
    out = enc(x)
    assert out.shape == (4, 20, 64)
