# Attention Pointer Network

> **Based on** [wouterkool/attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route) (Kool et al., ICLR 2019). This repo extends the Attention Model to the problem variant introduced in our paper. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the mapping.


Faithful PyTorch reproduction of *"Attention, Learn to Solve Routing Problems!"* (Kool et al., ICLR 2019). Encoder is multi-head attention with batch-norm (per the paper, not layer-norm). Decoder is a glimpse + pointer attention with tanh clip. Used as the starting point for the SED2AM, EFECTIW, and Edge-DIRECT papers.

## Use

```python
from ampn import AttentionModelEncoder, PointerDecoder, TSPEnv
env = TSPEnv(n_cities=20, batch_size=64)
encoder = AttentionModelEncoder(in_dim=2)
decoder = PointerDecoder()
```
<!-- notes 2022-02 -->

<!-- notes 2022-04 -->

<!-- notes 2022-06 -->

<!-- notes 2022-09 -->
