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

<!-- notes 2022-11 -->

<!-- maint 2025-01-24 -->

<!-- maint 2025-03-04 -->

<!-- maint 2025-04-13 -->

<!-- maint 2025-05-21 -->

<!-- maint 2025-06-30 -->

<!-- maint 2025-08-07 -->

<!-- maint 2025-09-16 -->

<!-- maint 2025-10-24 -->

<!-- maint 2025-12-02 -->

<!-- maint 2024-02-02 -->

<!-- maint 2024-03-25 -->

<!-- maint 2024-05-15 -->

<!-- maint 2024-07-06 -->

<!-- maint 2024-08-28 -->

<!-- maint 2024-10-19 -->

<!-- maint 2024-12-09 -->

<!-- maint 2023-02-13 -->

<!-- maint 2023-04-20 -->

<!-- iter 2023-06-26-09 -->

<!-- iter 2023-06-26-11 -->

<!-- iter 2023-06-26-13 -->

<!-- iter 2023-06-26-15 -->

<!-- iter 2023-06-26-17 -->

<!-- iter 2023-06-26-19 -->

<!-- iter 2023-06-26-21 -->

<!-- iter 2023-06-26-22 -->

<!-- iter 2023-11-20-09 -->

<!-- iter 2023-11-20-11 -->

<!-- iter 2023-11-20-13 -->

<!-- iter 2023-11-20-15 -->

<!-- iter 2023-11-20-17 -->

<!-- iter 2023-11-20-19 -->

<!-- iter 2023-11-20-21 -->

<!-- iter 2024-03-11-09 -->

<!-- iter 2024-03-11-11 -->

<!-- iter 2024-03-11-13 -->

<!-- iter 2024-03-11-15 -->

<!-- iter 2024-03-11-17 -->

<!-- iter 2024-03-11-19 -->

<!-- iter 2024-03-11-21 -->

<!-- iter 2024-09-23-09 -->

<!-- iter 2024-09-23-11 -->

<!-- iter 2024-09-23-13 -->

<!-- iter 2024-09-23-15 -->

<!-- iter 2024-09-23-17 -->

<!-- iter 2024-09-23-19 -->

<!-- iter 2024-09-23-21 -->

<!-- iter 2024-09-23-22 -->

<!-- iter 2026-03-16-09 -->

<!-- iter 2026-03-16-11 -->

<!-- iter 2026-03-16-13 -->

<!-- iter 2026-03-16-15 -->

<!-- iter 2026-03-16-17 -->

<!-- iter 2026-03-16-19 -->

<!-- iter 2026-03-16-21 -->

<!-- m 2025-05-03T21:53:00-06:00 -->

<!-- m 2024-06-16T20:24:00-06:00 -->

<!-- m 2023-10-14T17:35:00-06:00 -->

<!-- m 2023-03-10T11:19:00-06:00 -->

<!-- m 2023-01-21T17:59:00-06:00 -->

<!-- m 2023-12-27T17:25:00-06:00 -->

<!-- m 2025-12-11T13:04:00-06:00 -->

<!-- m 2023-01-25T22:06:00-06:00 -->

<!-- m 2023-03-07T20:43:00-06:00 -->

<!-- m 2024-12-06T19:19:00-06:00 -->

<!-- m 2025-04-04T18:43:00-06:00 -->

<!-- m 2025-12-13T16:46:00-06:00 -->

<!-- m 2023-12-23T21:49:00-06:00 -->

<!-- m 2023-04-14T19:17:00-06:00 -->

<!-- m 2025-04-05T18:15:00-06:00 -->

<!-- m 2025-08-02T20:39:00-06:00 -->

<!-- m 2024-06-14T23:44:00-06:00 -->

<!-- m 2026-01-22T23:14:00-06:00 -->

<!-- m 2025-02-27T15:44:00-06:00 -->

<!-- m 2025-08-01T23:28:00-06:00 -->

<!-- m 2023-02-16T19:51:00-06:00 -->

<!-- m 2024-06-15T19:50:00-06:00 -->

<!-- m 2023-01-23T20:18:00-06:00 -->

<!-- m 2023-04-13T15:40:00-06:00 -->

<!-- m 2024-09-10T16:02:00-06:00 -->

<!-- m 2025-08-07T20:58:00-06:00 -->

<!-- m 2023-04-11T15:22:00-06:00 -->

<!-- m 2023-10-13T19:26:00-06:00 -->

<!-- m 2025-03-04T17:09:00-06:00 -->
