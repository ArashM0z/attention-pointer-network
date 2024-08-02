# Architecture and base code

This implementation extends the **Attention Model** of Kool et al. (ICLR 2019).
The reference implementation is at:

> https://github.com/wouterkool/attention-learn-to-route

This repo inherits the same architectural skeleton and adapts it to the problem
introduced in our paper (Attention Pointer Network (Kool 2019 reproduction)).

## What we kept from the Kool 2019 base

- **Encoder**: Multi-head attention encoder over node embeddings with batch
  normalisation (per the paper, not layer-norm).
- **Decoder**: Glimpse + pointer attention with tanh-clipped logits.
- **REINFORCE with rollout baseline**: deepcopy of the current policy, swapped
  in when the current policy outperforms it on a held-out batch by a
  statistically-significant margin (paired t-test).
- **Training loop shape**: epoch-based, baseline-refresh schedule, gradient
  clipping at norm 1.0.

## What this repo changes

- **Problem-specific state and reward**: the environment (`env/`) implements
  the constraints unique to Attention Pointer Network (Kool 2019 reproduction).
- **Context features**: the pointer decoder receives an expanded context
  vector that carries the problem-specific state variables.
- **Reward shaping** where appropriate (e.g., time-window slack penalties for
  the HF-VRPTW variant).

## How files map to the Kool 2019 base

| This repo | Kool 2019 base |
|---|---|
| `src/.../policy/encoder.py` | `nets/graph_encoder.py` |
| `src/.../policy/decoder.py` | `nets/pointer_network.py` |
| `src/.../policy/agent.py`   | `nets/attention_model.py` |
| `src/.../trainer/reinforce.py` | `train.py` + `reinforce_baselines.py` |
| `src/.../env/*.py`          | `problems/cvrp/` (adapted) |

## Citation

Please cite **both** the Kool 2019 reference paper and our paper if you build
on this work:

```bibtex
@inproceedings{kool2019attention,
  title={Attention, Learn to Solve Routing Problems!},
  author={Kool, Wouter and van Hoof, Herke and Welling, Max},
  booktitle={International Conference on Learning Representations},
  year={2019}
}
```

For our paper, see CITATION.cff in the repo root.
