# fused_dispatch

A from-scratch CUDA implementation of Mixture-of-Experts (MoE) token dispatch
across GPUs on one node. GPUs write directly into each other's memory, with no
collective calls and no host round trip.

> Work in progress. See [Status](#status).

## The problem

For MoE small batch sizes, such as single-token decode during inference, one MoE layer might move only a few kilobytes between each pair of GPUs. Most of the latency comes from fixed overhead due to the NCCL library. Additionally, typical dispatch first permutes tokens, exchanges meta data, and then initializes buffers. So each layer can pay several kernel launches and a host synchronization to move data that takes well under a microsecond to cross the link. 


## The design

Each GPU writes its tokens straight into peer memory. It then announces how
many it sent with one 8-byte store. The receiver learns its counts without
waiting for a separate exchange.

**Per-source lanes.** Every GPU has one receive lane for each source GPU
(payload, metadata and a header word). A sender only ever writes to its own
lane on each destination. Senders never contend with each other, so allocating
a slot is a cheap atomic on the sender's local GPU, not a cross-GPU atomic.

**Single-writer headers.** Each lane's header is a 64-bit word that only its
source writes: `(seq << 32) | count`. One store publishes both the count and the
step number. The receiver polls until it sees the current `seq`, so the header
never needs resetting between steps.

**Three kernels, one stream, no host sync.**

```
each GPU's stream:   scatter  ──►  publish  ──►  wait  ──►  (expert compute)
```

- `scatter`: one block per (token, top-k choice). It claims a slot in the
  destination's lane and copies the token and its metadata there with peer
  writes.
- `publish`: one thread per destination. It writes the header word on that
  destination.
- `wait`: one thread per source. It spins on the local header until `seq`
  matches, then records the count. It has a spin limit, so a bug shows up as an
  error flag instead of a hang.

Stream order makes each GPU's scatter writes finish before its publish runs.
The host only enqueues kernels.

```
                 GPU 0                                   GPU 1
   ┌───────────────────────────────┐     ┌───────────────────────────────┐
   │ lane[src 0]  payload|meta|hdr │◄─┐  │ lane[src 0]  payload|meta|hdr │◄─┐
   │ lane[src 1]  payload|meta|hdr │◄┐│  │ lane[src 1]  payload|meta|hdr │◄┐│
   └───────────────────────────────┘ ││  └───────────────────────────────┘ ││
                                     ││                                    ││
   GPU 0 writes ─────────────────────┼┴────────────────────────────────────┼┘
   GPU 1 writes ─────────────────────┴─────────────────────────────────────┘
```

GPU 0 owns `lane[src 0]` on every GPU, including its own, and GPU 1 owns
`lane[src 1]` on every GPU. No two GPUs ever write to the same lane.

## Status

- [x] Scatter: peer writes into per-source lanes (tested against a CPU reference in `test_dispatch.py`)
- [x] Publish and wait: implemented, testing in progress
- [ ] NCCL all-to-all baseline for comparison
- [ ] Combine (the return path that gathers weighted expert outputs)
- [ ] Benchmarks across batch sizes and GPU counts

## Prior art

The idea is not new. [DeepEP](https://github.com/deepseek-ai/DeepEP) (DeepSeek)
and [pplx-kernels](https://github.com/ppl-ai/pplx-kernels) (Perplexity) are
production implementations of fused MoE dispatch and combine over NVLink and
RDMA, and they are what you should use in a real system. This project is a
from-scratch build so we can learn the design and measure where the time goes.

## Authors

Rockwell Jackson

## License

MIT. See [LICENSE](LICENSE).
