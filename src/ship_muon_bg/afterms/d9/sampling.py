"""D9 per-epoch deterministic minibatch sampling contract (Gate B.1).

Minibatch order within an epoch must depend only on (run seed, epoch index) --
never on how many permutations were drawn earlier in this process's lifetime
-- so that a resumed run reconstructs exactly the epoch-N ordering a clean,
uninterrupted run would have produced for that epoch.

A single long-lived ``torch.Generator`` seeded once at process start does not
have this property: its state at epoch N depends on how many epoch
permutations were already drawn *in this process*. A clean run draws N-1
permutations before epoch N; a resumed run starts a fresh process and draws
zero before epoch N, so the two see different generator states -- and
therefore different epoch-N orderings -- even though both are "epoch N" of
the same seed. Reseeding the generator from ``epoch_permutation_seed`` before
every epoch removes that process-history dependency.
"""

from __future__ import annotations

import hashlib

SAMPLING_CONTRACT_VERSION = "d9_epoch_seed_v1"

# torch.Generator.manual_seed accepts any Python int, but negative/huge values
# get folded internally in ways that vary by backend; masking to nonnegative
# int64 keeps the seed value unambiguous across CPU and CUDA generators.
_SEED_MASK = (1 << 63) - 1


def epoch_permutation_seed(run_seed: int, epoch_index: int) -> int:
    """Stable per-epoch generator seed, independent of process/resume boundaries.

    A pure function of (contract version, run_seed, epoch_index): the same
    run_seed and epoch_index always yield the same seed, regardless of which
    process or how many prior epochs ran in it. Different epoch_index or
    different run_seed values yield different seeds (collision probability
    negligible, sha256-derived).
    """

    payload = f"{SAMPLING_CONTRACT_VERSION}|{int(run_seed)}|{int(epoch_index)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & _SEED_MASK
