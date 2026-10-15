"""D9 Gate B.1: per-epoch deterministic minibatch sampling contract."""

import pytest

torch = pytest.importorskip("torch")

from ship_muon_bg.afterms.d9 import sampling as d9sampling


def test_same_seed_same_epoch_gives_same_seed_value():
    """Required test 1: same run seed + same epoch gives the same permutation."""

    a = d9sampling.epoch_permutation_seed(run_seed=7, epoch_index=2)
    b = d9sampling.epoch_permutation_seed(run_seed=7, epoch_index=2)
    assert a == b


def test_different_epochs_give_different_deterministic_seeds():
    """Required test 2: different epochs give different deterministic permutations."""

    seeds = {d9sampling.epoch_permutation_seed(run_seed=7, epoch_index=e) for e in range(1, 6)}
    assert len(seeds) == 5


def test_different_run_seeds_give_different_seeds():
    """Required test 3: different run seeds give different permutations."""

    a = d9sampling.epoch_permutation_seed(run_seed=1, epoch_index=2)
    b = d9sampling.epoch_permutation_seed(run_seed=2, epoch_index=2)
    assert a != b


def test_same_seed_same_epoch_gives_same_torch_permutation():
    """The derived seed reproduces the identical torch.randperm draw."""

    def _perm():
        gen = torch.Generator(device="cpu")
        gen.manual_seed(d9sampling.epoch_permutation_seed(run_seed=42, epoch_index=3))
        return torch.randperm(37, generator=gen)

    torch.testing.assert_close(_perm(), _perm())


def test_epoch_permutation_seed_independent_of_process_history():
    """The core Gate B.1 property: epoch N's seed does not depend on how many
    prior epochs were drawn from a shared generator in this process -- unlike
    a single long-lived Generator stream, where epoch N's draw is the Nth draw
    in a clean run but the 1st draw after a resume starting fresh at epoch N."""

    gen = torch.Generator(device="cpu")
    # Simulate "epoch 1 already happened in this process" by drawing once
    # from a fresh, differently-seeded generator first (mimicking unrelated
    # process-lifetime generator state) -- epoch_permutation_seed for epoch 2
    # must not depend on this.
    gen.manual_seed(d9sampling.epoch_permutation_seed(run_seed=5, epoch_index=1))
    torch.randperm(10, generator=gen)

    gen.manual_seed(d9sampling.epoch_permutation_seed(run_seed=5, epoch_index=2))
    perm_after_epoch_1 = torch.randperm(10, generator=gen)

    fresh_gen = torch.Generator(device="cpu")
    fresh_gen.manual_seed(d9sampling.epoch_permutation_seed(run_seed=5, epoch_index=2))
    perm_fresh_process = torch.randperm(10, generator=fresh_gen)

    torch.testing.assert_close(perm_after_epoch_1, perm_fresh_process)
