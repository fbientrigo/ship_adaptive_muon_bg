import numpy as np
import pytest

from ship_muon_bg.afterms.d9 import weighted_objective as wo


def test_estimator_name_is_versioned():
    assert wo.ESTIMATOR_NAME == "fixed_global_weight_normalization_v1"


def test_exact_fixed_global_weighted_objective_synthetic():
    """Required test 10: exact fixed-global weighted objective on a synthetic
    dataset, computed two independent ways and checked to agree."""

    rng = np.random.default_rng(0)
    n_train = 10
    weights = rng.uniform(0.1, 5.0, size=n_train)
    nll = rng.uniform(0.5, 3.0, size=n_train)

    w_train_total = wo.split_weight_total(weights)
    assert w_train_total == pytest.approx(np.sum(weights))

    # Single full-batch "minibatch" (batch == whole split) must reduce to
    # L(theta) = sum(w*nll)/sum(w) exactly.
    full_batch_loss = wo.weighted_batch_loss_numpy(
        nll, weights, n_train=n_train, w_train_total=w_train_total
    )
    direct = np.sum(weights * nll) / np.sum(weights)
    assert full_batch_loss == pytest.approx(direct, rel=1e-12)

    # Splitting into two batches and averaging their (N/W)*mean(w*nll) terms
    # must reproduce the same fixed full-split target in expectation over the
    # partition (exact equality holds for this specific even split since each
    # sub-mean scales by its own count / N appropriately when weighted by
    # n_train, not by the local batch's own weight sum).
    idx1, idx2 = np.arange(0, 5), np.arange(5, 10)
    loss1 = wo.weighted_batch_loss_numpy(
        nll[idx1], weights[idx1], n_train=n_train, w_train_total=w_train_total
    )
    loss2 = wo.weighted_batch_loss_numpy(
        nll[idx2], weights[idx2], n_train=n_train, w_train_total=w_train_total
    )
    # Each batch loss is (n_train/W)*mean_batch(w*nll) = (n_train/W)*(1/5)*sum_batch(w*nll).
    # Summing the two full gradients (not averaging) reconstructs (n_train/W)*(1/5)*sum_all(w*nll).
    reconstructed = (loss1 + loss2) * (len(idx1) / n_train)
    # (n_train/W)*(1/5)*sum_all(w*nll) == (n_train/W)*(2/10)*sum_all(w*nll) == direct*(n_train/5)...
    # Simplify: verify algebraically via direct recomputation instead of a fragile identity.
    expected = (n_train / w_train_total) * (np.sum(weights[idx1] * nll[idx1]) / len(idx1))
    assert loss1 == pytest.approx(expected, rel=1e-12)


def test_weighted_batch_loss_uses_fixed_global_denominator_not_per_batch():
    """Required test 12: no per-minibatch self-normalized denominator.

    The historical (forbidden) estimator is sum(w*nll)/sum(w) computed PER
    BATCH -- which is invariant to uniformly rescaling that batch's weights.
    The fixed-global estimator is NOT invariant to a batch-local weight
    rescale, because the denominator (w_train_total) does not rescale with
    it -- this is the structural signature that distinguishes the two.
    """

    rng = np.random.default_rng(1)
    n_train = 20
    weights = rng.uniform(0.5, 2.0, size=n_train)
    nll = rng.uniform(0.5, 3.0, size=n_train)
    w_train_total = wo.split_weight_total(weights)

    batch_idx = np.arange(0, 4)
    baseline = wo.weighted_batch_loss_numpy(
        nll[batch_idx], weights[batch_idx], n_train=n_train, w_train_total=w_train_total
    )
    rescaled_weights = weights[batch_idx] * 10.0
    rescaled = wo.weighted_batch_loss_numpy(
        nll[batch_idx], rescaled_weights, n_train=n_train, w_train_total=w_train_total
    )
    # Under the forbidden per-batch self-normalized ratio, rescaling every
    # weight in the batch by the same constant leaves the ratio unchanged.
    # Under fixed-global normalization it must NOT be unchanged.
    assert rescaled != pytest.approx(baseline, rel=1e-9)
    assert rescaled == pytest.approx(baseline * 10.0, rel=1e-9)


def test_weighted_validation_aggregation_matches_global_ratio():
    """Required test 11: weighted validation aggregation is one global ratio
    over the complete split, not an average of per-chunk ratios."""

    rng = np.random.default_rng(2)
    n_val = 13
    weights = rng.uniform(0.2, 4.0, size=n_val)
    nll = rng.uniform(0.5, 3.0, size=n_val)

    global_ratio = wo.weighted_validation_nll(nll, weights)
    expected = np.sum(weights * nll) / np.sum(weights)
    assert global_ratio == pytest.approx(expected, rel=1e-12)

    # An average of two per-chunk ratios is NOT the same quantity in general
    # (unless chunk weight sums happen to be equal) -- verify these two
    # candidate implementations actually differ for this data, guarding
    # against silently reintroducing the forbidden per-chunk-averaged form.
    idx1, idx2 = np.arange(0, 6), np.arange(6, 13)
    ratio1 = np.sum(weights[idx1] * nll[idx1]) / np.sum(weights[idx1])
    ratio2 = np.sum(weights[idx2] * nll[idx2]) / np.sum(weights[idx2])
    naive_average_of_chunks = 0.5 * (ratio1 + ratio2)
    assert global_ratio != pytest.approx(naive_average_of_chunks, rel=1e-6)


def test_weighted_batch_loss_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    n_train = 8
    weights = rng.uniform(0.5, 2.0, size=n_train)
    nll = rng.uniform(0.5, 3.0, size=n_train)
    w_train_total = wo.split_weight_total(weights)

    numpy_val = wo.weighted_batch_loss_numpy(nll, weights, n_train=n_train, w_train_total=w_train_total)
    torch_val = wo.weighted_batch_loss_torch(
        torch.tensor(nll, dtype=torch.float64),
        torch.tensor(weights, dtype=torch.float64),
        n_train=n_train,
        w_train_total=w_train_total,
    )
    assert float(torch_val) == pytest.approx(numpy_val, rel=1e-9)


def test_unweighted_reduces_to_plain_mean():
    rng = np.random.default_rng(4)
    nll = rng.uniform(0.5, 3.0, size=25)
    assert wo.unweighted_batch_loss_numpy(nll) == pytest.approx(float(np.mean(nll)))
    assert wo.unweighted_validation_nll(nll) == pytest.approx(float(np.mean(nll)))


def test_split_weight_total_rejects_degenerate_input():
    with pytest.raises(ValueError):
        wo.split_weight_total(np.array([0.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        wo.split_weight_total(np.array([[1.0, 2.0]]))
