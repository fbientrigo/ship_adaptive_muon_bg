"""D9-5: modern same-track generative model family arena (implementation only).

Compares NF_AC, GAUSS_DIAG, GAUSS_FULL and GMM under one shared data scope,
preprocessing contract and evaluation protocol, separately on
``TRK_PDG13_UW_ID`` and ``TRK_PDGM13_UW_ID``. See
docs/reviews/afterms_d9_5_model_family_arena_v0.md and
docs/reviews/afterms_d9_5_execution_runbook_v0.md.

This package implements the data-scope audit, the common model-adapter
contract, the Gaussian/GMM/NF_AC adapters, evaluation, aggregation and
reporting; it reuses ``ship_muon_bg.afterms.d9.runner.train_candidate_seed``
for NF_AC fitting rather than duplicating the training loop.
"""
