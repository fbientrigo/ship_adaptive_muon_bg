import inspect

from ship_muon_bg.afterms.d9 import contract as d9contract


def _base_kwargs():
    return dict(
        candidate_config={"model_family": "affine_coupling", "learning_rate": 0.001},
        preprocessing_contract={"preprocessing_name": "identity_standardized_v0", "serialization_hash": "abc"},
        dataset_identity={"train_split_hash": "t1", "validation_split_hash": "v1"},
        modeled_feature_order=["px", "py", "pz", "x", "y"],
        target_measure="physical_space_nll",
        weighting_estimator_version="not_applicable",
        optimizer_settings={"optimizer": "adam", "learning_rate": 0.001},
        module_fingerprints={"runner.py": "fp1"},
    )


def test_hash_is_stable_for_identical_inputs():
    a = d9contract.training_contract_hash(**_base_kwargs())
    b = d9contract.training_contract_hash(**_base_kwargs())
    assert a == b


def test_hash_has_no_producer_git_commit_parameter():
    """Required test 20: producer commit alone must not force a rerun --
    structurally guaranteed by never accepting it as an input at all."""

    params = inspect.signature(d9contract.training_contract_hash).parameters
    assert "producer_git_commit" not in params
    assert "git_commit" not in params
    assert "git_head" not in params


def test_hash_changes_when_candidate_config_changes():
    kwargs = _base_kwargs()
    a = d9contract.training_contract_hash(**kwargs)
    kwargs["candidate_config"] = dict(kwargs["candidate_config"], learning_rate=0.5)
    b = d9contract.training_contract_hash(**kwargs)
    assert a != b


def test_hash_changes_when_module_fingerprint_changes():
    """Required test 21 (proxy): a relevant training-code change invalidates resume."""

    kwargs = _base_kwargs()
    a = d9contract.training_contract_hash(**kwargs)
    kwargs["module_fingerprints"] = {"runner.py": "fp2_after_edit"}
    b = d9contract.training_contract_hash(**kwargs)
    assert a != b


def test_hash_changes_when_preprocessing_contract_changes():
    kwargs = _base_kwargs()
    a = d9contract.training_contract_hash(**kwargs)
    kwargs["preprocessing_contract"] = {"preprocessing_name": "cartesian_log1p_pz_v0", "serialization_hash": "abc"}
    b = d9contract.training_contract_hash(**kwargs)
    assert a != b


def test_hash_changes_when_weighting_estimator_changes():
    kwargs = _base_kwargs()
    a = d9contract.training_contract_hash(**kwargs)
    kwargs["weighting_estimator_version"] = "fixed_global_weight_normalization_v1"
    b = d9contract.training_contract_hash(**kwargs)
    assert a != b


def test_module_source_fingerprint_reads_current_repo_modules(tmp_path):
    (tmp_path / "a.py").write_text("print(1)\n", encoding="utf-8")
    fp1 = d9contract.module_source_fingerprint(tmp_path, ["a.py"])
    (tmp_path / "a.py").write_text("print(2)\n", encoding="utf-8")
    fp2 = d9contract.module_source_fingerprint(tmp_path, ["a.py"])
    assert fp1["a.py"] != fp2["a.py"]
