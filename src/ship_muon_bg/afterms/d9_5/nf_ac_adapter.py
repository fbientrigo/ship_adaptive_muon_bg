"""D9-5 NF_AC adapter (Sec 5, Sec 11).

Derives the scout-promoted architecture from the named
``AFFINE_COUPLING_CAPACITY_SCOUT_V0`` report (never a hardcoded legacy
candidate id) and reuses
``ship_muon_bg.afterms.d9.runner.train_candidate_seed`` for the actual
training loop -- this module never reimplements minibatch training, exact
resume, or checkpoint bundling. Each adapter instance is one (track,
model_config_id, seed): D9-5 trains three seeds from initialization per
track and never reuses a scout checkpoint (Sec 11).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ship_muon_bg.afterms import model_naming
from ship_muon_bg.afterms.d9 import checkpoint as d9ckpt
from ship_muon_bg.afterms.d9 import evaluate as d9evaluate
from ship_muon_bg.afterms.d9 import runner as d9runner
from ship_muon_bg.afterms.preprocessing import PreprocessingPipeline

from . import config as d9_5config
from . import model_adapter as ma

SCOUT_REPORT_PATH = (
    d9_5config.REPO_ROOT / "artifacts" / "afterms_d9_model_arena_v0" / "named" / "run_inventory_named.json"
)
SCOUT_EXPERIMENT_ID = "AFFINE_COUPLING_CAPACITY_SCOUT_V0"
SCOUT_SEED = 20260720
SCOUT_EPOCHS = 10
CAMPAIGN_ID = "afterms_d9_5_model_family_arena_v0"


class ScoutSelectionError(ValueError):
    pass


def select_scout_promoted_nf_config(track_id: str, *, scout_report_path: Path = SCOUT_REPORT_PATH) -> Dict[str, Any]:
    """Resolve the NF_AC ``model_config_id``/architecture for ``track_id``
    from the scout's own recorded evidence (best ``best_validation_metric``
    among its ``status == "completed"`` variants on this track)."""

    records = json.loads(Path(scout_report_path).read_text(encoding="utf-8"))
    candidates = [r for r in records if r.get("track_id") == track_id and r.get("status") == "completed"]
    if not candidates:
        raise ScoutSelectionError(
            f"no completed {SCOUT_EXPERIMENT_ID} run found for track_id={track_id!r} in {scout_report_path}"
        )
    best = min(candidates, key=lambda r: r["best_validation_metric"])
    architecture = model_naming.parse_nf_ac_model_config_id(best["model_config_id"])
    return {
        "track_id": track_id,
        "model_config_id": best["model_config_id"],
        "architecture": architecture,
        "source_scout_experiment": SCOUT_EXPERIMENT_ID,
        "source_run_id": best["internal_candidate_id"],
        "scout_seed": SCOUT_SEED,
        "scout_epochs": SCOUT_EPOCHS,
        "scout_validation_metric": best["best_validation_metric"],
        "selection_limitations": (
            f"single scout seed ({SCOUT_SEED}), {SCOUT_EPOCHS} scout epochs only; the scout selected "
            "architecture capacity within its track's sweep, it is not a converged final model and "
            "D9-5 never reuses its checkpoint."
        ),
        "config_label": "scout_promoted_nf_config",
    }


class NfAcAdapter(ma.ModelAdapter):
    model_family_id = "NF_AC"

    def __init__(
        self,
        *,
        track_id: str,
        pdg_value: int,
        architecture: Dict[str, int],
        execution_policy: Dict[str, int],
        optimizer_settings: Dict[str, Any],
        evaluation_policy: Dict[str, Any],
        artifact_root: Path,
        repo_root: Path = d9_5config.REPO_ROOT,
        device: str = "cpu",
        shard_dir: Optional[str] = None,
        train_shard_names: Optional[list] = None,
        validation_shard_names: Optional[list] = None,
        test_shard_names: Optional[list] = None,
    ) -> None:
        self.track_id = track_id
        self.pdg_value = int(pdg_value)
        self.architecture = dict(architecture)
        self.model_config_id = model_naming.nf_ac_model_config_id(
            number_of_blocks=architecture["number_of_blocks"],
            hidden_width=architecture["hidden_width"],
            hidden_depth=architecture["hidden_depth"],
        )
        self.execution_policy = dict(execution_policy)
        self.optimizer_settings = dict(optimizer_settings)
        self.evaluation_policy = dict(evaluation_policy)
        self.artifact_root = Path(artifact_root)
        self.repo_root = Path(repo_root)
        self.device = device
        self.shard_dir = shard_dir
        self.train_shard_names = list(train_shard_names or [])
        self.validation_shard_names = list(validation_shard_names or [])
        self.test_shard_names = list(test_shard_names or [])

        self._candidate_config: Optional[Dict[str, Any]] = None
        self._run_dir: Optional[Path] = None
        self._pipeline: Optional[PreprocessingPipeline] = None
        self._estimator = None
        self._bundle: Optional[Dict[str, Any]] = None

    @property
    def pipeline(self) -> Optional[PreprocessingPipeline]:
        """Public alias matching the Gaussian/GMM adapters' attribute name, so
        evaluation.py can read the fitted preprocessing contract uniformly
        across families."""
        return self._pipeline

    @property
    def candidate_id(self) -> str:
        # A slash-bearing candidate_id is a legitimate opaque identity string
        # to train_candidate_seed's Path(...) construction and gives the D9-5
        # runs/<track_id>/<model_config_id>/seed_<seed>/ layout for free,
        # without a second copy of run_directory's path logic.
        return f"{self.track_id}/{self.model_config_id}"

    def _build_candidate_config(self, *, seed: int) -> Dict[str, Any]:
        return {
            "campaign_id": CAMPAIGN_ID,
            "candidate_id": self.candidate_id,
            "model_family": "affine_coupling",
            # Deliberately NOT tagged with a capacity_label/alias string here:
            # candidate_config feeds semantic_training_hash wholesale (Sec 25,
            # required test 25), so a presentation-only label has no business
            # inside it -- "scout_promoted_nf_config" is recorded only in
            # select_scout_promoted_nf_config's own return dict and in
            # model_metadata(), never in the hashed training identity.
            "architecture": dict(self.architecture),
            "modeled_features": ["px", "py", "pz", "x", "y"],
            "feature_order": ["px", "py", "pz", "x", "y"],
            "pdg_policy": "pdg13" if self.pdg_value == 13 else "pdg_minus13",
            "pdg_value": self.pdg_value,
            "preprocessing_name": "identity_standardized_v0",
            "target_measure": "physical_space_nll",
            "weighting_policy": "row_empirical_unweighted",
            "weighting_estimator_version": "not_applicable",
            "shard_dir": self.shard_dir,
            "train_shards": self.train_shard_names,
            "validation_shards": self.validation_shard_names,
            "test_shards": self.test_shard_names,
            "seed_set": [int(seed)],
            "max_epochs": self.execution_policy["maximum_epochs"],
            "minimum_epochs": self.execution_policy["minimum_epochs"],
            "early_stopping_patience": self.execution_policy["early_stopping_patience"],
            "optimizer": self.optimizer_settings["optimizer"],
            "learning_rate": self.optimizer_settings["learning_rate"],
            "batch_size": self.optimizer_settings["batch_size"],
            "weight_decay": self.optimizer_settings["weight_decay"],
            "gradient_clipping": self.optimizer_settings["gradient_clipping"],
            "dtype": self.optimizer_settings["dtype"],
            "device_policy": self.optimizer_settings["device_policy"],
            "checkpoint_policy": {
                "save_best": True, "save_final": True, "save_last_resumable": True,
                "checkpoint_interval_epochs": 1,
            },
            "evaluation_policy": self.evaluation_policy,
        }

    def fit(
        self, train_raw: np.ndarray, validation_raw: np.ndarray, *, seed: int,
        resume: bool = False, extend_max_epochs: Optional[int] = None,
        interrupt_flag=None,
    ) -> Dict[str, Any]:
        candidate_config = self._build_candidate_config(seed=seed)
        result = d9runner.train_candidate_seed(
            candidate_config, int(seed), train_raw, validation_raw,
            artifact_root=self.artifact_root, repo_root=self.repo_root, device=self.device,
            resume=resume, extend_max_epochs=extend_max_epochs, interrupt_flag=interrupt_flag,
        )
        self._candidate_config = candidate_config
        self._run_dir = d9runner.run_directory(self.artifact_root, self.candidate_id, int(seed))
        if result["status"] == d9runner.STATUS_COMPLETED:
            self._load_estimator_from_checkpoint(d9ckpt.SCOPE_BEST)
        return result

    def _load_estimator_from_checkpoint(self, checkpoint_scope: str) -> None:
        from Nflow.registry import create_density_estimator

        bundle_path = self._run_dir / "checkpoints" / d9ckpt.FILENAME_BY_SCOPE[checkpoint_scope]
        bundle = d9ckpt.load_bundle(bundle_path)
        estimator = create_density_estimator(
            {"family": "affine_coupling", "params": d9runner._architecture_params(self._candidate_config)},
            dimension=5, device=self.device,
        )
        estimator._build_module(seed=int(bundle["seed"]))
        estimator._module.load_state_dict(bundle["model_state_dict"])
        estimator._module.eval()
        self._estimator = estimator
        self._bundle = bundle
        preprocessing_contract = json.loads(
            (self._run_dir / "preprocessing" / "preprocessing.json").read_text(encoding="utf-8")
        )
        self._pipeline = PreprocessingPipeline.from_dict(preprocessing_contract["fitted_state"])

    def _feature_log_prob(self, raw: np.ndarray) -> np.ndarray:
        import torch

        if self._estimator is None:
            raise RuntimeError("model is not fitted/loaded")
        normalized = self._pipeline.transform(raw)
        with torch.no_grad():
            log_prob = self._estimator._module.log_prob(
                torch.tensor(normalized, dtype=self._estimator.torch_dtype, device=self._estimator.device)
            ).cpu().numpy().astype(np.float64)
        return log_prob

    def validation_log_prob(self, validation_raw: np.ndarray) -> np.ndarray:
        return self._feature_log_prob(validation_raw)

    def test_log_prob(self, test_raw: np.ndarray) -> np.ndarray:
        return self._feature_log_prob(test_raw)

    def sample(self, n: int, *, seed: int) -> np.ndarray:
        normalized = d9evaluate.generate_deterministic_samples(self._estimator, int(n), int(seed))
        return self._pipeline.inverse(normalized)

    def capability_metadata(self) -> Dict[str, bool]:
        capabilities = {
            "supports_exact_log_prob": True,
            "supports_physical_log_prob": True,
            "stochastic_fit": True,
            "supports_cuda": True,
            "supports_resume": True,
            "deterministic_sampling_given_seed": True,
        }
        ma.assert_capability_metadata(capabilities)
        return capabilities

    def model_metadata(self) -> Dict[str, Any]:
        return {
            "model_family_id": self.model_family_id,
            "model_config_id": self.model_config_id,
            "track_id": self.track_id,
            "architecture": self.architecture,
            "semantic_training_hash": self._bundle.get("semantic_training_hash") if self._bundle else None,
            "execution_policy_hash": self._bundle.get("execution_policy_hash") if self._bundle else None,
            "evaluation_policy_hash": self._bundle.get("evaluation_policy_hash") if self._bundle else None,
            "best_validation_epoch": self._bundle.get("best_validation_epoch") if self._bundle else None,
            "best_validation_metric": self._bundle.get("best_validation_metric") if self._bundle else None,
        }

    def save_bundle(self, output_dir: Path) -> Dict[str, Any]:
        # train_candidate_seed already persisted checkpoints/preprocessing/
        # training_config atomically as part of fit(); this only adds the
        # common-contract metadata.json alongside for uniform discovery.
        output_dir = Path(output_dir)
        ma.atomic_write_json(output_dir / "metadata.json", self.model_metadata())
        return {"output_dir": str(output_dir), "checkpoint_dir": str(self._run_dir / "checkpoints") if self._run_dir else None}

    @classmethod
    def load_bundle(cls, input_dir: Path) -> "NfAcAdapter":
        run_dir = Path(input_dir)
        candidate_config = json.loads((run_dir / "training_config.json").read_text(encoding="utf-8"))
        architecture = {
            k: candidate_config["architecture"][k]
            for k in ("number_of_blocks", "hidden_width", "hidden_depth")
        }
        track_id = candidate_config["candidate_id"].split("/")[0]
        artifact_root = run_dir.parents[3]  # artifact_root/runs/<track_id>/<model_config_id>/seed_<seed>
        adapter = cls(
            track_id=track_id,
            pdg_value=candidate_config["pdg_value"],
            architecture=architecture,
            execution_policy={
                "minimum_epochs": candidate_config["minimum_epochs"],
                "maximum_epochs": candidate_config["max_epochs"],
                "early_stopping_patience": candidate_config["early_stopping_patience"],
            },
            optimizer_settings={
                "optimizer": candidate_config["optimizer"],
                "learning_rate": candidate_config["learning_rate"],
                "batch_size": candidate_config["batch_size"],
                "weight_decay": candidate_config["weight_decay"],
                "gradient_clipping": candidate_config["gradient_clipping"],
                "dtype": candidate_config["dtype"],
                "device_policy": candidate_config["device_policy"],
            },
            evaluation_policy=candidate_config["evaluation_policy"],
            artifact_root=artifact_root,
            shard_dir=candidate_config.get("shard_dir"),
            train_shard_names=candidate_config.get("train_shards"),
            validation_shard_names=candidate_config.get("validation_shards"),
            test_shard_names=candidate_config.get("test_shards"),
        )
        adapter._candidate_config = candidate_config
        adapter._run_dir = run_dir
        adapter._load_estimator_from_checkpoint(d9ckpt.SCOPE_BEST)
        return adapter
