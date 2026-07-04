# Legacy migration log

Every component migrated or relocated from a legacy source is recorded here
(required by the README).

| source repo | source file | destination file | change | reason | validation |
| --- | --- | --- | --- | --- | --- |
| mferril/NFlow (via this fork) | `deepflow.py` | `Nflow/legacy/deepflow.py` | moved; imports rewritten `utils.*` → `Nflow.legacy.utils.*` | quarantine untested fork code under module 2 | `py_compile`; not run (needs torch + HDF5 data) |
| mferril/NFlow (via this fork) | `config.yaml` | `Nflow/legacy/config.yaml` | moved unchanged | keep the legacy trainer's config next to it | n/a (data file) |
| mferril/NFlow (via this fork) | `utils/config.py` | `Nflow/legacy/utils/config.py` | moved unchanged | quarantine | `py_compile` |
| mferril/NFlow (via this fork) | `utils/data_handling.py` | `Nflow/legacy/utils/data_handling.py` | moved unchanged | quarantine (HDF5-era, untested) | `py_compile` |
| mferril/NFlow (via this fork) | `utils/flow_models.py` | `Nflow/legacy/utils/flow_models.py` | moved unchanged | quarantine; RealNVP is a promotion candidate once tested (skeleton doc §5) | `py_compile` |
| mferril/NFlow (via this fork) | `utils/logging_config.py` | `Nflow/legacy/utils/logging_config.py` | moved unchanged | quarantine | `py_compile` |
| mferril/NFlow (via this fork) | `utils/plotting.py` | `Nflow/legacy/utils/plotting.py` | moved; import rewritten `utils.data_handling` → `Nflow.legacy.utils.data_handling` | quarantine | `py_compile` |
| mferril/NFlow (via this fork) | `utils/run_management.py` | `Nflow/legacy/utils/run_management.py` | moved unchanged | quarantine | `py_compile` |
| mferril/NFlow (via this fork) | `utils/training.py` | `Nflow/legacy/utils/training.py` | moved unchanged | quarantine | `py_compile` |

Promotion rule: nothing leaves `Nflow/legacy/` without being adapted to the
`(N, 8)` PKL contract and gaining tests in the same commit
(`docs/architecture/ml_skeleton_local_pkl_v0.md`, §2 and §5).
