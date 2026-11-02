# green_field_notes

Greenfield split of `ship_adaptive_muon_bg` into two repos (user decision, 2026-11-02, "la propuesta de corte es buenisima y es hacia donde va"). Written while auditing the current tree; nothing here moves or deletes code. Status of each claim is marked VERIFIED / PROJECT DECISION / PROPOSAL / OPEN.

| File | What |
|---|---|
| `01_split_architecture.md` | the two repos, responsibilities, data contract, joint test |
| `02_audit_findings.md` | ponytail audit of the current tree against the split (what to carry, rewrite, drop) |
| `03_lessons_learned.md` | what the thesis work taught us, with evidence; the part not to lose |
| `04_data_contract_sketch.md` | what both sides must agree on; first sketch, to be settled |
| `05_open_questions.md` | decisions that remain with the user |

Rules carried over: AGENT_POLICY (decide before acting, FairShip remote-read-only, no push without an explicit request), no silent semantic changes, keep scientific evidence (do not delete outputs).
- `06_worktree_inventory.md`: inventario de worktrees, qué subir y qué es ruido.
