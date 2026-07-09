#!/usr/bin/env python
"""Sample small v0 muon-contract fixtures from NFlow full-MC assets.

Outputs gzip-PKL arrays with columns [px, py, pz, x, y, z, id, w]. These files are
for loader/model-development shape and range tests only, not rate estimates.
"""

from __future__ import annotations

import argparse, gzip, hashlib, json, pickle, random, sys
from pathlib import Path
from typing import Iterable

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
from ship_muon_bg.data_contracts import process_pkl, write_artifacts  # noqa: E402

COLUMNS = ("px", "py", "pz", "x", "y", "z", "id", "w")
ALIASES = {
    "px": ("px", "p_x", "momx", "mom_x", "momentumx", "momentum_x", "muonpx", "muon_px"),
    "py": ("py", "p_y", "momy", "mom_y", "momentumy", "momentum_y", "muonpy", "muon_py"),
    "pz": ("pz", "p_z", "momz", "mom_z", "momentumz", "momentum_z", "muonpz", "muon_pz"),
    "x": ("x", "posx", "pos_x", "positionx", "position_x", "muonx", "muon_x", "vx", "v_x"),
    "y": ("y", "posy", "pos_y", "positiony", "position_y", "muony", "muon_y", "vy", "v_y"),
    "z": ("z", "posz", "pos_z", "positionz", "position_z", "muonz", "muon_z", "vz", "v_z"),
    "id": ("id", "pid", "pdgid", "pdg_id", "pdgcode", "pdg_code", "particleid", "particle_id"),
    "w": ("w", "weight", "weights", "eventweight", "event_weight", "muonweight", "muon_weight"),
}
DATA_SUFFIXES = (".root", ".pkl", ".pkl.gz", ".pickle", ".pickle.gz", ".npy", ".npz")


def norm(name: str) -> str:
    return "".join(ch for ch in name.casefold() if ch.isalnum())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(p for p in path.rglob("*") if p.is_file())
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    return sorted({p.resolve() for p in files if p.name.casefold().endswith(DATA_SUFFIXES)})


def matrix_to_contract(a: np.ndarray, source: str):
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] < 6:
        return None
    out = np.ones((a.shape[0], 8), dtype=np.float64)
    n = min(a.shape[1], 8)
    out[:, :n] = np.asarray(a[:, :n], dtype=np.float64)
    filled = []
    if a.shape[1] < 7:
        out[:, 6] = 13.0
        filled.append("id")
    if a.shape[1] < 8:
        out[:, 7] = 1.0
        filled.append("w")
    return out, {"source": source, "layout": f"matrix_first_{n}_columns", "filled_columns": filled}


def iter_numpy_like(path: Path):
    name = path.name.casefold()
    try:
        if name.endswith((".pkl", ".pkl.gz", ".pickle", ".pickle.gz")):
            opener = gzip.open if name.endswith(".gz") else open
            with opener(path, "rb") as f:
                payload = pickle.load(f)
            if isinstance(payload, np.ndarray):
                hit = matrix_to_contract(payload, str(path))
                if hit:
                    yield hit
            if isinstance(payload, dict):
                for key, value in payload.items():
                    if isinstance(value, np.ndarray):
                        hit = matrix_to_contract(value, f"{path}:{key}")
                        if hit:
                            yield hit
        elif name.endswith(".npy"):
            hit = matrix_to_contract(np.load(path, allow_pickle=False), str(path))
            if hit:
                yield hit
        elif name.endswith(".npz"):
            with np.load(path, allow_pickle=False) as z:
                for key in z.files:
                    hit = matrix_to_contract(np.asarray(z[key]), f"{path}:{key}")
                    if hit:
                        yield hit
    except Exception as exc:  # noqa: BLE001
        print(f"warning: skipped {path}: {type(exc).__name__}: {exc}", file=sys.stderr)


def pick_branch(branches: list[str], column: str) -> str | None:
    by_norm = {norm(b): b for b in branches}
    for alias in ALIASES[column]:
        if norm(alias) in by_norm:
            return by_norm[norm(alias)]
    return None


def iter_root(path: Path, step_size: str):
    try:
        import uproot  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("ROOT assets require `pip install uproot`.") from exc
    try:
        with uproot.open(path) as f:
            for tree_name, tree in f.items(recursive=True):
                if not hasattr(tree, "keys") or not hasattr(tree, "iterate"):
                    continue
                branches = [str(k).split(";")[0] for k in tree.keys()]
                selected = {c: pick_branch(branches, c) for c in COLUMNS}
                if any(selected[c] is None for c in ("px", "py", "pz", "x", "y", "z")):
                    continue
                read_branches = [selected[c] for c in COLUMNS if selected[c] is not None]
                for arrays in tree.iterate(read_branches, step_size=step_size, library="np"):
                    n = len(arrays[selected["px"]])
                    cols = [np.asarray(arrays[selected[c]], dtype=np.float64) for c in ("px", "py", "pz", "x", "y", "z")]
                    ids = np.rint(np.asarray(arrays[selected["id"]], dtype=np.float64)) if selected["id"] else np.full(n, 13.0)
                    w = np.asarray(arrays[selected["w"]], dtype=np.float64) if selected["w"] else np.ones(n)
                    out = np.column_stack([*cols, ids, w])
                    yield out, {
                        "source": f"{path}:{tree_name}",
                        "layout": "root_named_branches",
                        "selected_columns": selected,
                        "filled_columns": [c for c in ("id", "w") if selected[c] is None],
                    }
    except Exception as exc:  # noqa: BLE001
        print(f"warning: skipped {path}: {type(exc).__name__}: {exc}", file=sys.stderr)


def iter_chunks(paths: list[Path], step_size: str):
    for path in paths:
        if path.name.casefold().endswith(".root"):
            yield from iter_root(path, step_size)
        else:
            yield from iter_numpy_like(path)


def reservoir(chunks, max_rows: int, seed: int):
    rng = random.Random(seed)
    rows: list[np.ndarray] = []
    n_seen = 0
    sources = []
    for chunk, meta in chunks:
        if chunk.size == 0:
            continue
        sources.append({**meta, "rows_seen": int(chunk.shape[0])})
        for row in chunk:
            n_seen += 1
            if len(rows) < max_rows:
                rows.append(np.asarray(row, dtype=np.float64))
            else:
                j = rng.randrange(n_seen)
                if j < max_rows:
                    rows[j] = np.asarray(row, dtype=np.float64)
    if not rows:
        raise RuntimeError("no compatible muon data found")
    return np.vstack(rows).astype(np.float64, copy=False), sources, n_seen


def write_capped_pkl(array: np.ndarray, path: Path, max_bytes: int) -> np.ndarray:
    current = np.ascontiguousarray(array, dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        with gzip.open(path, "wb", compresslevel=6) as f:
            pickle.dump(current, f, protocol=pickle.HIGHEST_PROTOCOL)
        if path.stat().st_size <= max_bytes or len(current) <= 1:
            return current
        current = current[: max(1, int(len(current) * 0.8))]


def build(label: str, inputs: list[Path], args, seed: int):
    files = find_files(inputs)
    sampled, sources, n_seen = reservoir(iter_chunks(files, args.root_step_size), args.max_rows, seed)
    sampled[:, 3:6] *= args.position_scale
    sampled[:, 5] += args.z_shift_m
    out = args.output_dir / f"{label}.pkl.gz"
    sampled = write_capped_pkl(sampled, out, int(args.max_mb * 1024 * 1024))
    contract_dir = args.output_dir / f"{label}_contract"
    artifacts = process_pkl(out, seed=seed, val_fraction=args.val_fraction)
    written = write_artifacts(artifacts, contract_dir)
    return {
        "label": label,
        "path": str(out),
        "size_bytes": out.stat().st_size,
        "sha256": sha256_file(out),
        "dataset_hash": artifacts["dataset_report"]["dataset_hash"],
        "n_rows_written": int(sampled.shape[0]),
        "n_rows_seen_before_sampling": int(n_seen),
        "input_files": [str(p) for p in files],
        "position_scale_applied": args.position_scale,
        "z_shift_m_applied": args.z_shift_m,
        "sources": sources,
        "contract_artifacts": written,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--after-input", nargs="+", type=Path, required=True)
    p.add_argument("--scoring-input", nargs="+", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("data/fixtures/full_mc"))
    p.add_argument("--max-mb", type=float, default=40.0)
    p.add_argument("--max-rows", type=int, default=250_000)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--position-scale", type=float, default=0.01, help="Default converts legacy cm to m.")
    p.add_argument("--z-shift-m", type=float, default=0.0)
    p.add_argument("--root-step-size", default="100 MB")
    p.add_argument("--val-fraction", type=float, default=0.2)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": "Small full-MC fixtures for shape/range/dev tests only; not for rate estimates.",
        "schema": list(COLUMNS),
        "seed": args.seed,
        "max_mb": args.max_mb,
        "datasets": [
            build("after_muon_shield", args.after_input, args, args.seed),
            build("scoring_plane", args.scoring_input, args, args.seed + 1),
        ],
    }
    manifest_path = args.output_dir / "full_mc_fixtures_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
