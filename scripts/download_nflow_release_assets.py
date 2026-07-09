#!/usr/bin/env python
"""Download the NFlow release assets needed to build small full-MC fixtures.

This script is intentionally a thin operational helper around the GitHub CLI.
It does not parse ROOT/PKL files and it does not make physics claims. Its job is
only to resolve the two release tags and download their assets into deterministic
local directories, with a small manifest for provenance.

Examples
--------
    python scripts/download_nflow_release_assets.py \
        --repo fbientrigo/NFlow \
        --output-dir .tmp/nflow_releases \
        --after-release v1.0-after-muon-shield \
        --scoring-release v1.0-scoring-plane

If release tags are omitted, the script tries a conservative name/tag match:
``after`` + (``shield`` or ``ms``) for the after-muon-shield sample, and
``scoring`` for the scoring-plane sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ReleaseChoice:
    """Resolved release metadata used for provenance."""

    label: str
    tag_name: str
    name: str | None
    published_at: str | None
    selection_reason: str


def _run_gh(args: list[str]) -> str:
    """Run ``gh`` and return stdout, failing with useful stderr context."""
    env = os.environ.copy()
    if not env.get("GH_TOKEN") and env.get("GITHUB_TOKEN"):
        env["GH_TOKEN"] = env["GITHUB_TOKEN"]
    proc = subprocess.run(
        ["gh", *args],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "gh command failed: "
            + " ".join(["gh", *args])
            + f"\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout


def _sha256_file(path: Path) -> str:
    """Compute a SHA-256 hash without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _list_releases(repo: str) -> list[dict[str, object]]:
    """Return non-draft releases from GitHub, newest first."""
    raw = _run_gh(
        [
            "release",
            "list",
            "--repo",
            repo,
            "--limit",
            "100",
            "--json",
            "tagName,name,isDraft,isPrerelease,publishedAt",
        ]
    )
    releases = json.loads(raw)
    return [r for r in releases if not bool(r.get("isDraft"))]


def _text_of_release(release: dict[str, object]) -> str:
    tag = str(release.get("tagName") or "")
    name = str(release.get("name") or "")
    return f"{tag} {name}".casefold()


def _score_after_muon_shield(text: str) -> int:
    """Heuristic score for the after-muon-shield full-MC release."""
    score = 0
    if "after" in text or "post" in text:
        score += 2
    if "muon" in text:
        score += 1
    if "shield" in text or " ms" in f" {text} " or "-ms" in text or "_ms" in text:
        score += 2
    if "scoring" in text:
        score -= 3
    if "full" in text or "monte" in text or "mc" in text:
        score += 1
    return score


def _score_scoring_plane(text: str) -> int:
    """Heuristic score for the scoring-plane full-MC release."""
    score = 0
    if "scoring" in text:
        score += 3
    if "plane" in text:
        score += 2
    if "muon" in text:
        score += 1
    if "shield" in text and "after" in text:
        score -= 3
    if "full" in text or "monte" in text or "mc" in text:
        score += 1
    return score


def _resolve_release(
    releases: list[dict[str, object]],
    *,
    explicit_tag: str | None,
    label: str,
) -> ReleaseChoice:
    """Resolve either an explicit tag or the best-scoring release."""
    if explicit_tag:
        for release in releases:
            if str(release.get("tagName")) == explicit_tag:
                return ReleaseChoice(
                    label=label,
                    tag_name=explicit_tag,
                    name=release.get("name"),
                    published_at=release.get("publishedAt"),
                    selection_reason="explicit tag input",
                )
        known = ", ".join(str(r.get("tagName")) for r in releases)
        raise RuntimeError(f"explicit tag {explicit_tag!r} not found. Known tags: {known}")

    scorer = _score_after_muon_shield if label == "after_muon_shield" else _score_scoring_plane
    scored = []
    for release in releases:
        text = _text_of_release(release)
        scored.append((scorer(text), release))
    scored.sort(key=lambda pair: pair[0], reverse=True)

    if not scored or scored[0][0] <= 0:
        known = "\n".join(
            f"- tag={r.get('tagName')!r}, name={r.get('name')!r}" for r in releases
        )
        raise RuntimeError(
            f"could not auto-resolve {label}. Provide the release tag explicitly.\n"
            f"Available releases:\n{known}"
        )

    best_score, release = scored[0]
    return ReleaseChoice(
        label=label,
        tag_name=str(release.get("tagName")),
        name=release.get("name"),
        published_at=release.get("publishedAt"),
        selection_reason=f"auto-selected by release name/tag score={best_score}",
    )


def _download_release(repo: str, choice: ReleaseChoice, output_dir: Path) -> list[dict[str, object]]:
    """Download all assets for one release and return per-file provenance."""
    release_dir = output_dir / choice.label
    release_dir.mkdir(parents=True, exist_ok=True)
    _run_gh(
        [
            "release",
            "download",
            choice.tag_name,
            "--repo",
            repo,
            "--dir",
            str(release_dir),
            "--clobber",
        ]
    )
    files = []
    for path in sorted(p for p in release_dir.rglob("*") if p.is_file()):
        files.append(
            {
                "path": str(path),
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not files:
        raise RuntimeError(f"release {choice.tag_name!r} downloaded no files")
    return files


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="fbientrigo/NFlow", help="Source repository owner/name.")
    parser.add_argument("--output-dir", required=True, help="Directory for downloaded assets.")
    parser.add_argument("--after-release", default=None, help="Exact tag for the after-muon-shield release.")
    parser.add_argument("--scoring-release", default=None, help="Exact tag for the scoring-plane release.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    releases = _list_releases(args.repo)
    if not releases:
        raise RuntimeError(f"repository {args.repo!r} has no visible releases")

    choices = [
        _resolve_release(releases, explicit_tag=args.after_release, label="after_muon_shield"),
        _resolve_release(releases, explicit_tag=args.scoring_release, label="scoring_plane"),
    ]

    manifest: dict[str, object] = {
        "source_repository": args.repo,
        "downloads": {},
    }
    for choice in choices:
        files = _download_release(args.repo, choice, output_dir)
        manifest[choice.label] = {
            "tag_name": choice.tag_name,
            "name": choice.name,
            "published_at": choice.published_at,
            "selection_reason": choice.selection_reason,
            "files": files,
        }
        manifest["downloads"][choice.label] = str(output_dir / choice.label)

    manifest_path = output_dir / "downloads_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
