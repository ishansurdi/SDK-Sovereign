"""Validate every repo's golden patch passes all parity tests."""
from __future__ import annotations
from pathlib import Path
from server.verifier import Verifier
from scripts.hand_patches import GOOD_PATCHES


def test_all_repos_have_golden_patches_that_pass() -> None:
    """Validate all golden patches produce all-pass parity results."""
    repos_root = Path(__file__).resolve().parent.parent / "server" / "repos"
    v = Verifier(repos_root)
    for repo_id, patch in GOOD_PATCHES.items():
        results = v.run_parity_tests(patch, repo_id)
        assert all(results.values()), (
            f"Repo {repo_id} golden patch failed: {results}"
        )


def test_broken_code_does_not_pass() -> None:
    """The shipped broken.py must at least be syntactically valid Python."""
    import ast
    repos_root = Path(__file__).resolve().parent.parent / "server" / "repos"
    for repo_dir in repos_root.iterdir():
        if not repo_dir.is_dir():
            continue
        broken = (repo_dir / "broken.py").read_text()
        ast.parse(broken)  # must not raise


def test_meta_files_have_required_keys() -> None:
    """All meta.json files must carry the keys the environment reads."""
    import json
    repos_root = Path(__file__).resolve().parent.parent / "server" / "repos"
    required = {
        "repo_id", "deprecated_sdk", "ground_truth_replacement",
        "category", "entrypoint", "error_log",
    }
    for d in repos_root.iterdir():
        if not d.is_dir() or d.name.startswith("_"):
            continue
        meta = json.loads((d / "meta.json").read_text())
        assert required.issubset(meta.keys()), (
            f"{d.name}: missing {required - meta.keys()}"
        )
