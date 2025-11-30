from pathlib import Path


def test_repo_structure_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / "src").exists()
    assert (root / "data").exists()
    assert (root / "tests").exists()
