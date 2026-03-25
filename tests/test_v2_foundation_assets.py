from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_v2_roadmap_exists():
    roadmap = REPO_ROOT / "docs" / "airtrace_v2_roadmap.md"
    assert roadmap.exists()


def test_stage0_checklist_exists():
    checklist = REPO_ROOT / "docs" / "stage0_baseline_checklist.md"
    assert checklist.exists()


def test_adr_scaffold_exists():
    adr_root = REPO_ROOT / "docs" / "adr"
    assert (adr_root / "README.md").exists()
    assert (adr_root / "templates" / "adr-template.md").exists()


def test_initial_v2_adr_set_exists():
    adr_root = REPO_ROOT / "docs" / "adr"
    expected = [
        "001-modular-monolith.md",
        "002-application-layer.md",
        "003-timescaledb-for-history.md",
        "004-readonly-api-first.md",
        "005-python-ssr-migration.md",
        "006-feature-flags-for-migration.md",
        "007-v1-deprecation-policy.md",
        "008-provenance-and-confidence-model.md",
        "009-alert-write-paths-after-readonly-v2.md",
    ]
    for name in expected:
        assert (adr_root / name).exists(), name
