from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTECTED_DOCS = [
    REPO_ROOT / "docs" / "airtrace_v2_roadmap.md",
    REPO_ROOT / "docs" / "stage0_baseline_checklist.md",
    REPO_ROOT / "docs" / "adr" / "README.md",
    REPO_ROOT / "docs" / "adr" / "templates" / "adr-template.md",
    REPO_ROOT / "docs" / "adr" / "001-modular-monolith.md",
    REPO_ROOT / "docs" / "adr" / "002-application-layer.md",
    REPO_ROOT / "docs" / "adr" / "003-timescaledb-for-history.md",
    REPO_ROOT / "docs" / "adr" / "004-readonly-api-first.md",
    REPO_ROOT / "docs" / "adr" / "005-python-ssr-migration.md",
    REPO_ROOT / "docs" / "adr" / "006-feature-flags-for-migration.md",
    REPO_ROOT / "docs" / "adr" / "007-v1-deprecation-policy.md",
    REPO_ROOT / "docs" / "adr" / "008-provenance-and-confidence-model.md",
    REPO_ROOT / "docs" / "adr" / "009-alert-write-paths-after-readonly-v2.md",
]

MOJIBAKE_MARKERS = ("Ð", "Ñ", "Ã", "â€™", "â€œ", "â€", "\ufffd")


def test_v2_docs_are_valid_utf8_without_bom():
    for path in PROTECTED_DOCS:
        content = path.read_text(encoding="utf-8")
        assert not content.startswith("\ufeff"), path


def test_v2_docs_do_not_contain_common_mojibake_markers():
    for path in PROTECTED_DOCS:
        content = path.read_text(encoding="utf-8")
        for marker in MOJIBAKE_MARKERS:
            assert marker not in content, f"{path} contains marker {marker!r}"
