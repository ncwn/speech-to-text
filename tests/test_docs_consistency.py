"""Guards against documentation drifting away from the code.

Every check here was a grep typed by hand during a cleanup that found ten
locations contradicting the code and three different values for the same
measurement. Greps do not run themselves; tests do.

The rule this repo settled on is *one home per fact*: measured numbers live in
`docs/findings.md`, and everything else links to it. These enforce the parts of
that rule a machine can check.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from stt.evidence import UNVERIFIED_MESSAGE, check_evidence

ROOT = Path(__file__).resolve().parents[1]
FINDINGS = ROOT / "docs" / "findings.md"

#: Prose that is allowed to restate a measured number, because a reader needs
#: some figure before deciding to click through.
SUMMARY_FILES = {ROOT / "README.md"}


def _prose_files() -> list[Path]:
    return [ROOT / "README.md", ROOT / "CLAUDE.md", *sorted((ROOT / "docs").glob("*.md"))]


def _slug(heading: str) -> str:
    text = re.sub(r"[^\w\s-]", "", heading.lower().strip())
    return re.sub(r"\s+", "-", text)


def _findings_anchors() -> set[str]:
    headings = re.findall(r"^#{2,3}\s+(.+)$", FINDINGS.read_text(), re.M)
    return {_slug(h) for h in headings}


def _searchable() -> list[Path]:
    return [
        *_prose_files(),
        *sorted((ROOT / "src").rglob("*.py")),
        ROOT / "data" / "reference" / "README.md",
    ]


def test_every_findings_anchor_resolves():
    """A link into findings.md must land on a heading that exists."""
    anchors = _findings_anchors()
    broken = []
    for path in _searchable():
        for anchor in re.findall(r"findings\.md#([\w-]+)", path.read_text()):
            if anchor not in anchors:
                broken.append(f"{path.relative_to(ROOT)} → #{anchor}")
    assert broken == [], f"dangling anchors: {broken}"


def test_every_relative_doc_link_resolves():
    link = re.compile(r"\[[^\]]+\]\((?!https?://|#)([^)#]+)(?:#[^)]*)?\)")
    broken = []
    for path in [*_prose_files(), ROOT / "data" / "reference" / "README.md"]:
        for target in link.findall(path.read_text()):
            if not (path.parent / target).resolve().exists():
                broken.append(f"{path.relative_to(ROOT)} → {target}")
    assert broken == [], f"broken links: {broken}"


def test_generated_evidence_blocks_match_artifacts():
    """Measured blocks are keyed to artifacts, models, corpora, and metrics.

    The current files predate canonical audio identity, so the checked-in block
    must explicitly say that they are unverified.  Once trusted runs replace
    them, this same check compares the generated numeric table byte-for-byte.
    """
    check = check_evidence(ROOT / "evidence" / "manifest.json")
    assert check.matches, "docs/findings.md evidence blocks need `stt evidence --update`"
    if not check.publishable:
        assert UNVERIFIED_MESSAGE in FINDINGS.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "claim",
    [
        "no validated Metal",
        "has no validated MPS",
        "there is no GPU path",
        "so this runs on CPU",
    ],
)
def test_no_stale_device_claims(claim):
    """`omniasr-torch` defaults to Metal; ten places once said otherwise."""
    guilty = [
        str(path.relative_to(ROOT))
        for path in _searchable()
        if path.is_file() and claim.lower() in path.read_text().lower()
    ]
    assert guilty == [], f"{claim!r} contradicts _resolve_device in omniasr_torch.py: {guilty}"


def test_documented_commands_match_the_cli():
    """A command that exists but is undocumented is as bad as the reverse."""
    from stt.cli import app

    registered = {
        (command.name or command.callback.__name__).replace("_", "-")
        for command in app.registered_commands
    }
    documented = set(re.findall(r"`([a-z][a-z-]+)`", (ROOT / "CLAUDE.md").read_text()))
    missing = registered - documented
    assert missing == set(), f"commands absent from CLAUDE.md: {sorted(missing)}"


def test_findings_is_the_only_place_with_a_results_table():
    """Result tables belong in findings.md; the README keeps a short summary."""
    header = re.compile(r"^\|(?:[^|\n]*\|)*\s*\**CER\**\s*\|", re.M)
    offenders = []
    for path in _prose_files():
        if path == FINDINGS or path in SUMMARY_FILES:
            continue
        if header.search(path.read_text()):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == [], f"CER tables outside findings.md: {offenders}"
