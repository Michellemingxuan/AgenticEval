"""The doctor has to survive the environment it exists to diagnose.

These assertions hold in dev, where safechain is absent and most checks fail,
AND in the private environment, where they pass — so the test cannot encode
"safechain is missing" as an expectation. What it pins is the part that must be
true either way: every check is reported, nothing escapes as a traceback, and a
diagnosed problem is a non-zero exit rather than a cheerful one.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_DOCTOR = Path(__file__).resolve().parents[2] / "bin" / "safechain-doctor"


def _run(tmp_path, *args, config_exists=True):
    target = tmp_path / "ee.yaml"
    if config_exists:
        target.write_text("endpoint: https://example.invalid\n", encoding="utf-8")
    env_file = tmp_path / "probe.env"
    env_file.write_text(
        f"LLM_BACKEND=safechain\nCONFIG_PATH={target}\n", encoding="utf-8",
    )
    return subprocess.run(
        [sys.executable, str(_DOCTOR), "--env-file", str(env_file), *args],
        capture_output=True, text=True, timeout=120,
        # A clean environment, so the developer's own exports cannot decide
        # what this test sees.
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )


def test_it_reports_every_stage_without_ever_raising(tmp_path):
    result = _run(tmp_path)
    assert "Traceback" not in result.stderr, result.stderr
    for heading in ("== environment", "== imports", "== ee_config", "== build"):
        assert heading in result.stdout, result.stdout


def test_a_present_config_path_is_reported_as_found(tmp_path):
    """The env file supplies it, so this passes in dev and in the private env."""
    result = _run(tmp_path)
    assert "] CONFIG_PATH  " in result.stdout
    assert "CONFIG_PATH target" not in result.stdout   # the file exists


def test_a_dangling_config_path_is_caught_before_anything_builds(tmp_path):
    """The variable being SET is not the check; the file existing is.

    A path pointing nowhere fails the same way a missing variable does, only
    later and with ee_config's wording instead of ours.
    """
    result = _run(tmp_path, config_exists=False)
    assert "[ FAIL ] CONFIG_PATH target" in result.stdout
    assert result.returncode == 1


def test_a_diagnosed_problem_exits_non_zero(tmp_path):
    """Whatever the environment, a FAIL line and exit 0 must not coexist."""
    result = _run(tmp_path, config_exists=False)
    assert ("FAILED:" in result.stdout) == (result.returncode == 1)
