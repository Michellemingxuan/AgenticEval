import pytest

from agentic_eval import envfile


def _write(tmp_path, text, name=".env"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_the_exported_environment_beats_the_file(tmp_path, monkeypatch):
    """The opposite of `set -a; . file`, and deliberately so.

    One of these values decides where judging traffic goes. Someone who
    exported LLM_BACKEND meant it; a file silently overruling that could send
    a private-environment run to OpenAI.
    """
    path = _write(tmp_path, "LLM_BACKEND=openai\nCONFIG_PATH=/from/file\n")
    monkeypatch.setenv("LLM_BACKEND", "safechain")
    monkeypatch.delenv("CONFIG_PATH", raising=False)

    loaded, applied = envfile.load(str(path))

    assert loaded == path
    assert applied == 1                              # only the gap was filled
    import os
    assert os.environ["LLM_BACKEND"] == "safechain"  # untouched
    assert os.environ["CONFIG_PATH"] == "/from/file"


def test_shell_isms_in_the_file_are_understood(tmp_path, monkeypatch):
    """It is the same file a shell sources, so it carries shell syntax."""
    path = _write(tmp_path, "\n".join([
        "# a comment",
        "",
        "export LLM_BACKEND=safechain",
        'CONFIG_PATH="/opt/ee/config.yaml"',
        "SAFECHAIN_MODEL='gpt-4.1'",
        "NOT_A_PAIR",
    ]))
    for key in ("LLM_BACKEND", "CONFIG_PATH", "SAFECHAIN_MODEL"):
        monkeypatch.delenv(key, raising=False)

    envfile.load(str(path))

    import os
    assert os.environ["LLM_BACKEND"] == "safechain"
    assert os.environ["CONFIG_PATH"] == "/opt/ee/config.yaml"
    assert os.environ["SAFECHAIN_MODEL"] == "gpt-4.1"
    assert "NOT_A_PAIR" not in os.environ


def test_a_named_file_that_is_missing_is_an_error(tmp_path, monkeypatch):
    """--env-file names a file the caller believes in.

    Continuing without it would fail later, in the judge, with a message about
    configuration rather than about the file that was not there.
    """
    monkeypatch.delenv("AGENTIC_EVAL_ENV", raising=False)
    with pytest.raises(FileNotFoundError, match="--env-file"):
        envfile.load(str(tmp_path / "absent.env"))

    monkeypatch.setenv("AGENTIC_EVAL_ENV", str(tmp_path / "absent.env"))
    with pytest.raises(FileNotFoundError, match="AGENTIC_EVAL_ENV"):
        envfile.load()


def test_the_flag_outranks_the_variable(tmp_path, monkeypatch):
    explicit = _write(tmp_path, "PICKED=flag\n", name="flag.env")
    fallback = _write(tmp_path, "PICKED=variable\n", name="var.env")
    monkeypatch.setenv("AGENTIC_EVAL_ENV", str(fallback))
    monkeypatch.delenv("PICKED", raising=False)

    envfile.load(str(explicit))

    import os
    assert os.environ["PICKED"] == "flag"


def test_no_file_anywhere_is_not_an_error(tmp_path, monkeypatch):
    """Only a file someone NAMED is required. The repo default is optional."""
    monkeypatch.delenv("AGENTIC_EVAL_ENV", raising=False)
    monkeypatch.setattr(envfile, "_ROOT", tmp_path)
    assert envfile.load() is None
