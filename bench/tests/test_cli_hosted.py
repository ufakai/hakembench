"""The harness's hosted routes: the parser takes them, and the key and address come from the
environment."""

import json

import pytest

from bench.harness import cli


def parsed(monkeypatch, argv):
    """The namespace `cli.main` builds from argv, without running the command."""
    import argparse

    seen = {}
    original = argparse.ArgumentParser.parse_args

    def capture(self, args=None, namespace=None):
        out = original(self, args, namespace)
        out.func = lambda a: seen.setdefault("args", a) and 0
        return out

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    cli.main(argv)
    return seen["args"]


def test_the_run_parser_takes_the_hosted_routes(monkeypatch, tmp_path):
    argv = ["run", "--adapter", "hosted-stated", "--items", "x.jsonl", "--out",
            str(tmp_path / "o.jsonl"), "--model", "m", "--extra", '{"reasoning_effort": "low"}',
            "--max-tokens", "900"]  # fmt: skip
    args = parsed(monkeypatch, argv)
    assert (args.adapter, args.max_tokens) == ("hosted-stated", 900)
    assert json.loads(args.extra) == {"reasoning_effort": "low"}


def test_the_key_comes_from_the_environment_or_a_file(monkeypatch, tmp_path):
    for name in (cli.API_KEY_ENV, cli.API_KEY_FILE_ENV, cli.API_URL_ENV):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit, match=cli.API_KEY_ENV):
        cli.api_key()
    with pytest.raises(SystemExit, match=cli.API_URL_ENV):
        cli.api_url()
    key_file = tmp_path / "key"
    key_file.write_text("from-file\n")
    monkeypatch.setenv(cli.API_KEY_FILE_ENV, str(key_file))
    assert cli.api_key() == "from-file"
    monkeypatch.setenv(cli.API_KEY_ENV, "from-env")
    assert cli.api_key() == "from-env"
    monkeypatch.setenv(cli.API_URL_ENV, "https://api.test/v1/")
    assert cli.api_url() == "https://api.test/v1"
