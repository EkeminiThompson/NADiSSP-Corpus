"""Shared fixtures. Everything here is SYNTHETIC (tests/fixtures.py): it exercises the code, it is not data."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import fixtures  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data import build_corpus, load_corpus  # noqa: E402
from src.training.data import Bundle  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config(str(ROOT / "configs" / "smoke.yaml"))


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    return fixtures.make_all(tmp_path_factory.mktemp("raw"))


@pytest.fixture(scope="session")
def corpus_dir(cfg, raw_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("proc")
    build_corpus(cfg, raw_dir, out)
    return out


@pytest.fixture(scope="session")
def bundle_meta(corpus_dir):
    r = load_corpus(corpus_dir)
    return Bundle(r["corpus"], r["meta"]), r["meta"]


@pytest.fixture(scope="session")
def bundle(bundle_meta):
    return bundle_meta[0]


@pytest.fixture(scope="session")
def meta(bundle_meta):
    return bundle_meta[1]


@pytest.fixture(scope="session")
def full_model(cfg, bundle, meta, tmp_path_factory):
    from src.training.runner import train_variant
    from src.training.variants import VARIANTS
    ck = tmp_path_factory.mktemp("ckpt")
    return train_variant(cfg, bundle, meta, VARIANTS["full_nadip"], 0, ckpt_root=ck, log=lambda *a: None), ck
