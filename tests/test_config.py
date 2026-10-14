import shutil
from pathlib import Path

import pytest
import yaml

from core.config import ENV_MAP, REQUIRED_SECTIONS, ConfigError, load_config

REPO_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV_MAP:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "config.yaml"
    shutil.copy(REPO_CONFIG, path)
    return path


def test_repo_config_loads(config_path, tmp_path):
    config = load_config(str(config_path), env_file=str(tmp_path / "missing.env"))
    for section in REQUIRED_SECTIONS:
        assert isinstance(config[section], dict)
    assert "azure_api_key" not in config["llm"]


def test_secrets_merged_from_env_file(config_path, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("AZURE_OPENAI_API_KEY=from-file\nDASHBOARD_USER=admin\n")
    config = load_config(str(config_path), env_file=str(env_file))
    assert config["llm"]["azure_api_key"] == "from-file"
    assert config["dashboard"]["user"] == "admin"


def test_environment_wins_over_env_file(config_path, tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("AZURE_OPENAI_ENDPOINT=https://file.example\n")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://env.example")
    config = load_config(str(config_path), env_file=str(env_file))
    assert config["llm"]["azure_endpoint"] == "https://env.example"


def test_missing_required_env_fails_fast(config_path, tmp_path):
    with pytest.raises(ConfigError, match="AZURE_OPENAI_API_KEY"):
        load_config(str(config_path), env_file=str(tmp_path / "missing.env"),
                    required_env=["AZURE_OPENAI_API_KEY"])


def test_missing_section_fails_fast(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"gas": {}}))
    with pytest.raises(ConfigError, match="vision"):
        load_config(str(path))


def test_missing_file_fails_fast(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(str(tmp_path / "nope.yaml"))
