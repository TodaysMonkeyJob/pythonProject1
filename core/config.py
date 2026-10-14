"""
Configuration loader: ``config.yaml`` merged with secrets from the environment / ``.env``.

Secrets never live in config.yaml. They are read from environment variables (``.env`` is loaded
first without overriding variables that are already set) and placed into the config tree via ENV_MAP.
"""
import os
from typing import Any, Dict, Iterable

import yaml
from dotenv import load_dotenv

REQUIRED_SECTIONS = ("vision", "actions", "identity", "audio", "gas", "aggregator", "llm", "dashboard")

# env var -> (section, key) in the merged config
ENV_MAP = {
    "AZURE_OPENAI_ENDPOINT": ("llm", "azure_endpoint"),
    "AZURE_OPENAI_API_KEY": ("llm", "azure_api_key"),
    "AZURE_OPENAI_API_VERSION": ("llm", "azure_api_version"),
    "DASHBOARD_USER": ("dashboard", "user"),
    "DASHBOARD_PASS": ("dashboard", "password"),
}


class ConfigError(RuntimeError):
    """Raised when the configuration is missing or incomplete."""


def load_config(path: str = "config.yaml",
                env_file: str = ".env",
                required_env: Iterable[str] = ()) -> Dict[str, Any]:
    """
    Loads config.yaml and merges secrets from the environment.

    :param path: Path to the YAML config
    :param env_file: Path to the .env file; silently skipped if it does not exist
    :param required_env: Env var names (keys of ENV_MAP) that must be set, e.g. for the modules being started
    :return: Config as nested dicts
    """
    if not os.path.isfile(path):
        raise ConfigError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")

    missing_sections = [s for s in REQUIRED_SECTIONS if not isinstance(config.get(s), dict)]
    if missing_sections:
        raise ConfigError(f"{path} is missing sections: {', '.join(missing_sections)}")

    if env_file and os.path.isfile(env_file):
        load_dotenv(env_file, override=False)

    for env_name, (section, key) in ENV_MAP.items():
        value = os.environ.get(env_name)
        if value:
            config[section][key] = value

    missing_env = [name for name in required_env if not os.environ.get(name)]
    if missing_env:
        raise ConfigError(f"Missing required environment variables: {', '.join(missing_env)} "
                          f"(set them in {env_file} or the environment; see .env.example)")
    return config
