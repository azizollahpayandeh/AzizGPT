"""Configuration: config.yaml for everything, .env for keys only.

Model names, provider URLs and the app allowlist live in config.yaml so they can
be swapped without touching code. Keys are read from the environment by name and
are never returned in any string that gets logged or printed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"

REQUIRED_SECTIONS = ("providers", "llm", "apps", "weather", "email", "alarms", "state")

MIN_KEY_LENGTH = 20


def key_problem(key: str | None) -> str | None:
    """Why a key is unusable, phrased so the key itself is never revealed."""
    if not key:
        return "not set"
    if not key.isascii():
        return (
            "contains non-ASCII characters, so it cannot be sent as an HTTP "
            "header - this looks like placeholder text rather than a real key"
        )
    if not key.isprintable():
        return "contains control characters"
    if len(key) < MIN_KEY_LENGTH:
        return f"is only {len(key)} characters long, too short to be an API key"
    return None


class ConfigError(RuntimeError):
    """Raised when config.yaml is missing or structurally wrong."""


class Config:
    """Thin read-only wrapper over the parsed config.yaml."""

    def __init__(self, data: dict[str, Any], path: Path) -> None:
        self._data = data
        self.path = path

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    # ------------------------------------------------------------ providers --
    @property
    def providers(self) -> list[dict[str, Any]]:
        return list(self._data.get("providers", []))

    def enabled_providers(self) -> list[dict[str, Any]]:
        return [p for p in self.providers if p.get("enabled")]

    @staticmethod
    def api_key_for(provider: dict[str, Any]) -> str | None:
        """Look up a provider's key. The value is never logged or printed."""
        env_name = provider.get("api_key_env")
        if not env_name:
            return None
        key = os.getenv(env_name)
        return key.strip() if key else None

    # ----------------------------------------------------------------- apps --
    @property
    def app_keys(self) -> list[str]:
        return sorted(self._data.get("apps", {}))

    def app_argv(self, key: str) -> list[str] | None:
        """Return the exact argv list for an allowlisted key, else None."""
        argv = self._data.get("apps", {}).get(key)
        if isinstance(argv, list) and argv and all(isinstance(a, str) for a in argv):
            return list(argv)
        return None

    # ---------------------------------------------------------------- state --
    def state_dir(self) -> Path:
        raw = self._data.get("state", {}).get("dir", "~/.local/state/azizgpt")
        path = Path(os.path.expanduser(raw))
        path.mkdir(parents=True, exist_ok=True)
        return path


def load_config(path: str | Path | None = None) -> Config:
    """Load .env then config.yaml, with enough validation to fail loudly."""
    # override=True: the spec says keys come from .env only. Without this a
    # stale exported variable in the shell silently shadows the file.
    load_dotenv(ENV_PATH, override=True)

    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise ConfigError(f"config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ConfigError(f"{cfg_path} did not parse to a mapping")

    missing = [s for s in REQUIRED_SECTIONS if s not in data]
    if missing:
        raise ConfigError(f"{cfg_path} is missing section(s): {', '.join(missing)}")

    if not isinstance(data["providers"], list) or not data["providers"]:
        raise ConfigError("config 'providers' must be a non-empty list")

    for i, prov in enumerate(data["providers"]):
        for field in ("name", "base_url", "api_key_env", "model"):
            if not prov.get(field):
                raise ConfigError(f"provider #{i} is missing '{field}'")

    for key, argv in data["apps"].items():
        if not isinstance(argv, list) or not argv:
            raise ConfigError(f"app '{key}' must map to a non-empty argv list")

    return Config(data, cfg_path)
