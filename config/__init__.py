"""Configuration management for detection pipeline."""
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages pipeline configuration from YAML file and environment overrides."""

    def __init__(self, config_path: str = None):
        """Initialize config manager.

        Args:
            config_path: Path to YAML config file. If None, uses default_config.yaml.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "default_config.yaml"
        else:
            config_path = Path(config_path)

        self.config_path = config_path
        self.config = self._load_config()
        self._apply_env_overrides()
        logger.info(f"Configuration loaded from {self.config_path}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config if config else {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")

    def _apply_env_overrides(self):
        """Apply environment variable overrides to config."""
        if "source" not in self.config:
            self.config["source"] = {}
        if "detector" not in self.config:
            self.config["detector"] = {}
        if "visualization" not in self.config:
            self.config["visualization"] = {}
        
        # Source overrides
        if "PIPELINE_SOURCE_TYPE" in os.environ:
            self.config["source"]["type"] = os.environ["PIPELINE_SOURCE_TYPE"]
        if "PIPELINE_VIDEO_PATH" in os.environ:
            self.config["source"]["video_path"] = os.environ["PIPELINE_VIDEO_PATH"]
        if "PIPELINE_NICLA_SERIAL_PORT" in os.environ:
            self.config["source"]["nicla_serial_port"] = os.environ["PIPELINE_NICLA_SERIAL_PORT"]
        if "PIPELINE_NICLA_BAUDRATE" in os.environ:
            try:
                self.config["source"]["nicla_baudrate"] = int(os.environ["PIPELINE_NICLA_BAUDRATE"])
            except ValueError:
                logger.warning("Invalid PIPELINE_NICLA_BAUDRATE, using default")
        if "PIPELINE_NICLA_BOOT_DELAY_SEC" in os.environ:
            try:
                self.config["source"]["nicla_boot_delay_sec"] = int(os.environ["PIPELINE_NICLA_BOOT_DELAY_SEC"])
            except ValueError:
                logger.warning("Invalid PIPELINE_NICLA_BOOT_DELAY_SEC, using default")

        # Detector overrides
        if "PIPELINE_DETECTOR_TYPE" in os.environ:
            self.config["detector"]["type"] = os.environ["PIPELINE_DETECTOR_TYPE"]
        if "PIPELINE_MODEL_PATH" in os.environ:
            self.config["detector"]["model_path"] = os.environ["PIPELINE_MODEL_PATH"]
        if "PIPELINE_CONF_THRESHOLD" in os.environ:
            try:
                self.config["detector"]["confidence_threshold"] = float(
                    os.environ["PIPELINE_CONF_THRESHOLD"]
                )
            except ValueError:
                logger.warning("Invalid PIPELINE_CONF_THRESHOLD, using default")

        # Visualization override
        if "PIPELINE_VISUALIZATION_ENABLED" in os.environ:
            enabled = os.environ["PIPELINE_VISUALIZATION_ENABLED"].lower() in ("true", "1", "yes")
            self.config["visualization"]["enabled"] = enabled

    def get(self, key_path: str, default=None):
        """Get config value using dot notation.

        Args:
            key_path: Dot-separated path, e.g., "detector.confidence_threshold"
            default: Default value if key not found

        Returns:
            Config value or default
        """
        keys = key_path.split(".")
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise KeyError(f"Config key not found: {key_path}")

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire config section.

        Args:
            section: Section name, e.g., "detector", "candidate_scoring"

        Returns:
            Dictionary of config values
        """
        return self.config.get(section, {})

    def __repr__(self) -> str:
        return f"ConfigManager(path={self.config_path})"
