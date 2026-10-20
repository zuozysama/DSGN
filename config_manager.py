# config_manager.py
import json
import os

import yaml
from utils import logger as logger


class ConfigManager:
    """Manage shared configuration state for the project."""
    _instance = None
    _config = {}

    def __new__(cls, *args, **kwargs):
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def load_config(self, file_path):
        """Load a configuration file, which can be in YAML or JSON format.

        Args:
            file_path (str): The path to the configuration file.
        """
        logger.info(f"Process {os.getpid()} Loading configuration from {file_path}")
        if file_path.endswith('.yaml') or file_path.endswith('.yml'):
            self._load_yaml(file_path)
        elif file_path.endswith('.json'):
            self._load_json(file_path)
        else:
            raise ValueError("Unsupported file format. Use .yaml, .yml or .json files.")

    def _load_yaml(self, file_path):
        """Load a YAML configuration file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            self._config.update(config)

    def _load_json(self, file_path):
        """Load a JSON configuration file."""
        with open(file_path, 'r') as file:
            config = json.load(file)
            self._config.update(config)

    def get(self, key, default=None):
        """Get a configuration parameter, supporting nested key access.

        Args:
            key (str): The key to retrieve, using dot notation for nested keys (e.g., 'a.b.c').
            default (Any, optional): The default value to return if the key is not found. Defaults to None.

        Returns:
            Any: The value associated with the key.

        Raises:
            KeyError: If the key is not found in the configuration.
        """
        keys = key.split('.')
        value = self._config
        try:
            for k in keys:
                value = value[k]
            return value
        except KeyError:
            raise KeyError(f"Key '{key}' not found in configuration.")

    def set(self, key, value):
        """Set a configuration parameter, supporting nested key setting.

        Args:
            key (str): The key to set, using dot notation for nested keys (e.g., 'a.b.c').
            value (Any): The value to set for the key.
        """
        keys = key.split('.')
        d = self._config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    def get_config(self):
        """Get the complete configuration dictionary.

        Returns:
            dict: The full configuration dictionary.
        """
        return self._config

    def update_config(self, new_config):
        """Update the configuration dictionary.

        Args:
            new_config (dict): The new configuration data to merge into the existing configuration.
        """
        self._config.update(new_config)

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of ConfigManager.

        Returns:
            ConfigManager: The singleton instance of the ConfigManager class.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
