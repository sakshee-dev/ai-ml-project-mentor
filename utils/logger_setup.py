from __future__ import annotations

import json
import logging
import logging.config
from pathlib import Path


def setup_logging(
    project_root: str | Path,
    config_path: str | Path,
    logger_name: str = "log_console_file",
) -> logging.Logger:
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()

    log_dir = project_root / "log"
    log_dir.mkdir(parents=True, exist_ok=True)

    if not config_path.exists():
        raise FileNotFoundError(f"Logger config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Keep your desired filename: log/log.txt
    if "handlers" in config and "file_handler" in config["handlers"]:
        config["handlers"]["file_handler"]["filename"] = str((project_root / "log" / "log.txt").resolve())

    logging.config.dictConfig(config)
    logger = logging.getLogger(logger_name)
    logger.info("Logging configured successfully in current process")
    logger.info("Project root: %s", project_root)
    logger.info("Logger config path: %s", config_path)
    logger.info("Log file path: %s", project_root / "log" / "log.txt")
    return logger