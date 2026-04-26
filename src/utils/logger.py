# src/utils/logger.py
import logging
import os
from logging.handlers import RotatingFileHandler
import yaml

def get_logger(name: str) -> logging.Logger:
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    log_dir = cfg["paths"]["logs"]
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Avoid duplicate handlers

    logger.setLevel(cfg["logging"]["level"])

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler (rotating)
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, f"{name}.log"),
        maxBytes=cfg["logging"]["max_bytes"],
        backupCount=cfg["logging"]["backup_count"]
    )
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
