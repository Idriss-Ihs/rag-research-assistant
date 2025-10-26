import logging
from pathlib import Path

def setup_logger(name: str, log_file: str = "logs/project.log", level=logging.INFO):
    Path("logs").mkdir(exist_ok=True)
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

    handler = logging.FileHandler(log_file, encoding="utf-8")
    console = logging.StreamHandler()

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[handler, console]
    )

    return logging.getLogger(name)
