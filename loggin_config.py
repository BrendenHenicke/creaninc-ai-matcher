import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging(log_dir="logs", log_file="app.log"):
    """
    Sets up structured logging with console + rotating file handlers.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, log_file)

    # Formatter for both console and file
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler (rotating to prevent size blowup)
    file_handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=5)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)

    # Root logger setup
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Reduce noise from external libs
    for noisy in ["faiss", "werkzeug", "urllib3", "openai"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.info("✅ Logging initialized — writing to %s", log_path)
