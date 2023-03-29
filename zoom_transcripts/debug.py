import os
import sys
import logging
from logging.handlers import RotatingFileHandler


def get_program_name():
    return os.path.basename(sys.argv[0])


def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)

    # Add stdout handler if DEBUG=1
    if os.environ.get("DEBUG") == "1":
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def debug_log(message, mode="info"):
    program_name = get_program_name()
    log_file = f"logs/{program_name}.txt"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = setup_logger(program_name, log_file)

    if mode == "error":
        logger.error(message)
    else:
        logger.info(message)

    # Remove logger handlers to avoid duplicate logs
    logger.handlers.clear()


if __name__ == "__main__":
    debug_log("This is an info message.")
    debug_log("This is an error message.", mode="error")
