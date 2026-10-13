import logging
import sys

LOGGER_NAME = "speech_to_text"
LOG_FORMAT = "%(asctime)s - %(processName)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_LEVEL = "WARNING"


def init_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """
    Returns the shared project logger with a console handler attached.

    :param name: Logger name
    :return: Configured logger
    """
    logger = logging.getLogger(name)
    logger.propagate = False
    if not logger.handlers:
        set_logger_params(logger, no_log_file=True)
    return logger


def set_logger_params(logger: logging.Logger,
                      log_file: str = None,
                      log_level=DEFAULT_LOG_LEVEL,
                      no_log_file: bool = False) -> logging.Logger:
    """
    (Re)configures logger level and handlers. Safe to call repeatedly: old handlers are replaced.

    :param logger: Logger returned by init_logger()
    :param log_file: Path to the log file; ignored when no_log_file is True or None
    :param log_level: Level name ("DEBUG", "INFO", ...) or logging int constant
    :param no_log_file: Log to console only
    :return: The same logger
    """
    if isinstance(log_level, str):
        log_level = logging.getLevelName(log_level.upper())
        if not isinstance(log_level, int):
            log_level = logging.getLevelName(DEFAULT_LOG_LEVEL)

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file and not no_log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.setLevel(log_level)
    return logger
