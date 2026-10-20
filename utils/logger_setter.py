import logging
import os
import platform
from datetime import datetime


def setup_logger(name, log_file, level=logging.DEBUG):
    """
    Set up a logger

    Args:
        name (str): Name of the logger
        log_file (str): Path to the log file
        level (int): Logging level

    Returns:
        logging.Logger: Configured logger instance
    """
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:  # Ensure handlers are not added repeatedly
        # Set up file handler with UTF-8 encoding
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def create_log_dir(log_file_path):
    """
    Ensure the log directory exists

    Args:
        log_file_path (str): Path to the log file
    """
    log_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except OSError as e:
            raise EnvironmentError(f"Failed to create log directory: {e}")


def get_log_file_path(base_name):
    """
    Get the log file path, using the parent directory of the current script as the project root, and generate a timestamped log filename

    Args:
        base_name (str): Base name for the log file

    Returns:
        str: Path to the log file
    """
    # Generate a timestamped log filename
    log_filename = datetime.now().strftime(f"{base_name}_%Y-%m-%d-%H-%M-%S.log")

    # Get the project root directory, which is the parent directory of the logger_setter.py file
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Create the logs folder path under the project root directory
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)  # Ensure the logs folder exists

    # Combine to form the full log file path
    log_file_path = os.path.join(log_dir, log_filename)

    return log_file_path


# Create log directory
log_file_path_train = get_log_file_path("log")
create_log_dir(log_file_path_train)


try:
    logger = setup_logger('logger', log_file_path_train)
except (OSError, IOError) as e:
    raise EnvironmentError(f"UNABLE TO SET LOGGER: {e}")

logger.info(f"Process {os.getpid()} Train logger is successfully set up.")
