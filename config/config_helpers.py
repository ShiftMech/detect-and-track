import os
from enum import Enum
from flask import current_app


class Mode(Enum):
    """Mapping of modes (specified during runtime) and their configurations"""
    development = 'config.config.DevelopmentConfig'
    production = 'config.config.ProductionConfig'
    testing = 'config.config.TestingConfig'


def get_mode() -> Mode:
    """
    Returns the intended mode(based on environment variable) in which application will run.

    Returns:
        Aforementioned
    """
    if os.environ['FLASK_ENV'] is None:
        # Case with no environment variable defined, default to development mode
        current_app._get_current_object().logger.info("Warning! Environment variable either not set or incorrect! "
                                                      "It is advisable to set it before proceeding.")
        return Mode.dev.value
    return Mode[os.environ['FLASK_ENV']].value
