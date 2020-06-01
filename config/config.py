

class Config(object):
    RECOG = True


class ProductionConfig(Config):
    """Config to be used in production environment"""

    def __repr__(self):
        return "Production mode"

    ENV = 'production'
    DEBUG = False
    TESTING = False


class DevelopmentConfig(Config):
    """Config to be used for development environment"""

    def __repr__(self):
        return "Development mode"

    ENV = 'development'
    DEBUG = True
    TESTING = False


class TestingConfig(Config):
    """Config to be used for testing environment"""

    def __repr__(self):
        return "Testing mode"

    ENV = 'testing'
    DEBUG = True
    TESTING = True