from flask import Flask
import pickle
import tensorflow as tf
from global_variable.globals import globalClass
from logs.extractor_logging import setup_logging
from config.config_helpers import get_mode
from service.main_service import start_service

setup_logging()
app = Flask(__name__, instance_relative_config = True)
global globalClass

with app.app_context():
    # Load config
    MODE = get_mode()
    app.logger.info("Running in {} mode".format(MODE))
    app.config.from_object(MODE)

    globalClass = globalClass()

    saved_embeds, names = None, None
    if app.config['RECOG']:
        # load distance
        with open("embeddings/embeddings.pkl", "rb") as f:
            (saved_embeds, names) = pickle.load(f)

    start_service(globalClass, saved_embeds, names)
