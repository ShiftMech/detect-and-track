from collections import defaultdict
import dlib
import onnx
import onnxruntime as ort
from imutils import face_utils
from onnx_tf.backend import prepare
import tensorflow as tf
from tracker.centroidtracker_test import CentroidTracker
from flask import current_app

class globalClass:
    """Class, object of  which holds globally required variables"""

    def __init__(self):
        self.box_face = defaultdict(lambda: defaultdict(int))
        onnx_path = 'models/ultra_light/ultra_light_models/ultra_light_640.onnx'
        onnx_model = onnx.load(onnx_path)
        self.predictor = prepare(onnx_model)
        self.ort_session = ort.InferenceSession(onnx_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.processframe = 0
        self.threshold = 0.63

        self.images_placeholder = None
        self.embeddings = None
        self.phase_train_placeholder = None
        self.embedding_size = None

        self.ct = CentroidTracker()

        self.shape_predictor_al = dlib.shape_predictor('models/facial_landmarks/shape_predictor_5_face_landmarks.dat')
        # shape_predictor = dlib.shape_predictor('models/facial_landmarks/shape_predictor_68_face_landmarks.dat')
        self.fa = face_utils.facealigner.FaceAligner(self.shape_predictor_al, desiredFaceWidth=112, desiredLeftEye=(0.3, 0.3))

    def load_models(self, sess):
        saver = tf.train.import_meta_graph('models/mfn/m1/mfn.ckpt.meta')
        saver.restore(sess, 'models/mfn/m1/mfn.ckpt')

        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        self.embedding_size = self.embeddings.get_shape()[1]