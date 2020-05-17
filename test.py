import cv2
import dlib
from imutils import face_utils
import tensorflow as tf
import pickle
from tracker.centroidtracker_test import CentroidTracker
from helpers.tools import predict, showtoscreen, globalClass, preprocess, check_area, face_finder, \
    face_match
from collections import defaultdict
# from memory_profiler import profile
from extractor_logging import setup_logging
logger = setup_logging()

shape_predictor_al = dlib.shape_predictor('models/facial_landmarks/shape_predictor_5_face_landmarks.dat')
# shape_predictor = dlib.shape_predictor('models/facial_landmarks/shape_predictor_68_face_landmarks.dat')
fa = face_utils.facealigner.FaceAligner(shape_predictor_al, desiredFaceWidth=112, desiredLeftEye=(0.3, 0.3))

ct = CentroidTracker()
global globalClass
globalClass = globalClass()
# load distance
with open("embeddings/embeddings.pkl", "rb") as f:
    (saved_embeds, names) = pickle.load(f)


def face_points_68(frame, predboxes, shape_predictor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for (i, rect) in enumerate(predboxes):
        left = rect[0]
        top = rect[1]
        right = rect[2]
        bottom = rect[3]
        dlibRect = dlib.rectangle(left, top, right, bottom)
        shape = shape_predictor(gray, dlibRect)
        shape = face_utils.shape_to_np(shape)
        # Draw on our image, all the finded cordinate points (x,y)
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)


def load_models(tf, sess):
    saver = tf.train.import_meta_graph('models/mfn/m1/mfn.ckpt.meta')
    saver.restore(sess, 'models/mfn/m1/mfn.ckpt')

    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]
    return images_placeholder, embeddings, phase_train_placeholder


# def run():
with tf.Graph().as_default():
    with tf.Session() as sess:

        globalClass.load_models(sess)

        video_capture = cv2.VideoCapture('/dev/video1')
        # video_capture = cv2.VideoCapture('/dev/video1')
        globalClass.box_face = defaultdict(lambda: defaultdict(int))
        while True:
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            ret, frame = video_capture.read()
            # frame = cv2.imread("/home/ravirajprajapat/Desktop/11.jpg")
            if globalClass.processframe % 1 == 0 and ret:
                # preprocess faces
                img, h, w = preprocess(frame)

                # detect faces
                confidences, predboxes = globalClass.ort_session.run(None, {globalClass.input_name: img})
                predboxes, labels, probs = predict(w, h, confidences, predboxes, 0.6)

                predboxes[predboxes < 0] = 0  # [[297 116 519 401]]

                boxes = check_area(predboxes)
                boxes = predboxes

                objects = ct.update(boxes)  # OrderedDict([(0, array([227, 123])), (1, array([361, 219]))])
                logger.info(f"Detected total {len(predboxes)} boxes and Objects are {objects}")

                # Track the obj ID with box
                for i, j in objects.items():
                    globalClass.box_face[i]["box"] = j.tolist()
                    if j.tolist() in boxes.tolist():
                        globalClass.box_face[i]["found"] = True
                    else:
                        globalClass.box_face[i]["found"] = False
                # Delete the Untracked Ids
                for i in list(globalClass.box_face.keys()):
                    if i not in objects.keys():
                        del globalClass.box_face[i]

                logger.info(boxes, objects.items())
                logger.info(f"number of objects  coming : {len(objects)} while boxes sent are : {boxes.shape} ")

                face_dict = face_finder(globalClass, frame, fa,logger)

                # face_dict = {} #Only if you want to disable the Recognition
                # face embedding
                if len(face_dict) > 0:
                    face_match(face_dict, globalClass, sess, saved_embeds, names,logger)

                showtoscreen(globalClass.box_face, frame,logger)

            globalClass.processframe += 1
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()


# run()
