import cv2
import dlib
import numpy as np
from imutils import face_utils
import tensorflow as tf
import pickle
import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare
from tracker.centroidtracker_test import CentroidTracker
from helpers.tools import area_of, predict

onnx_path = 'models/ultra_light/ultra_light_models/ultra_light_640.onnx'
onnx_model = onnx.load(onnx_path)
predictor = prepare(onnx_model)
ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name

shape_predictor_al = dlib.shape_predictor('models/facial_landmarks/shape_predictor_5_face_landmarks.dat')
# shape_predictor = dlib.shape_predictor('models/facial_landmarks/shape_predictor_68_face_landmarks.dat')
fa = face_utils.facealigner.FaceAligner(shape_predictor_al, desiredFaceWidth=112, desiredLeftEye=(0.3, 0.3))
threshold = 0.63
ct = CentroidTracker()
# load distance
with open("embeddings/embeddings.pkl", "rb") as f:
    (saved_embeds, names) = pickle.load(f)


def load_models(tf, sess):
    saver = tf.train.import_meta_graph('models/mfn/m1/mfn.ckpt.meta')
    saver.restore(sess, 'models/mfn/m1/mfn.ckpt')

    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]
    return images_placeholder, embeddings, phase_train_placeholder


with tf.Graph().as_default():
    with tf.Session() as sess:
        images_placeholder, embeddings, phase_train_placeholder = load_models(tf, sess)

        video_capture = cv2.VideoCapture(0)
        # video_capture = cv2.VideoCapture('/dev/video1')
        box_face = {}
        processframe = 0
        while True:
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            ret, frame = video_capture.read()
            # frame = cv2.imread("/home/ravirajprajapat/Desktop/11.jpg")
            if processframe % 1 == 0:
                # preprocess faces
                h, w, _ = frame.shape
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTERim_CUBIC)
                img = cv2.resize(img, (640, 480))
                # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                # img = cv2.filter2D(img, -1, kernel)

                img_mean = np.array([127, 127, 127])
                img = (img - img_mean) / 128
                img = np.transpose(img, [2, 0, 1])
                img = np.expand_dims(img, axis=0)
                img = img.astype(np.float32)

                # detect faces
                confidences, predboxes = ort_session.run(None, {input_name: img})
                predboxes, labels, probs = predict(w, h, confidences, predboxes, 0.6)
                print(f"Detected total {len(predboxes)} boxes")

                faces = []
                predboxes[predboxes < 0] = 0  # [[297 116 519 401]]

                ## Check the Area of the box
                # boxes = []
                # for i in range(predboxes.shape[0]):
                #     predbox = predboxes[i, :]
                #     x1, y1, x2, y2 = predbox
                #     predbox_area = area_of(np.array([[x1, y1]]), np.array([[x2, y2]]))
                #     if predbox_area > 5000:
                #         boxes.append(predbox)
                # boxes = np.array(boxes)
                boxes = predboxes

                objects = ct.update(boxes)  # OrderedDict([(0, array([227, 123])), (1, array([361, 219]))])

                # Track the obj ID with box
                for i, j in objects.items():
                    if j.tolist() in boxes.tolist():
                        index = boxes.tolist().index(j.tolist())
                        box_face[i] = {"box": j.tolist}
                # Delete the Untracked Ids
                for i,j in box_face.items():
                    if i not in objects.keys():
                        del box_face[i]

                print(boxes, objects.items())
                if len(objects) != boxes.shape[0]:
                    print(f"number of objects  coming : {len(objects)} while boxes sent are : {boxes.shape} ")
                    print(f"{boxes}")
                    # pass

                for i in range(boxes.shape[0]):
                    box = boxes[i, :]
                    x1, y1, x2, y2 = box

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    aligned_face = fa.align(frame, gray, dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2))
                    aligned_face = cv2.resize(aligned_face, (112, 112))

                    aligned_face = aligned_face - 127.5
                    aligned_face = aligned_face * 0.0078125
                    # box_face[i]['face'] = aligned_face
                    # box_face[i]['box'] = box

                    faces.append(aligned_face)

                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # for (i, rect) in enumerate(predboxes):
                #     left = rect[0]
                #     top = rect[1]
                #     right = rect[2]
                #     bottom = rect[3]
                #     dlibRect = dlib.rectangle(left, top, right, bottom)
                #     shape = shape_predictor(gray, dlibRect)
                #     shape = face_utils.shape_to_np(shape)

                    # Draw on our image, all the finded cordinate points (x,y)
                    # for (x, y) in shape:
                    #     cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                # face embedding
                if len(faces) > 0:
                    predictions = []

                    faces = np.array(faces)
                    feed_dict = {images_placeholder: faces, phase_train_placeholder: False}

                    embeds = sess.run(embeddings, feed_dict=feed_dict)

                    for embedding in embeds:
                        diff = np.subtract(saved_embeds, embedding)
                        dist = np.sum(np.square(diff), 1)
                        idx = np.argmin(dist)
                        print(dist[idx], ":", names[idx])
                        if dist[idx] < threshold:
                            predictions.append(names[idx])
                        else:
                            predictions.append("unknown")

                    for i in range(boxes.shape[0]):
                        box = boxes[i, :]

                        text = f"{predictions[i]}"
                        # text = f"RRP"

                        x1, y1, x2, y2 = box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 18, 236), 2)
                        # Draw a label with a name below the face
                        cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (80, 18, 236), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(frame, text, (x1 + 6, y2 - 6), font, 1.8, (255, 255, 255), 1)

                        # print(f'Frame No. {a} found {text}')

                    if len(objects) > 0:
                        for i, (objectID, centroid) in enumerate(objects.items()):
                            cv2.putText(frame, str(objectID), (centroid[0] - 10, centroid[1] - 10), font, 1.8,
                                        (255, 241, 3), 2)
                            # cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 241, 3), 2)

            processframe += 1
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

