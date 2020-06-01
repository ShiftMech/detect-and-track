from collections import defaultdict,deque
from time import time

import cv2
import tensorflow as tf
from helpers.tools import preprocess, predict, check_area, face_finder, face_match, showtoscreen
from flask import current_app


def start_service(globalClass, saved_embeds, names):
    app = current_app._get_current_object()
    with tf.Graph().as_default():
        with tf.Session() as sess:
            if app.config['RECOG']:
                globalClass.load_models(sess)

            video_capture = cv2.VideoCapture('/dev/video1')
            # video_capture = cv2.VideoCapture('/dev/video1')
            globalClass.box_face = defaultdict(lambda: defaultdict(int))
            while True:
                fps = video_capture.get(cv2.CAP_PROP_FPS)
                ret, frame = video_capture.read()
                # ret = True
                # frame = cv2.imread(f"/home/ravirajprajapat/Desktop/30.jpg")
                if globalClass.processframe % 1 == 0 and ret:
                    # preprocess faces
                    img, h, w = preprocess(frame)

                    start = time()
                    # detect faces
                    confidences, predboxes = globalClass.ort_session.run(None, {globalClass.input_name: img})
                    predboxes, labels, probs = predict(w, h, confidences, predboxes, 0.6)

                    predboxes[predboxes < 0] = 0  # [[297 116 519 401]]

                    boxes = check_area(predboxes)
                    # boxes = predboxes
                    end = time()
                    app.logger.info(f"Total time at 5 ------------------- : {end - start}")

                    start = time()
                    objects = globalClass.ct.update(
                        boxes)  # OrderedDict([(8, array([450,  53, 503, 401], dtype=int32))])
                    app.logger.info(f"Detected total {len(predboxes)} boxes and Objects are {objects}")
                    end = time()
                    app.logger.info(f"Total time at 6 ------------------- : {end - start}")

                    # start = time()
                    # Track the obj ID with box
                    for i, j in objects.items():
                        start = time()
                        globalClass.box_face[i]["box"] = j['rect'].tolist()
                        if globalClass.box_face[i]['move'] == 0:
                            globalClass.box_face[i]["move"] = deque(j['move'])
                            globalClass.box_face[i]["walk"] = deque(j['walk'])
                        elif len(globalClass.box_face[i]['move']) > 6:
                            globalClass.box_face[i]["move"].popleft()
                            globalClass.box_face[i]["move"].append(j['move'])
                            globalClass.box_face[i]["walk"].popleft()
                            globalClass.box_face[i]["walk"].append(j['walk'])
                        else:
                            globalClass.box_face[i]["move"].append(j['move'])
                            globalClass.box_face[i]["walk"].append(j['walk'])

                        end = time()
                        app.logger.info(f"Total time at 71 ------------------- : {end - start}")

                        start = time()
                        if j['rect'].tolist() in boxes.tolist():
                            globalClass.box_face[i]["found"] = True
                        else:
                            globalClass.box_face[i]["found"] = False

                        end = time()
                        app.logger.info(f"Total time at 72 ------------------- : {end - start}")
                    # Delete the Untracked Ids
                    for i in list(globalClass.box_face.keys()):
                        if i not in objects.keys():
                            del globalClass.box_face[i]



                    app.logger.info(f"{boxes}, {objects.items()}")
                    app.logger.info(f"number of objects  coming : {len(objects)} while boxes sent are : {boxes.shape} ")

                    start = time()
                    face_dict = face_finder(globalClass, frame)
                    end = time()
                    app.logger.info(f"Total time at 8 ------------------- : {end - start}")

                    # face_dict = {} #Only if you want to disable the Recognition
                    # face embedding
                    start = time()
                    if len(face_dict) and app.config['RECOG'] > 0:
                        face_match(face_dict, globalClass, sess, saved_embeds, names)
                    end = time()
                    app.logger.info(f"Total time at 9 ------------------- : {end - start}")

                    start = time()

                    showtoscreen(globalClass.box_face, frame)

                    end = time()
                    app.logger.info(f"Total time at 10 ------------------- : {end - start}")

                globalClass.processframe += 1

                cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
                cv2.imshow('Video', frame)

                # Hit 'q' on the keyboard to quit!
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Release handle to the webcam
            video_capture.release()
            cv2.destroyAllWindows()
