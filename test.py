import cv2
import tensorflow as tf
import pickle
from helpers.tools import predict, showtoscreen, preprocess, check_area, face_finder, \
    face_match
from collections import defaultdict
from logs.extractor_logging import setup_logging
from global_variable.globals import globalClass

logger = setup_logging()
global globalClass
globalClass = globalClass()

recog = True
if recog:
    # load distance
    with open("embeddings/embeddings.pkl", "rb") as f:
        (saved_embeds, names) = pickle.load(f)


def run():
    with tf.Graph().as_default():
        with tf.Session() as sess:
            if recog:
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

                    # detect faces
                    confidences, predboxes = globalClass.ort_session.run(None, {globalClass.input_name: img})
                    predboxes, labels, probs = predict(w, h, confidences, predboxes, 0.6)

                    predboxes[predboxes < 0] = 0  # [[297 116 519 401]]

                    boxes = check_area(predboxes)
                    # boxes = predboxes

                    objects = globalClass.ct.update(boxes)  # OrderedDict([(8, array([450,  53, 503, 401], dtype=int32))])
                    print(f"Detected total {len(predboxes)} boxes and Objects are {objects}")

                    # Track the obj ID with box
                    for i, j in objects.items():
                        globalClass.box_face[i]["box"] = j['rect'].tolist()
                        if globalClass.box_face[i]['move'] == 0:
                            globalClass.box_face[i]["move"] = [j['move']]
                            globalClass.box_face[i]["walk"] = [j['walk']]
                        elif len(globalClass.box_face[i]['move']) > 6:
                            globalClass.box_face[i]["move"].pop(0)
                            globalClass.box_face[i]["move"].append(j['move'])
                            globalClass.box_face[i]["walk"].pop(0)
                            globalClass.box_face[i]["walk"].append(j['walk'])
                        else:
                            globalClass.box_face[i]["move"].append(j['move'])
                            globalClass.box_face[i]["walk"].append(j['walk'])

                        if j['rect'].tolist() in boxes.tolist():
                            globalClass.box_face[i]["found"] = True
                        else:
                            globalClass.box_face[i]["found"] = False
                    # Delete the Untracked Ids
                    for i in list(globalClass.box_face.keys()):
                        if i not in objects.keys():
                            del globalClass.box_face[i]

                    print(boxes, objects.items())
                    print(f"number of objects  coming : {len(objects)} while boxes sent are : {boxes.shape} ")

                    face_dict = face_finder(globalClass, frame)

                    # face_dict = {} #Only if you want to disable the Recognition
                    # face embedding
                    if len(face_dict) and recog > 0:
                        face_match(face_dict, globalClass, sess, saved_embeds, names)

                    showtoscreen(globalClass.box_face, frame)

                globalClass.processframe += 1

                cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
                cv2.imshow('Video', frame)

                # Hit 'q' on the keyboard to quit!
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


run()
