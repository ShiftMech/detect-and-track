from memory_profiler import profile

@profile()
def run():
    import time
    import cv2
    import dlib
    import numpy as np
    from imutils import face_utils
    import tensorflow as tf
    import onnx
    import onnxruntime as ort
    from onnx_tf.backend import prepare
    from tracker.centroidtracker import CentroidTracker
    from helpers.tools import predict


    start =time.time()
    onnx_path = 'models/ultra_light/ultra_light_models/ultra_light_640.onnx'
    # onnx_model = onnx.load(onnx_path)
    # predictor = prepare(onnx_model)
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name


    shape_predictor = dlib.shape_predictor('models/facial_landmarks/shape_predictor_5_face_landmarks.dat')
    # shape_predictor = dlib.shape_predictor('models/facial_landmarks/shape_predictor_68_face_landmarks.dat')
    fa = face_utils.facealigner.FaceAligner(shape_predictor, desiredFaceWidth=112, desiredLeftEye=(0.3, 0.3))
    threshold = 0.63
    ct = CentroidTracker()
    # load distance

    end =time.time()
    print(f"Time to load setup is {end-start}")
    with tf.Graph().as_default():
        with tf.Session() as sess:
            video_capture = cv2.VideoCapture(0)
            # video_capture = cv2.VideoCapture('/dev/video1')
            a = 0
            while True:
                fps = video_capture.get(cv2.CAP_PROP_FPS)
                ret, frame = video_capture.read()
                if a % 1 == 0:
                    start = time.time()
                    # preprocess faces
                    h, w, _ = frame.shape
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTERim_CUBIC)
                    # img = cv2.resize(img, (640, 480))
                    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                    img = cv2.filter2D(img, -1, kernel)


                    img_mean = np.array([127, 127, 127])
                    # img = np.subtract(img,img_mean) / 128
                    # img = (img - img_mean) / 128
                    img = np.divide(np.subtract(img, img_mean), 128)
                    img = np.transpose(img, [2, 0, 1])
                    img = np.expand_dims(img, axis=0)
                    img = img.astype(np.float32)

                    end = time.time()
                    print(f"Time to pre-process Image is {end - start}")
                    # detect faces
                    start = time.time()
                    confidences, predboxes = ort_session.run(None, {input_name: img})
                    predboxes, labels, probs = predict(w, h, confidences, predboxes, 0.6)

                    faces = []
                    predboxes[predboxes < 0] = 0  # [[297 116 519 401]]
                    boxes = predboxes
                    objects = ct.update(boxes)

                    for i in range(boxes.shape[0]):
                        box = boxes[i, :]

                        # text = f"{predictions[i]}"
                        text = f"RRP"

                        x1, y1, x2, y2 = box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 18, 236), 2)
                        # Draw a label with a name below the face
                        cv2.rectangle(frame, (x1, y2 - 30), (x2, y2), (80, 18, 236), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(frame, text, (x1 + 6, y2 - 6), font, 1.8, (255, 255, 255), 1)

                        # print(f'Frame No. {a} found {text}')

                    for i, (objectID, centroid) in enumerate(objects.items()):
                        cv2.putText(frame, str(objectID), (centroid[0] - 10, centroid[1] - 10), font, 1.8,
                                    (255, 241, 3), 2)
                        cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 241, 3), 2)

                a += 1
                cv2.imshow('Video', frame)

                # Hit 'q' on the keyboard to quit!
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Release handle to the webcam
            video_capture.release()
            cv2.destroyAllWindows()

run()