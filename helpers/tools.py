import numpy as np
import cv2
from collections import defaultdict
# from helpers.tools import most_common
import dlib
import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare
import tensorflow as tf

def area_of(left_top, right_bottom):
    """
    Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Perform hard non-maximum-supression to filter out boxes with iou greater
    than threshold
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
        picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
    """
    Select boxes that contain human faces
    Args:
        width: original image width
        height: original image height
        confidences (N, 2): confidence array
        boxes (N, 4): boxes array in corner-form
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
        boxes (k, 4): an array of boxes kept
        labels (k): an array of labels for each boxes kept
        probs (k): an array of probabilities for each boxes being in corresponding labels
    """
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
           iou_threshold=iou_threshold,
           top_k=top_k,
           )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

def showtoscreen(box_face,frame):
    for id in list(box_face.keys()):
        box = box_face[id].get("box")
        if box and len(box_face[id].get("names")) > 0:
            print(f"Found names for {id} : {box_face[id].get('names')}")
            text = most_common(box_face[id].get("names"))
        else:
            text = "unknown"
        print(f"Naming {text} for id {id}")

        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 18, 236), 2)
        # Draw a label with a name below the face
        cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (80, 18, 236), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, text, (x1 + 6, y2 - 6), font, 1.8, (255, 255, 255), 1)
        cv2.putText(frame, str(id), (x1 + 56, y2 - 6), font, 1.8, (255, 241, 3), 2)


def most_common(lst):
    return max(set(lst), key=lst.count)


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

        self.images_placeholder= None
        self.embeddings = None
        self.phase_train_placeholder=None
        self.embedding_size=None

    def load_models(self, sess):
        saver = tf.train.import_meta_graph('models/mfn/m1/mfn.ckpt.meta')
        saver.restore(sess, 'models/mfn/m1/mfn.ckpt')

        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        self.embedding_size = self.embeddings.get_shape()[1]


def preprocess(frame):
    h, w, _ = frame.shape
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTERim_CUBIC)
    img = cv2.resize(img, (640, 480))
    img_mean = np.array([127, 127, 127])
    img = (img - img_mean) / 128
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)
    return img, h, w

def check_area(predboxes):
    boxes = []
    for i in range(predboxes.shape[0]):
        predbox = predboxes[i, :]
        x1, y1, x2, y2 = predbox
        predbox_area = area_of(np.array([[x1, y1]]), np.array([[x2, y2]]))
        if predbox_area > 5000:
            boxes.append(predbox)
    return np.array(boxes)


def face_finder(globalClass,frame,fa):
    face_dict = {}
    for id in list(globalClass.box_face.keys()):
        found = globalClass.box_face[id].get("found")
        box = globalClass.box_face[id].get("box")
        print(f"for {id} found {found} and box is {box}")
        if found:
            count = globalClass.box_face[id].get("count")
            print(f"Total count for {id} is {count}")
            if count is None or count < 30:
                x1, y1, x2, y2 = box
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                aligned_face = fa.align(frame, gray, dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2))
                aligned_face = cv2.resize(aligned_face, (112, 112))

                aligned_face = aligned_face - 127.5
                aligned_face = aligned_face * 0.0078125
                globalClass.box_face[id]["face"] = aligned_face
                face_dict[id] = aligned_face
                print(f"Aligned Face for id {id} is {len(aligned_face)}")
                if count is not None:
                    print(f"Adding more for the count of ID : {id}")
                    globalClass.box_face[id]["count"] += 1
                else:
                    print(f"New Entrant with ID : {id}")
                    globalClass.box_face[id]["count"] = 1
                    globalClass.box_face[id]["names"] = []
            else:
                fixname = most_common(globalClass.box_face[id].get("names"))
                print(f"ID number {id} has achieved total count of {count} and Name is {fixname}")
    return face_dict


def face_match(face_dict,globalClass,sess,saved_embeds,names):
    print(f"Total Length of face dict is {len(face_dict)} with keys {len(face_dict.keys())}")
    # predictions = []
    faces = np.array(list(face_dict.values()))
    feed_dict = {globalClass.images_placeholder: faces, globalClass.phase_train_placeholder: False}
    embeds = sess.run(globalClass.embeddings, feed_dict=feed_dict)
    print(f"Shape of face dict is {len(face_dict)} and embeds is {len(embeds)}")
    for i, embedding in enumerate(embeds):
        print(f"fetching embedding for ID: {list(face_dict.keys())[i]}")
        diff = np.subtract(saved_embeds, embedding)
        dist = np.sum(np.square(diff), 1)
        idx = np.argmin(dist)
        # print(dist[idx], ":", names[idx])
        if dist[idx] < globalClass.threshold:
            globalClass.box_face[list(face_dict.keys())[i]]["names"].append(names[idx])
        else:
            globalClass.box_face[list(face_dict.keys())[i]]["names"].append("unknown")