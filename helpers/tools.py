import numpy as np
import cv2
import dlib

from imutils import face_utils

from flask import current_app


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


def showtoscreen(box_face, frame):
    app = current_app._get_current_object()

    total_people = {}
    for id in list(box_face.keys()):
        box = box_face[id].get("box")
        if box and len(box_face[id].get("names")) > 0:
            app.logger.info(f"Found names for {id} : {box_face[id].get('names')}")
            text = most_common(box_face[id].get('names'))
        else:
            text = "unknown"

        text = f"Name of the Person : {text}"
        movement = f"Laterally Moving Towards : {most_common(box_face[id].get('move'))}"
        walk = f"Longitudinal Movment : {most_common(box_face[id].get('walk'))}"

        area = area_of(np.array([[box_face[id]['box'][0], box_face[id]['box'][1]]]),
                       np.array([[box_face[id]['box'][2], box_face[id]['box'][3]]]))[0]
        dist = f"Ditance from Camera : {round(12774.13*area**(-0.5134785),1)} cms"
        total_people[id] = text
        app.logger.info(f"Naming {text} for id {id}")

        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 18, 236), 2)
        # Draw a label with a name below the face
        # cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (80, 18, 236), cv2.FILLED)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(frame, text, (x1, y2 + 32), font, 1, (255, 255, 255), 1)
        cv2.putText(frame, f"ID of the Person : {str(id)}", (x1, y2 + 16), font, 1, (255, 241, 3), 1)
        cv2.putText(frame, movement, (x1, y2 + 48), font, 1, (255, 241, 3), 1)
        cv2.putText(frame, dist, (x1, y2 + 64), font, 1, (255, 241, 3), 1)
        cv2.putText(frame, walk, (x1, y2 + 80), font, 1, (255, 241, 3), 1)

        # stats
        text1 = f"Total Number of people: {len(total_people)}"
        people_names = str(list(total_people.values())).replace("[", "").replace("]", "")
        text2 = f"Names of people: {people_names}"
        cv2.rectangle(frame, (0, 0), (350, 100), (80, 18, 236), 2)
        cv2.putText(frame, text1, (2, 12), cv2.FONT_HERSHEY_PLAIN, 1, (88, 255, 5), 1)
        cv2.putText(frame, text2, (2, 25), cv2.FONT_HERSHEY_PLAIN, 1, (88, 255, 5), 1)


def most_common(lst):
    if len(lst) > 0:
        return max(set(lst), key=lst.count)
    else:
        return None





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
        if predbox_area > 5001:
            boxes.append(predbox)
    return np.array(boxes)


def face_finder(globalClass, frame):
    app = current_app._get_current_object()
    face_dict = {}
    for id in list(globalClass.box_face.keys()):
        found = globalClass.box_face[id].get("found")
        box = globalClass.box_face[id].get("box")
        app.logger.info(f"for {id} found {found} and box is {box}")
        if found:
            count = globalClass.box_face[id].get("count")
            app.logger.info(f"Total count for {id} is {count}")
            if count is None or count < 30:
                x1, y1, x2, y2 = box
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                aligned_face = globalClass.fa.align(frame, gray, dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2))
                aligned_face = cv2.resize(aligned_face, (112, 112))

                aligned_face = aligned_face - 127.5
                aligned_face = aligned_face * 0.0078125
                globalClass.box_face[id]["face"] = aligned_face
                face_dict[id] = aligned_face
                app.logger.info(f"Aligned Face for id {id} is {len(aligned_face)}")
                if count is not None:
                    app.logger.info(f"Adding more for the count of ID : {id}")
                    globalClass.box_face[id]["count"] += 1
                else:
                    app.logger.info(f"New Entrant with ID : {id}")
                    globalClass.box_face[id]["count"] = 1
                    globalClass.box_face[id]["names"] = []
            else:
                fixname = most_common(globalClass.box_face[id].get("names"))
                app.logger.info(f"ID number {id} has achieved total count of {count} and Name is {fixname}")
    return face_dict


def face_match(face_dict, globalClass, sess, saved_embeds, names):
    app = current_app._get_current_object()
    app.logger.info(f"Total Length of face dict is {len(face_dict)} with keys {len(face_dict.keys())}")
    # predictions = []
    faces = np.array(list(face_dict.values()))
    feed_dict = {globalClass.images_placeholder: faces, globalClass.phase_train_placeholder: False}
    embeds = sess.run(globalClass.embeddings, feed_dict=feed_dict)
    app.logger.info(f"Shape of face dict is {len(face_dict)} and embeds is {len(embeds)}")
    for i, embedding in enumerate(embeds):
        app.logger.info(f"fetching embedding for ID: {list(face_dict.keys())[i]}")
        diff = np.subtract(saved_embeds, embedding)
        dist = np.sum(np.square(diff), 1)
        idx = np.argmin(dist)
        # logging.info(dist[idx], ":", names[idx])
        if dist[idx] < globalClass.threshold:
            globalClass.box_face[list(face_dict.keys())[i]]["names"].append(names[idx])
        else:
            globalClass.box_face[list(face_dict.keys())[i]]["names"].append("unknown")

#
# def load_models(tf, sess):
#     saver = tf.train.import_meta_graph('models/mfn/m1/mfn.ckpt.meta')
#     saver.restore(sess, 'models/mfn/m1/mfn.ckpt')
#
#     images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
#     embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
#     phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
#     embedding_size = embeddings.get_shape()[1]
#     return images_placeholder, embeddings, phase_train_placeholder


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