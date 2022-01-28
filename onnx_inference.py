import onnxruntime as ort
import os
import argparse
import cv2
import numpy as np
import vision.utils.box_utils_numpy as box_utils




# face detection setting
def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
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
        box_probs = box_utils.hard_nms(box_probs, iou_threshold=iou_threshold, top_k=top_k)
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


if __name__ == "__main__":

    session = ort.InferenceSession("models/onnx/version-RFB-320.onnx")
    input_name = session.get_inputs()[0].name
    


    input_image = cv2.imread("testimage.png")
    resize_image = cv2.resize(input_image, (320, 240))
    cv2.imshow("input", input_image)

    image_mean = np.array([127, 127, 127])
    image = (resize_image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)


    confidences, boxes = session.run(None, {"input": image})
    print(confidences.shape, boxes.shape)

    boxes, labels, probs = predict(input_image.shape[1], input_image.shape[0], confidences, boxes, 0.7)    
    print(boxes.shape, labels.shape, probs.shape)

    print(boxes)


    cv2.waitKey(0)
    cv2.destroyAllWindows()