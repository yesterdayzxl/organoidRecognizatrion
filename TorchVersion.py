import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort

def letterbox_image(image, new_size):
    """
    Resize the image with unchanged aspect ratio using padding.
    """
    old_size = image.shape[:2]  # old_size is in (height, width) format
    ratio = min(new_size[0] / old_size[0], new_size[1] / old_size[1])
    new_size = tuple([int(x * ratio) for x in old_size])
    image = cv2.resize(image, (new_size[1], new_size[0]))

    delta_w = new_size[1] - new_size[1]
    delta_h = new_size[0] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_img, ratio, (top, left)

def preprocess_image(image, input_size=(640, 640)):
    """
    Preprocess the image to match the input size of the model.
    """
#    img, _, _ = letterbox_image(image, input_size)
#    img = img[:, :, ::-1].transpose((2, 0, 1))  # BGR to RGB, HWC to CHW
#    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size).astype(np.float32)
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    return img

def postprocess_output(output, confidence_threshold=0.5, iou_threshold=0.5):
    """
    Postprocess the output from the model.
    """
    boxes = output[0][0][:4][:8400]
    scores = output[0][0][4][:8400]
    #labels = output[0][]
    masks = output[0][0][5:][:8400]

    # Apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), confidence_threshold, iou_threshold)

    filtered_boxes = []
    filtered_scores = []
    filtered_labels = []
    filtered_masks = []

    for i in indices:
        idx = i[0]
        filtered_boxes.append(boxes[idx])
        filtered_scores.append(scores[idx])
        filtered_labels.append(labels[idx])
        filtered_masks.append(masks[idx])
    return filtered_boxes, filtered_scores, filtered_labels, filtered_masks

def visualize(image, boxes, scores, labels, masks, orig_image_shape, ratio, padding):
    """
    Visualize the detection results.
    """
    top, left = padding
    for box, score, label, mask in zip(boxes, scores, labels, masks):
        box = np.array(box).astype(int)
        box /= ratio
        box[[0, 2]] -= left
        box[[1, 3]] -= top
        box = box.clip(min=0)

        # Draw bounding box
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        # Draw segmentation mask
        mask = (mask > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(mask, (orig_image_shape[1], orig_image_shape[0]))
        image_masked = cv2.bitwise_and(image, image, mask=mask)
        image = cv2.addWeighted(image, 1, image_masked, 0.5, 0)

    return image

# Load the ONNX model
ort_session = ort.InferenceSession("G:\\活细胞成像\\Project\\runs\segment\\train26\\weights\\model_OLD.onnx")

# Load an example image
image_path = 'G:\CellImage\D4 (22).png'
image = cv2.imread(image_path)
orig_image_shape = image.shape[:2]

# Preprocess the image
input_image = preprocess_image(image)
input_image = np.expand_dims(input_image, axis=0)

# Perform inference
outputs = ort_session.run(None, {'images': input_image})

# Postprocess the output
filtered_boxes, filtered_scores, filtered_labels, filtered_masks = postprocess_output(outputs)

# Visualize the results
visualized_image = visualize(image, filtered_boxes, filtered_scores, filtered_labels, filtered_masks, orig_image_shape, 1.0, (0, 0))

# Display the result
cv2.imshow('Detection and Segmentation', visualized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()