import onnxruntime as ort
import cv2
import numpy as np

session = ort.InferenceSession(r"G:\Projects\CellImaging\cellImaging\Project\runs\segment\\train72\\weights\\best.onnx", providers=["CUDAExecutionProvider"])

for input in session.get_inputs():
    print("input name: ", input.name)
    print("input shape: ", input.shape)
    print("input type: ", input.type)

for output in session.get_outputs():
    print("output name: ", output.name)
    print("output shape: ", output.shape)
    print("output type: ", output.type)




def prepare_input(bgr_image, width, height):
    image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height)).astype(np.float32)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    input_tensor = np.expand_dims(image, axis=0)
    return input_tensor

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_mask(row, box, img_width, img_height):
    mask0 = row.reshape(160, 160)
    x10, y10, x20, y20 = box
    # box坐标是相对于原始图像大小，需转换到相对于160*160的大小
    mask_x1 = round(x10 / img_width * 160)
    mask_y1 = round(y10 / img_height * 160)
    mask_x2 = round(x20 / img_width * 160)
    mask_y2 = round(y20 / img_height * 160)
    mask0 = mask0[mask_y1:mask_y2, mask_x1:mask_x2]
    mask0 = sigmoid(mask0)
    # 把mask的尺寸调整到相对于原始图像大小
    mask0 = cv2.resize(mask0, (round(x20 - x10), round(y20 - y10)))
    mask0 = (mask0 > 0.5).astype("uint8") * 255
    return mask0


if __name__ == '__main__':
    image = cv2.imread(r"G:\Picture02\chip2-res\\47D3.png")
    image_height, image_width, _ = image.shape
    input_tensor = prepare_input(image, 1440, 1440)
    print("input_tensor shape: ", input_tensor.shape)
    outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
    output0 = np.squeeze(outputs[0])
    print("output0 shape:", output0.shape)

    boxes = output0[:, 0:5]
    masks = output0[:, 5:]
    print("boxes shape:", boxes.shape)
    print("masks shape:", masks.shape)
    detections = output0
    print("detection shape:", detections.shape)
    objects = []
    for row in detections:
        prob = row[4]
        if prob < 0.5:
            continue
        class_id = row[4].argmax()
        #label = COCO_CLASSES[class_id]
        xc, yc, w, h = row[:4]
        #把x1, y1, x2, y2的坐标恢复到原始图像坐标
        x1 = (xc - w / 2) / 640 * image_width
        y1 = (yc - h / 2) / 640 * image_height
        x2 = (xc + w / 2) / 640 * image_width
        y2 = (yc + h / 2) / 640 * image_height
        # 获取实例分割mask
        mask = get_mask(row[8:25608], (x1, y1, x2, y2), image_width, image_height)
        # 从mask中提取轮廓
        #polygon = get_polygon(mask, x1, y1)
        #objects.append([x1, y1, x2, y2, label, prob, polygon, mask])

    # NMS
    #objects.sort(key=lambda x: x[5], reverse=True)
    #results = []
    #while len(objects) > 0:
    #    results.append(objects[0])
    #    objects = [object for object in objects if iou(object, objects[0]) < 0.5]





