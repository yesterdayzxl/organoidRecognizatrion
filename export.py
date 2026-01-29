from ultralytics import YOLO
import onnx
import onnxruntime as ort
import numpy as np


# Load the YOLOv8 model
model = YOLO(r"G:\Projects\CellImaging\cellImaging\Project\runs\\segment\\train72\\weights\\best.pt")



# Export the model to ONNX format
model.export(format="onnx", nms=True)

# 检查 onnx 计算图
onnx_model = onnx.load(r"G:\Projects\CellImaging\cellImaging\Project\runs\\segment\\train72\\weights\\best.onnx")

# 检查模型
onnx.checker.check_model(onnx_model)



print("ONNX模型验证成功！")

# 使用 ONNX Runtime 进行推理测试
ort_session = ort.InferenceSession(r"G:\Projects\CellImaging\cellImaging\Project\runs\\segment\\train72\\weights\\best.onnx")

# 创建测试输入数据
# 输入数据形状需要根据模型调整 (例如 [1, 3, 640, 640] 对于 YOLOv8)
input_shape = (1, 3, 1440, 1440)
dummy_input = np.random.randn(*input_shape).astype(np.float32)

# 获取模型输入的名称
input_name = ort_session.get_inputs()[0].name

# 运行推理
outputs = ort_session.run(None, {input_name: dummy_input})

# 输出推理结果
print("ONNX模型推理结果: ", outputs)