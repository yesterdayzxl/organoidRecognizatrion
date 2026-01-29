from ultralytics import YOLO


if __name__ == '__main__':
#下面是三种训练的方式，使用其中一个的时候要注释掉另外两个
# Load a model  加载模型
#这种方式是选择.yaml模型文件，从零开始训练
#model = YOLO('yolov8n.yaml')  # build a new model from YAML
#这种方式是选择.pt文件，利用预训练模型训练
    model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)
#这种方式是可以加载自己想要使用的预训练模型
    #model = YOLO('yolov8x.yaml').load('yolov8x-seg.pt')  # build from YAML and transfer weights

# Train the model   data = 这里用来设置你的data.yaml路径。即第一步设置的文件。后面的就是它的一些属性设置，你也可以增加一些比如batch=16等等。
    model.train(data='my_data.yaml', epochs=50, imgsz=1440, plots=True, batch= 16, mask_ratio=6, mixup=0.5, flipud=0.5, degrees=180, shear=180, perspective=0.0005, bgr=0.5, copy_paste=0.5, close_mosaic=0)

