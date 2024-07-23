import torch

# 使用torch.hub加载yolov5的预训练模型训练
model = torch.hub.load('ultralytics/yolov9', 'yolov9t')  # or yolov5m, yolov5x, custom

# 加载自己训练好的模型及相关参数
cpkt = torch.load("./best.pt",map_location=torch.device("cuda:0"))

# 将预训练的模型的骨干替换成自己训练好的
yolov9_load = model
yolov9_load.model = cpkt["model"]

# 进行模型调用测试
img_path = '/0007_slice_0_123.png'  # or file, PIL, OpenCV, numpy, multiple
results = yolov9_load(img_path) # 得到预测结果

print(results.xyxy) # 输出预测出的bbox_list
results.show() # 预测结果展示