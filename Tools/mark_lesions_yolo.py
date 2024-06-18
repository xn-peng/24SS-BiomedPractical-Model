import cv2
import numpy as np
import os

def find_white_regions(image_path):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")
    
    # 检测白色区域
    _, thresholded = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)

    # 找到白色区域的轮廓
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours, image.shape

def convert_to_yolo_format(contours, image_shape):
    image_height, image_width = image_shape
    
    yolo_labels = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # 计算Yolo格式的边界框中心和宽高 (相对值)
        x_center = (x + w / 2) / image_width
        y_center = (y + h / 2) / image_height
        width = w / image_width
        height = h / image_height
        
        yolo_labels.append((x_center, y_center, width, height))
    
    return yolo_labels

def save_yolo_labels(yolo_labels, label_path):
    with open(label_path, 'w') as file:
        for label in yolo_labels:
            x_center, y_center, width, height = label
            # 写入标签文件 (class_id 0 表示白色区域)
            file.write(f"0 {x_center} {y_center} {width} {height}\n")

def process_image(image_path, labels_root_dir, relative_dir):
    contours, image_shape = find_white_regions(image_path)
    yolo_labels = convert_to_yolo_format(contours, image_shape)
    
    label_name = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
    label_subdir = os.path.join(labels_root_dir, relative_dir)
    os.makedirs(label_subdir, exist_ok=True)
    label_path = os.path.join(label_subdir, label_name)
    
    save_yolo_labels(yolo_labels, label_path)

def process_lesion_folders(input_dir, labels_root_dir):
    for root, dirs, files in os.walk(input_dir):
        for dir_name in dirs:
            if dir_name.endswith("Lesion"):
                lesion_folder_path = os.path.join(root, dir_name)
                relative_dir = os.path.relpath(lesion_folder_path, input_dir)
                for filename in os.listdir(lesion_folder_path):
                    if filename.endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        image_path = os.path.join(lesion_folder_path, filename)
                        process_image(image_path, labels_root_dir, relative_dir)

if __name__ == "__main__":
    input_dir = r"D:/Deutschland/Data and Computer Science/24SS/Bio Practical/BMP Slices/output-256"
    labels_root_dir = r"D:/Deutschland/Data and Computer Science/24SS/Bio Practical/BMP Slices/labels"
    
    process_lesion_folders(input_dir, labels_root_dir)