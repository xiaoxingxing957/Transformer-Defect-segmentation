import cv2
import numpy as np

def calculate_iou(binary_image, reference_image):
    # 确保输入图像是二值化图像
    if len(binary_image.shape) != 2 or len(reference_image.shape) != 2:
        raise ValueError("输入的图像应为二值化图像")
    
    # 计算交集
    intersection = np.logical_and(binary_image, reference_image)
    
    # 计算并集
    union = np.logical_or(binary_image, reference_image)

    # 计算IoU
    iou = np.sum(intersection) / np.sum(union)
    return iou

# 读取图像
binary_image = cv2.imread('result_image.png', cv2.IMREAD_GRAYSCALE)  # 二值化图像
reference_image = cv2.imread('reference_image.png', cv2.IMREAD_GRAYSCALE)  # 参考图像

# 确保图像大小相同
if binary_image.shape != reference_image.shape:
    raise ValueError("二值化图像和参考图像必须具有相同的大小")

# 计算准确率
iou = calculate_iou(binary_image, reference_image)

print(f'二值化图像的IOU为: {iou:.2%}')