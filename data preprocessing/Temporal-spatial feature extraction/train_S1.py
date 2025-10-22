import numpy as np
import cv2
import glob
import pickle

def load_images(path_to_images):
    #image_files = sorted(glob.glob(f'{path_to_images}/*.png'))
    #images = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in image_files]
    images = np.load(path_to_images)
    return np.array(images,dtype=np.float32)

def create_samples(images):
    num_frames, height, width = images.shape
    samples = []

    for i in range(55, 59):
        for j in range(98, 101):
            # 构建包含中心像素及其邻域像素的多变量时间序列
            multivariate_time_series = images[:, i-1:i+2, j-1:j+2].reshape(num_frames, 9)
            label = 1
            samples.append((multivariate_time_series, label))
    
    for i in range(55, 59):
        for j in range(171, 175):
            # 构建包含中心像素及其邻域像素的多变量时间序列
            multivariate_time_series = images[:, i-1:i+2, j-1:j+2].reshape(num_frames, 9)
            label = 1
            samples.append((multivariate_time_series, label))
    
    for i in range(53, 61):
        for j in range(241, 249):
            # 构建包含中心像素及其邻域像素的多变量时间序列
            multivariate_time_series = images[:, i-1:i+2, j-1:j+2].reshape(num_frames, 9)
            label = 1
            samples.append((multivariate_time_series, label))

    for i in range(90, 94):
        for j in range(98, 100):
            # 构建包含中心像素及其邻域像素的多变量时间序列
            multivariate_time_series = images[:, i-1:i+2, j-1:j+2].reshape(num_frames, 9)
            label = 1
            samples.append((multivariate_time_series, label))    

    for i in range(88, 95):
        for j in range(146, 154):
            # 构建包含中心像素及其邻域像素的多变量时间序列
            multivariate_time_series = images[:, i-1:i+2, j-1:j+2].reshape(num_frames, 9)
            label = 1
            samples.append((multivariate_time_series, label))

    for i in range(122, 128):
        for j in range(97, 103):
            # 构建包含中心像素及其邻域像素的多变量时间序列
            multivariate_time_series = images[:, i-1:i+2, j-1:j+2].reshape(num_frames, 9)
            label = 1
            samples.append((multivariate_time_series, label))

    for i in range(121, 128):
        for j in range(147, 155):
            # 构建包含中心像素及其邻域像素的多变量时间序列
            multivariate_time_series = images[:, i-1:i+2, j-1:j+2].reshape(num_frames, 9)
            label = 1
            samples.append((multivariate_time_series, label))

    for i in range(156, 160):
        for j in range(123, 127):
            # 构建包含中心像素及其邻域像素的多变量时间序列
            multivariate_time_series = images[:, i-1:i+2, j-1:j+2].reshape(num_frames, 9)
            label = 1
            samples.append((multivariate_time_series, label))

    for i in range(154, 162):
        for j in range(148, 154):
            # 构建包含中心像素及其邻域像素的多变量时间序列
            multivariate_time_series = images[:, i-1:i+2, j-1:j+2].reshape(num_frames, 9)
            label = 1
            samples.append((multivariate_time_series, label))

    for i in range(153, 163):
        for j in range(169, 178):
            # 构建包含中心像素及其邻域像素的多变量时间序列
            multivariate_time_series = images[:, i-1:i+2, j-1:j+2].reshape(num_frames, 9)
            label = 1
            samples.append((multivariate_time_series, label))

    for i in range(151, 165):
        for j in range(187, 200):
            # 构建包含中心像素及其邻域像素的多变量时间序列
            multivariate_time_series = images[:, i-1:i+2, j-1:j+2].reshape(num_frames, 9)
            label = 1
            samples.append((multivariate_time_series, label))

    for i in range(155, 162):
        for j in range(208, 216):
            # 构建包含中心像素及其邻域像素的多变量时间序列
            multivariate_time_series = images[:, i-1:i+2, j-1:j+2].reshape(num_frames, 9)
            label = 1
            samples.append((multivariate_time_series, label))

    for i in range(188, 193):
        for j in range(171, 176):
            # 构建包含中心像素及其邻域像素的多变量时间序列
            multivariate_time_series = images[:, i-1:i+2, j-1:j+2].reshape(num_frames, 9)
            label = 1
            samples.append((multivariate_time_series, label))

    for i in range(80, 104):
        for j in range(185, 213):
            # 构建包含中心像素及其邻域像素的多变量时间序列
            multivariate_time_series = images[:, i-1:i+2, j-1:j+2].reshape(num_frames, 9)
            label = 0
            samples.append((multivariate_time_series, label))

    return samples



# 主函数
def main(path_to_images):
    images = load_images(path_to_images)
    samples = create_samples(images)
    
    return samples

# 使用示例
if __name__ == '__main__':
    path_to_images = 'first_derivative_train.npy'
    stacked_array = main(path_to_images)

    with open('reconstruct_train_samples.pkl', 'wb') as f:
        pickle.dump(stacked_array, f)

