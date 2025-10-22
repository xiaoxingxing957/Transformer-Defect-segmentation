import numpy as np
import pickle

def load_images(path_to_images):
    #image_files = sorted(glob.glob(f'{path_to_images}/*.png'))
    #images = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in image_files]
    images = np.load(path_to_images)
    return np.array(images,dtype=np.float32)

def create_samples(images):
    num_frames, height, width = images.shape
    samples = []

    for i in range(73, 173):
        for j in range(85, 255):
            # 构建包含中心像素及其邻域像素的多变量时间序列
            multivariate_time_series = images[:, i-1:i+2, j-1:j+2].reshape(num_frames, 9)
            samples.append(multivariate_time_series)

    return samples

# 主函数
def main(path_to_images):
    images = load_images(path_to_images)
    samples = create_samples(images)
    
    return samples

# 使用示例
path_to_images = 'first_derivative_test.npy'
stacked_array = main(path_to_images)

with open('reconstruct_val_samples.pkl', 'wb') as f:
    pickle.dump(stacked_array, f)

