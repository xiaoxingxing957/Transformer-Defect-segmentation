import numpy as np
import cv2
import scipy.io as scio
import matplotlib.pyplot as plt

def load_images(path_to_images):
    data=scio.loadmat(path_to_images)
    thermal_images_3D=np.array(data['imageArray']) 
    thermal_images_3D = np.transpose(thermal_images_3D, (2,0,1))
    thermal_images_3D = thermal_images_3D[14:, :, :]
    thermal_images_3D=thermal_images_3D.astype(np.float32)
    return thermal_images_3D

def log_transform(images):
    return np.log(images)  # Adding 1 to avoid log(0)

def fit_polynomial(log_images, order=8):
    num_images, height, width = log_images.shape
    coeffs = np.zeros((height, width, order + 1))
    
    for i in range(height):
        for j in range(width):
            y = log_images[:, i, j]
            x = np.log(np.linspace(0.4, 0.4 + (1796 - 1) * 0.1, 1796))
            coeffs[i, j, :] = np.polyfit(x, y, order)
    
    return coeffs

def reconstruct_images(coeffs, num_frames):
    height, width, order_plus_one = coeffs.shape
    reconstructed = np.zeros((num_frames, height, width))
    time = np.linspace(0.4, 0.4 + (1796 - 1) * 0.1, 1796)
    
    for i in range(height):
        for j in range(width):
            for t in range(num_frames):
                ln_t = np.log(time[t])
                ln_T = sum(coeffs[i, j, 8-k] * (ln_t ** k) for k in range(order_plus_one))
                #reconstructed[t, i, j] = np.exp(ln_T)
                reconstructed[t, i, j] =ln_T
    
    return reconstructed

def compute_first_derivative(coeffs, num_frames):
    height, width, order_plus_one = coeffs.shape
    first_derivative = np.zeros((num_frames, height, width))
    time = np.linspace(0.4, 0.4 + (1796 - 1) * 0.1, 1796)
    
    for i in range(height):
        for j in range(width):
            for t in range(num_frames):
                ln_t = np.log(time[t])
                d_ln_T_d_ln_t = sum(k * coeffs[i, j, 8-k] * (ln_t ** (k - 1)) for k in range(1, order_plus_one))
                first_derivative[t, i, j] = d_ln_T_d_ln_t
    
    return first_derivative

#def save_images(images, output_path):
    for i, img in enumerate(images):
        cv2.imwrite(f'{output_path}/first_derivative_image_{i}.png', img)

# Main function
if __name__ == '__main__':
    path_to_images = 'S1.mat'
    images = load_images(path_to_images)
    log_images = log_transform(images)
    coeffs = fit_polynomial(log_images)
    reconstructed_images = reconstruct_images(coeffs, len(images))
    first_derivative_images = compute_first_derivative(coeffs, len(images))
    #save_images(first_derivative_images, output_path)
    np.save('first_derivative_train.npy', first_derivative_images)
    #output_path = 'first_derivative_images'
    
    print("Shape of images:", images.shape)
    defect_series = images[:, 157, 193]
    sound_series = images[:, 108, 195]

    print("Shape of reconstructed_images:", reconstructed_images.shape)
    reconstructed_defect_series = reconstructed_images[:, 157, 193]
    reconstructed_sound_series = reconstructed_images[:, 108, 195]

    print("Shape of first_derivative_images:", first_derivative_images.shape)
    first_derivative_defect_series = first_derivative_images[:, 157, 193]
    first_derivative_sound_series = first_derivative_images[:, 108, 195]

    np.savetxt('defect_series.csv', defect_series, delimiter=',')
    np.savetxt('sound_series.csv', sound_series, delimiter=',')
    np.savetxt('reconstructed_defect_series.csv', reconstructed_defect_series, delimiter=',')
    np.savetxt('reconstructed_sound_series.csv', reconstructed_sound_series, delimiter=',')
    np.savetxt('first_derivative_defect_series.csv', first_derivative_defect_series, delimiter=',')
    np.savetxt('first_derivative_sound_series.csv', first_derivative_sound_series, delimiter=',')


    #time_steps_array = np.arange(first_derivative_images.shape[0])
    #plt.figure(figsize=(10, 6))
    #plt.plot(time_steps_array, time_series1, marker='o', linestyle='-', color='b', label=f'Point 1 ({157}, {193})')
    #plt.plot(time_steps_array, time_series2, marker='o', linestyle='-', color='r', label=f'Point 2 ({108}, {195})')
    #plt.title('first-derivative time Series of Two Points')
    #plt.xlabel('Time Step')
    #plt.ylabel('Value')
    #plt.legend()
    #plt.grid(True)
    #plt.tight_layout()
    #plt.show()