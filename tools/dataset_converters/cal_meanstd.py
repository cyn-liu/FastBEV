import cv2
import numpy as np
import os
import tqdm
import concurrent.futures
import random
# 遍历文件夹获取图像路径列表
def get_image_paths(folder):
    file_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if any(file.endswith(ext) for ext in file_extensions):
                image_paths.append(os.path.join(root, file))
    random.shuffle(image_paths)
    image_paths = image_paths[:int(len(image_paths)*cal_part)]
    return image_paths

# 计算均值和方差
def compute_mean_and_std(image_path):
    img = cv2.imread(image_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    mean = np.mean(img, axis=(0,1))
    std = np.std(img, axis=(0,1))
    return mean, std

# 读取文件夹中的所有图像并计算均值和方差
folder = '/data/fuyu/fastbev/roadside/images/'
cal_part = 1 # 计算部分

image_paths = get_image_paths(folder)
mean_sum = np.zeros(3)
std_sum = np.zeros(3)

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(compute_mean_and_std, path) for path in image_paths]
    bar = tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(image_paths))
    for future in bar:
        mean, std = future.result()
        mean_sum += mean
        std_sum += std

mean = mean_sum / len(image_paths)
std = std_sum / len(image_paths)

print(f'BGR mean = {mean}, std = {std}')
