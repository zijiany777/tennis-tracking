import cv2
import numpy as np

def is_sunny(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # Convert an image from RGB to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义晴天和阴天的HSV阈值范围
    sunny_lower = np.array([20, 50, 50])
    sunny_upper = np.array([35, 255, 255])
    cloudy_lower = np.array([0, 0, 0])
    cloudy_upper = np.array([180, 50, 200])

    # 判断图像中是否存在晴天的颜色
    sunny_mask = cv2.inRange(hsv_image, sunny_lower, sunny_upper)
    sunny_pixels = cv2.countNonZero(sunny_mask)

    # 判断图像中是否存在阴天的颜色
    cloudy_mask = cv2.inRange(hsv_image, cloudy_lower, cloudy_upper)
    cloudy_pixels = cv2.countNonZero(cloudy_mask)

    # 根据阈值判断天气
    if sunny_pixels > cloudy_pixels:
        return "Sunny"
    else:
        return "Cloudy"

# 示例用法
image_path = "images/yin.png"
weather = is_sunny(image_path)
print("The weather is:", weather)