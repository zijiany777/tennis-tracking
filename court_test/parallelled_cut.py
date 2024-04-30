import cv2
import numpy as np

# 定义全局变量
mask_points = []  # 用于存储选定区域的顶点坐标

# 鼠标回调函数
def mouse_callback(event, x, y, flags, param):
    global mask_points, img

    if event == cv2.EVENT_LBUTTONDOWN:
        mask_points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("image", img)

# 读取图像
img = cv2.imread("greencourt96.png")

# 获取图像大小
height, width, _ = img.shape

# 创建一个窗口并设置大小为图像大小
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("image", width, height)

# 创建窗口并设置鼠标回调函数
cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_callback)

while True:
    cv2.imshow("image", img)
    key = cv2.waitKey(1) & 0xFF

    # 按 'esc' 键退出循环
    if key == 27:
        break

# 创建图像掩码
mask = np.zeros_like(img[:, :, 0])

# 将选定区域设为255（白色）
cv2.fillPoly(mask, [np.int32(mask_points)], 255)

# 将原图与掩码进行与运算，保留选定区域的像素值
result = cv2.bitwise_and(img, img, mask=mask)

# 获取图像大小
height, width, _ = result.shape

# 创建一个窗口并设置大小为图像大小
cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Result", width, height)


# 显示结果
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()