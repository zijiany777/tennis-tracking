import cv2
import numpy as np

# 定义全局变量
drawing = False  # 是否正在绘制矩形标志
ix, iy = -1, -1  # 矩形左上角坐标
roi = None  # 选定的区域

# 鼠标回调函数
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img, roi

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        roi = img[iy:y, ix:x]
        print("Selected region pixel values:")
        print(roi)

# 读取图像
img = cv2.imread("greencourt96.png")

# 获取图像大小
height, width, _ = img.shape

# 创建一个窗口并设置大小为图像大小
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("image", width, height)

# 创建窗口并设置鼠标回调函数
cv2.namedWindow("image")
cv2.setMouseCallback("image", draw_rectangle)

while True:
    cv2.imshow("image", img)
    key = cv2.waitKey(1) & 0xFF

    # 按 's' 键保存选定区域，并显示截图后的图像
    if key == ord('s'):
        if roi is not None:
            # 显示截图后的图像
            cv2.imshow("Selected Region", roi)
            cv2.waitKey(0)
            print("Selected region saved as selected_region.jpg")
        break

    # 按 'esc' 键退出循环
    elif key == 27:
        break

cv2.destroyAllWindows()
