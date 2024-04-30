import matplotlib.pyplot as plt
import cv2
import numpy as np
def clickpos(image):
    def on_click(event):
        # 输出鼠标点击的像素位置
        print(f'点击位置： x={event.xdata}, y={event.ydata}')

    # 创建图像窗口
    fig, ax = plt.subplots()

    # 显示图像
    ax.imshow(image)

    # 将点击事件绑定到on_click函数
    fig.canvas.mpl_connect('button_press_event', on_click)

    # 显示图像
    plt.show()


img = cv2.imread("img_lines.jpg")
clickpos(img)