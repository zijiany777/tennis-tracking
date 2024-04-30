import cv2
import numpy as np

def _classify_lines(lines):
    # 这里应该是你的 _classify_lines 函数的实现，暂时省略，直接返回lines作为水平和垂直线的示例
    return lines, lines  # 假设所有线都既是水平又是垂直，仅作为示例

def draw_lines(image, lines, color=(0, 255, 0), thickness=2):
    """
    Draws lines on an image.
    """
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)

def _detect_lines(gray):
    """
    Finds all line in frame using Hough transform.
    """
    minLineLength = 100
    maxLineGap = 20
    # Detect all lines
    lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 80, minLineLength=minLineLength, maxLineGap=maxLineGap)
    lines = np.squeeze(lines)

    # Classify the lines using their slope
    horizontal, vertical = _classify_lines(lines)

    return horizontal, vertical

# 加载图像
image = cv2.imread('bluecourt86.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 检测线条
horizontal_lines, vertical_lines = _detect_lines(gray)

# 在原图上绘制检测到的线条
draw_lines(image, horizontal_lines, color=(255, 0, 0), thickness=2)  # 用红色绘制水平线
draw_lines(image, vertical_lines, color=(0, 0, 255), thickness=2)  # 用蓝色绘制垂直线

# 显示结果
cv2.imshow('Lines on Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
