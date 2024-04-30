import cv2
import numpy as np
from court_reference import CourtReference
# 初始化空的点列表
points_on_court = []


def click_event(event, x, y, flags, param):


    # 如果点击了鼠标左键，则记录下坐标
    if event == cv2.EVENT_LBUTTONDOWN:
        points_on_court.append((x, y))

        # 在图像上标记点并显示
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('image', img)

        # 如果收集了四个点，输出它们的坐标
        if len(points_on_court) == 4:
            print("选定的四个点的坐标是：", points_on_court)


# 读取图像
img = cv2.imread('../court_test/images/1.png')  # 替换为您的图片路径

# 获取图像大小
height, width, channels = img.shape  # 修正为接收三个返回值

# 创建一个窗口并设置大小为图像大小
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("image", width, height)
cv2.imshow('image', img)

# 设置鼠标回调函数
cv2.setMouseCallback('image', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 确保我们已经收集了四个点
if len(points_on_court) == 4:
    # 将收集的点转换为所需的格式
    points_on_court = np.float32(points_on_court)

    top_inner_line = ((423, 1110), (1242, 1110))
    bottom_inner_line = ((423, 2386), (1242, 2386))
    rect_coords = np.float32([
        top_inner_line[0],
        top_inner_line[1],
        bottom_inner_line[1],
        bottom_inner_line[0]
    ])

    #matrix, _ = cv2.findHomography(np.float32(configuration), np.float32(intersections), method=0)与下面的区别
    # 计算透视变换矩阵
    matrix = cv2.getPerspectiveTransform(rect_coords, points_on_court)  # 这里需要您提供目标点

    court = cv2.cvtColor(cv2.imread('../court_configurations/court_reference.png'), cv2.COLOR_BGR2GRAY)
    # 应用透视变换
    transformed_image = cv2.warpPerspective(court, matrix, (img.shape[1], img.shape[0]))


    #对直线进行投射变换
    p = np.array(CourtReference().get_important_lines(), dtype=np.float32).reshape((-1, 1, 2))
    lines = cv2.perspectiveTransform(p,matrix).reshape(-1)
    for i in range(0, len(lines), 4):
        x1, y1, x2, y2 = lines[i],lines[i+1], lines[i+2], lines[i+3]
        pt1 = (int(round(x1)), int(round(y1)))
        pt2 = (int(round(x2)), int(round(y2)))
        cv2.line(img, pt1, pt2, (255, 0, 0), 3)

    # 获取图像大小
    height, width, channels = img.shape  # 修正为接收三个返回值

    # 创建一个窗口并设置大小为图像大小
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", width, height)
    cv2.imshow('image', img)

    # 获取图像大小
    height, width = transformed_image.shape  # 修正为接收2个返回值

    # 创建一个窗口并设置大小为图像大小
    cv2.namedWindow("Transformed Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Transformed Image", width, height)
    # 显示变换后的图像
    cv2.imshow('Transformed Image', transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
