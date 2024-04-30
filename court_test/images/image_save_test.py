#
# # import os
# # import cv2
# #
# # # 创建新文件夹
# # output_folder = "../output_images"
# # os.makedirs(output_folder, exist_ok=True)
# #
# # # 读取图像
# # image = cv2.imread("../image2/bluecourt86.png")
# #
# # # 进行图像处理操作...
# #
# # # 保存图像到新文件夹中
# # output_path = os.path.join(output_folder, "output_image.jpg")
# # cv2.imwrite(output_path, image)
#
# import os
# import cv2
#
# # 设置原始图像文件夹的路径和新图像文件夹的路径
# original_folder_path = "../image2"
# new_folder_path = "../output_images"
#
# # 获取原始图像文件夹中所有文件的名称
# file_names = [f for f in os.listdir(original_folder_path) if os.path.isfile(os.path.join(original_folder_path, f))]
#
# # 遍历所有图像文件
# for file_name in file_names:
#     # 构建原始图像和新图像的完整路径
#     original_image_path = os.path.join(original_folder_path, file_name)
#     new_image_path = os.path.join(new_folder_path, file_name+"_changed.jpg")
#
#     # 读取原始图像
#     original_image = cv2.imread(original_image_path)
#
#     # 进行图像处理操作...
#
#     # 保存修改后的图像到新文件夹中，并使用相应的文件名
#     cv2.imwrite(new_image_path, original_image)

file_name = "image.jpg"
file_name_without_extension = file_name.split(".")[0]
print(file_name_without_extension)