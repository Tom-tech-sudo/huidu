import cv2
import numpy as np
import os


def cv_imread(file_path):
    """支持中文路径的图片读取"""
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)


def cv_imwrite(file_path, img):
    """支持中文路径的图片保存"""
    _, ext = os.path.splitext(file_path)
    cv2.imencode(ext, img)[1].tofile(file_path)


def resize_to_64x36_color(input_path, output_path, crop_black_border=True, margin=15):
    """
    通用图像压缩函数：自动去黑边、保持原始颜色和亮度，精准压缩到 64 * 36 像素（彩色）。

    参数:
    input_path: 输入图片的绝对路径
    output_path: 压缩后的彩色矩阵图片保存路径
    crop_black_border: 是否自动切除四周的黑色背景（默认开启）
    margin: 去除黑边时的安全内缩边距
    """
    # 1. 读取原始彩色图像 (BGR 三通道)
    img = cv_imread(input_path)
    if img is None:
        print(f"错误：无法读取图片，请检查路径: {input_path}")
        return None

    # 2. 如果开启了去黑边，先定位核心晶格区域的坐标
    if crop_black_border:
        # 转灰度仅用于寻找边界轮廓
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # 安全边距裁剪坐标
            img_h, img_w, _ = img.shape
            x_new = max(0, x + margin)
            y_new = max(0, y + margin)
            w_new = min(img_w - x_new, w - 2 * margin)
            h_new = min(img_h - y_new, h - 2 * margin)

            # 关键点：直接在【原始彩色图】上截取去掉黑边后的矩阵
            img_target = img[y_new:y_new + h_new, x_new:x_new + w_new]
        else:
            print("警告：未检测到明显的晶格边界，将对全图直接进行彩色压缩。")
            img_target = img
    else:
        img_target = img

    # 3. 核心步骤：缩放到 64 * 36 像素
    # 使用 cv2.INTER_AREA，确保缩放后每个宏观网格的亮度和颜色比例与原图高精对齐
    compressed_color_matrix = cv2.resize(img_target, (64, 36), interpolation=cv2.INTER_AREA)

    # 4. 保存压缩后的彩色特征矩阵图像
    cv_imwrite(output_path, compressed_color_matrix)

    print(f"彩色压缩成功！")
    print(
        f"最终矩阵分辨率: {compressed_color_matrix.shape[1]}x{compressed_color_matrix.shape[0]}，通道数: {compressed_color_matrix.shape[2]}")
    print(f"彩色矩阵保存路径: {output_path}")

    return compressed_color_matrix


if __name__ == "__main__":
    # 配置你的输入和输出路径
    input_img = r"D:\huidubuchang\number2\divide\cropped_blur_mark_result.JPG"
    output_img = r"D:\huidubuchang\number2\divide\compressed_64x36_color.jpg"

    # 执行彩色压缩
    color_matrix = resize_to_64x36_color(
        input_path=input_img,
        output_path=output_img,
        crop_black_border=True,  # 去黑边后再缩放
        margin=15
    )