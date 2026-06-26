import cv2
import numpy as np
import os


def cv_imread(file_path):
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)


def cv_imwrite(file_path, img):
    _, ext = os.path.splitext(file_path)
    cv2.imencode(ext, img)[1].tofile(file_path)


def auto_crop_lattice_region(img, margin=15):
    """
    1. 自动裁切黄色晶格区域，去掉黑色背景。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 提取非黑色区域
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # 闭运算，填补边缘小断裂
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError("未能识别到晶格区域，请检查图片")

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # 加 margin，进一步去掉边缘黑边
    x = max(0, x + margin)
    y = max(0, y + margin)
    w = max(1, w - 2 * margin)
    h = max(1, h - 2 * margin)

    cropped_img = img[y:y + h, x:x + w]

    return cropped_img, (x, y, w, h)


def detect_blur_regions_from_cropped(
        cropped_img,
        blur_percentile=12,
        min_area_ratio=0.002,
        local_window=151,
        morph_kernel_size=45
):
    """
    2. 在裁切后的图像中检测模糊区域。
    """
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

    # 轻微降噪
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Laplacian 响应
    lap = cv2.Laplacian(gray_blur, cv2.CV_64F, ksize=3)

    # 计算局部 Laplacian 方差
    mean = cv2.blur(lap, (local_window, local_window))
    mean_sq = cv2.blur(lap ** 2, (local_window, local_window))
    local_var = mean_sq - mean ** 2

    var_low = np.percentile(local_var, 2)
    var_high = np.percentile(local_var, 98)
    local_var_clip = np.clip(local_var, var_low, var_high)

    norm_var = (local_var_clip - var_low) / (var_high - var_low + 1e-8)
    norm_var = np.clip(norm_var, 0.0, 1.0)

    threshold = np.percentile(norm_var, blur_percentile)

    blur_mask = np.zeros_like(gray, dtype=np.uint8)
    blur_mask[norm_var <= threshold] = 255

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    blur_mask = cv2.morphologyEx(blur_mask, cv2.MORPH_CLOSE, kernel_close)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size // 2 | 1, morph_kernel_size // 2 | 1))
    blur_mask = cv2.morphologyEx(blur_mask, cv2.MORPH_OPEN, kernel_open)

    h, w = gray.shape
    min_area = int(h * w * min_area_ratio)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(blur_mask, connectivity=8)

    final_mask = np.zeros_like(blur_mask)
    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        aspect_ratio = max(bw / (bh + 1e-8), bh / (bw + 1e-8))
        if aspect_ratio > 8:
            continue
        final_mask[labels == i] = 255

    return final_mask, norm_var


# def draw_blur_on_cropped_img(
#         cropped_img,
#         blur_mask,
#         draw_bbox=True,
#         draw_contour=True
# ):
#     """
#     修改点：直接在【裁切图】的坐标系中绘制模糊边缘和矩形框
#     """
#     # 拷贝一份裁切图，防止污染原图矩阵
#     result = cropped_img.copy()
#
#     contours, _ = cv2.findContours(
#         blur_mask,
#         cv2.RETR_EXTERNAL,
#         cv2.CHAIN_APPROX_SIMPLE
#     )
#
#     for contour in contours:
#         area = cv2.contourArea(contour)
#         if area <= 0:
#             continue
#
#         # 因为 blur_mask 本身就是从 cropped_img 算出来的，
#         # 所以这里不需要加任何坐标偏移量，直接画即可！
#         if draw_contour:
#             cv2.drawContours(
#                 result,
#                 [contour],
#                 -1,
#                 (0, 0, 255),  # 红色边缘
#                 3
#             )
#
#         if draw_bbox:
#             x, y, w, h = cv2.boundingRect(contour)
#             cv2.rectangle(
#                 result,
#                 (x, y),
#                 (x + w, y + h),
#                 (0, 255, 255),  # 黄色矩形框
#                 3
#             )
#
#             cv2.putText(
#                 result,
#                 "Blur Region",
#                 (x, max(0, y - 10)),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1.0,
#                 (0, 0, 255),
#                 2
#             )
#
#     return result
def draw_blur_on_cropped_img(
        cropped_img,
        blur_mask,
        draw_bbox=True,
        draw_contour=True
):
    """
    修改点：直接在【裁切图】的坐标系中绘制模糊边缘和矩形框，
    并加入了【动态自适应线宽】，解决高分辨率图片线太细的问题。
    """
    # 拷贝一份裁切图，防止污染原图矩阵
    result = cropped_img.copy()

    # 获取当前图像的宽高
    img_h, img_w = result.shape[:2]

    # ==================== 【核心修改：自适应线宽与字体】 ====================
    # 根据图像宽度动态计算线宽（取宽度的 0.5%，保证高分辨率图上线条足够粗）
    # 最细不低于 3 个像素
    dynamic_thickness = max(3, int(img_w * 0.005))

    # 字体大小也跟着线宽动态放大
    dynamic_font_scale = max(1.0, dynamic_thickness / 3.0)
    # ========================================================================

    contours, _ = cv2.findContours(
        blur_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 0:
            continue

        if draw_contour:
            cv2.drawContours(
                result,
                [contour],
                -1,
                (0, 0, 255),  # 红色边缘
                dynamic_thickness  # 使用动态线宽
            )

        if draw_bbox:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(
                result,
                (x, y),
                (x + w, y + h),
                (0, 255, 255),  # 黄色矩形框
                dynamic_thickness  # 使用动态线宽
            )

            cv2.putText(
                result,
                "Blur Region",
                (x, max(0, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                dynamic_font_scale,  # 动态字体缩放
                (0, 0, 255),
                max(2, int(dynamic_thickness * 0.6))  # 字体线条也适当加粗
            )

    return result


def detect_and_mark_cropped_blur_regions(
        input_image_path,
        output_image_path,
        crop_margin=15,
        blur_percentile=12,
        min_area_ratio=0.002,
        local_window=151,
        morph_kernel_size=45,
        save_debug=False
):
    """
    主流程函数
    """
    img = cv_imread(input_image_path)
    if img is None:
        print(f"无法读取图片，请检查文件是否存在: {input_image_path}")
        return

    # 1. 自动裁切黄色晶格区域 (去掉黑边)
    cropped_img, crop_box = auto_crop_lattice_region(img, margin=crop_margin)

    # 2. 从裁切图中检测模糊区域
    blur_mask, clarity_map = detect_blur_regions_from_cropped(
        cropped_img,
        blur_percentile=blur_percentile,
        min_area_ratio=min_area_ratio,
        local_window=local_window,
        morph_kernel_size=morph_kernel_size
    )

    # 3. 修改点：直接标注在去黑边的裁切图上
    result_cropped_marked = draw_blur_on_cropped_img(
        cropped_img=cropped_img,
        blur_mask=blur_mask,
        draw_bbox=True,
        draw_contour=True
    )

    # 4. 保存最终的去黑边标注图
    cv_imwrite(output_image_path, result_cropped_marked)
    print(f"处理完成，去黑边的模糊区域标注图已保存至: {output_image_path}")

    # 可选：保存调试图
    if save_debug:
        base_dir = os.path.dirname(output_image_path)
        base_name = os.path.splitext(os.path.basename(output_image_path))[0]

        mask_path = os.path.join(base_dir, base_name + "_blur_mask.jpg")
        clarity_path = os.path.join(base_dir, base_name + "_clarity_map.jpg")

        clarity_vis = (clarity_map * 255).astype(np.uint8)
        clarity_vis = cv2.applyColorMap(clarity_vis, cv2.COLORMAP_JET)

        cv_imwrite(mask_path, blur_mask)
        cv_imwrite(clarity_path, clarity_vis)
        print(f"调试图（掩膜和热力图）已保存至: {base_dir}")


if __name__ == "__main__":
    input_img = r"D:\huidubuchang\number2\divide\number2.jpg"

    # 最终输出的图片
    output_img = r"D:\huidubuchang\number2\divide\cropped_blur_mark_result.JPG"

    detect_and_mark_cropped_blur_regions(
        input_image_path=input_img,
        output_image_path=output_img,
        crop_margin=15,

        # 核心检测参数
        blur_percentile=3,
        min_area_ratio=0.002,
        local_window=151,
        morph_kernel_size=45,

        # 是否保存调试图
        save_debug=True
    )