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


def transfer_and_darken_compensation(mask_img_path, target_img_path, output_path, max_comp_value=5):
    """
    通用版：从图一提取红框，平移到图三，并在图三内进行“中心高、边缘低”的【灰度降低】补偿。
    """
    # 1. 读取图一（模具/坐标提取）和图三（干净底片）
    mask_img = cv_imread(mask_img_path)
    bg_img = cv_imread(target_img_path)

    if mask_img is None or bg_img is None:
        print("错误：无法读取图片，请检查路径。")
        return

    # 强制对齐尺寸，确保像素一一对应
    bg_h, bg_w = bg_img.shape[:2]
    if mask_img.shape[:2] != (bg_h, bg_w):
        mask_img = cv2.resize(mask_img, (bg_w, bg_h), interpolation=cv2.INTER_AREA)

    # 确保底图是灰度模式
    bg_gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY) if len(bg_img.shape) == 3 else bg_img

    # # ==================== 【步骤 1：在图一中提取红框，无视其背景】 ====================
    # hsv = cv2.cvtColor(mask_img, cv2.COLOR_BGR2HSV)
    # # 提取红色 (宽容度较高，防止 JPG 压缩导致漏算)
    # mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([15, 255, 255]))
    # mask2 = cv2.inRange(hsv, np.array([160, 50, 50]), np.array([180, 255, 255]))
    # red_mask = mask1 | mask2
    #
    # # 形态学闭运算：把断裂的红线强行缝合起来
    # kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # red_mask_clean = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_clean)
    # # ==================================================================================
    # ==================== 【步骤 1：在图一中提取红框，无视其背景】 ====================
    hsv = cv2.cvtColor(mask_img, cv2.COLOR_BGR2HSV)

    # 【核心修改】：极其宽容的色彩阈值！
    # Hue(色相) 拓宽到 30 (抓取橙色和棕色)
    # Saturation(饱和度) 和 Value(明度) 降低到 30 (允许颜色发灰、发暗)
    mask1 = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([30, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([150, 30, 30]), np.array([180, 255, 255]))
    red_mask = mask1 | mask2

    # 为了应对线条严重断裂，加大闭运算的核尺寸，强行把碎渣糊成一个完整的框
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    red_mask_clean = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_clean)
    # ==================================================================================

    # 寻找所有的独立红色轮廓
    contours, _ = cv2.findContours(red_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 【全局补偿地图】：用来记录每个像素该减去多少灰度
    global_comp_map = np.zeros(bg_gray.shape, dtype=np.float32)
    # 【全局实心掩膜】：用来标记哪些像素在框内（包含了红框线本身的像素）
    global_solid_mask = np.zeros_like(bg_gray)

    box_count = 0
    # ==================== 【步骤 2：生成外接矩形，铺设 3D 压暗梯度】 ====================
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # 过滤掉细小的误识别噪点 (长宽大于 4 才算真实红框)
        if w > 4 and h > 4:
            # 针对当前框，创建一个局部画布
            local_mask = np.zeros_like(bg_gray)

            # 【核心平移映射】：在对应坐标画一个实心矩形（这就把红框线及其内部全包了）
            cv2.rectangle(local_mask, (x, y), (x + w, y + h), 255, thickness=cv2.FILLED)
            cv2.rectangle(global_solid_mask, (x, y), (x + w, y + h), 255, thickness=cv2.FILLED)

            # 距离变换：计算矩形内每一个点到矩形边缘的距离
            dist_transform = cv2.distanceTransform(local_mask, cv2.DIST_L2, 3)

            # 归一化 (中心=1.0, 边缘=0)
            if dist_transform.max() > 0:
                norm_dist = dist_transform / dist_transform.max()
            else:
                norm_dist = dist_transform

            # 乘以设定的峰值，生成梯度图 (中心为 max_comp_value，边缘趋近于 0)
            local_comp = norm_dist * max_comp_value

            # 汇总到全局地图中 (如果有多个框重叠，取补偿值大的)
            global_comp_map = np.maximum(global_comp_map, local_comp)
            box_count += 1
    # ====================================================================================

    # ==================== 【步骤 3：执行灰度压暗（减法运算）】 ====================
    # 使用 np.float32 防止 uint8 溢出，核心逻辑：原灰度 - 梯度补偿值
    new_gray_float = bg_gray.astype(np.float32) - global_comp_map

    # 强制把数值截断在 0~255 的安全范围内，防止出现负数
    new_gray = np.clip(new_gray_float, 0, 255).astype(np.uint8)

    # 仅替换被框选区域的像素，图三的其余干净部分绝对不动
    final_img = bg_gray.copy()
    final_img[global_solid_mask == 255] = new_gray[global_solid_mask == 255]
    # ==============================================================================

    # 输出统计与日志验证
    y_indices, x_indices = np.where(global_solid_mask == 255)
    total_points = len(y_indices)

    print("=" * 65)
    print(f"【坐标平移 3D压暗梯度补偿 已完成】")
    print(f" 设定矩形中心最高降低峰值: {max_comp_value}")
    print(f" 从图一成功提取并平移的红框数量: {box_count} 个")
    print(f" 在图三中覆盖补偿的总像素点数: {total_points} 个")
    print("-" * 65)

    if total_points > 0:
        step = max(1, total_points // 5)
        for i in range(0, total_points, step)[:5]:
            y, x = y_indices[i], x_indices[i]
            orig = bg_gray[y, x]
            comp = final_img[y, x]
            diff_val = int(orig) - int(comp)  # 注意这里换成了 原值-新值，计算减少量
            print(f" 映射坐标: [{y:3d}, {x:3d}] | 底片灰度: {orig:3d} -> 补偿后: {comp:3d} | 灰度降低: {diff_val:2d}")
    else:
        print(" [警告] 未能从图一提取到红色边界，请检查图一是否存在红线！")

    print("=" * 65)
    cv_imwrite(output_path, final_img)
    print(f" 最终的补偿灰度矩阵已保存至: {output_path}")


if __name__ == "__main__":
    # 1. 含有红框的压缩图 (仅用于提供坐标位置)
    mask_reference_img = r"D:\huidubuchang\number2\divide\compressed_64x36_color.jpg"

    # 2. 纯净的原版图三 (在此图上进行灰度压暗)
    original_bg = r"D:\huidubuchang\number2\divide\1.00.png"

    # 3. 最终输出路径
    output_result = r"D:\huidubuchang\number2\divide\final_compensated_matrix.png"

    # 执行补偿，传入峰值 5 (中心点最多降低 5 个灰度阶)
    transfer_and_darken_compensation(
        mask_img_path=mask_reference_img,
        target_img_path=original_bg,
        output_path=output_result,
        max_comp_value=5
    )