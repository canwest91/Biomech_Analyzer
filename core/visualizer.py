import cv2
import numpy as np

def draw_analysis_overlay(image, p1, p2, p3, angle_value, color=(0, 255, 0)):
    """
    繪製專業級的運動分析疊加層 (含扇形圖與 HUD)
    """
    # 轉換座標為整數 (繪圖需要)
    p1 = (int(p1[0]), int(p1[1]))
    p2 = (int(p2[0]), int(p2[1]))
    p3 = (int(p3[0]), int(p3[1]))

    # --- 1. 繪製半透明扇形 (Visual Arc) ---
    # 建立一個與原圖一樣大的透明層
    overlay = image.copy()
    
    # 計算起始角度與結束角度
    vec1 = np.array(p1) - np.array(p2)
    vec2 = np.array(p3) - np.array(p2)
    angle_start = np.degrees(np.arctan2(vec1[1], vec1[0]))
    angle_end = np.degrees(np.arctan2(vec2[1], vec2[0]))
    
    # OpenCV ellipse 需要的角度順序調整
    if angle_start < 0: angle_start += 360
    if angle_end < 0: angle_end += 360
    
    # 確保扇形是畫在內角
    diff = angle_end - angle_start
    if diff > 180: angle_end, angle_start = angle_start, angle_end
    if diff < -180: angle_end, angle_start = angle_start, angle_end
    
    # 畫實心扇形
    radius = 40
    cv2.ellipse(overlay, p2, (radius, radius), 0, angle_start, angle_end, color, -1, cv2.LINE_AA)
    
    # 混合透明層 (alpha = 0.3)
    cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)

    # --- 2. 繪製骨架連線 ---
    # 外框線 (黑) 增加對比
    thickness = 4
    cv2.line(image, p1, p2, (50, 50, 50), thickness+2, cv2.LINE_AA)
    cv2.line(image, p2, p3, (50, 50, 50), thickness+2, cv2.LINE_AA)
    # 內芯線 (彩色)
    cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)
    cv2.line(image, p2, p3, color, thickness, cv2.LINE_AA)
    
    # --- 3. 繪製關節點 (發光效果) ---
    for p in [p1, p2, p3]:
        cv2.circle(image, p, 8, (255, 255, 255), -1, cv2.LINE_AA) # 白核
        cv2.circle(image, p, 9, color, 2, cv2.LINE_AA) # 彩環

    # --- 4. 繪製 HUD 數據標籤 ---
    text = f"{int(angle_value)}"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.8
    font_thickness = 2
    
    # 計算文字大小
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    
    # 標籤位置 (在關節旁)
    label_x = p2[0] + 45
    label_y = p2[1] - 10
    
    # 繪製半透明文字背景框
    pad = 10
    overlay = image.copy()
    cv2.rectangle(overlay, 
                  (label_x - pad, label_y - text_h - pad), 
                  (label_x + text_w + pad + 15, label_y + pad), 
                  (20, 20, 20), -1) # 深灰底
    # 畫一個左邊的彩色條
    cv2.rectangle(overlay,
                  (label_x - pad, label_y - text_h - pad),
                  (label_x - pad + 5, label_y + pad),
                  color, -1)
                  
    cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
    
    # 繪製文字
    cv2.putText(image, text, (label_x + 5, label_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    # 小度數符號
    cv2.putText(image, "o", (label_x + text_w + 5, label_y - text_h + 5), font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    
    return image