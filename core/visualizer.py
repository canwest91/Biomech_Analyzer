import cv2
import numpy as np
import math

def draw_dashed_line(img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
    """輔助函式：繪製虛線"""
    dist = ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5
    pts = []
    for i in  np.arange(0, dist, gap):
        r = i/dist
        x = int((pt1[0]*(1-r) + pt2[0]*r) + .5)
        y = int((pt1[1]*(1-r) + pt2[1]*r) + .5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i%2==1:
                cv2.line(img, s, e, color, thickness)
            i += 1

def draw_analysis_overlay(image, p1, p2, p3, angle_value, color=(0, 255, 0)):
    """
    繪製廣播級的運動分析疊加層
    特點：動態顏色、陰影文字、虛線引導、半透明扇形
    """
    # 確保座標為整數
    p1 = (int(p1[0]), int(p1[1]))
    p2 = (int(p2[0]), int(p2[1]))
    p3 = (int(p3[0]), int(p3[1]))

    # --- 0. 動態顏色邏輯 (可選) ---
    # 例如：角度小於 90 度顯示紅色(警告)，否則綠色
    # if angle_value < 90: color = (0, 0, 255) 

    # --- 1. 繪製半透明扇形 (Visual Arc) ---
    overlay = image.copy()
    
    vec1 = np.array(p1) - np.array(p2)
    vec2 = np.array(p3) - np.array(p2)
    angle_start = np.degrees(np.arctan2(vec1[1], vec1[0]))
    angle_end = np.degrees(np.arctan2(vec2[1], vec2[0]))
    
    # 修正角度順序
    if angle_start < 0: angle_start += 360
    if angle_end < 0: angle_end += 360
    
    # 確保畫的是內角
    if abs(angle_end - angle_start) > 180:
        if angle_end > angle_start: angle_start += 360
        else: angle_end += 360

    # 扇形半徑動態調整 (根據肢體長度)
    limb_len = min(np.linalg.norm(vec1), np.linalg.norm(vec2))
    radius = int(limb_len * 0.3) # 扇形大小是肢體長度的 30%
    radius = max(30, min(radius, 80)) # 限制最大最小範圍

    # 繪製扇形填滿
    cv2.ellipse(overlay, p2, (radius, radius), 0, angle_start, angle_end, color, -1, cv2.LINE_AA)
    # 繪製扇形邊框 (更清晰)
    cv2.ellipse(image, p2, (radius, radius), 0, angle_start, angle_end, (255, 255, 255), 1, cv2.LINE_AA)
    
    # 混合扇形透明度
    cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)

    # --- 2. 繪製骨架連線 (帶陰影與邊框) ---
    # 陰影 (Shadow) - 增加立體感
    shadow_offset = 3
    cv2.line(image, (p1[0]+shadow_offset, p1[1]+shadow_offset), (p2[0]+shadow_offset, p2[1]+shadow_offset), (0,0,0), 4, cv2.LINE_AA)
    cv2.line(image, (p2[0]+shadow_offset, p2[1]+shadow_offset), (p3[0]+shadow_offset, p3[1]+shadow_offset), (0,0,0), 4, cv2.LINE_AA)

    # 主線條
    thickness = 4
    # 白邊
    cv2.line(image, p1, p2, (255, 255, 255), thickness+2, cv2.LINE_AA)
    cv2.line(image, p2, p3, (255, 255, 255), thickness+2, cv2.LINE_AA)
    # 彩芯
    cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)
    cv2.line(image, p2, p3, color, thickness, cv2.LINE_AA)
    
    # --- 3. 繪製關節點 (同心圓設計) ---
    for p in [p1, p2, p3]:
        # 外陰影
        cv2.circle(image, (p[0]+2, p[1]+2), 8, (0,0,0), -1, cv2.LINE_AA)
        # 外白環
        cv2.circle(image, p, 8, (255, 255, 255), -1, cv2.LINE_AA)
        # 內彩點
        cv2.circle(image, p, 5, color, -1, cv2.LINE_AA)

    # --- 4. 繪製 HUD 數據標籤 ---
    text = f"{int(angle_value)}"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.9
    font_thickness = 2
    
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    
    # 智慧標籤位置：根據角度自動決定標籤要在左邊還是右邊，避免擋住肢體
    # 這裡簡化處理，固定在 p2 旁
    label_x = p2[0] + radius + 10
    label_y = p2[1]
    
    # HUD 背景框 (圓角矩形模擬)
    pad = 12
    overlay = image.copy()
    
    # 背景黑框
    bg_pt1 = (label_x - pad, label_y - text_h - pad)
    bg_pt2 = (label_x + text_w + pad + 10, label_y + pad)
    
    cv2.rectangle(overlay, bg_pt1, bg_pt2, (30, 30, 30), -1, cv2.LINE_AA)
    
    # 左側彩色指示條 (Indicator Strip)
    cv2.rectangle(overlay, (bg_pt1[0], bg_pt1[1]), (bg_pt1[0]+4, bg_pt2[1]), color, -1)
    
    # 混合 HUD 背景透明度 (讓它看起來像玻璃)
    cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)
    
    # 繪製文字 (帶微弱陰影)
    cv2.putText(image, text, (label_x + 2, label_y + 2), font, font_scale, (0,0,0), font_thickness, cv2.LINE_AA)
    cv2.putText(image, text, (label_x, label_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    # 度數符號 (較小)
    cv2.putText(image, "o", (label_x + text_w + 2, label_y - text_h + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    
    return image