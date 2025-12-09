import cv2
import numpy as np

def draw_analysis_overlay(image, p1, p2, p3, angle_value, color=(0, 255, 0)):
    """在圖像上繪製骨架連線、關節點與角度標籤"""
    
    # 1. 繪製骨架連線 (白底+彩線)
    cv2.line(image, p1, p2, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.line(image, p2, p3, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.line(image, p1, p2, color, 2, cv2.LINE_AA)
    cv2.line(image, p2, p3, color, 2, cv2.LINE_AA)
    
    # 2. 繪製關節點 (同心圓)
    for p in [p1, p2, p3]:
        cv2.circle(image, p, 10, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, p, 6, color, -1, cv2.LINE_AA)
    
    # 3. 繪製角度標籤 (帶背景框)
    text_pos = (p2[0] + 30, p2[1]) 
    label = f"{int(angle_value)} deg"
    
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(image, (text_pos[0]-5, text_pos[1]-text_h-5), (text_pos[0]+text_w+5, text_pos[1]+5), (0,0,0), -1)
    cv2.putText(image, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    return image