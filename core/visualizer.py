import cv2
import numpy as np

def draw_angle_overlay(image, p1, p2, p3, angle_value, color=(0, 255, 0), label_text=""):
    """通用化的角度繪圖函式"""
    # 畫連線
    cv2.line(image, p1, p2, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.line(image, p2, p3, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.line(image, p1, p2, color, 2, cv2.LINE_AA)
    cv2.line(image, p2, p3, color, 2, cv2.LINE_AA)
    
    # 畫關節點
    for p in [p1, p2, p3]:
        cv2.circle(image, p, 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, p, 4, color, -1, cv2.LINE_AA)
    
    # 畫標籤
    text = f"{int(angle_value)}"
    # 文字位置稍微錯開，避免重疊
    text_pos = (p2[0] + 20, p2[1]) 
    
    cv2.putText(image, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    
    return image

def draw_com_overlay(image, com_coord, com_y_history):
    """繪製重心點與軌跡"""
    # 1. 畫重心點 (紅色大圓點)
    cv2.circle(image, com_coord, 8, (0, 0, 255), -1, cv2.LINE_AA) # 紅色實心
    cv2.circle(image, com_coord, 10, (255, 255, 255), 2, cv2.LINE_AA) # 白色外框

    # 2. 畫重心文字
    cv2.putText(image, "COM", (com_coord[0]+15, com_coord[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    # 3. (進階) 畫出一條淡淡的垂直軌跡線，顯示這一小段時間的波動
    # 我們只畫最近 30 幀的軌跡
    if len(com_y_history) > 1:
        # 這裡我們只用 y 軸變化，x 軸固定在當前 COM 位置，形成垂直振幅示意圖
        current_x = com_coord[0]
        pts = []
        for i, y in enumerate(com_y_history[-30:]): # 取最後30點
            pts.append([current_x, y])
        
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], False, (0, 255, 255), 1, cv2.LINE_AA)

    return image