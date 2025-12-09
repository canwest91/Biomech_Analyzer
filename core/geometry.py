import numpy as np

def get_landmark_coords(landmarks, image_shape, idx):
    """將 MediaPipe 的正規化座標 (0~1) 轉為像素座標 (Pixel)"""
    h, w, _ = image_shape
    return (
        int(landmarks[idx].x * w),
        int(landmarks[idx].y * h)
    )

def calculate_angle(a, b, c):
    """計算三點夾角 (a=起點, b=頂點, c=終點)"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    # 使用 atan2 計算向量角度，確保方向性正確
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    # 正規化至 0-360 度，並取較小的內角
    if angle > 180.0:
        angle = 360 - angle
        
    return angle