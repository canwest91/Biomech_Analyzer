import numpy as np

def get_landmark_coords(landmarks, image_shape, idx):
    """將 MediaPipe 正規化座標轉為像素座標"""
    h, w, _ = image_shape
    return (int(landmarks[idx].x * w), int(landmarks[idx].y * h))

def calculate_angle(a, b, c):
    """計算三點夾角"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

def calculate_approx_com(landmarks, image_shape):
    """
    計算近似重心 (Center of Mass)
    在運動快篩中，通常取 '左右髖關節的中點' 作為骨盆核心位置的近似。
    """
    h, w, _ = image_shape
    left_hip = landmarks[23]
    right_hip = landmarks[24]
    
    # 計算中點 (正規化座標)
    cx = (left_hip.x + right_hip.x) / 2
    cy = (left_hip.y + right_hip.y) / 2
    
    # 轉為像素座標
    return (int(cx * w), int(cy * h))