import numpy as np
import math

# --- 1. OneEuroFilter 濾波器類別 (消除抖動的神器) ---
class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = x0
        self.dx_prev = dx0
        self.t_prev = t0

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        t_e = t - self.t_prev

        # 避免時間倒流或重複導致錯誤
        if t_e <= 0:
            return self.x_prev

        # 估算訊號變化率 (Jitter)
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # 根據變化率動態調整截止頻率
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        
        # 進行平滑
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

# --- 2. 角度計算函式 ---
def calculate_angle(a, b, c):
    """
    計算三點夾角 (a, b, c)，b 為頂點。
    輸入格式: (x, y) 像素座標
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    # 計算向量
    # ba = a - b
    # bc = c - b
    
    # 使用 atan2 計算絕對角度 (更穩定)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle