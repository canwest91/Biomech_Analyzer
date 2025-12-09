import streamlit as st
import tempfile
import mediapipe as mp
import cv2
import time
import numpy as np
from core.geometry import calculate_angle, get_landmark_coords, calculate_approx_com
from core.visualizer import draw_angle_overlay, draw_com_overlay

# --- 1. 頁面基礎設定 (必須是第一行) ---
st.set_page_config(
    layout="wide", 
    page_title="Coach's Eye Pro", 
    page_icon="🏃",
    initial_sidebar_state="expanded"
)

# --- 2. CSS 魔改 (HUD 風格) ---
# 這裡定義了所有的視覺樣式
st.markdown("""
    <style>
    /* 全局背景色與字體 */
    .stApp {
        background-color: #0E1117;
        font-family: 'Roboto', sans-serif;
    }
    
    /* 去除頂部空白 */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    
    /* 隱藏選單 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* === HUD 數據卡片樣式 === */
    .dashboard-container {
        display: flex;
        flex-direction: column;
        gap: 12px;
        padding: 10px;
    }
    
    .metric-card {
        background: rgba(38, 39, 48, 0.6); /* 半透明黑底 */
        backdrop-filter: blur(10px);       /* 毛玻璃特效 */
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 5px solid #00FF00;    /* 預設綠色邊框 */
        border-radius: 8px;
        padding: 15px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 12px rgba(0, 255, 0, 0.2); /* 懸浮發光 */
    }

    .metric-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 5px;
    }

    .metric-label {
        color: #aaaaaa;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    .metric-value {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
        font-family: 'Courier New', monospace; /* 科技感等寬字體 */
        text-shadow: 0 0 10px rgba(0, 255, 0, 0.5); /* 霓虹光暈 */
    }

    .metric-unit {
        font-size: 0.9rem;
        color: #888;
        margin-left: 5px;
    }
    
    /* 自定義進度條顏色 */
    .stProgress > div > div > div > div {
        background-color: #00FF00;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. 關節設定檔 ---
JOINT_CONFIG = {
    "右膝 (R. Knee)":     (24, 26, 28, (0, 255, 0)),     # 綠色
    "右髖 (R. Hip)":      (12, 24, 26, (0, 255, 255)),   # 青色
    "右肘 (R. Elbow)":    (12, 14, 16, (255, 0, 255)),   # 紫色
    "左膝 (L. Knee)":     (23, 25, 27, (255, 165, 0)),   # 橘色 (左側)
}

# --- MediaPipe Setup ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- 側邊欄 UI ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2586/2586885.png", width=50)
    st.markdown("### ⚙️ 控制中心")
    uploaded_file = st.file_uploader("上傳影片", type=['mp4', 'mov', 'avi'])
    
    st.markdown("---")
    st.markdown("#### 🎯 分析目標")
    selected_joints = st.multiselect(
        "選擇關節", 
        options=list(JOINT_CONFIG.keys()), 
        default=["右膝 (R. Knee)"]
    )
    show_com = st.toggle("顯示重心 (COM)", value=True)
    
    st.markdown("---")
    play_speed = st.select_slider("⚡ 播放速度", options=[0.1, 0.25, 0.5, 1.0], value=0.5)

# --- 主畫面邏輯 ---
if uploaded_file:
    # 處理影片暫存
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- 佈局調整：改為 7:3 比例，讓數據欄寬一點 ---
    col1, col2 = st.columns([0.7, 0.3], gap="medium")

    with col1:
        st.markdown(f"##### 📹 影像分析 ({int(width)}x{int(height)})")
        image_placeholder = st.empty()
        
        # 播放控制器 (整合在一起)
        c1, c2 = st.columns([0.15, 0.85])
        with c1:
            is_playing = st.toggle("播放", value=False)
        with c2:
            if 'frame_index' not in st.session_state: st.session_state.frame_index = 0
            if not is_playing:
                st.session_state.frame_index = st.slider("時間軸", 0, total_frames-1, st.session_state.frame_index, label_visibility="collapsed")
            else:
                st.progress(st.session_state.frame_index / max(1, total_frames-1))

    with col2:
        st.markdown("##### 📊 即時數據 (Live Data)")
        metrics_placeholder = st.empty()

    # --- 分析迴圈 ---
    if 'com_history' not in st.session_state: st.session_state.com_history = []
    if st.session_state.frame_index == 0: st.session_state.com_history = []

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_index)
        ret, frame = cap.read()
        if not ret: 
            st.session_state.frame_index = 0
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        # 收集數據用於渲染 HTML
        data_cards = []

        # 1. 時間卡片
        curr_time = st.session_state.frame_index / fps
        data_cards.append({
            "label": "TIME CODE",
            "value": f"{curr_time:.2f}",
            "unit": "s",
            "color": "#FFFFFF" # 白色
        })

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            dims = (height, width, 3)
            
            # 畫骨架底圖 (灰色)
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(50,50,50), thickness=2, circle_radius=1),
                mp_drawing.DrawingSpec(color=(200,200,200), thickness=1, circle_radius=1)
            )

            # 關節角度計算
            for joint in selected_joints:
                p1, p2, p3, color_rgb = JOINT_CONFIG[joint]
                try:
                    coords = [get_landmark_coords(lm, dims, p) for p in [p1, p2, p3]]
                    angle = calculate_angle(*coords)
                    
                    # 畫圖
                    image = draw_angle_overlay(image, *coords, angle, color_rgb)
                    
                    # 準備數據卡片 (轉換 RGB tuple 到 Hex 顏色碼以用於 CSS)
                    hex_color = '#%02x%02x%02x' % color_rgb
                    data_cards.append({
                        "label": joint.split('(')[-1].strip(')'), # 取括號內的英文
                        "value": str(int(angle)),
                        "unit": "°",
                        "color": hex_color
                    })
                except: pass

            # 重心計算
            if show_com:
                try:
                    com = calculate_approx_com(lm, dims)
                    st.session_state.com_history.append(com[1])
                    if len(st.session_state.com_history) > 60: st.session_state.com_history.pop(0)
                    image = draw_com_overlay(image, com, st.session_state.com_history)
                    
                    # 計算振幅
                    recent = st.session_state.com_history[-30:]
                    if len(recent) > 1:
                        amp = (max(recent) - min(recent)) / height * 100
                        data_cards.append({
                            "label": "COM AMP",
                            "value": f"{amp:.1f}",
                            "unit": "%",
                            "color": "#FF4B4B" # 紅色
                        })
                except: pass

        # --- 渲染影像 ---
        image_placeholder.image(image, channels="RGB", use_container_width=True)

        # --- 渲染 HTML 數據儀表板 (核心修復點) ---
        html_code = '<div class="dashboard-container">'
        
        for card in data_cards:
            # 這裡我們動態生成每個卡片的 HTML
            # 注意 style 中的 text-shadow 和 border-color 會根據關節顏色改變
            html_code += f"""
            <div class="metric-card" style="border-left: 5px solid {card['color']};">
                <div class="metric-header">
                    <span class="metric-label">{card['label']}</span>
                </div>
                <div>
                    <span class="metric-value" style="text-shadow: 0 0 10px {card['color']}80;">{card['value']}</span>
                    <span class="metric-unit">{card['unit']}</span>
                </div>
            </div>
            """
        html_code += "</div>"
        
        # !!! 關鍵修復 !!! 
        # 必須使用 st.markdown 並開啟 unsafe_allow_html=True
        # 這樣瀏覽器才會把它當作網頁渲染，而不是當作文字印出來
        metrics_placeholder.markdown(html_code, unsafe_allow_html=True)

        # 播放邏輯
        if is_playing:
            st.session_state.frame_index += 1
            if st.session_state.frame_index >= total_frames: st.session_state.frame_index = 0
            time.sleep(1.0 / (fps * play_speed))
        else:
            break
            
    cap.release()
else:
    # 空白狀態
    st.markdown("""
    <div style='display: flex; justify-content: center; align-items: center; height: 300px; border: 2px dashed #333; border-radius: 10px; color: #555;'>
        <h3>👈 請從左側上傳影片以啟動 HUD 系統</h3>
    </div>
    """, unsafe_allow_html=True)
    