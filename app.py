import streamlit as st
import tempfile
import mediapipe as mp
import cv2
import time
import numpy as np
from core.geometry import calculate_angle, get_landmark_coords, calculate_approx_com
from core.visualizer import draw_angle_overlay, draw_com_overlay

# --- CSS 優化注入 ---
def inject_custom_css():
    st.markdown("""
        <style>
        /* 1. 移除頂部討厭的空白，讓畫面更緊湊 */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            padding-left: 5rem;
            padding-right: 5rem;
        }
        
        /* 2. 隱藏 Streamlit 預設的漢堡選單與 Footer (看起來更像獨立 App) */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* 3. 自定義數據卡片風格 */
        .metric-card {
            background-color: #262730;
            border-left: 5px solid #00FF00;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 10px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
        }
        .metric-label {
            color: #aaaaaa;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .metric-value {
            color: #ffffff;
            font-size: 1.5rem;
            font-weight: bold;
            font-family: 'Consolas', monospace;
        }
        
        /* 4. 優化滑桿，使其更像影片時間軸 */
        .stSlider [data-baseweb="slider"] {
            height: 10px;
        }
        .stSlider .st-ae { 
            background-color: #00FF00; /* 拉動條顏色 */
        }
        </style>
    """, unsafe_allow_html=True)

import streamlit as st
import tempfile
import mediapipe as mp
import cv2
import time
import numpy as np
from core.geometry import calculate_angle, get_landmark_coords, calculate_approx_com
from core.visualizer import draw_angle_overlay, draw_com_overlay

# --- 設定頁面 (必須是第一行) ---
st.set_page_config(layout="wide", page_title="Coach's Eye Pro", page_icon="🏃")

# --- 注入 CSS ---
def inject_custom_css():
    st.markdown("""
        <style>
        .block-container { padding-top: 1rem; padding-bottom: 2rem; }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* 數據儀表板樣式 */
        .dashboard-container {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }
        .metric-card {
            background-color: #1E1E1E;
            border-left: 4px solid #00FF00;
            padding: 10px;
            border-radius: 8px;
        }
        .metric-label { color: #888; font-size: 0.75rem; }
        .metric-value { color: #FFF; font-size: 1.2rem; font-weight: 700; font-family: monospace; }
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# --- 1. 定義全身關節設定檔 ---
JOINT_CONFIG = {
    "左肘 (L. Elbow)":    (11, 13, 15, (255, 165, 0)),
    "右肘 (R. Elbow)":    (12, 14, 16, (147, 112, 219)),
    "左肩 (L. Shoulder)": (23, 11, 13, (255, 165, 0)),
    "右肩 (R. Shoulder)": (24, 12, 14, (147, 112, 219)),
    "左髖 (L. Hip)":      (11, 23, 25, (255, 165, 0)),
    "右髖 (R. Hip)":      (12, 24, 26, (147, 112, 219)),
    "左膝 (L. Knee)":     (23, 25, 27, (255, 165, 0)),
    "右膝 (R. Knee)":     (24, 26, 28, (147, 112, 219)),
}

# --- MediaPipe Setup ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- Sidebar ---
with st.sidebar:
    st.markdown("## ⚙️ 控制中心")
    uploaded_file = st.file_uploader("上傳影片", type=['mp4', 'mov', 'avi'])
    
    st.markdown("---")
    st.markdown("### 🎯 分析目標")
    selected_joints = st.multiselect(
        "關節角度", 
        options=list(JOINT_CONFIG.keys()), 
        default=["右膝 (R. Knee)", "右髖 (R. Hip)"]
    )
    show_com = st.toggle("顯示重心 (COM) 軌跡", value=True)
    
    st.markdown("---")
    play_speed = st.select_slider("⚡ 播放速度", options=[0.1, 0.25, 0.5, 1.0], value=0.5)

# --- Main Area ---
if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 使用兩欄佈局：左邊是影片(大)，右邊是數據(小)
    col1, col2 = st.columns([0.75, 0.25], gap="medium")

    with col1:
        st.markdown(f"### 📹 影像分析 ({int(width)}x{int(height)})")
        image_placeholder = st.empty()
        
        # 播放控制條放在影片正下方
        ctrl_col1, ctrl_col2 = st.columns([0.15, 0.85])
        with ctrl_col1:
            is_playing = st.toggle("▶ 播放", value=False)
        with ctrl_col2:
            if 'frame_index' not in st.session_state: st.session_state.frame_index = 0
            
            if not is_playing:
                st.session_state.frame_index = st.slider("Frame Scrubber", 0, total_frames-1, st.session_state.frame_index, label_visibility="collapsed")
            else:
                st.progress(st.session_state.frame_index / max(1, total_frames-1))
    
    with col2:
        st.markdown("### 📊 即時數據 (Live Data)")
        # 這裡用一個空的容器，等一下用 HTML 填充
        metrics_placeholder = st.empty()

    # --- Loop ---
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
        
        # 數據收集字典
        dashboard_data = {}

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            image_dims = (height, width, 3)
            
            # 1. 畫底圖骨架
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80,80,80), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=1)
            )

            # 2. 畫關節
            for joint_name in selected_joints:
                p1_id, p2_id, p3_id, color = JOINT_CONFIG[joint_name]
                try:
                    p1 = get_landmark_coords(lm, image_dims, p1_id)
                    p2 = get_landmark_coords(lm, image_dims, p2_id)
                    p3 = get_landmark_coords(lm, image_dims, p3_id)
                    angle = calculate_angle(p1, p2, p3)
                    
                    image = draw_angle_overlay(image, p1, p2, p3, angle, color)
                    
                    # 將英文名稱簡化以顯示在儀表板
                    short_name = joint_name.split('(')[-1].replace(')', '')
                    dashboard_data[short_name] = f"{int(angle)}°"
                except: pass

            # 3. 畫重心
            if show_com:
                try:
                    com_coord = calculate_approx_com(lm, image_dims)
                    st.session_state.com_history.append(com_coord[1])
                    if len(st.session_state.com_history) > 60: st.session_state.com_history.pop(0)
                    image = draw_com_overlay(image, com_coord, st.session_state.com_history)
                    
                    # 計算數據
                    recent = st.session_state.com_history[-30:]
                    if len(recent) > 1:
                        amp = max(recent) - min(recent)
                        amp_pct = (amp / height) * 100
                        dashboard_data["COM Amp"] = f"{amp_pct:.1f}%"
                except: pass

        # 渲染影像
        image_placeholder.image(image, channels="RGB", use_container_width=True)
        
        # 渲染右側數據儀表板 (使用 HTML/CSS)
        html_content = '<div class="dashboard-container">'
        
        # 加入時間數據
        curr_time = st.session_state.frame_index / fps
        html_content += f"""
        <div class="metric-card" style="border-left: 4px solid #FFFFFF;">
            <div class="metric-label">TIME</div>
            <div class="metric-value">{curr_time:.2f}s</div>
        </div>
        """
        
        # 加入關節數據
        for k, v in dashboard_data.items():
            # 根據不同關節給不同邊框色
            border_color = "#00FF00"
            if "Knee" in k: border_color = "#FFA500"
            if "Hip" in k: border_color = "#9370DB"
            
            html_content += f"""
            <div class="metric-card" style="border-left: 4px solid {border_color};">
                <div class="metric-label">{k}</div>
                <div class="metric-value">{v}</div>
            </div>
            """
        html_content += "</div>"
        
        metrics_placeholder.markdown(html_content, unsafe_allow_html=True)

        if is_playing:
            st.session_state.frame_index += 1
            if st.session_state.frame_index >= total_frames: st.session_state.frame_index = 0
            time.sleep(1.0 / (fps * play_speed))
        else:
            break
            
    cap.release()
else:
    # 空白狀態的歡迎畫面
    st.markdown("""
    <div style='text-align: center; padding: 50px;'>
        <h1>👋 歡迎使用 Coach's Eye Pro</h1>
        <p>請從左側側邊欄上傳您的運動影片以開始分析。</p>
    </div>
    """, unsafe_allow_html=True)