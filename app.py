import streamlit as st
import tempfile
import mediapipe as mp
import cv2
import time
import numpy as np
import os
from core.geometry import calculate_angle, get_landmark_coords
from core.visualizer import draw_analysis_overlay

# --- 1. 系統設定 ---
st.set_page_config(layout="wide", page_title="Coach's Eye Pro - Replay Mode")

# 初始化 Session State (關鍵：防止網頁刷新後資料遺失)
if 'result_video_path' not in st.session_state:
    st.session_state.result_video_path = None
if 'frame_index' not in st.session_state:
    st.session_state.frame_index = 0
if 'is_analyzing' not in st.session_state:
    st.session_state.is_analyzing = False

# --- CSS 優化 (HUD 風格) ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    [data-testid="stSidebar"] { background-color: #262730; border-right: 1px solid #333; }
    
    /* 滑桿與按鈕優化 */
    div.stSlider > div[data-baseweb="slider"] > div > div { background-color: #00FF00 !important; }
    .stButton > button { border: 1px solid #00FF00; color: #00FF00; background: transparent; width: 100%; font-weight: bold; }
    .stButton > button:hover { background-color: #00FF00; color: #000; box-shadow: 0 0 15px rgba(0,255,0,0.6); }
    
    /* 進度條 */
    .stProgress > div > div > div > div { background-color: #00FF00; }
    
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 全身關節設定 ---
JOINT_CONFIG = {
    # 下肢
    "右膝 (R. Knee)":    (24, 26, 28, (147, 112, 219)),
    "左膝 (L. Knee)":     (23, 25, 27, (255, 165, 0)),
    "右髖 (R. Hip)":      (12, 24, 26, (147, 112, 219)),
    "左髖 (L. Hip)":      (11, 23, 25, (255, 165, 0)),
    "右踝 (R. Ankle)":    (26, 28, 32, (147, 112, 219)), 
    "左踝 (L. Ankle)":    (25, 27, 31, (255, 165, 0)),
    # 上肢
    "右肘 (R. Elbow)":    (12, 14, 16, (147, 112, 219)),
    "左肘 (L. Elbow)":    (11, 13, 15, (255, 165, 0)),
    "右肩 (R. Shoulder)": (14, 12, 24, (147, 112, 219)),
    "左肩 (L. Shoulder)": (13, 11, 23, (255, 165, 0)),
}

# MediaPipe 初始化 (使用 High Quality 模式)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- 核心：背景分析引擎 ---
def run_analysis_pipeline(input_path, output_path, selected_joints, progress_bar, status_text):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 使用 mp4v 編碼
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # AI 運算
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 繪圖 (燒錄進影片)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # 畫骨架
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80,80,80), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(200,200,200), thickness=2, circle_radius=2)
            )
            # 畫數據
            for joint_name in selected_joints:
                p1_id, p2_id, p3_id, color = JOINT_CONFIG[joint_name]
                try:
                    p1 = get_landmark_coords(landmarks, (height, width, 3), p1_id)
                    p2 = get_landmark_coords(landmarks, (height, width, 3), p2_id)
                    p3 = get_landmark_coords(landmarks, (height, width, 3), p3_id)
                    angle = calculate_angle(p1, p2, p3)
                    image = draw_analysis_overlay(image, p1, p2, p3, angle, color=color)
                except IndexError: continue

        out.write(image)
        
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"AI 分析中... {int(progress*100)}%")

    cap.release()
    out.release()

# --- UI 介面 ---
st.sidebar.title("🔧 設定中心")
uploaded_file = st.sidebar.file_uploader("1. 上傳影片", type=['mp4', 'mov', 'avi'])

st.sidebar.markdown("---")
selected_joints = st.sidebar.multiselect(
    "2. 選擇關節數據:",
    options=list(JOINT_CONFIG.keys()),
    default=["右膝 (R. Knee)", "右髖 (R. Hip)", "右踝 (R. Ankle)"]
)

st.title("🏃 Coach's Eye: 分析 & 慢速回放系統")

if uploaded_file:
    # 1. 處理上傳
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    
    # 2. 分析按鈕
    if st.sidebar.button("🚀 開始分析 (Analyze)"):
        # 建立輸出路徑
        output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        st.session_state.result_video_path = output_temp.name
        
        # 執行分析
        prog_bar = st.progress(0)
        status = st.empty()
        with st.spinner("正在進行全身動力鍊分析..."):
            run_analysis_pipeline(tfile.name, st.session_state.result_video_path, selected_joints, prog_bar, status)
        
        status.success("✅ 分析完成！進入回放模式。")
        prog_bar.empty()
        st.session_state.frame_index = 0 # 重置播放器

# --- 3. 智慧播放器 (Smart Player) ---
# 只有當分析完成，且有影片路徑時才顯示播放器
if st.session_state.result_video_path and os.path.exists(st.session_state.result_video_path):
    st.divider()
    
    # 讀取已處理的影片
    cap = cv2.VideoCapture(st.session_state.result_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 播放器佈局
    col1, col2 = st.columns([0.7, 0.3])
    
    with col2:
        st.subheader("🎛️ 回放控制")
        
        # 速度滑桿 (這就是你要的！)
        playback_speed = st.select_slider(
            "變速播放 (x)", 
            options=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 
            value=0.5
        )
        
        # 播放開關
        is_playing = st.toggle("▶ 開始播放", value=False)
        
        # 下載按鈕
        with open(st.session_state.result_video_path, 'rb') as f:
            st.download_button("⬇️ 下載分析影片", f, file_name="analysis_result.mp4", mime="video/mp4")

    with col1:
        image_spot = st.empty()
        
        # 時間軸 (如果沒在播放，允許手動拖拉)
        if not is_playing:
            st.session_state.frame_index = st.slider(
                "Frame Scrubber", 0, total_frames-1, st.session_state.frame_index, label_visibility="collapsed"
            )
            
            # 顯示靜態單幀
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_index)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_spot.image(frame, channels="RGB", use_container_width=True)
                
        else:
            # === 播放迴圈 (現在非常快，因為只是讀圖，不算AI) ===
            slider_placeholder = st.empty() # 用來顯示跑動的進度條
            
            while is_playing:
                start_time = time.time()
                
                # 設定讀取位置
                cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_index)
                ret, frame = cap.read()
                if not ret:
                    st.session_state.frame_index = 0 # 循環播放
                    break
                
                # 顯示
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_spot.image(frame, channels="RGB", use_container_width=True)
                
                # 更新進度條 UI
                slider_placeholder.progress(st.session_state.frame_index / max(1, total_frames - 1))
                
                # 下一幀
                st.session_state.frame_index += 1
                if st.session_state.frame_index >= total_frames:
                    st.session_state.frame_index = 0
                
                # 智慧延遲 (Smart Sleep)
                process_time = time.time() - start_time
                target_interval = 1.0 / (fps * playback_speed)
                wait_time = max(0, target_interval - process_time)
                time.sleep(wait_time)
                
                # 為了讓 Stop 按鈕能隨時生效，需要重新檢查
                # (Streamlit 的限制，通常需要按兩下暫停，或使用 Rerun，這裡使用簡單迴圈)

    cap.release()

elif not uploaded_file:
    st.info("👈 請先上傳影片，並點擊「開始分析」。")