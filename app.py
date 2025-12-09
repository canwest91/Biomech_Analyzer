import streamlit as st
import tempfile
import cv2
import time
import numpy as np
import os
from ultralytics import YOLO
from core.geometry import calculate_angle, OneEuroFilter
from core.visualizer import draw_analysis_overlay

# --- 1. 系統設定 ---
st.set_page_config(layout="wide", page_title="動作捕捉分析系統demo版")

# 初始化 Session State
if 'result_video_path' not in st.session_state: st.session_state.result_video_path = None
if 'frame_index' not in st.session_state: st.session_state.frame_index = 0

# --- CSS 優化 ---
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

# --- 2. YOLO 專用關節設定 (COCO 17 Keypoints) ---
JOINT_CONFIG = {
    # 下肢
    "右膝 (R. Knee)":    (12, 14, 16, (147, 112, 219)),
    "左膝 (L. Knee)":     (11, 13, 15, (255, 165, 0)),
    "右髖 (R. Hip)":      (6, 12, 14, (147, 112, 219)),
    "左髖 (L. Hip)":      (5, 11, 13, (255, 165, 0)),
    # 上肢
    "右肘 (R. Elbow)":    (6, 8, 10, (147, 112, 219)),
    "左肘 (L. Elbow)":    (5, 7, 9, (255, 165, 0)),
    "右肩 (R. Shoulder)": (8, 6, 12, (147, 112, 219)),
    "左肩 (L. Shoulder)": (7, 5, 11, (255, 165, 0)),
}

# --- 初始化 YOLO 模型 ---
@st.cache_resource
def load_model():
    return YOLO('yolov8n-pose.pt')

model = load_model()

# --- 核心分析引擎 (整合 YOLO + OneEuroFilter) ---
def run_analysis_pipeline(input_path, output_path, selected_joints, progress_bar, status_text):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 初始化濾波器字典
    filters = {}
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 1. YOLO 推論
        results = model(frame, verbose=False)
        
        if len(results[0].keypoints) > 0:
            kpts = results[0].keypoints.xy.cpu().numpy()[0]
            confs = results[0].keypoints.conf.cpu().numpy()[0] 
            
            # 2. 處理每個選擇的關節
            for joint_name in selected_joints:
                p1_idx, p2_idx, p3_idx, color = JOINT_CONFIG[joint_name]
                
                # 初始化該關節的濾波器 (若尚未存在)
                if joint_name not in filters:
                    f_params = {'min_cutoff': 0.5, 'beta': 0.2} 
                    filters[joint_name] = {
                        'p1x': OneEuroFilter(frame_count, kpts[p1_idx][0], **f_params),
                        'p1y': OneEuroFilter(frame_count, kpts[p1_idx][1], **f_params),
                        'p2x': OneEuroFilter(frame_count, kpts[p2_idx][0], **f_params),
                        'p2y': OneEuroFilter(frame_count, kpts[p2_idx][1], **f_params),
                        'p3x': OneEuroFilter(frame_count, kpts[p3_idx][0], **f_params),
                        'p3y': OneEuroFilter(frame_count, kpts[p3_idx][1], **f_params),
                    }
                
                # 信心度檢查
                if (confs[p1_idx] > 0.5 and confs[p2_idx] > 0.5 and confs[p3_idx] > 0.5):
                    raw_p1 = (kpts[p1_idx][0], kpts[p1_idx][1])
                    raw_p2 = (kpts[p2_idx][0], kpts[p2_idx][1])
                    raw_p3 = (kpts[p3_idx][0], kpts[p3_idx][1])
                    
                    # 執行濾波 (Smoothing)
                    f = filters[joint_name]
                    smooth_p1 = (f['p1x'](frame_count, raw_p1[0]), f['p1y'](frame_count, raw_p1[1]))
                    smooth_p2 = (f['p2x'](frame_count, raw_p2[0]), f['p2y'](frame_count, raw_p2[1]))
                    smooth_p3 = (f['p3x'](frame_count, raw_p3[0]), f['p3y'](frame_count, raw_p3[1]))
                    
                    # 計算角度與繪圖
                    angle = calculate_angle(smooth_p1, smooth_p2, smooth_p3)
                    frame = draw_analysis_overlay(frame, smooth_p1, smooth_p2, smooth_p3, angle, color=color)

        out.write(frame)
        
        frame_count += 1
        if total_frames > 0:
            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"AI精確分析中... {int(progress*100)}%")

    cap.release()
    out.release()

# --- UI 介面 ---
st.sidebar.title("設定中心")
uploaded_file = st.sidebar.file_uploader("1. 上傳影片", type=['mp4', 'mov', 'avi'])

st.sidebar.markdown("---")
selected_joints = st.sidebar.multiselect(
    "2. 選擇關節數據:",
    options=list(JOINT_CONFIG.keys()),
    default=["右膝 (R. Knee)", "右髖 (R. Hip)"]
)

st.title("動作捕捉分析平台")

if uploaded_file:
    # 1. 處理上傳
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    
    # 2. 分析按鈕
    if st.sidebar.button("開始分析 (Analyze)"):
        output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        st.session_state.result_video_path = output_temp.name
        
        prog_bar = st.progress(0)
        status = st.empty()
        with st.spinner("正在啟動 YOLOv8 進行全身掃描..."):
            run_analysis_pipeline(tfile.name, st.session_state.result_video_path, selected_joints, prog_bar, status)
        
        status.success("分析完成 YOLO 模型已生成影片。")
        prog_bar.empty()
        st.session_state.frame_index = 0 

# --- 3. 智慧播放器 (Smart Player) ---
if st.session_state.result_video_path and os.path.exists(st.session_state.result_video_path):
    st.divider()
    cap = cv2.VideoCapture(st.session_state.result_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    col1, col2 = st.columns([0.7, 0.3])
    
    with col2:
        st.subheader("回放控制")
        playback_speed = st.select_slider(
            "變速播放 (x)", 
            options=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 
            value=0.5
        )
        is_playing = st.toggle("▶ 開始播放", value=False)
        with open(st.session_state.result_video_path, 'rb') as f:
            st.download_button("⬇下載分析影片", f, file_name="yolo_analysis.mp4", mime="video/mp4")

    with col1:
        image_spot = st.empty()
        slider_placeholder = st.empty()
        
        if not is_playing:
            # 暫停模式：顯示滑桿與靜態圖
            st.session_state.frame_index = st.slider(
                "Frame Scrubber", 0, total_frames-1, st.session_state.frame_index, label_visibility="collapsed"
            )
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_index)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_spot.image(frame, channels="RGB", use_container_width=True)
                
        else:
            # 播放模式：執行優化迴圈
            while is_playing:
                start_time = time.time()
                
                # 讀取
                cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_index)
                ret, frame = cap.read()
                if not ret:
                    st.session_state.frame_index = 0 # 循環播放
                    break
                
                # 轉色
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 顯示優化 (降維打擊：寬度大於800則縮小顯示，大幅降低延遲)
                h, w = frame.shape[:2]
                if w > 800:
                    display_scale = 800 / w
                    frame_display = cv2.resize(frame, (0, 0), fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)
                else:
                    frame_display = frame

                image_spot.image(frame_display, channels="RGB", use_container_width=True)
                
                # 更新進度
                slider_placeholder.progress(st.session_state.frame_index / max(1, total_frames - 1))
                
                st.session_state.frame_index += 1
                if st.session_state.frame_index >= total_frames:
                    st.session_state.frame_index = 0
                
                # 智慧延遲
                process_time = time.time() - start_time
                target_interval = 1.0 / (fps * playback_speed)
                wait_time = max(0, target_interval - process_time)
                time.sleep(wait_time)
                
    cap.release()

elif not uploaded_file:
    st.info("請先上傳影片，並點擊「開始分析」。(Powered by YOLOv8)")