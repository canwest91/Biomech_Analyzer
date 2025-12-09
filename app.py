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
st.set_page_config(layout="wide", page_title="系統Demo")

# 初始化 Session State
if 'result_video_path' not in st.session_state: st.session_state.result_video_path = None
if 'frame_index' not in st.session_state: st.session_state.frame_index = 0

# --- CSS 優化 ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    [data-testid="stSidebar"] { background-color: #262730; border-right: 1px solid #333; }
    
    /* 滑桿優化 */
    div.stSlider > div[data-baseweb="slider"] > div > div { background-color: #00FF00 !important; height: 12px !important; }
    div.stSlider > div[data-baseweb="slider"] > div { background-color: #444 !important; height: 12px !important; }
    div.stSlider > div[data-baseweb="slider"] > div > div > div {
        width: 24px !important; height: 24px !important; margin-top: -6px !important;
        background-color: #FFFFFF !important; border: 3px solid #00FF00 !important;
        box-shadow: 0 0 15px rgba(0,255,0,0.8); cursor: grab;
    }
    
    .stButton > button { border: 1px solid #00FF00; color: #00FF00; background: transparent; width: 100%; font-weight: bold; }
    .stButton > button:hover { background-color: #00FF00; color: #000; box-shadow: 0 0 15px rgba(0,255,0,0.6); }
    .stProgress > div > div > div > div { background-color: #00FF00; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 關節設定 ---
JOINT_CONFIG = {
    "右膝 (R. Knee)":    (12, 14, 16, (147, 112, 219)),
    "左膝 (L. Knee)":     (11, 13, 15, (255, 165, 0)),
    "右髖 (R. Hip)":      (6, 12, 14, (147, 112, 219)),
    "左髖 (L. Hip)":      (5, 11, 13, (255, 165, 0)),
    "右肘 (R. Elbow)":    (6, 8, 10, (147, 112, 219)),
    "左肘 (L. Elbow)":    (5, 7, 9, (255, 165, 0)),
    "右肩 (R. Shoulder)": (8, 6, 12, (147, 112, 219)),
    "左肩 (L. Shoulder)": (7, 5, 11, (255, 165, 0)),
}

# --- 初始化模型 ---
@st.cache_resource
def load_model():
    return YOLO('yolov8n-pose.pt')

model = load_model()

# --- 背景分析引擎 (含編碼器防當機修正) ---
def run_analysis_pipeline(input_path, output_path, selected_joints, progress_bar, status_text):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 1. 嘗試 H.264 (avc1) - 瀏覽器相容性最佳
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 2. 自動故障轉移 (Failover) - 若 avc1 失敗，改用 mp4v
    if not out.isOpened():
        print("警告: avc1 編碼器啟動失敗，切換至 mp4v...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
    if not out.isOpened():
        st.error("❌ 嚴重錯誤：無法初始化影片寫入器，請檢查系統編碼環境。")
        return

    filters = {}
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        results = model(frame, verbose=False)
        
        if len(results[0].keypoints) > 0:
            kpts = results[0].keypoints.xy.cpu().numpy()[0]
            confs = results[0].keypoints.conf.cpu().numpy()[0] 
            
            for joint_name in selected_joints:
                p1_idx, p2_idx, p3_idx, color = JOINT_CONFIG[joint_name]
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
                
                if (confs[p1_idx] > 0.5 and confs[p2_idx] > 0.5 and confs[p3_idx] > 0.5):
                    raw_p1 = (kpts[p1_idx][0], kpts[p1_idx][1])
                    raw_p2 = (kpts[p2_idx][0], kpts[p2_idx][1])
                    raw_p3 = (kpts[p3_idx][0], kpts[p3_idx][1])
                    
                    f = filters[joint_name]
                    smooth_p1 = (f['p1x'](frame_count, raw_p1[0]), f['p1y'](frame_count, raw_p1[1]))
                    smooth_p2 = (f['p2x'](frame_count, raw_p2[0]), f['p2y'](frame_count, raw_p2[1]))
                    smooth_p3 = (f['p3x'](frame_count, raw_p3[0]), f['p3y'](frame_count, raw_p3[1]))
                    
                    angle = calculate_angle(smooth_p1, smooth_p2, smooth_p3)
                    frame = draw_analysis_overlay(frame, smooth_p1, smooth_p2, smooth_p3, angle, color=color)

        out.write(frame)
        frame_count += 1
        if total_frames > 0:
            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"AI 正在逐幀繪製骨架... {int(progress*100)}%")

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

st.title("運動分析平台")

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    
    if st.sidebar.button("開始分析"):
        output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        st.session_state.result_video_path = output_temp.name
        
        prog_bar = st.progress(0)
        status = st.empty()
        with st.spinner("正在進行 AI 運算..."):
            run_analysis_pipeline(tfile.name, st.session_state.result_video_path, selected_joints, prog_bar, status)
        
        status.success("分析完成！")
        prog_bar.empty()
        st.session_state.frame_index = 0

# --- 雙模式播放器 ---
if st.session_state.result_video_path and os.path.exists(st.session_state.result_video_path):
    st.divider()
    
    tab1, tab2 = st.tabs(["流暢回放", "逐幀分析"])
    
    # 模式 1: 原生播放器
    with tab1:
        st.markdown("##### 最佳體驗：使用下方播放條可快速拖動")
        st.video(st.session_state.result_video_path)
        with open(st.session_state.result_video_path, 'rb') as f:
            st.download_button("⬇下載影片", f, file_name="analysis_result.mp4", mime="video/mp4")

    # 模式 2: 自定義播放器
    with tab2:
        cap = cv2.VideoCapture(st.session_state.result_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        col1, col2 = st.columns([0.75, 0.25])
        
        with col2:
            st.info("此模式用於精確的慢動作分析")
            playback_speed = st.select_slider("變速 (x)", options=[0.1, 0.2, 0.3, 0.5, 1.0], value=0.5)
            is_playing = st.toggle("▶ 播放 / 暫停", value=False)

        with col1:
            image_spot = st.empty()
            
            if not is_playing:
                st.session_state.frame_index = st.slider("拖動時間軸", 0, total_frames-1, st.session_state.frame_index, label_visibility="collapsed")
                cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_index)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w = frame.shape[:2]
                    if w > 800:
                        s = 800/w
                        frame = cv2.resize(frame, (0,0), fx=s, fy=s)
                    image_spot.image(frame, channels="RGB", use_container_width=True)
            else:
                while is_playing:
                    start = time.time()
                    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_index)
                    ret, frame = cap.read()
                    if not ret:
                        st.session_state.frame_index = 0
                        break
                    
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w = frame.shape[:2]
                    if w > 800:
                        s = 800/w
                        frame = cv2.resize(frame, (0,0), fx=s, fy=s, interpolation=cv2.INTER_AREA)
                    
                    image_spot.image(frame, channels="RGB", use_container_width=True)
                    st.session_state.frame_index += 1
                    
                    dt = time.time() - start
                    target = 1.0 / (fps * playback_speed)
                    time.sleep(max(0, target - dt))
        cap.release()

elif not uploaded_file:
    # --- 您的錯誤就是在這裡 ---
    # 之前是 st.info() 空的，現在我幫您補上文字了
    st.info("請先上傳影片，並點擊「開始分析」。")