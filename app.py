import streamlit as st
import tempfile
import cv2
import time
import numpy as np
import os
from ultralytics import YOLO  # <--- 核心改變：改用 YOLO
from core.geometry import calculate_angle
from core.visualizer import draw_analysis_overlay

# --- 1. 系統設定 ---
st.set_page_config(layout="wide", page_title="Coach's Eye Pro (YOLOv8 Edition)")

# 初始化 Session State
if 'result_video_path' not in st.session_state: st.session_state.result_video_path = None
if 'frame_index' not in st.session_state: st.session_state.frame_index = 0

# --- CSS 優化 (保持賽博龐克風格) ---
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
# YOLO 的 ID 跟 MediaPipe 完全不同，請參考 COCO 格式：
# 0:鼻 5:左肩 6:右肩 11:左髖 12:右髖 13:左膝 14:右膝 15:左踝 16:右踝
# 注意：YOLO 沒有腳尖點，所以無法計算精確的踝關節角度，這裡移除了踝關節選項
JOINT_CONFIG = {
    # 下肢
    "右膝 (R. Knee)":    (12, 14, 16, (147, 112, 219)), # 右髖-右膝-右踝
    "左膝 (L. Knee)":     (11, 13, 15, (255, 165, 0)),   # 左髖-左膝-左踝
    "右髖 (R. Hip)":      (6, 12, 14, (147, 112, 219)),  # 右肩-右髖-右膝
    "左髖 (L. Hip)":      (5, 11, 13, (255, 165, 0)),    # 左肩-左髖-左膝
    
    # 上肢
    "右肘 (R. Elbow)":    (6, 8, 10, (147, 112, 219)),   # 右肩-右肘-右腕
    "左肘 (L. Elbow)":    (5, 7, 9, (255, 165, 0)),      # 左肩-左肘-左腕
    "右肩 (R. Shoulder)": (8, 6, 12, (147, 112, 219)),   # 右肘-右肩-右髖
    "左肩 (L. Shoulder)": (7, 5, 11, (255, 165, 0)),     # 左肘-左肩-左髖
}

# --- 初始化 YOLO 模型 ---
# 第一次執行會自動下載 'yolov8n-pose.pt' (Nano版，速度最快)
@st.cache_resource
def load_model():
    return YOLO('yolov8n-pose.pt')

model = load_model()

# --- 核心：背景分析引擎 (YOLO版) ---
def run_analysis_pipeline(input_path, output_path, selected_joints, progress_bar, status_text):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 1. YOLO 推論 (verbose=False 關閉終端機洗版)
        results = model(frame, verbose=False)
        
        # 取得關鍵點 (Keypoints)
        # results[0] 是第一張圖, keypoints.xy 是座標, cpu().numpy() 轉成陣列
        # shape: (num_people, 17, 2)
        if len(results[0].keypoints) > 0:
            # 預設抓第一個人 (Index 0)
            kpts = results[0].keypoints.xy.cpu().numpy()[0]
            confs = results[0].keypoints.conf.cpu().numpy()[0] # 信心分數
            
            # 2. 繪圖 (直接畫在 frame 上)
            # YOLO 原生繪圖有點雜，我們用自己的 visualizer 保持風格統一
            
            # 先畫基礎骨架連線 (簡化版，只畫四肢)
            # 為了效能與美觀，這裡我們只畫分析的關節連線，或者你可以自己定義 skeleton 連線
            # 這裡簡單畫所有關鍵點
            for i, (x, y) in enumerate(kpts):
                if confs[i] > 0.5: # 只有信心度 > 0.5 才畫
                    cv2.circle(frame, (int(x), int(y)), 3, (200, 200, 200), -1)

            # 3. 計算並繪製角度
            for joint_name in selected_joints:
                p1_idx, p2_idx, p3_idx, color = JOINT_CONFIG[joint_name]
                
                # 檢查這三個點的信心度是否都足夠
                if (confs[p1_idx] > 0.5 and confs[p2_idx] > 0.5 and confs[p3_idx] > 0.5):
                    # YOLO 輸出的座標直接就是像素 (Pixel)，不需要再乘 width/height
                    p1 = (int(kpts[p1_idx][0]), int(kpts[p1_idx][1]))
                    p2 = (int(kpts[p2_idx][0]), int(kpts[p2_idx][1]))
                    p3 = (int(kpts[p3_idx][0]), int(kpts[p3_idx][1]))
                    
                    angle = calculate_angle(p1, p2, p3)
                    frame = draw_analysis_overlay(frame, p1, p2, p3, angle, color=color)

        out.write(frame)
        
        frame_count += 1
        # 避免除以零
        if total_frames > 0:
            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"YOLO 分析中... {int(progress*100)}%")

    cap.release()
    out.release()

# --- UI 介面 ---
st.sidebar.title("🔧 設定中心 (YOLOv8)")
uploaded_file = st.sidebar.file_uploader("1. 上傳影片", type=['mp4', 'mov', 'avi'])

st.sidebar.markdown("---")
selected_joints = st.sidebar.multiselect(
    "2. 選擇關節數據:",
    options=list(JOINT_CONFIG.keys()),
    default=["右膝 (R. Knee)", "右髖 (R. Hip)"]
)

st.title("🏃 Coach's Eye: YOLOv8 抗遮擋分析")

if uploaded_file:
    # 1. 處理上傳
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    
    # 2. 分析按鈕
    if st.sidebar.button("🚀 開始分析 (Analyze)"):
        output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        st.session_state.result_video_path = output_temp.name
        
        prog_bar = st.progress(0)
        status = st.empty()
        with st.spinner("正在啟動 YOLOv8 進行全身掃描..."):
            run_analysis_pipeline(tfile.name, st.session_state.result_video_path, selected_joints, prog_bar, status)
        
        status.success("✅ 分析完成！YOLO 模型已生成影片。")
        prog_bar.empty()
        st.session_state.frame_index = 0 

# --- 3. 智慧播放器 (Smart Player) - 保持不變 ---
if st.session_state.result_video_path and os.path.exists(st.session_state.result_video_path):
    st.divider()
    cap = cv2.VideoCapture(st.session_state.result_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    col1, col2 = st.columns([0.7, 0.3])
    
    with col2:
        st.subheader("🎛️ 回放控制")
        playback_speed = st.select_slider(
            "變速播放 (x)", 
            options=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 
            value=0.5
        )
        is_playing = st.toggle("▶ 開始播放", value=False)
        with open(st.session_state.result_video_path, 'rb') as f:
            st.download_button("⬇️ 下載分析影片", f, file_name="yolo_analysis.mp4", mime="video/mp4")

    with col1:
        image_spot = st.empty()
        
        if not is_playing:
            st.session_state.frame_index = st.slider(
                "Frame Scrubber", 0, total_frames-1, st.session_state.frame_index, label_visibility="collapsed"
            )
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_index)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_spot.image(frame, channels="RGB", use_container_width=True)
                
        else:
            slider_placeholder = st.empty()
            while is_playing:
                start_time = time.time()
                cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_index)
                ret, frame = cap.read()
                if not ret:
                    st.session_state.frame_index = 0
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_spot.image(frame, channels="RGB", use_container_width=True)
                slider_placeholder.progress(st.session_state.frame_index / max(1, total_frames - 1))
                
                st.session_state.frame_index += 1
                if st.session_state.frame_index >= total_frames: st.session_state.frame_index = 0
                
                process_time = time.time() - start_time
                target_interval = 1.0 / (fps * playback_speed)
                wait_time = max(0, target_interval - process_time)
                time.sleep(wait_time)
    cap.release()
elif not uploaded_file:
    st.info("👈 請先上傳影片，並點擊「開始分析」。(Powered by YOLOv8)")