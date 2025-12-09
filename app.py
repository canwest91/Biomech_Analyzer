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
st.set_page_config(layout="wide", page_title="Coach's Eye Pro - Full Body Export")

# 變數初始化
uploaded_file = None
tfile = None

# --- CSS 優化 (保留原本的 HUD 風格) ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    [data-testid="stSidebar"] { background-color: #262730; border-right: 1px solid #333; }
    
    /* 按鈕樣式 */
    .stButton > button { 
        border: 1px solid #00FF00; color: #00FF00; background: transparent; width: 100%;
        font-weight: bold; padding: 10px;
    }
    .stButton > button:hover { background-color: #00FF00; color: #000; box-shadow: 0 0 15px rgba(0,255,0,0.6); }
    
    /* 進度條樣式 */
    .stProgress > div > div > div > div { background-color: #00FF00; }
    
    /* 多選單樣式 */
    .stMultiSelect [data-baseweb="tag"] { background-color: #333 !important; border: 1px solid #00FF00 !important; }
    
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 關鍵更新：全身生物力學關節設定 (Full Body Config) ---
# 格式: (起點, 頂點, 終點, 顏色RGB)
# 顏色: 橘色(左側)=(255, 165, 0), 紫色(右側)=(147, 112, 219)
JOINT_CONFIG = {
    # --- 下肢 (Lower Body) ---
    "右膝 (R. Knee)":    (24, 26, 28, (147, 112, 219)), # 髖-膝-踝
    "左膝 (L. Knee)":     (23, 25, 27, (255, 165, 0)),
    "右髖 (R. Hip)":      (12, 24, 26, (147, 112, 219)), # 肩-髖-膝 (軀幹角度)
    "左髖 (L. Hip)":      (11, 23, 25, (255, 165, 0)),
    "右踝 (R. Ankle)":    (26, 28, 32, (147, 112, 219)), # 膝-踝-足尖 (推蹬分析)
    "左踝 (L. Ankle)":    (25, 27, 31, (255, 165, 0)),

    # --- 上肢 (Upper Body) ---
    "右肘 (R. Elbow)":    (12, 14, 16, (147, 112, 219)), # 肩-肘-腕
    "左肘 (L. Elbow)":    (11, 13, 15, (255, 165, 0)),
    "右肩 (R. Shoulder)": (14, 12, 24, (147, 112, 219)), # 肘-肩-髖 (擺臂幅度)
    "左肩 (L. Shoulder)": (13, 11, 23, (255, 165, 0)),
}

# MediaPipe 初始化 (使用 Full 模式以獲得最高精度)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False, 
    model_complexity=1, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# --- 核心處理函式 (背景轉檔引擎) ---
def process_video_background(input_path, output_path, selected_joints, progress_bar, status_text):
    cap = cv2.VideoCapture(input_path)
    
    # 取得影片資訊
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 設定影片寫入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 1. 偵測骨架
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 2. 繪圖
        if results.pose_landmarks:
            # 畫基礎骨架 (淡灰色，避免搶眼)
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80,80,80), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(200,200,200), thickness=2, circle_radius=2)
            )
            
            landmarks = results.pose_landmarks.landmark
            
            # 畫所有被勾選的關節
            for joint_name in selected_joints:
                p1_id, p2_id, p3_id, color = JOINT_CONFIG[joint_name]
                try:
                    p1 = get_landmark_coords(landmarks, (height, width, 3), p1_id)
                    p2 = get_landmark_coords(landmarks, (height, width, 3), p2_id)
                    p3 = get_landmark_coords(landmarks, (height, width, 3), p3_id)
                    
                    angle = calculate_angle(p1, p2, p3)
                    
                    # 繪製疊加層
                    image = draw_analysis_overlay(image, p1, p2, p3, angle, color=color)
                except IndexError:
                    continue

        # 3. 寫入
        out.write(image)
        
        # 4. 更新 UI
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"正在運算全身生物力學數據... {int(progress*100)}% ({frame_count}/{total_frames})")

    cap.release()
    out.release()
    return True

# --- 主程式 UI ---
st.sidebar.title("🔧 設定中心")
uploaded_file = st.sidebar.file_uploader("1. 上傳影片", type=['mp4', 'mov', 'avi'])

st.sidebar.markdown("---")
# 預設勾選常用的下肢關節
selected_joints = st.sidebar.multiselect(
    "2. 選擇要疊加的數據:",
    options=list(JOINT_CONFIG.keys()),
    default=["右膝 (R. Knee)", "右髖 (R. Hip)", "右踝 (R. Ankle)"]
)

if uploaded_file:
    # 處理暫存路徑
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    
    # 建立輸出檔案路徑
    output_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = output_temp_file.name
    
    cap = cv2.VideoCapture(tfile.name)
    st.info(f"影片已載入: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {int(cap.get(cv2.CAP_PROP_FPS))}FPS")
    cap.release()

    if st.button("🚀 開始背景運算 (Start Processing)"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("AI 正在逐幀分析全身關節，請稍候..."):
            success = process_video_background(tfile.name, output_path, selected_joints, progress_bar, status_text)
        
        if success:
            status_text.success("✅ 分析完成！影片已生成。")
            progress_bar.empty()
            
            st.divider()
            col1, col2 = st.columns([0.7, 0.3])
            
            with col1:
                st.subheader("🎬 分析結果")
                # 這裡使用原生的 Streamlit 播放器，支援拖拉、全螢幕
                st.video(output_path)
            
            with col2:
                st.subheader("📥 匯出報告")
                st.write("影片已包含完整的關節角度數據。")
                
                with open(output_path, 'rb') as f:
                    video_bytes = f.read()
                    
                st.download_button(
                    label="⬇️ 下載分析影片 (MP4)",
                    data=video_bytes,
                    file_name="full_body_analysis.mp4",
                    mime="video/mp4"
                )
else:
    st.info("👈 請從左側上傳影片以開始。")