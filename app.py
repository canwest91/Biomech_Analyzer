import streamlit as st
import tempfile
import mediapipe as mp
import cv2
import time
import numpy as np
from core.geometry import calculate_angle, get_landmark_coords
from core.visualizer import draw_analysis_overlay

# --- 1. 系統設定與全域變數 ---
st.set_page_config(layout="wide", page_title="Coach's Eye Pro")

# --- 自定義 CSS 以優化介面 (暗色模式/HUD風格) ---
st.markdown("""
<style>
    /* 1. 全域背景與字體設定 */
    .stApp {
        background-color: #0E1117; /*極深灰背景*/
        color: #FAFAFA;
    }
    
    /* 2. 側邊欄優化 */
    [data-testid="stSidebar"] {
        background-color: #262730;
        border-right: 1px solid #333;
    }
    
    /* 3. 滑桿 (Slider) 大改造 - 霓虹綠風格 */
    div.stSlider > div[data-baseweb="slider"] > div > div {
        background-color: #00FF00 !important;
    }
    div.stSlider > div[data-baseweb="slider"] > div {
        background-color: #444 !important;
    }
    div.stSlider > div[data-baseweb="slider"] > div > div > div {
        background-color: #FFFFFF !important; 
        border: 2px solid #00FF00 !important;
        box-shadow: 0 0 10px rgba(0,255,0,0.5);
    }
    
    /* 4. Checkbox/Multiselect 優化 */
    .stCheckbox span { color: #E0E0E0; }
    .stCheckbox [data-baseweb="checkbox"] div {
        background-color: #00FF00 !important;
        border-color: #00FF00 !important;
    }
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #333 !important;
        border: 1px solid #00FF00 !important;
    }
    
    /* 5. 按鈕優化 */
    .stButton > button {
        border: 1px solid #00FF00;
        background-color: transparent;
        color: #00FF00;
        border-radius: 4px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #00FF00;
        color: #000000;
        box-shadow: 0 0 15px rgba(0,255,0,0.6);
    }

    /* 6. 去除頂部空白 */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# 定義關節組合與對應的 MediaPipe ID
JOINT_CONFIG = {
    "右膝 (Right Knee)":    (24, 26, 28, (147, 112, 219)),
    "左膝 (Left Knee)":     (23, 25, 27, (255, 165, 0)),
    "右肘 (Right Elbow)":   (12, 14, 16, (147, 112, 219)),
    "左肘 (Left Elbow)":    (11, 13, 15, (255, 165, 0)),
    "右髖 (Right Hip)":     (12, 24, 26, (147, 112, 219)),
    "左髖 (Left Hip)":      (11, 23, 25, (255, 165, 0)),
    "右肩 (Right Shoulder)":(14, 12, 24, (147, 112, 219)),
    "左肩 (Left Shoulder)": (13, 11, 23, (255, 165, 0)),
}

# --- MediaPipe 初始化 ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False, 
    model_complexity=0, # <--- 改用 0 (Lite) 提升速度，若覺得不準可改回 1
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# --- 2. 側邊欄控制中心 ---
st.sidebar.title("🔧 控制面板")

# A. 模式選擇
mode = st.sidebar.radio("選擇模式", ["📁 影片分析", "📷 即時影像 (Webcam)"])

st.sidebar.markdown("---")
st.sidebar.subheader("分析設定")

# B. 關節選擇
selected_joints = st.sidebar.multiselect(
    "選擇要顯示的關節角度:",
    options=list(JOINT_CONFIG.keys()),
    default=["右膝 (Right Knee)"]
)

st.sidebar.markdown("---")

# --- 3. 主邏輯 ---
st.title("運動生物力學分析")
col1, col2 = st.columns([3, 1])
image_placeholder = col1.empty() 
data_placeholder = col2.empty() 

def process_frame(frame):
    """處理單一影格 (包含自動縮放優化)"""
    # 取得原始尺寸
    h, w = frame.shape[:2]
    
    # === 效能優化關鍵：如果圖片太大，就縮小來算 ===
    # 限制最大寬度為 640px (對於姿勢分析來說通常夠用了)
    if w > 640:
        scale = 640 / w
        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        h, w = frame.shape[:2] # 更新尺寸

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True

    angle_data = {} 

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
        
        landmarks = results.pose_landmarks.landmark
        
        for joint_name in selected_joints:
            p1_id, p2_id, p3_id, color = JOINT_CONFIG[joint_name]
            try:
                # 這裡傳入新的 h, w 確保座標正確
                p1 = get_landmark_coords(landmarks, (h, w, 3), p1_id)
                p2 = get_landmark_coords(landmarks, (h, w, 3), p2_id)
                p3 = get_landmark_coords(landmarks, (h, w, 3), p3_id)
                
                angle = calculate_angle(p1, p2, p3)
                angle_data[joint_name] = int(angle)
                
                image = draw_analysis_overlay(image, p1, p2, p3, angle, color=color)
            except IndexError:
                continue 

    return image, angle_data

# --- 模式 A: 影片分析 ---
if mode == "📁 影片分析":
    uploaded_file = st.sidebar.file_uploader("上傳影片 (MP4/MOV)", type=['mp4', 'mov', 'avi'])
    
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 播放控制 (已更新速度選項)
        st.sidebar.subheader("播放控制")
        
        # === 這裡修正了速度選項 ===
        play_speed = st.sidebar.select_slider(
            "⚡ 播放速度 (x)", 
            options=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 
            value=1.0
        )
        # =======================
        
        is_playing = st.sidebar.checkbox("▶ 開始播放")
        
        if 'frame_index' not in st.session_state:
            st.session_state.frame_index = 0
            
        if not is_playing:
            st.session_state.frame_index = st.slider(
                "時間軸", 0, total_frames - 1, st.session_state.frame_index
            )
        else:
            st.progress(st.session_state.frame_index / max(1, total_frames - 1))

# 影片處理迴圈
        while True:
            # 1. 紀錄開始時間 (用於計算運算延遲)
            start_time = time.time()
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_index)
            ret, frame = cap.read()
            if not ret:
                st.session_state.frame_index = 0
                break
            
            # 呼叫處理函數 (這是最花時間的步驟)
            processed_image, angle_data = process_frame(frame)
            
            image_placeholder.image(processed_image, channels="RGB", use_container_width=True)
            
            with data_placeholder.container():
                st.markdown("### 📊 即時數據")
                st.markdown(f"**時間:** {st.session_state.frame_index/fps:.2f} s")
                for name, val in angle_data.items():
                    st.metric(name, f"{val}°")

            if is_playing:
                st.session_state.frame_index += 1
                if st.session_state.frame_index >= total_frames:
                    st.session_state.frame_index = 0
                
                # --- 關鍵修正：動態睡眠時間計算 ---
                # 計算剛剛處理那張圖花了多久
                process_duration = time.time() - start_time
                
                # 計算理論上每一幀應該間隔多久
                target_interval = 1.0 / (fps * play_speed)
                
                # 真正的休息時間 = 理論間隔 - 已經花掉的運算時間
                # 如果運算超時 (結果小於0)，就不休息 (0秒)，全速跑下一張
                wait_time = max(0, target_interval - process_duration)
                
                time.sleep(wait_time) 
            else:
                break
                
        cap.release()

# --- 模式 B: 即時影像 (Webcam) ---
elif mode == "📷 即時影像 (Webcam)":
    st.sidebar.info("請確保瀏覽器允許使用鏡頭。點擊下方按鈕開始/停止。")
    run_camera = st.sidebar.checkbox("啟動鏡頭", value=False)
    
    if run_camera:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("無法開啟鏡頭，請檢查連接設定。")
        else:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.warning("無法接收鏡頭畫面")
                    break
                
                frame = cv2.flip(frame, 1)
                processed_image, angle_data = process_frame(frame, height, width)
                
                image_placeholder.image(processed_image, channels="RGB", use_container_width=True)
                
                with data_placeholder.container():
                    st.markdown("### 🔴 LIVE 數據")
                    for name, val in angle_data.items():
                        if "左" in name:
                            st.markdown(f"<span style='color:orange'>**{name}:** {val}°</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<span style='color:#9370DB'>**{name}:** {val}°</span>", unsafe_allow_html=True)
            
            cap.release()
    else:
        image_placeholder.info("等待啟動鏡頭...")