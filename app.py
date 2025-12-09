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

# 自定義 CSS 以優化介面 (暗色模式優化)
# --- 自定義 CSS 以優化介面 (暗色模式/HUD風格) ---
st.markdown("""
<style>
    /* 1. 全域背景與字體設定 */
    .stApp {
        background-color: #0E1117; /*極深灰背景*/
        color: #FAFAFA;
    }
    
    /* 2. 側邊欄優化 - 讓它更像工具箱 */
    [data-testid="stSidebar"] {
        background-color: #262730; /* 稍淺的深灰 */
        border-right: 1px solid #333;
    }
    
    /* 3. 滑桿 (Slider) 大改造 - 變成霓虹綠風格 */
    div.stSlider > div[data-baseweb="slider"] > div > div {
        background-color: #00FF00 !important; /* 拉動後的顏色 (霓虹綠) */
    }
    div.stSlider > div[data-baseweb="slider"] > div {
        background-color: #444 !important; /* 軌道底色 */
    }
    /* 滑桿圓點 */
    div.stSlider > div[data-baseweb="slider"] > div > div > div {
        background-color: #FFFFFF !important; 
        border: 2px solid #00FF00 !important;
        box-shadow: 0 0 10px rgba(0,255,0,0.5); /* 發光特效 */
    }
    
    /* 4. 單選/多選框 (Checkbox/Multiselect) - 統一強調色 */
    .stCheckbox span {
        color: #E0E0E0;
    }
    /* 讓勾選框變成綠色 */
    .stCheckbox [data-baseweb="checkbox"] div {
        background-color: #00FF00 !important;
        border-color: #00FF00 !important;
    }
    /* 多選標籤的顏色 */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #333 !important;
        border: 1px solid #00FF00 !important;
    }
    
    /* 5. 按鈕優化 (Button) - 實心按鈕改為邊框風格 */
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

    /* 6. 去除頂部討厭的空白 */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    
    /* 7. 隱藏 Streamlit 預設選單 (讓 App 看起來更像獨立軟體) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# 定義關節組合與對應的 MediaPipe ID
# 格式: "顯示名稱": (起點ID, 頂點ID, 終點ID, 顏色RGB)
# 顏色: 橘色=(255, 165, 0), 紫色=(147, 112, 219), 綠色=(0, 255, 0)
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
    model_complexity=1, 
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

# B. 關節選擇 (使用多選選單)
# 預設勾選右膝
selected_joints = st.sidebar.multiselect(
    "選擇要顯示的關節角度:",
    options=list(JOINT_CONFIG.keys()),
    default=["右膝 (Right Knee)"]
)

st.sidebar.markdown("---")

# --- 3. 主邏輯 ---
st.title("運動生物力學分析")
col1, col2 = st.columns([3, 1])
image_placeholder = col1.empty() # 創建影像容器
data_placeholder = col2.empty()  # 創建數據容器

def process_frame(frame, height, width):
    """處理單一影格的通用函數：偵測、計算角度、繪圖"""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True

    angle_data = {} # 儲存計算結果

    if results.pose_landmarks:
        # 1. 繪製基礎骨架
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
        
        landmarks = results.pose_landmarks.landmark
        
        # 2. 根據勾選的關節進行計算與繪圖
        for joint_name in selected_joints:
            p1_id, p2_id, p3_id, color = JOINT_CONFIG[joint_name]
            
            try:
                # 取得座標
                p1 = get_landmark_coords(landmarks, (height, width, 3), p1_id)
                p2 = get_landmark_coords(landmarks, (height, width, 3), p2_id)
                p3 = get_landmark_coords(landmarks, (height, width, 3), p3_id)
                
                # 計算角度
                angle = calculate_angle(p1, p2, p3)
                angle_data[joint_name] = int(angle)
                
                # 繪製視覺疊加
                image = draw_analysis_overlay(image, p1, p2, p3, angle, color=color)
            except IndexError:
                continue # 若人物部分出鏡導致無法抓取座標，則跳過該關節

    return image, angle_data

# --- 模式 A: 影片分析 ---
if mode == "📁 影片分析":
    uploaded_file = st.sidebar.file_uploader("上傳影片 (MP4/MOV)", type=['mp4', 'mov', 'avi'])
    
    if uploaded_file:
        # 處理暫存檔
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 播放控制
        st.sidebar.subheader("播放控制")
        play_speed = st.sidebar.select_slider("播放速度", options=[0.1, 0.25, 0.5, 0.75, 1.0], value=0.5)
        is_playing = st.sidebar.checkbox("▶ 開始播放")
        
        # 時間軸滑桿
        if 'frame_index' not in st.session_state:
            st.session_state.frame_index = 0
            
        if not is_playing:
            # 暫停時，顯示滑桿讓使用者手動拖拉
            st.session_state.frame_index = st.slider(
                "時間軸", 0, total_frames - 1, st.session_state.frame_index
            )
        else:
            # 播放時，顯示進度條
            st.progress(st.session_state.frame_index / max(1, total_frames - 1))

        # 影片處理迴圈
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_index)
            ret, frame = cap.read()
            if not ret:
                st.session_state.frame_index = 0 # 循環播放
                break
            
            # 呼叫處理函數
            processed_image, angle_data = process_frame(frame, height, width)
            
            # 更新畫面
            image_placeholder.image(processed_image, channels="RGB", use_container_width=True)
            
            # 更新數據面板
            with data_placeholder.container():
                st.markdown("### 📊 即時數據")
                st.markdown(f"**時間:** {st.session_state.frame_index/fps:.2f} s")
                for name, val in angle_data.items():
                    st.metric(name, f"{val}°")

            # 播放邏輯控制
            if is_playing:
                st.session_state.frame_index += 1
                if st.session_state.frame_index >= total_frames:
                    st.session_state.frame_index = 0
                time.sleep(1.0 / (fps * play_speed)) # 控制播放速度
            else:
                break # 暫停模式下，只渲染當前幀就停止，避免無限迴圈占用資源
                
        cap.release()

# --- 模式 B: 即時影像 (Webcam) ---
elif mode == "📷 即時影像 (Webcam)":
    st.sidebar.info("請確保瀏覽器允許使用鏡頭。點擊下方按鈕開始/停止。")
    run_camera = st.sidebar.checkbox("啟動鏡頭", value=False)
    
    if run_camera:
        cap = cv2.VideoCapture(0) # 0 通常是預設鏡頭
        
        if not cap.isOpened():
            st.error("無法開啟鏡頭，請檢查連接設定。")
        else:
            # 取得鏡頭參數
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.warning("無法接收鏡頭畫面")
                    break
                
                # 鏡頭畫面通常需要水平翻轉 (像鏡子一樣)
                frame = cv2.flip(frame, 1)
                
                # 呼叫處理函數
                processed_image, angle_data = process_frame(frame, height, width)
                
                # 更新畫面
                image_placeholder.image(processed_image, channels="RGB", use_container_width=True)
                
                # 更新數據
                with data_placeholder.container():
                    st.markdown("### 🔴 LIVE 數據")
                    for name, val in angle_data.items():
                        # 使用字體顏色區分左右側，增加可讀性
                        if "左" in name:
                            st.markdown(f"<span style='color:orange'>**{name}:** {val}°</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<span style='color:#9370DB'>**{name}:** {val}°</span>", unsafe_allow_html=True)
                
                # Web 模式下不需要 time.sleep，全速跑
            
            cap.release()
    else:
        image_placeholder.info("等待啟動鏡頭...")