# 🧬 Coach's Eye: AI Biomechanics Analyzer

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-00BFFF?style=for-the-badge&logo=google&logoColor=white)](https://google.github.io/mediapipe/)
[![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![Chart.js](https://img.shields.io/badge/Chart.js-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white)](https://www.chartjs.org/)

> **A professional-grade, high-performance motion analysis platform combining Python's flexibility with WebGL's speed.** > 專為運動員與教練設計的 AI 動作捕捉分析平台，實現 60FPS 零延遲的即時骨架疊加與生物力學數據分析。

![Project Demo](https://via.placeholder.com/800x400?text=Please+Replace+With+Your+App+Screenshot)
*(建議在此處放一張您程式運作時的 GIF 或截圖，展示雙視窗與圖表)*

---

## 📖 Introduction (專案介紹)

**Coach's Eye** 是一個基於 **Streamlit** 與 **MediaPipe (JS)** 的混合架構應用程式。有別於傳統純 Python 的影像處理方案（容易卡頓、延遲），本專案採用 **Hybrid Computing** 架構：

1.  **Python (Backend)**: 負責檔案處理、UI 框架與邏輯控制。
2.  **JavaScript (Client-side)**: 利用瀏覽器 GPU 加速進行 MediaPipe 推論與 Canvas 繪圖，達成 **Real-time 60FPS** 的滑順體驗。

本工具專注於短跑、跳躍等高動態運動的生物力學分析，提供精確的關節角度測量與重心振幅監測。

## ✨ Key Features (核心功能)

* **⚡ 零延遲即時運算 (Zero-Latency Rendering)**
    * 隨意拖動時間軸，骨架與數據即時同步顯示，無須等待伺服器回傳。
* **📐 多關節角度分析 (Multi-Joint Kinematics)**
    * 支援膝蓋 (Knee)、髖部 (Hip)、手肘 (Elbow)、肩膀 (Shoulder) 的角度計算。
    * **左右對照模式 (L/R Compare)**：同時顯示左右側數據，分析對稱性。
* **🟡 重心振幅監測 (COM Analysis)**
    * 視覺化重心 (Center of Mass) 軌跡。
    * 即時計算垂直振幅 (Vertical Oscillation) 與平均高度，評估跑步效率。
* **📈 專業互動式圖表 (Interactive Charts)**
    * 採用線性時間軸 (Linear Time Scale)，數據與影片秒數精確對應。
    * 自動去重與排序算法，解決暫停/倒帶時的數據誤判問題。
* **📥 數據與影片導出**
    * **CSV 下載**：導出每一幀的時間、角度與重心高度數據。
    * **WebM 錄製**：直接錄製帶有骨架疊加的分析影片。

## 🛠️ Tech Stack (技術棧)

* **Frontend**: HTML5 Canvas, JavaScript, [Chart.js](https://www.chartjs.org/) (Data Visualization), [MediaPipe Pose JS](https://developers.google.com/mediapipe) (On-device ML).
* **Backend**: Python, [Streamlit](https://streamlit.io/) (App Framework).
* **Core Logic**: Vector Algebra (Angle Calculation), Signal Smoothing.

## 🚀 Installation & Usage (安裝與執行)

### Prerequisites (前置需求)
* Python 3.8+
* Web Browser (Chrome/Edge recommended for WebGL support)

### Steps (步驟)

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/your-username/coachs-eye-analyzer.git](https://github.com/your-username/coachs-eye-analyzer.git)
    cd coachs-eye-analyzer
    ```

2.  **Install dependencies**
    ```bash
    pip install streamlit
    ```

3.  **Run the app**
    ```bash
    streamlit run app.py
    ```

4.  **Open your browser**
    The app should automatically open at `http://localhost:8501`.

## 📂 Project Structure (專案結構)

```text
├── app.py                # 主要應用程式入口 (Python + Embedded JS)
├── requirements.txt      # 依賴套件列表
├── README.md             # 專案說明文件
└── data/                 # (Optional) 範例影片

劉昱昇-動作捕捉系統
