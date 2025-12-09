
# AI-Powered Biomechanics Analysis
> 一個基於電腦視覺的運動生物力學分析平台，專為教練與運動員設計。

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![MediaPipe](https://img.shields.io/badge/AI-MediaPipe-green)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-yellow)

## 📖 專案簡介 (Introduction)
此程式是一個自動化的動作分析工具。旨在解決傳統運動分析軟體需要大量手動標記（Manual Digitizing）的痛點。

本系統利用 **Google MediaPipe** 進行人體骨架偵測，結合 **OpenCV** 進行幾何運算，能夠即時計算關節角度與身體重心（COM）變化。特別針對短跑（Sprinting）與舉重等週期性運動進行優化，提供逐幀（Frame-by-Frame）的量化數據。

## ✨ 核心功能 (Key Features)
* **多關節角度測量 (Multi-Joint Kinematics)**: 支援全身主要關節（如膝、髖、肘、肩）的角度計算。
* **重心振幅分析 (COM Analysis)**: 自動估算身體重心 (Approximate COM) 並追蹤垂直位移，用於評估跑步經濟性。
* **逐幀控制 (Frame-by-Frame Control)**: 透過滑桿精確定位至「觸地期 (Contact Phase)」或「推蹬期 (Propulsion Phase)」。
* **視覺化疊加 (Visual Overlay)**: 在原始影片上繪製動態骨架與數據儀表板。

## ⚙️ 安裝與執行 (Installation)

### 1. 複製專案 (Clone)
```bash
git clone [https://github.com/您的帳號/biomech-ai-coach.git](https://github.com/您的帳號/biomech-ai-coach.git)
cd biomech-ai-coach
````

### 2\. 建立虛擬環境 (Virtual Environment)

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3\. 安裝依賴套件 (Dependencies)

```bash
pip install -r requirements.txt
```

### 4\. 啟動系統 (Run)

```bash
streamlit run app.py
```

## 📂 專案結構 (Project Structure)

```text
Biomech_Analyzer/
├── core/
│   ├── geometry.py       # 幾何運算核心 (向量角度、COM計算)
│   └── visualizer.py     # OpenCV 繪圖引擎
├── app.py                # Streamlit 前端入口與邏輯控制
├── requirements.txt      # 專案依賴清單
└── README.md             # 專案說明文件
```

## 🚀 未來展望 (Roadmap)

  * [ ] **時間序列圖表**: 繪製角度 vs 時間的連續波形圖 (Angle-Time Plot)。
  * [ ] **3D 姿態校正**: 利用 MediaPipe 3D 座標修正攝影機視角誤差。
  * [ ] **自動化步態分割**: 自動偵測觸地 (Touch-down) 與離地 (Toe-off) 瞬間。
  * [ ] **C++ 效能優化**: 計畫將核心運算模組移植至 C++ 以提升 FPS。

## 🛠️ 技術棧 (Tech Stack)

  * **Language**: Python 3.12
  * **Frontend**: Streamlit
  * **Computer Vision**: OpenCV, MediaPipe Pose
  * **Data Processing**: NumPy

-----

劉昱昇-動作捕捉系統
