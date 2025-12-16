import streamlit as st
import streamlit.components.v1 as components
import base64
import tempfile
import json # 新增 json 用於處理陣列傳遞

# --- 1. 系統設定 ---
st.set_page_config(layout="wide", page_title="Coach's Eye: Multi-Joint", page_icon="🧬")

# CSS 美化
st.markdown("""
<style>
    .stApp { background-color: #0D1117; color: #C9D1D9; }
    [data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
    .stButton>button { 
        background-color: #238636; color: white; border: none; font-weight: bold;
    }
    iframe { width: 100% !important; }
</style>
""", unsafe_allow_html=True)

# --- 2. 輔助函式 ---
def get_video_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
        return base64.b64encode(data).decode()

# --- 3. HTML/JS 播放器模板 (支援多選陣列) ---
def get_html_player(video_base64, joint_parts_json, display_mode, trail_target):
    return f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/pose/pose.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@2.1.0/dist/chartjs-plugin-annotation.min.js"></script>
    
    <style>
        body {{ margin: 0; background-color: #0d1117; color: #c9d1d9; font-family: 'Segoe UI', sans-serif; overflow-y: auto; padding-bottom: 20px; }}
        
        .workspace {{ display: flex; gap: 20px; width: 100%; margin-bottom: 20px; height: 500px; }}
        .view-container {{ flex: 1; background: #000; position: relative; border-radius: 12px; border: 1px solid #30363d; display: flex; align-items: center; justify-content: center; overflow: hidden; }}
        .view-title {{ position: absolute; top: 15px; left: 15px; background: rgba(0,0,0,0.7); padding: 6px 12px; border-radius: 6px; font-size: 14px; z-index: 10; pointer-events: none; color: white; }}
        canvas {{ position: absolute; width: 100%; height: 100%; object-fit: contain; }}
        
        .controls-panel {{ background: #161b22; padding: 15px; border-radius: 12px; border: 1px solid #30363d; margin-top: 10px; }}
        .playback-row {{ display: flex; gap: 15px; align-items: center; }}
        
        /* 導出區 */
        .export-panel {{ background: #161b22; padding: 15px; border-radius: 12px; border: 1px solid #30363d; margin-top: 20px; display: flex; align-items: center; justify-content: flex-end; gap: 15px; }}
        
        button {{ background: #238636; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: bold; font-size: 14px; white-space: nowrap; transition: background 0.2s; }}
        button:hover {{ background: #2ea043; }}
        button.record-btn {{ background: #da3633; }}
        button.download-btn {{ background: #1f6feb; }}
        input[type="range"] {{ flex: 1; accent-color: #238636; cursor: pointer; height: 8px; }}
        
        .charts-wrapper {{ display: flex; flex-direction: column; gap: 15px; margin-top: 15px; }}
        .chart-card {{ background: #161b22; border-radius: 12px; border: 1px solid #30363d; padding: 15px; height: 240px; position: relative; }}
        .chart-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; font-size: 14px; color: #8b949e; }}
        .live-value {{ color: #fff; font-family: monospace; font-size: 14px; font-weight: bold; }}
    </style>
</head>
<body>
    <video id="sourceVideo" style="display:none" playsinline muted>
        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
    </video>

    <div class="workspace" id="workspace">
        <div class="view-container">
            <div class="view-title">Blueprint (Skeleton)</div>
            <canvas id="blueprintCanvas"></canvas>
        </div>
        <div class="view-container">
            <div class="view-title">Video Overlay</div>
            <canvas id="overlayCanvas"></canvas>
        </div>
    </div>

    <div class="controls-panel">
        <div class="playback-row">
            <button id="playBtn">▶ 播放</button>
            <input type="range" id="progressBar" value="0" min="0" max="100" step="0.1">
            <span id="timeDisplay" style="font-family: monospace; font-size: 16px;">00.00s</span>
        </div>
    </div>

    <div class="charts-wrapper">
        <div class="chart-card">
            <div class="chart-header">
                <span>📐 多關節角度分析 (Multi-Joint Angles)</span>
                <span id="currentAngleVal" class="live-value" style="text-align: right;">--</span>
            </div>
            <div style="position: relative; height: 200px; width: 100%">
                <canvas id="angleChart"></canvas>
            </div>
        </div>

        <div class="chart-card">
            <div class="chart-header">
                <span>🟡 重心垂直振幅 (COM Vertical)</span>
                <div>
                    <span style="font-size:12px; color:#8b949e;">Avg: </span>
                    <span id="avgComVal" class="live-value" style="color: #FF4081; margin-right:10px;">--</span>
                    <span style="font-size:12px; color:#8b949e;">Curr: </span>
                    <span id="currentComVal" class="live-value" style="color: #FFD600;">--</span>
                </div>
            </div>
            <div style="position: relative; height: 200px; width: 100%">
                <canvas id="comChart"></canvas>
            </div>
        </div>
    </div>

    <div class="export-panel">
        <span style="color: #8b949e; font-size: 14px; margin-right: auto;">📥 導出工具：</span>
        <button id="downloadCsvBtn" class="download-btn">📊 下載數據 (.csv)</button>
        <button id="recordBtn" class="record-btn">🔴 錄製分析影片</button>
    </div>

    <script>
        // 接收 Python 傳來的陣列字串 (JSON 格式)
        const CONFIG_PARTS = JSON.parse('{joint_parts_json}'); // e.g. ["Knee", "Hip"]
        const CONFIG_MODE = "{display_mode}";
        const CONFIG_TRAIL = "{trail_target}";

        const video = document.getElementById('sourceVideo');
        const overlayCanvas = document.getElementById('overlayCanvas');
        const blueprintCanvas = document.getElementById('blueprintCanvas');
        const ctxOverlay = overlayCanvas.getContext('2d');
        const ctxBlueprint = blueprintCanvas.getContext('2d');
        const playBtn = document.getElementById('playBtn');
        const progressBar = document.getElementById('progressBar');
        const timeDisplay = document.getElementById('timeDisplay');
        const currentAngleVal = document.getElementById('currentAngleVal');
        const currentComVal = document.getElementById('currentComVal');
        const avgComVal = document.getElementById('avgComVal');
        const downloadCsvBtn = document.getElementById('downloadCsvBtn');
        const recordBtn = document.getElementById('recordBtn');

        let angleChart = null;
        let comChart = null;
        let animationFrameId;
        let trailQueue = []; 
        const MAX_TRAIL_LEN = 40;
        let dataStore = new Map();
        let comValues = [];
        let mediaRecorder;
        let recordedChunks = [];
        let isRecording = false;

        const JOINT_MAP = {{
            "Knee":     {{ "R": [24, 26, 28], "L": [23, 25, 27] }},
            "Hip":      {{ "R": [12, 24, 26], "L": [11, 23, 25] }},
            "Elbow":    {{ "R": [12, 14, 16], "L": [11, 13, 15] }},
            "Shoulder": {{ "R": [14, 12, 24], "L": [13, 11, 23] }}
        }};

        // 配色表：為每個關節定義基本色，左右用深淺或色相區分
        const COLORS = {{
            "R": {{ "Knee": "#00E676", "Hip": "#00B0FF", "Elbow": "#00E5FF", "Shoulder": "#1DE9B6" }}, // 冷色系
            "L": {{ "Knee": "#FF4081", "Hip": "#FF9100", "Elbow": "#FF5252", "Shoulder": "#FFC400" }}  // 暖色系
        }};

        const TRAIL_MAP = {{
            "R.Ankle": 28, "L.Ankle": 27, "R.Knee": 26, "L.Knee": 25,
            "R.Hip": 24, "L.Hip": 23, "R.Elbow": 14, "L.Elbow": 13,
            "Head": 0, "COM": "COM"
        }};

        const pose = new Pose({{locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${{file}}`}});
        pose.setOptions({{modelComplexity: 1, smoothLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5}});
        pose.onResults(onResults);

        const commonOptions = {{
            responsive: true, maintainAspectRatio: false, animation: false,
            interaction: {{ mode: 'nearest', axis: 'x', intersect: false }},
            scales: {{
                x: {{ type: 'linear', display: true, grid: {{ color: '#30363d' }}, ticks: {{ color: '#8b949e', callback: v => v.toFixed(1)+'s' }} }},
                y: {{ grid: {{ color: '#30363d' }}, ticks: {{ color: '#8b949e' }} }}
            }},
            plugins: {{
                legend: {{ labels: {{ color: 'white', boxWidth: 12, font: {{size: 11}} }} }},
                annotation: {{ annotations: {{ line1: {{ type: 'line', xMin: 0, xMax: 0, borderColor: 'rgba(255,255,255,0.5)', borderWidth: 2, borderDash: [5,5] }} }} }}
            }}
        }};

        function initCharts() {{
            const ctxAngle = document.getElementById('angleChart').getContext('2d');
            let angleDatasets = [];

            // 根據選取的關節列表動態生成 Dataset
            CONFIG_PARTS.forEach(part => {{
                if (CONFIG_MODE === "Compare") {{
                    angleDatasets.push({{ label: `R.${{part}}`, data: [], borderColor: COLORS["R"][part], borderWidth: 2, pointRadius: 0, tension: 0.1 }});
                    angleDatasets.push({{ label: `L.${{part}}`, data: [], borderColor: COLORS["L"][part], borderWidth: 2, pointRadius: 0, tension: 0.1 }});
                }} else {{
                    const side = (CONFIG_MODE === "Left") ? "L" : "R";
                    const label = (CONFIG_MODE === "Left") ? "左" : "右";
                    angleDatasets.push({{ label: `${{label}}.${{part}}`, data: [], borderColor: COLORS[side][part], borderWidth: 2, pointRadius: 0, tension: 0.1 }});
                }}
            }});

            angleChart = new Chart(ctxAngle, {{ type: 'line', data: {{ datasets: angleDatasets }}, options: commonOptions }});

            const ctxCom = document.getElementById('comChart').getContext('2d');
            comChart = new Chart(ctxCom, {{
                type: 'scatter',
                data: {{ 
                    datasets: [
                        {{ label: '重心 (COM)', data: [], backgroundColor: '#FFD600', pointRadius: 2 }},
                        {{ type: 'line', label: '平均線', data: [], borderColor: '#FF4081', borderWidth: 1, borderDash: [5,5], pointRadius: 0 }}
                    ] 
                }},
                options: commonOptions
            }});
        }}
        initCharts();

        function calculateAngle(a, b, c) {{
            const rad = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
            let ang = Math.abs(rad * 180.0 / Math.PI);
            if (ang > 180.0) ang = 360 - ang;
            return parseInt(ang);
        }}

        function onResults(results) {{
            overlayCanvas.width = video.videoWidth; overlayCanvas.height = video.videoHeight;
            blueprintCanvas.width = video.videoWidth; blueprintCanvas.height = video.videoHeight;

            ctxOverlay.drawImage(results.image, 0, 0, overlayCanvas.width, overlayCanvas.height);
            ctxBlueprint.fillStyle = "black"; ctxBlueprint.fillRect(0, 0, blueprintCanvas.width, blueprintCanvas.height);

            const currentTime = parseFloat(video.currentTime.toFixed(2));
            // 動態建立數據結構： newData = {{ time: 1.2, comY: 100, R_Knee: 160, L_Knee: 150, ... }}
            let newData = {{ time: currentTime, comY: null }};

            if (results.poseLandmarks) {{
                const lm = results.poseLandmarks;
                const styleLine = {{color: '#FACE87', lineWidth: 4}};
                const stylePoint = {{color: '#FFFF64', lineWidth: 2, radius: 4}};

                drawConnectors(ctxOverlay, lm, POSE_CONNECTIONS, styleLine);
                drawLandmarks(ctxOverlay, lm, stylePoint);
                drawConnectors(ctxBlueprint, lm, POSE_CONNECTIONS, styleLine);
                drawLandmarks(ctxBlueprint, lm, stylePoint);

                // --- COM & Trail ---
                let pt = null;
                if (CONFIG_TRAIL === "COM" && lm[24] && lm[23]) {{
                    const cx = (lm[24].x + lm[23].x) / 2;
                    const cy = (lm[24].y + lm[23].y) / 2;
                    pt = {{x: cx * overlayCanvas.width, y: cy * overlayCanvas.height}};
                    newData.comY = Math.round(overlayCanvas.height - pt.y);
                    
                    ctxOverlay.beginPath(); ctxOverlay.arc(pt.x, pt.y, 8, 0, 2*Math.PI); 
                    ctxOverlay.fillStyle = "#FF4081"; ctxOverlay.fill();
                }} else if (CONFIG_TRAIL !== "None") {{
                    const idx = TRAIL_MAP[CONFIG_TRAIL];
                    if (lm[idx]) pt = {{x: lm[idx].x * overlayCanvas.width, y: lm[idx].y * overlayCanvas.height}};
                }}

                if (pt) {{
                    trailQueue.push(pt);
                    if (trailQueue.length > MAX_TRAIL_LEN) trailQueue.shift();
                    for (let i = 1; i < trailQueue.length; i++) {{
                        ctxOverlay.beginPath(); ctxOverlay.moveTo(trailQueue[i-1].x, trailQueue[i-1].y);
                        ctxOverlay.lineTo(trailQueue[i].x, trailQueue[i].y);
                        ctxOverlay.strokeStyle = (CONFIG_TRAIL === "COM") ? `rgba(255, 64, 129, ${{i/trailQueue.length}})` : `rgba(255, 215, 0, ${{i/trailQueue.length}})`;
                        ctxOverlay.lineWidth = 4; ctxOverlay.stroke();
                    }}
                }}

                // --- Multi-Joint Angle Calculation ---
                let displayTexts = [];
                
                CONFIG_PARTS.forEach(part => {{
                    const jointData = JOINT_MAP[part];
                    let sides = [];
                    if (CONFIG_MODE === "Compare") sides = ["R", "L"];
                    else if (CONFIG_MODE === "Right") sides = ["R"];
                    else sides = ["L"];

                    sides.forEach(side => {{
                        const ids = jointData[side];
                        if (lm[ids[0]] && lm[ids[1]] && lm[ids[2]]) {{
                            const ang = calculateAngle(lm[ids[0]], lm[ids[1]], lm[ids[2]]);
                            
                            // 儲存數據: Key 例如 "R_Knee"
                            newData[`${{side}}_${{part}}`] = ang;
                            displayTexts.push(`${{side}}.${{part}}: ${{ang}}°`);

                            // 繪製文字
                            const center = lm[ids[1]];
                            // 簡單錯位避免重疊：依據部位不同給予不同偏移
                            let xOffset = (side === "R" ? 15 : -65);
                            let yOffset = 0;
                            // 如果選多個部位，可微調 yOffset 避免擠在一起 (這裡簡化處理)
                            
                            const txtX = center.x * overlayCanvas.width + xOffset;
                            const txtY = center.y * overlayCanvas.height + yOffset;
                            const fontSize = Math.max(18, Math.floor(overlayCanvas.width / 50));
                            const color = COLORS[side][part];
                            
                            [ctxOverlay, ctxBlueprint].forEach(ctx => {{
                                ctx.font = `bold ${{fontSize}}px Arial`; ctx.fillStyle = color; ctx.strokeStyle = "black"; ctx.lineWidth = 3;
                                ctx.strokeText(ang + "°", txtX, txtY); ctx.fillText(ang + "°", txtX, txtY);
                            }});
                        }}
                    }});
                }});

                // --- Update UI ---
                currentAngleVal.innerText = displayTexts.join(" | ");
                if(newData.comY) currentComVal.innerText = newData.comY + " px";

                if (!video.paused) {{
                    dataStore.set(currentTime.toFixed(2), newData);
                    
                    if(newData.comY) {{
                        comValues.push(newData.comY);
                        if (comValues.length > 200) comValues.shift();
                    }}
                }}

                const sortedData = Array.from(dataStore.values()).sort((a, b) => a.time - b.time);
                
                [angleChart, comChart].forEach(chart => {{
                    chart.options.plugins.annotation.annotations.line1.xMin = currentTime;
                    chart.options.plugins.annotation.annotations.line1.xMax = currentTime;
                }});

                const avgCom = comValues.length > 0 ? (comValues.reduce((a,b)=>a+b,0)/comValues.length) : 0;
                avgComVal.innerText = Math.round(avgCom);

                if(sortedData.length > 0) {{
                    // 更新角度圖表 Datasets
                    // 注意：這裡我們需要依照 initCharts 時建立 dataset 的順序來推數據
                    let dsIndex = 0;
                    CONFIG_PARTS.forEach(part => {{
                        if (CONFIG_MODE === "Compare") {{
                            // R
                            angleChart.data.datasets[dsIndex++].data = sortedData.map(d => ({{x: d.time, y: d[`R_${{part}}`]}}));
                            // L
                            angleChart.data.datasets[dsIndex++].data = sortedData.map(d => ({{x: d.time, y: d[`L_${{part}}`]}}));
                        }} else {{
                            const side = (CONFIG_MODE === "Left") ? "L" : "R";
                            angleChart.data.datasets[dsIndex++].data = sortedData.map(d => ({{x: d.time, y: d[`${{side}}_${{part}}`]}}));
                        }}
                    }});
                    
                    comChart.data.datasets[0].data = sortedData.map(d => ({{x: d.time, y: d.comY}}));
                    comChart.data.datasets[1].data = sortedData.map(d => ({{x: d.time, y: avgCom}}));
                    
                    const xMin = Math.max(0, currentTime - 3);
                    const xMax = Math.max(5, currentTime + 2);
                    
                    angleChart.options.scales.x.min = xMin; angleChart.options.scales.x.max = xMax;
                    comChart.options.scales.x.min = xMin; comChart.options.scales.x.max = xMax;
                    
                    angleChart.update('none');
                    comChart.update('none');
                }}
            }}
            
            if (!isScrubbing) {{
                progressBar.value = (video.currentTime / video.duration) * 100;
                timeDisplay.innerText = video.currentTime.toFixed(2) + "s";
            }}
        }}

        function adjustLayout() {{
            if (video.videoWidth) {{
                const containerWidth = (document.body.clientWidth - 40) / 2; 
                const aspectRatio = video.videoWidth / video.videoHeight;
                const newHeight = Math.min(containerWidth / aspectRatio, 600); 
                workspace.style.height = newHeight + 'px';
            }}
        }}
        video.onloadedmetadata = () => {{ adjustLayout(); pose.send({{image: video}}); }};
        window.addEventListener('resize', adjustLayout);

        async function renderFrame() {{
            if (video.paused || video.ended) return;
            await pose.send({{image: video}});
            animationFrameId = requestAnimationFrame(renderFrame);
        }}

        playBtn.onclick = () => {{
            if (video.paused) {{ video.play(); renderFrame(); playBtn.innerText = "⏸ 暫停"; }} 
            else {{ video.pause(); cancelAnimationFrame(animationFrameId); playBtn.innerText = "▶ 播放"; }}
        }};

        let isScrubbing = false;
        progressBar.oninput = () => {{
            isScrubbing = true;
            video.currentTime = (progressBar.value / 100) * video.duration;
            timeDisplay.innerText = video.currentTime.toFixed(2) + "s";
            pose.send({{image: video}});
        }};
        progressBar.onchange = () => {{ isScrubbing = false; if(!video.paused) renderFrame(); }};

        // CSV 下載邏輯升級：支援動態欄位
        downloadCsvBtn.onclick = () => {{
            const sortedData = Array.from(dataStore.values()).sort((a, b) => a.time - b.time);
            if (sortedData.length === 0) {{ alert("沒有數據！"); return; }}
            
            // 動態產生 Header
            let headers = ["Time(s)", "COM_Height_px"];
            CONFIG_PARTS.forEach(part => {{
                if(CONFIG_MODE === "Compare") {{ headers.push(`R_${{part}}`, `L_${{part}}`); }}
                else {{ headers.push(`${{CONFIG_MODE === "Left" ? "L" : "R"}}_${{part}}`); }}
            }});
            
            let csv = headers.join(",") + "\\n";
            
            sortedData.forEach(r => {{
                let row = [r.time, r.comY || ''];
                CONFIG_PARTS.forEach(part => {{
                    if(CONFIG_MODE === "Compare") {{
                        row.push(r[`R_${{part}}`] || '', r[`L_${{part}}`] || '');
                    }} else {{
                        const side = (CONFIG_MODE === "Left") ? "L" : "R";
                        row.push(r[`${{side}}_${{part}}`] || '');
                    }}
                }});
                csv += row.join(",") + "\\n";
            }});
            
            const link = document.createElement("a");
            link.href = "data:text/csv;charset=utf-8," + encodeURI(csv);
            link.download = "biomech_multi_joint_data.csv";
            link.click();
        }};

        recordBtn.onclick = () => {{
            if (isRecording) {{
                mediaRecorder.stop();
                recordBtn.innerText = "🔴 錄製分析影片"; recordBtn.style.background = "#da3633";
                isRecording = false;
            }} else {{
                const stream = overlayCanvas.captureStream(30);
                mediaRecorder = new MediaRecorder(stream, {{ mimeType: 'video/webm' }});
                recordedChunks = [];
                mediaRecorder.ondataavailable = e => {{ if (e.data.size > 0) recordedChunks.push(e.data); }};
                mediaRecorder.onstop = () => {{
                    const blob = new Blob(recordedChunks, {{ type: 'video/webm' }});
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url; a.download = 'analysis_video.webm';
                    a.click();
                }};
                mediaRecorder.start();
                video.play(); renderFrame(); playBtn.innerText = "⏸ 暫停";
                recordBtn.innerText = "⏹ 停止錄製"; recordBtn.style.background = "#bf2c29";
                isRecording = true;
            }}
        }};
    </script>
</body>
</html>
    """

# --- 4. 主程式介面 ---

st.sidebar.title("🎛️ 參數設定")
uploaded_file = st.sidebar.file_uploader("1. 上傳影片", type=['mp4', 'mov', 'avi'])

st.sidebar.markdown("---")
st.sidebar.subheader("2. 分析設定 (多選)")

joint_options = {
    "膝蓋 (Knee)": "Knee", 
    "髖部 (Hip)": "Hip", 
    "手肘 (Elbow)": "Elbow", 
    "肩膀 (Shoulder)": "Shoulder"
}

# [升級點 1] 改為 multiselect，預設選 Knee
selected_joint_labels = st.sidebar.multiselect(
    "選擇分析部位 (可複選):", 
    list(joint_options.keys()), 
    default=["膝蓋 (Knee)"]
)
# 轉換為對應的 Key 列表，例如 ["Knee", "Hip"]
selected_joints = [joint_options[label] for label in selected_joint_labels]

# 將 Python 列表轉換為 JSON 字串傳給 JS
selected_joints_json = json.dumps(selected_joints)

mode_options = {
    "右側 (Right Only)": "Right", 
    "左側 (Left Only)": "Left", 
    "左右對照 (Compare L/R)": "Compare"
}
selected_mode_label = st.sidebar.selectbox("顯示模式:", list(mode_options.keys()))
selected_mode = mode_options[selected_mode_label]

st.sidebar.markdown("---")
st.sidebar.subheader("3. 軌跡與重心")
trail_options = {
    "重心 (Center of Mass)": "COM", "無 (None)": "None",
    "右腳踝 (R.Ankle)": "R.Ankle", "左腳踝 (L.Ankle)": "L.Ankle",
    "右膝 (R.Knee)": "R.Knee", "左膝 (L.Knee)": "L.Knee",
    "右髖 (R.Hip)": "R.Hip", "左髖 (L.Hip)": "L.Hip",
    "右肘 (R.Elbow)": "R.Elbow", "左肘 (L.Elbow)": "L.Elbow", "頭部 (Head)": "Head"
}
selected_trail_label = st.sidebar.selectbox("軌跡追蹤目標:", list(trail_options.keys()), index=0)
selected_trail = trail_options[selected_trail_label]

st.title("劉昱昇動作捕捉分析demo")

if uploaded_file:
    # 確保有選至少一個關節，否則可能會出錯
    if not selected_joints:
        st.warning("請至少選擇一個分析部位")
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
        tfile.write(uploaded_file.read())
        
        with st.spinner("建立多關節分析引擎..."):
            video_b64 = get_video_base64(tfile.name)
            # 傳遞 JSON 格式的陣列
            html_code = get_html_player(video_b64, selected_joints_json, selected_mode, selected_trail)

        st.success(f"✅ 分析就緒: {', '.join(selected_joint_labels)} | {selected_mode_label}")
        components.html(html_code, height=1600)

else:
    st.info("請先從左側上傳影片並調整參數。")