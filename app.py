import streamlit as st
import streamlit.components.v1 as components
import base64
import tempfile
import json

# --- 1. 系統設定 ---
st.set_page_config(layout="wide", page_title="Coach's Eye: Pro Speed", page_icon="🚀")
st.markdown("""
<style>
    body { overflow: hidden; }
    .stApp { background-color: #0D1117; color: #C9D1D9; height: 100vh; overflow: hidden; }
    .block-container { padding: 0 !important; max-width: 100% !important; }
    header { background-color: transparent !important; }
    footer { visibility: hidden; height: 0px !important; }
    [data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; padding-top: 1rem; }
    iframe { width: 100% !important; height: 100vh !important; }
</style>
""", unsafe_allow_html=True)

# --- 2. 輔助函式 ---
def get_video_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
        return base64.b64encode(data).decode()

# --- 3. HTML/JS 播放器模板 ---
def get_html_player(video_base64, joint_parts_json, display_mode, trail_target, user_height):
    return f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/pose/pose.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@2.1.0/dist/chartjs-plugin-annotation.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/hammerjs@2.0.8"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1/dist/chartjs-plugin-zoom.min.js"></script>
    
    <style>
        body {{ margin: 0; background-color: #0d1117; color: #c9d1d9; font-family: 'Segoe UI', sans-serif; height: 100vh; width: 100vw; overflow: hidden; }}
        #dashboard-container {{ display: flex; height: 100%; width: 100%; gap: 8px; padding: 8px; box-sizing: border-box; }}
        #left-panel {{ flex: 55; display: flex; flex-direction: column; gap: 8px; height: 100%; min-width: 0; }}
        #right-panel {{ flex: 45; display: flex; flex-direction: column; gap: 8px; height: 100%; min-width: 0; }}
        .workspace {{ display: flex; flex-direction: column; gap: 8px; flex: 85; min-height: 0; }}
        .view-container {{ flex: 1; background: #000; position: relative; border-radius: 8px; border: 1px solid #30363d; display: flex; align-items: center; justify-content: center; overflow: hidden; }}
        .view-title {{ position: absolute; top: 5px; left: 5px; background: rgba(0,0,0,0.6); padding: 2px 6px; border-radius: 4px; font-size: 11px; color: #eee; pointer-events: none; }}
        canvas {{ position: absolute; width: 100%; height: 100%; object-fit: contain; }}
        .controls-wrapper {{ background: #161b22; padding: 8px; border-radius: 8px; border: 1px solid #30363d; flex: 15; display: flex; flex-direction: column; justify-content: center; gap: 5px; min-height: 0; }}
        .playback-row {{ display: flex; gap: 8px; align-items: center; }}
        .export-row {{ display: flex; align-items: center; justify-content: space-between; gap: 8px; }}
        .chart-card {{ background: #161b22; border-radius: 8px; border: 1px solid #30363d; padding: 5px 8px; flex: 1; min-height: 0; display: flex; flex-direction: column; }}
        .chart-container {{ flex: 1; position: relative; width: 100%; min-height: 0; }}
        .chart-header {{ display: flex; justify-content: space-between; align-items: center; font-size: 11px; color: #8b949e; height: 18px; margin-bottom: 2px; }}
        .header-left {{ display: flex; gap: 5px; align-items: center; }}
        .live-value {{ color: #fff; font-family: monospace; font-size: 13px; font-weight: bold; }}
        button {{ background: #238636; color: white; border: none; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 11px; white-space: nowrap; }}
        button:hover {{ background: #2ea043; }}
        button.auto-export-btn {{ background: linear-gradient(90deg, #8957e5, #b355e6); color: white; flex: 1; }}
        button.icon-btn {{ background: #30363d; color: #c9d1d9; border: 1px solid #8b949e; }}
        input[type="range"] {{ flex: 1; accent-color: #238636; cursor: pointer; height: 5px; }}
        #compositionCanvas {{ display: none; }}
    </style>
</head>
<body>
    <video id="sourceVideo" style="display:none" playsinline muted>
        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
    </video>
    <canvas id="compositionCanvas"></canvas>

    <div id="dashboard-container">
        <div id="left-panel">
            <div class="workspace">
                <div class="view-container">
                    <div class="view-title">Blueprint</div>
                    <canvas id="blueprintCanvas"></canvas>
                </div>
                <div class="view-container">
                    <div class="view-title">Video Overlay</div>
                    <canvas id="overlayCanvas"></canvas>
                </div>
            </div>
            <div class="controls-wrapper">
                <div class="playback-row">
                    <button id="playBtn" style="width: 60px;">▶ Play</button>
                    <input type="range" id="progressBar" value="0" min="0" max="100" step="0.1">
                    <span id="timeDisplay" style="font-family: monospace; font-size: 14px; color: #00E676; width: 50px; text-align: right;">0.00s</span>
                </div>
                <div class="export-row">
                    <button id="autoExportBtn" class="auto-export-btn">🎬 生成分析影片</button>
                    <button id="snapshotBtn" class="icon-btn">📷 截圖</button>
                    <button id="downloadCsvBtn" class="icon-btn">📊 數據</button>
                </div>
            </div>
        </div>

        <div id="right-panel">
            <div class="chart-card">
                <div class="chart-header">
                    <div class="header-left">
                        <span>關節角度</span>
                        <button class="icon-btn" style="padding: 1px 5px; font-size: 10px;" onclick="resetZoom(angleChart)">⟲</button>
                    </div>
                    <span id="currentAngleVal" class="live-value">--°</span>
                </div>
                <div class="chart-container"><canvas id="angleChart"></canvas></div>
            </div>
            <div class="chart-card">
                <div class="chart-header">
                    <div class="header-left">
                        <span>水平速度 (COM X)</span>
                        <button class="icon-btn" style="padding: 1px 5px; font-size: 10px;" onclick="resetZoom(velocityChart)">⟲</button>
                    </div>
                    <span id="currentVelVal" class="live-value" style="color: #00E5FF;">-- m/s</span>
                </div>
                <div class="chart-container"><canvas id="velocityChart"></canvas></div>
            </div>
            <div class="chart-card">
                <div class="chart-header">
                    <div class="header-left">
                        <span>🟡 垂直振幅 (Vertical Oscillation)</span>
                        <button class="icon-btn" style="padding: 1px 5px; font-size: 10px;" onclick="resetZoom(comChart)">⟲</button>
                    </div>
                    <span id="currentComVal" class="live-value" style="color: #FFD600;">-- cm</span>
                </div>
                <div class="chart-container"><canvas id="comChart"></canvas></div>
            </div>
        </div>
    </div>

    <script>
        const CONFIG_PARTS = JSON.parse('{joint_parts_json}');
        const CONFIG_MODE = "{display_mode}";
        const CONFIG_TRAIL = "{trail_target}";
        const REAL_HEIGHT = {user_height};

        // --- 核心狀態變數 ---
        let prevData = {{ time: null, comX: null, comY: null }};
        let pixelToMeterRatio = null;
        let maxPixelHeight = 0; // [修正3] 用最大身高來鎖定比例尺

        // 垂直振幅緩衝區 (Oscillation Buffer)
        let oscBuffer = []; 
        const OSC_BUFFER_SIZE = 15; // 縮小視窗，讓反應更靈敏

        // 速度平滑變數 (EMA)
        let smoothedSpeed = 0;
        const SPEED_ALPHA = 0.15; 

        // 畫布與元件
        const video = document.getElementById('sourceVideo');
        const overlayCanvas = document.getElementById('overlayCanvas');
        const blueprintCanvas = document.getElementById('blueprintCanvas');
        const ctxOverlay = overlayCanvas.getContext('2d');
        const ctxBlueprint = blueprintCanvas.getContext('2d');
        const compositionCanvas = document.getElementById('compositionCanvas');
        const ctxComp = compositionCanvas.getContext('2d');
        const playBtn = document.getElementById('playBtn');
        const progressBar = document.getElementById('progressBar');
        const timeDisplay = document.getElementById('timeDisplay');
        const currentAngleVal = document.getElementById('currentAngleVal');
        const currentVelVal = document.getElementById('currentVelVal');
        const currentComVal = document.getElementById('currentComVal');
        const autoExportBtn = document.getElementById('autoExportBtn');

        let angleChart, comChart, velocityChart;
        let animationFrameId;
        let trailQueue = []; 
        const MAX_TRAIL_LEN = 40;
        let dataStore = new Map();
        let isExporting = false; 
        let mediaRecorder, recordedChunks = [];

        // 關節與顏色定義
        const JOINT_MAP = {{ "Knee": {{ "R": [24, 26, 28], "L": [23, 25, 27] }}, "Hip": {{ "R": [12, 24, 26], "L": [11, 23, 25] }}, "Elbow": {{ "R": [12, 14, 16], "L": [11, 13, 15] }}, "Shoulder": {{ "R": [14, 12, 24], "L": [13, 11, 23] }} }};
        const COLORS = {{ "R": {{ "Knee": "#00E676", "Hip": "#00B0FF", "Elbow": "#00E5FF", "Shoulder": "#1DE9B6" }}, "L": {{ "Knee": "#FF4081", "Hip": "#FF9100", "Elbow": "#FF5252", "Shoulder": "#FFC400" }} }};
        const SEGMENT_WEIGHTS = [{{ indices: [0], weight: 0.081 }}, {{ indices: [11, 12, 23, 24], weight: 0.497 }}, {{ indices: [11, 13], weight: 0.028 }}, {{ indices: [12, 14], weight: 0.028 }}, {{ indices: [13, 15], weight: 0.016 }}, {{ indices: [14, 16], weight: 0.016 }}, {{ indices: [23, 25], weight: 0.100 }}, {{ indices: [24, 26], weight: 0.100 }}, {{ indices: [25, 27], weight: 0.0465 }}, {{ indices: [26, 28], weight: 0.0465 }}, {{ indices: [27, 31], weight: 0.0145 }}, {{ indices: [28, 32], weight: 0.0145 }}];

        const pose = new Pose({{locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${{file}}`}});
        pose.setOptions({{modelComplexity: 1, smoothLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5}});
        pose.onResults(onResults);

        const commonOptions = {{
            responsive: true, maintainAspectRatio: false, animation: false,
            interaction: {{ mode: 'nearest', axis: 'x', intersect: false }},
            scales: {{ x: {{ type: 'linear', display: true, grid: {{ color: '#30363d', lineWidth: 0.5 }}, ticks: {{ color: '#8b949e', font: {{size: 10}}, callback: v=>v.toFixed(1)+'s' }} }}, y: {{ grid: {{ color: '#30363d', lineWidth: 0.5 }}, ticks: {{ color: '#8b949e', font: {{size: 10}} }} }} }},
            plugins: {{ legend: {{ labels: {{ color: 'white', font: {{size: 10}}, boxWidth: 10 }} }}, annotation: {{ annotations: {{ line1: {{ type: 'line', xMin: 0, xMax: 0, borderColor: 'rgba(255,255,255,0.8)', borderWidth: 2, borderDash: [0,0] }} }} }}, zoom: {{ pan: {{ enabled: true, mode: 'x' }}, zoom: {{ wheel: {{ enabled: true }}, pinch: {{ enabled: true }}, mode: 'x' }} }} }}
        }};

        function initCharts() {{
            const ctxAngle = document.getElementById('angleChart').getContext('2d');
            const ctxVel = document.getElementById('velocityChart').getContext('2d');
            const ctxCom = document.getElementById('comChart').getContext('2d');
            
            let angleDatasets = [];
            CONFIG_PARTS.forEach(part => {{
                if (CONFIG_MODE === "Compare") {{
                    angleDatasets.push({{ label: `R.${{part}}`, data: [], borderColor: COLORS["R"][part], borderWidth: 1.5, pointRadius: 0, tension: 0.1 }});
                    angleDatasets.push({{ label: `L.${{part}}`, data: [], borderColor: COLORS["L"][part], borderWidth: 1.5, pointRadius: 0, tension: 0.1 }});
                }} else {{
                    const side = (CONFIG_MODE === "Left") ? "L" : "R";
                    angleDatasets.push({{ label: `${{side}}.${{part}}`, data: [], borderColor: COLORS[side][part], borderWidth: 1.5, pointRadius: 0, tension: 0.1 }});
                }}
            }});
            angleChart = new Chart(ctxAngle, {{ type: 'line', data: {{ datasets: angleDatasets }}, options: commonOptions }});
            
            velocityChart = new Chart(ctxVel, {{ type: 'line', data: {{ datasets: [{{ label: 'X Speed (m/s)', data: [], borderColor: '#00E5FF', borderWidth: 1.5, pointRadius: 0, tension: 0.3 }}] }}, options: commonOptions }});
            
            comChart = new Chart(ctxCom, {{ 
                type: 'line', 
                data: {{ datasets: [{{ label: 'Oscillation (cm)', data: [], borderColor: '#FFD600', borderWidth: 1.5, pointRadius: 0, tension: 0.3, fill: true, backgroundColor: 'rgba(255, 214, 0, 0.1)' }}] }}, 
                options: {{ ...commonOptions, scales: {{ ...commonOptions.scales, y: {{ ...commonOptions.scales.y, title: {{ display: true, text: 'cm' }}, suggestedMin: 0, suggestedMax: 15 }} }} }} 
            }});
        }}
        initCharts();
        window.resetZoom = function(chart) {{ chart.resetZoom(); }};

        function calculateAngle(a, b, c) {{
            const rad = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
            let ang = Math.abs(rad * 180.0 / Math.PI); if (ang > 180.0) ang = 360 - ang; return parseInt(ang);
        }}

        function calculateWeightedCOM(lm, width, height) {{
            let totalX = 0, totalY = 0, totalWeight = 0;
            SEGMENT_WEIGHTS.forEach(seg => {{
                let segX = 0, segY = 0, validPoints = 0;
                seg.indices.forEach(idx => {{ if(lm[idx] && lm[idx].visibility > 0.5) {{ segX += lm[idx].x; segY += lm[idx].y; validPoints++; }} }});
                if (validPoints > 0) {{ totalX += (segX/validPoints)*seg.weight; totalY += (segY/validPoints)*seg.weight; totalWeight += seg.weight; }}
            }});
            return totalWeight > 0.5 ? {{ x: (totalX/totalWeight)*width, y: (totalY/totalWeight)*height }} : null;
        }}

        function onResults(results) {{
            ctxOverlay.drawImage(results.image, 0, 0, overlayCanvas.width, overlayCanvas.height);
            ctxBlueprint.fillStyle = "black"; ctxBlueprint.fillRect(0, 0, blueprintCanvas.width, blueprintCanvas.height);

            const currentTime = parseFloat(video.currentTime.toFixed(2));
            let newData = {{ time: currentTime, comY: null, comX: null, vertOscCm: 0, speed: 0 }};

            if (results.poseLandmarks) {{
                const lm = results.poseLandmarks;
                drawConnectors(ctxOverlay, lm, POSE_CONNECTIONS, {{color: '#FACE87', lineWidth: 4}});
                drawLandmarks(ctxOverlay, lm, {{color: '#FFFF64', lineWidth: 2, radius: 4}});
                drawConnectors(ctxBlueprint, lm, POSE_CONNECTIONS, {{color: '#FACE87', lineWidth: 4}});
                drawLandmarks(ctxBlueprint, lm, {{color: '#FFFF64', lineWidth: 2, radius: 4}});

                // --- [修正3] PPM: 改回「最大身高鎖定」策略 (解決起跑蹲踞導致 PPM 過小問題) ---
                // 只有當人體比例合理時才更新，且只會變大不會變小 (鎖定在站立時的身高)
                const headY = lm[0].y * overlayCanvas.height;
                const ankleY = ((lm[27].y + lm[28].y) / 2) * overlayCanvas.height; 
                const currentPxHeight = Math.abs(ankleY - headY);
                
                // 過濾明顯錯誤的偵測 (太小或太誇張)
                if (currentPxHeight > overlayCanvas.height * 0.2) {{
                     if (currentPxHeight > maxPixelHeight) {{
                        maxPixelHeight = currentPxHeight;
                        pixelToMeterRatio = maxPixelHeight / REAL_HEIGHT;
                     }}
                }}
                
                // 如果還沒偵測到有效身高，先用當前頂著，避免除以零
                const activePPM = pixelToMeterRatio || (currentPxHeight / REAL_HEIGHT);

                // --- 2. COM 與 垂直振幅 ---
                const com = calculateWeightedCOM(lm, overlayCanvas.width, overlayCanvas.height);
                if (com) {{
                    newData.comX = com.x;
                    newData.comY = com.y;

                    if (CONFIG_TRAIL === "COM") {{
                        ctxOverlay.beginPath(); ctxOverlay.arc(newData.comX, newData.comY, 8, 0, 2*Math.PI); ctxOverlay.fillStyle = "#FF4081"; ctxOverlay.fill();
                        trailQueue.push({{x: newData.comX, y: newData.comY}}); 
                        if(trailQueue.length > MAX_TRAIL_LEN) trailQueue.shift();
                        if(trailQueue.length > 1) {{
                            ctxOverlay.beginPath(); ctxOverlay.moveTo(trailQueue[0].x, trailQueue[0].y);
                            for(let i=1; i<trailQueue.length; i++) ctxOverlay.lineTo(trailQueue[i].x, trailQueue[i].y);
                            ctxOverlay.strokeStyle = "#FFD600"; ctxOverlay.lineWidth = 4; ctxOverlay.stroke();
                        }}
                    }}

                    // --- [修正1] 垂直振幅: 解決 240cm 暴衝問題 ---
                    // 使用 COM Y 軸的像素變化，而非絕對高度
                    // 加入邊界檢查：如果 COM Y 突然變成 0 或極大，忽略該幀
                    if (newData.comY > 0 && newData.comY < overlayCanvas.height) {{
                        oscBuffer.push(newData.comY);
                        if (oscBuffer.length > OSC_BUFFER_SIZE) oscBuffer.shift();

                        // 只有當緩衝區填滿一半以上才開始算，避免初期震盪
                        if (oscBuffer.length > 3) {{
                            const minPx = Math.min(...oscBuffer);
                            const maxPx = Math.max(...oscBuffer);
                            const oscPx = maxPx - minPx; // 像素振幅
                            
                            // 轉成 cm
                            if (activePPM > 0) {{
                                newData.vertOscCm = (oscPx / activePPM) * 100;
                            }}
                        }}
                    }}
                }}

                // --- [修正2] 速度計算: 解決歸零斷崖問題 ---
                if (activePPM && newData.comX !== null && prevData.comX !== null && currentTime !== prevData.time) {{
                    const dt = currentTime - prevData.time;
                    if (dt > 0.016) {{ 
                        const dx = newData.comX - prevData.comX;
                        const distPx = Math.abs(dx); 
                        
                        if (distPx > 1.5) {{ // 降低一點門檻
                            const distM = distPx / activePPM;
                            let rawSpeed = distM / dt;
                            // 物理夾具：人類極限過濾
                            if (rawSpeed > 13.0) rawSpeed = smoothedSpeed;
                            smoothedSpeed = (rawSpeed * SPEED_ALPHA) + (smoothedSpeed * (1 - SPEED_ALPHA));
                        }} else {{
                            // [關鍵] 柔性衰減 (Soft Decay)，不要直接歸零
                            smoothedSpeed = smoothedSpeed * 0.92; 
                            if (smoothedSpeed < 0.05) smoothedSpeed = 0;
                        }}
                        newData.speed = smoothedSpeed;
                    }} else {{
                        newData.speed = smoothedSpeed;
                    }}
                }}
                
                // --- 5. 角度與 UI ---
                let displayTexts = [];
                CONFIG_PARTS.forEach(part => {{
                    const jointData = JOINT_MAP[part];
                    let sides = (CONFIG_MODE === "Compare") ? ["R", "L"] : [(CONFIG_MODE === "Left") ? "L" : "R"];
                    sides.forEach(side => {{
                        const ids = jointData[side];
                        if (lm[ids[0]] && lm[ids[1]] && lm[ids[2]]) {{
                            const ang = calculateAngle(lm[ids[0]], lm[ids[1]], lm[ids[2]]);
                            newData[`${{side}}_${{part}}`] = ang;
                            displayTexts.push(`${{side}}:${{ang}}`);
                            const center = lm[ids[1]];
                            const txtX = center.x * overlayCanvas.width + (side === "R" ? 15 : -65);
                            const txtY = center.y * overlayCanvas.height;
                            const color = COLORS[side][part];
                            [ctxOverlay, ctxBlueprint].forEach(ctx => {{
                                ctx.font = "bold 20px Arial"; ctx.fillStyle = color; ctx.strokeStyle = "black"; ctx.lineWidth = 3;
                                ctx.strokeText(ang + "°", txtX, txtY); ctx.fillText(ang + "°", txtX, txtY);
                            }});
                        }}
                    }});
                }});

                currentAngleVal.innerText = displayTexts.join(" | ");
                currentVelVal.innerText = newData.speed.toFixed(2) + " m/s";
                currentComVal.innerText = (newData.vertOscCm || 0).toFixed(1) + " cm";

                if (!video.paused) {{
                    prevData = {{ ...newData }};
                    dataStore.set(currentTime.toFixed(2), {{ ...newData }});
                }}

                const sortedData = Array.from(dataStore.values()).sort((a, b) => a.time - b.time);
                
                [angleChart, velocityChart, comChart].forEach(c => {{
                    c.options.scales.x.min = 0; c.options.scales.x.max = video.duration || 10;
                    c.options.plugins.annotation.annotations.line1.xMin = currentTime; c.options.plugins.annotation.annotations.line1.xMax = currentTime;
                    c.update('none');
                }});

                if(sortedData.length > 0) {{
                    let dsIndex = 0;
                    CONFIG_PARTS.forEach(part => {{
                        if (CONFIG_MODE === "Compare") {{
                            angleChart.data.datasets[dsIndex++].data = sortedData.map(d => ({{x: d.time, y: d[`R_${{part}}`]}}));
                            angleChart.data.datasets[dsIndex++].data = sortedData.map(d => ({{x: d.time, y: d[`L_${{part}}`]}}));
                        }} else {{
                            const side = (CONFIG_MODE === "Left") ? "L" : "R";
                            angleChart.data.datasets[dsIndex++].data = sortedData.map(d => ({{x: d.time, y: d[`${{side}}_${{part}}`]}}));
                        }}
                    }});
                    velocityChart.data.datasets[0].data = sortedData.map(d => ({{x: d.time, y: d.speed}}));
                    comChart.data.datasets[0].data = sortedData.map(d => ({{x: d.time, y: d.vertOscCm}}));
                    [angleChart, velocityChart, comChart].forEach(c => {{ c.update('none'); }});
                }}
            }}
            
            if (!isScrubbing) {{
                progressBar.value = (video.currentTime / video.duration) * 100;
                timeDisplay.innerText = video.currentTime.toFixed(2) + "s";
            }}
        }}

        function adjustLayout() {{
            if (video.videoWidth) {{
                singleVideoWidth = video.videoWidth; singleVideoHeight = video.videoHeight;
                compWidth = singleVideoWidth * 2; compHeight = singleVideoHeight * 2; 
                compositionCanvas.width = compWidth; compositionCanvas.height = compHeight;
                overlayCanvas.width = singleVideoWidth; overlayCanvas.height = singleVideoHeight;
                blueprintCanvas.width = singleVideoWidth; blueprintCanvas.height = singleVideoHeight;
                [angleChart, velocityChart, comChart].forEach(c => {{ c.options.scales.x.max = video.duration; c.update(); }});
            }}
        }}
        video.onloadedmetadata = () => {{ adjustLayout(); pose.send({{image: video}}); }};
        window.addEventListener('resize', adjustLayout);

        function drawCompositionFrame() {{
            if (!compWidth || !compHeight) return;
            ctxComp.fillStyle = '#0d1117'; ctxComp.fillRect(0, 0, compWidth, compHeight);
            ctxComp.drawImage(blueprintCanvas, 0, 0, singleVideoWidth, singleVideoHeight);
            ctxComp.drawImage(overlayCanvas, 0, singleVideoHeight, singleVideoWidth, singleVideoHeight);
            
            ctxComp.font = 'bold 30px Arial'; ctxComp.fillStyle = 'rgba(255,255,255,0.8)';
            ctxComp.fillText("Blueprint", 20, 50); ctxComp.fillText("Overlay", 20, singleVideoHeight + 50);

            const eachChartHeight = compHeight / 3; const rightColX = singleVideoWidth;
            const charts = [
                {{ canvas: angleChart.canvas, title: "Angle", value: currentAngleVal.innerText, color: "white" }},
                {{ canvas: velocityChart.canvas, title: "Speed X (m/s)", value: currentVelVal.innerText, color: "#00E5FF" }},
                {{ canvas: comChart.canvas, title: "Oscillation (cm)", value: currentComVal.innerText, color: "#FFD600" }}
            ];
            charts.forEach((item, idx) => {{
                const yPos = idx * eachChartHeight;
                ctxComp.fillStyle = '#161b22'; ctxComp.fillRect(rightColX, yPos, singleVideoWidth, eachChartHeight);
                ctxComp.drawImage(item.canvas, rightColX, yPos, singleVideoWidth, eachChartHeight);
                ctxComp.font = 'bold 28px Arial'; ctxComp.fillStyle = '#c9d1d9'; ctxComp.fillText(item.title, rightColX + 20, yPos + 50);
                ctxComp.font = 'bold 40px monospace'; ctxComp.fillStyle = item.color; ctxComp.textAlign = 'right'; ctxComp.fillText(item.value, compWidth - 20, yPos + 60); ctxComp.textAlign = 'left'; 
            }});
            ctxComp.font = 'bold 40px monospace'; ctxComp.fillStyle = '#00E676'; ctxComp.fillText(timeDisplay.innerText, compWidth - 20, 50); 
        }}

        async function renderFrame() {{
            if (video.paused || video.ended) return;
            await pose.send({{image: video}});
            if (isExporting) drawCompositionFrame();
            animationFrameId = requestAnimationFrame(renderFrame);
        }}
        playBtn.onclick = () => {{ if (video.paused) {{ video.play(); renderFrame(); playBtn.innerText = "⏸"; }} else {{ video.pause(); cancelAnimationFrame(animationFrameId); playBtn.innerText = "▶"; }} }};
        
        let isScrubbing = false;
        progressBar.oninput = () => {{ isScrubbing = true; prevData = {{ time: null, comX: null }}; video.currentTime = (progressBar.value / 100) * video.duration; timeDisplay.innerText = video.currentTime.toFixed(2) + "s"; pose.send({{image: video}}); }};
        progressBar.onchange = () => {{ isScrubbing = false; if(!video.paused) renderFrame(); }};

        document.getElementById('downloadCsvBtn').onclick = () => {{
            const arr = Array.from(dataStore.values()).sort((a,b)=>a.time-b.time);
            if(!arr.length) return alert("無數據");
            let csv = "Time,Angle,Speed_X_mps,Oscillation_cm\\n";
            arr.forEach(r=>{{ csv+=`${{r.time}},${{r.angleR||r.angleL||''}},${{r.speed||0}},${{r.vertOscCm||''}}\\n` }});
            const a = document.createElement("a"); a.href="data:text/csv;charset=utf-8,"+encodeURI(csv); a.download="data.csv"; a.click();
        }};
        
        autoExportBtn.onclick = () => {{
            if (isExporting) return;
            isExporting = true; autoExportBtn.innerText = "⏳ Recording...";
            const stream = compositionCanvas.captureStream(60); 
            mediaRecorder = new MediaRecorder(stream, {{ mimeType: 'video/webm;codecs=vp9', videoBitsPerSecond: 8000000 }});
            recordedChunks = [];
            mediaRecorder.ondataavailable = e => {{ if (e.data.size > 0) recordedChunks.push(e.data); }};
            mediaRecorder.onstop = () => {{
                const blob = new Blob(recordedChunks, {{ type: 'video/webm' }}); const url = URL.createObjectURL(blob);
                const a = document.createElement('a'); a.href = url; a.download = 'CoachsEye_Analysis.webm'; a.click();
                isExporting = false; autoExportBtn.innerText = "🎬 自動生成影片";
            }};
            video.currentTime = 0; video.onseeked = () => {{ video.onseeked = null; mediaRecorder.start(); video.play(); renderFrame(); }}; video.onended = () => {{ mediaRecorder.stop(); video.onended = null; }};
        }};
    </script>
</body>
</html>
    """

# --- 4. 主程式介面 ---
st.sidebar.title("Settings")
uploaded_file = st.sidebar.file_uploader("1. Video", type=['mp4', 'mov', 'avi'])
st.sidebar.markdown("---")
reference_height = st.sidebar.number_input("2. Subject Height (m):", min_value=1.0, max_value=2.5, value=1.75, step=0.01)
st.sidebar.markdown("---")
joint_options = {"膝蓋 (Knee)": "Knee", "髖部 (Hip)": "Hip", "手肘 (Elbow)": "Elbow"}
selected_joint_labels = st.sidebar.multiselect("3. Joint Analysis:", list(joint_options.keys()), default=["膝蓋 (Knee)"])
selected_joints = [joint_options[label] for label in selected_joint_labels]
selected_joints_json = json.dumps(selected_joints)
mode_options = {"右側 (Right)": "Right", "左側 (Left)": "Left", "對照 (Compare)": "Compare"}
selected_mode_label = st.sidebar.selectbox("4. Side:", list(mode_options.keys()))
selected_mode = mode_options[selected_mode_label]
st.sidebar.markdown("---")
trail_options = {"重心 (COM)": "COM", "無": "None", "右膝": "R.Knee"}
selected_trail_label = st.sidebar.selectbox("5. Trail:", list(trail_options.keys()), index=0)
selected_trail = trail_options[selected_trail_label]

if uploaded_file:
    if not selected_joints:
        st.warning("請選擇一個身體關節點進行分析。")
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
        tfile.write(uploaded_file.read())
        with st.spinner("分析中..."):
            video_b64 = get_video_base64(tfile.name)
            html_code = get_html_player(video_b64, selected_joints_json, selected_mode, selected_trail, reference_height)
        components.html(html_code, height=1000, scrolling=False) 
else:
    st.info("請上傳影片")