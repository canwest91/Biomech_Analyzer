import streamlit as st
import streamlit.components.v1 as components
import base64
import tempfile
import json

# --- 1. 系統設定 ---
st.set_page_config(layout="wide", page_title="Coach's Eye: Pro Speed", page_icon="🚀")

# CSS 強制設定
st.markdown("""
<style>
    body { overflow: hidden; }
    .stApp { background-color: #0D1117; color: #C9D1D9; height: 100vh; overflow: hidden; }
    
    .block-container { 
        padding: 0 !important; 
        max-width: 100% !important; 
    }
    
    header { background-color: transparent !important; }
    footer { visibility: hidden; height: 0px !important; }
    
    [data-testid="stSidebar"] { 
        background-color: #161B22; 
        border-right: 1px solid #30363D; 
        padding-top: 1rem;
    }
    
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

        /* 影片區 */
        .workspace {{ display: flex; flex-direction: column; gap: 8px; flex: 85; min-height: 0; }}
        .view-container {{ flex: 1; background: #000; position: relative; border-radius: 8px; border: 1px solid #30363d; display: flex; align-items: center; justify-content: center; overflow: hidden; }}
        .view-title {{ position: absolute; top: 5px; left: 5px; background: rgba(0,0,0,0.6); padding: 2px 6px; border-radius: 4px; font-size: 11px; color: #eee; pointer-events: none; }}
        canvas {{ position: absolute; width: 100%; height: 100%; object-fit: contain; }}
        
        /* 控制區 */
        .controls-wrapper {{ background: #161b22; padding: 8px; border-radius: 8px; border: 1px solid #30363d; flex: 15; display: flex; flex-direction: column; justify-content: center; gap: 5px; min-height: 0; }}
        .playback-row {{ display: flex; gap: 8px; align-items: center; }}
        .export-row {{ display: flex; align-items: center; justify-content: space-between; gap: 8px; }}
        
        /* 圖表區 */
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
                    <button id="autoExportBtn" class="auto-export-btn">🎬 生成分析影片 (Auto Export)</button>
                    <button id="snapshotBtn" class="icon-btn">📷 截圖</button>
                    <button id="downloadCsvBtn" class="icon-btn">📊 數據</button>
                </div>
            </div>
        </div>

        <div id="right-panel">
            <div class="chart-card">
                <div class="chart-header">
                    <div class="header-left">
                        <span>📐 Angle</span>
                        <button class="icon-btn" style="padding: 1px 5px; font-size: 10px;" onclick="resetZoom(angleChart)">⟲</button>
                    </div>
                    <span id="currentAngleVal" class="live-value">--°</span>
                </div>
                <div class="chart-container"><canvas id="angleChart"></canvas></div>
            </div>

            <div class="chart-card">
                <div class="chart-header">
                    <div class="header-left">
                        <span>🚀 Linear Speed (m/s)</span>
                        <button class="icon-btn" style="padding: 1px 5px; font-size: 10px;" onclick="resetZoom(velocityChart)">⟲</button>
                    </div>
                    <span id="currentVelVal" class="live-value" style="color: #00E5FF;">-- m/s</span>
                </div>
                <div class="chart-container"><canvas id="velocityChart"></canvas></div>
            </div>

            <div class="chart-card">
                <div class="chart-header">
                    <div class="header-left">
                        <span>🟡 COM</span>
                        <button class="icon-btn" style="padding: 1px 5px; font-size: 10px;" onclick="resetZoom(comChart)">⟲</button>
                    </div>
                    <span id="currentComVal" class="live-value" style="color: #FFD600;">-- px</span>
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

        // 狀態變數
        let prevData = {{ time: null, angleR: null, angleL: null, comX: null, comY: null }};
        let pixelToMeterRatio = null;
        let maxPixelHeight = 0;
        let speedBuffer = []; // [New] 速度平滑緩衝區
        const BUFFER_SIZE = 6; // 平滑窗口大小 (數值越大越平滑但延遲越高)

        const video = document.getElementById('sourceVideo');
        const overlayCanvas = document.getElementById('overlayCanvas');
        const blueprintCanvas = document.getElementById('blueprintCanvas');
        const ctxOverlay = overlayCanvas.getContext('2d');
        const ctxBlueprint = blueprintCanvas.getContext('2d');
        const compositionCanvas = document.getElementById('compositionCanvas');
        const ctxComp = compositionCanvas.getContext('2d');
        
        let compWidth, compHeight, singleVideoWidth, singleVideoHeight;
        const playBtn = document.getElementById('playBtn');
        const progressBar = document.getElementById('progressBar');
        const timeDisplay = document.getElementById('timeDisplay');
        const currentAngleVal = document.getElementById('currentAngleVal');
        const currentVelVal = document.getElementById('currentVelVal');
        const currentComVal = document.getElementById('currentComVal');

        let angleChart, comChart, velocityChart;
        let animationFrameId;
        let trailQueue = []; 
        const MAX_TRAIL_LEN = 40;
        let dataStore = new Map();
        let comValues = [];
        let isExporting = false; 
        let mediaRecorder; 
        let recordedChunks = [];
        const autoExportBtn = document.getElementById('autoExportBtn');

        const JOINT_MAP = {{ "Knee": {{ "R": [24, 26, 28], "L": [23, 25, 27] }}, "Hip": {{ "R": [12, 24, 26], "L": [11, 23, 25] }}, "Elbow": {{ "R": [12, 14, 16], "L": [11, 13, 15] }}, "Shoulder": {{ "R": [14, 12, 24], "L": [13, 11, 23] }} }};
        const COLORS = {{ "R": {{ "Knee": "#00E676", "Hip": "#00B0FF", "Elbow": "#00E5FF", "Shoulder": "#1DE9B6" }}, "L": {{ "Knee": "#FF4081", "Hip": "#FF9100", "Elbow": "#FF5252", "Shoulder": "#FFC400" }} }};
        const TRAIL_MAP = {{ "R.Ankle": 28, "L.Ankle": 27, "R.Knee": 26, "L.Knee": 25, "R.Hip": 24, "L.Hip": 23, "R.Elbow": 14, "L.Elbow": 13, "Head": 0, "COM": "COM" }};

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
            
            // 速度圖表
            velocityChart = new Chart(ctxVel, {{ 
                type: 'line', 
                data: {{ datasets: [{{ label: 'Speed (m/s)', data: [], borderColor: '#00E5FF', borderWidth: 1.5, pointRadius: 0, tension: 0.3 }}] }}, 
                options: commonOptions 
            }});
            
            comChart = new Chart(ctxCom, {{ type: 'scatter', data: {{ datasets: [{{ label: 'COM', data: [], borderColor: '#FFD600', pointRadius: 1.5 }}, {{ type: 'line', label: 'Avg', data: [], borderColor: '#FF4081', borderWidth: 1, borderDash: [5,5], pointRadius: 0 }}] }}, options: commonOptions }});
        }}
        initCharts();
        window.resetZoom = function(chart) {{ chart.resetZoom(); }};

        function calculateAngle(a, b, c) {{
            const rad = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
            let ang = Math.abs(rad * 180.0 / Math.PI); if (ang > 180.0) ang = 360 - ang; return parseInt(ang);
        }}

        function onResults(results) {{
            ctxOverlay.drawImage(results.image, 0, 0, overlayCanvas.width, overlayCanvas.height);
            ctxBlueprint.fillStyle = "black"; ctxBlueprint.fillRect(0, 0, blueprintCanvas.width, blueprintCanvas.height);

            const currentTime = parseFloat(video.currentTime.toFixed(2));
            let newData = {{ time: currentTime, comY: null, comX: null }};

            if (results.poseLandmarks) {{
                const lm = results.poseLandmarks;
                const styleLine = {{color: '#FACE87', lineWidth: 4}};
                const stylePoint = {{color: '#FFFF64', lineWidth: 2, radius: 4}};
                drawConnectors(ctxOverlay, lm, POSE_CONNECTIONS, styleLine);
                drawLandmarks(ctxOverlay, lm, stylePoint);
                drawConnectors(ctxBlueprint, lm, POSE_CONNECTIONS, styleLine);
                drawLandmarks(ctxBlueprint, lm, stylePoint);

                // --- 1. 自動校正 (PPM 計算) ---
                const headY = lm[0].y * overlayCanvas.height;
                const ankleY = ((lm[27].y + lm[28].y) / 2) * overlayCanvas.height;
                const currentPxHeight = Math.abs(ankleY - headY);
                if (currentPxHeight > maxPixelHeight) {{
                    maxPixelHeight = currentPxHeight;
                    pixelToMeterRatio = maxPixelHeight / REAL_HEIGHT;
                }}

                // --- 2. COM (重心) ---
                if (lm[23] && lm[24]) {{
                    const cx = (lm[23].x + lm[24].x) / 2; 
                    const cy = (lm[23].y + lm[24].y) / 2;
                    newData.comX = cx * overlayCanvas.width;
                    newData.comY = cy * overlayCanvas.height;
                    
                    if (CONFIG_TRAIL === "COM") {{
                        ctxOverlay.beginPath(); ctxOverlay.arc(newData.comX, newData.comY, 8, 0, 2*Math.PI); ctxOverlay.fillStyle = "#FF4081"; ctxOverlay.fill();
                        trailQueue.push({{x: newData.comX, y: newData.comY}}); 
                        if(trailQueue.length > MAX_TRAIL_LEN) trailQueue.shift();
                        if (trailQueue.length > 1) {{
                            ctxOverlay.beginPath(); ctxOverlay.moveTo(trailQueue[0].x, trailQueue[0].y);
                            for (let i = 1; i < trailQueue.length; i++) ctxOverlay.lineTo(trailQueue[i].x, trailQueue[i].y);
                            ctxOverlay.strokeStyle = "#FFD600"; ctxOverlay.lineWidth = 4; ctxOverlay.stroke();
                        }}
                    }}
                }}

                // --- 3. 速度計算 (物理引擎 V = d/t) ---
                let speedMps = 0;
                
                // 只有在數據有效且已經有上一幀數據時才計算
                if (pixelToMeterRatio && newData.comX !== null && prevData.comX !== null && currentTime !== prevData.time) {{
                    const dt = currentTime - prevData.time; // 秒 (時間差)
                    
                    if (dt > 0.01) {{ // 避免時間差過小導致除以零或無限大
                        const dx = newData.comX - prevData.comX;
                        const dy = newData.comY - prevData.comY;
                        const distPx = Math.sqrt(dx*dx + dy*dy); // 像素位移
                        
                        // [雜訊閘]：如果移動小於 2 像素，視為靜止
                        if (distPx > 2) {{
                            const distM = distPx / pixelToMeterRatio; // 位移 (公尺)
                            const rawSpeed = distM / dt; // 原始速度 (m/s)
                            
                            // [平滑化]：加入緩衝區計算移動平均
                            speedBuffer.push(rawSpeed);
                            if (speedBuffer.length > BUFFER_SIZE) speedBuffer.shift();
                            const sum = speedBuffer.reduce((a, b) => a + b, 0);
                            speedMps = sum / speedBuffer.length;
                        }} else {{
                            speedMps = 0;
                            speedBuffer = []; // 重置緩衝
                        }}
                    }}
                }}
                
                // --- 4. 角度 ---
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

                // 更新 UI
                currentAngleVal.innerText = displayTexts.join(" | ");
                currentVelVal.innerText = speedMps.toFixed(2) + " m/s"; // 顯示公尺/秒
                if(newData.comY) currentComVal.innerText = newData.comY + " px";

                if (!video.paused) {{
                    prevData = {{ ...newData }};
                    dataStore.set(currentTime.toFixed(2), {{ ...newData, speed: speedMps }});
                    if(newData.comY) {{ comValues.push(newData.comY); if (comValues.length > 200) comValues.shift(); }}
                }}

                const sortedData = Array.from(dataStore.values()).sort((a, b) => a.time - b.time);
                
                [angleChart, velocityChart, comChart].forEach(c => {{
                    c.options.scales.x.min = 0; c.options.scales.x.max = video.duration || 10;
                    c.options.plugins.annotation.annotations.line1.xMin = currentTime; c.options.plugins.annotation.annotations.line1.xMax = currentTime;
                    c.update('none');
                }});

                const avgCom = comValues.length > 0 ? (comValues.reduce((a,b)=>a+b,0)/comValues.length) : 0;

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
                    
                    // [關鍵] 更新速度圖表 (m/s)
                    velocityChart.data.datasets[0].data = sortedData.map(d => ({{x: d.time, y: d.speed}}));

                    comChart.data.datasets[0].data = sortedData.map(d => ({{x: d.time, y: d.comY}}));
                    comChart.data.datasets[1].data = sortedData.map(d => ({{x: d.time, y: avgCom}}));
                    
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
                {{ canvas: velocityChart.canvas, title: "Speed (m/s)", value: currentVelVal.innerText, color: "#00E5FF" }},
                {{ canvas: comChart.canvas, title: "COM", value: currentComVal.innerText, color: "#FFD600" }}
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
        progressBar.oninput = () => {{ isScrubbing = true; prevData = {{ time: null, angleR: null, angleL: null }}; video.currentTime = (progressBar.value / 100) * video.duration; timeDisplay.innerText = video.currentTime.toFixed(2) + "s"; pose.send({{image: video}}); }};
        progressBar.onchange = () => {{ isScrubbing = false; if(!video.paused) renderFrame(); }};

        document.getElementById('downloadCsvBtn').onclick = () => {{
            const arr = Array.from(dataStore.values()).sort((a,b)=>a.time-b.time);
            if(!arr.length) return alert("無數據");
            let csv = "Time,Angle,Speed_mps,COM_Y\\n";
            arr.forEach(r=>{{ csv+=`${{r.time}},${{r.angleR||r.angleL||''}},${{r.speed||0}},${{r.comY||''}}\\n` }});
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
                const a = document.createElement('a'); a.href = url; a.download = 'CoachsEye_Speed_Analysis.webm'; a.click();
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
# [新增] 身高輸入
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
trail_options = {"重心": "COM", "無": "None", "右膝": "R.Knee"}
selected_trail_label = st.sidebar.selectbox("5. Trail:", list(trail_options.keys()), index=0)
selected_trail = trail_options[selected_trail_label]

if uploaded_file:
    if not selected_joints:
        st.warning("Please select a body part.")
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
        tfile.write(uploaded_file.read())
        with st.spinner("Initializing Physics Engine..."):
            video_b64 = get_video_base64(tfile.name)
            # 傳入 user_height 參數
            html_code = get_html_player(video_b64, selected_joints_json, selected_mode, selected_trail, reference_height)
        components.html(html_code, height=1000, scrolling=False) 
else:
    st.info("👈 Please upload video.")