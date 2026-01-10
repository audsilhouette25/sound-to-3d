let audioCtx, analyser, microphone, mediaRecorder, sourceNode;
let audioChunks = [];
let audioTag = null;
let brain;
let scene, camera, renderer, currentMesh;
let state = 'IDLE'; 
let isPlaying = false; 
let recordedX = { loudness: 0, pitch: 0, brightness: 0, roughness: 0, count: 0 };
let currentX = { loudness: 0, pitch: 0, brightness: 0, roughness: 0 };
let targetY = { y1: 0.5, y2: 0.5, y3: 0.5, y4: 0.5, shape: 0 };
let currentY = { y1: 0.5, y2: 0.5, y3: 0.5, y4: 0.5, shape: 0 };
let customTrainingData = [];
let currentLang = 'KR';
let micStream = null;

// --- GPU Shader Code (Fixed Noise Function) ---
const vertexShader = `
    varying float vDisplacement;
    varying vec3 vNormal;
    uniform float uTime;
    uniform float uLoudness;
    uniform float uY1, uY2, uY3, uY4;

    float hash(float n) { return fract(sin(n) * 43758.5453123); }
    float noise(vec3 x) {
        vec3 p = floor(x); vec3 f = fract(x);
        f = f*f*(3.0-2.0*f);
        float n = p.x + p.y*57.0 + 113.0*p.z;
        // Fixed: Corrected 3D noise mix logic
        return mix(
            mix(mix(hash(n+0.0), hash(n+1.0), f.x), mix(hash(n+57.0), hash(n+58.0), f.x), f.y),
            mix(mix(hash(n+113.0), hash(n+114.0), f.x), mix(hash(n+170.0), hash(n+171.0), f.x), f.y),
            f.z
        );
    }

    void main() {
        vNormal = normal;
        vec3 pos = position;
        float noiseVal = noise(pos * (2.0 + uY4 * 8.0) + uTime * 0.4);
        float angular = floor(noiseVal * (1.0 + (1.0-uY1)*12.0)) / (1.0 + (1.0-uY1)*12.0);
        float finalNoise = mix(noiseVal, angular, uY1);
        float wave = sin(pos.x * 12.0 + uTime) * uY2 * 0.45;

        float displacement = (finalNoise * uY3 * 0.7) + (uLoudness * 0.6) + wave;
        vDisplacement = displacement;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(pos + normal * displacement, 1.0);
    }
`;

const fragmentShader = `
    varying float vDisplacement;
    void main() {
        vec3 colorA = vec3(0.0, 1.0, 0.7);
        vec3 colorB = vec3(0.1, 0.05, 0.4);
        vec3 finalColor = mix(colorB, colorA, vDisplacement + 0.25);
        gl_FragColor = vec4(finalColor, 0.9);
    }
`;

let shaderUniforms = {
    uTime: { value: 0 }, uLoudness: { value: 0 },
    uY1: { value: 0.5 }, uY2: { value: 0.5 }, uY3: { value: 0.5 }, uY4: { value: 0.5 }
};

// --- 3D Initialization ---
function initThree() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x050505);

    const rightPanelWidth = window.innerWidth - 320;
    camera = new THREE.PerspectiveCamera(75, rightPanelWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 3.5;

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(rightPanelWidth, window.innerHeight);
    renderer.domElement.style.position = 'absolute';
    renderer.domElement.style.top = '0';
    renderer.domElement.style.left = '320px';
    renderer.domElement.style.zIndex = '1';
    document.body.appendChild(renderer.domElement);

    window.addEventListener('resize', onWindowResize);
    createShape(0);
    animate();
}

function onWindowResize() {
    const rightPanelWidth = window.innerWidth - 320;
    camera.aspect = rightPanelWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(rightPanelWidth, window.innerHeight);
}

function createShape(type) {
    if (currentMesh) { 
        scene.remove(currentMesh); 
        if(currentMesh.geometry) currentMesh.geometry.dispose(); 
        if(currentMesh.material) currentMesh.material.dispose();
    }
    let geo;
    type = parseInt(type);
    if (type === 0) geo = new THREE.SphereGeometry(1, 128, 128);
    else if (type === 1) geo = new THREE.BoxGeometry(1.4, 1.4, 1.4, 32, 32, 32);
    else if (type === 2) geo = new THREE.TorusGeometry(0.8, 0.4, 64, 128);
    else if (type === 3) geo = new THREE.ConeGeometry(1, 2, 64, 64);
    else if (type === 4) geo = new THREE.CylinderGeometry(0.8, 0.8, 2, 64, 64);
    else geo = new THREE.OctahedronGeometry(1.2, 32);

    const mat = new THREE.ShaderMaterial({ 
        uniforms: shaderUniforms, 
        vertexShader, 
        fragmentShader, 
        wireframe: true, 
        transparent: true 
    });
    currentMesh = new THREE.Mesh(geo, mat);
    scene.add(currentMesh);
}

// --- Audio & ML ---
async function initEngine() {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === 'suspended') await audioCtx.resume();
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048;

    brain = ml5.neuralNetwork({
        inputs: ['loudness', 'pitch', 'brightness', 'roughness'],
        outputs: ['y1', 'y2', 'y3', 'y4', 'shape'],
        task: 'regression', debug: false
    });

    document.getElementById('btn-engine').style.display = 'none';
    document.getElementById('btn-main').style.display = 'block';
    document.getElementById('save-load-zone').style.display = 'block';
    
    loadTrainingData();
    initThree();
    updateAllUIText();
    updateStatus(translations[currentLang].statusReady, 'status-idle');
}

async function handleRecord() {
    const t = translations[currentLang];
    if (audioCtx && audioCtx.state === 'suspended') await audioCtx.resume();

    if (state === 'IDLE' || state === 'REVIEWING') {
        state = 'RECORDING'; audioChunks = [];
        recordedX = { loudness: 0, pitch: 0, brightness: 0, roughness: 0, count: 0 };
        if(audioTag) { audioTag.pause(); isPlaying = false; }
        
        try {
            if (!micStream) {
                micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                microphone = audioCtx.createMediaStreamSource(micStream);
                mediaRecorder = new MediaRecorder(micStream);
                mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
                mediaRecorder.onstop = saveRecording;
            }
            microphone.connect(analyser);
            mediaRecorder.start();
            document.getElementById('btn-main').innerText = t.btnStop;
            updateStatus(t.statusRecording, 'status-recording');
            document.getElementById('labeling-zone').style.display = 'none';
        } catch (err) { alert('마이크 권한이 필요합니다.'); state = 'IDLE'; }
    } else {
        mediaRecorder.stop();
        state = 'REVIEWING';
        if (microphone) microphone.disconnect();
        
        document.getElementById('labeling-zone').style.display = 'block';
        document.getElementById('btn-play').style.display = 'block';
        document.getElementById('btn-confirm').style.display = 'block';
        
        // 텍스트 강제 갱신 (Play 버튼 등)
        updateAllUIText();
        updateStatus(t.statusReviewing, 'status-review');
    }
}

function saveRecording() {
    const blob = new Blob(audioChunks, { type: 'audio/wav' });
    audioTag = new Audio(URL.createObjectURL(blob));
    audioTag.loop = true;
    if(recordedX.count > 0) {
        recordedX.loudness /= recordedX.count; 
        recordedX.pitch /= recordedX.count;
        recordedX.brightness /= recordedX.count; 
        recordedX.roughness /= recordedX.count;
    }
}

function togglePlayback() {
    const playBtn = document.getElementById('btn-play');
    const t = translations[currentLang];
    if (!audioTag) return;
    if (audioTag.paused) {
        if (!sourceNode) {
            sourceNode = audioCtx.createMediaElementSource(audioTag);
            sourceNode.connect(analyser); 
            analyser.connect(audioCtx.destination);
        }
        audioTag.play(); isPlaying = true;
        playBtn.innerText = t.btnPause;
    } else {
        audioTag.pause(); isPlaying = false;
        playBtn.innerText = t.btnPlay;
    }
}

function animate() {
    requestAnimationFrame(animate);
    if (!analyser) return;

    analyzeAudio();
    
    shaderUniforms.uTime.value += 0.05;
    shaderUniforms.uLoudness.value = currentX.loudness;

    if (state === 'REVIEWING') {
        targetY.y1 = parseFloat(document.getElementById('y1').value);
        targetY.y2 = parseFloat(document.getElementById('y2').value);
        targetY.y3 = parseFloat(document.getElementById('y3').value);
        targetY.y4 = parseFloat(document.getElementById('y4').value);
        targetY.shape = parseInt(document.getElementById('shape-selector').value);
    } else if (state === 'IDLE' && customTrainingData.length >= 1) {
        brain.predict([currentX.loudness, currentX.pitch, currentX.brightness, currentX.roughness], (err, res) => {
            if(!err) {
                targetY.y1 = res[0].value; targetY.y2 = res[1].value;
                targetY.y3 = res[2].value; targetY.y4 = res[3].value;
                let predShape = Math.round(res[4].value * 5);
                predShape = Math.max(0, Math.min(5, predShape));
                if(predShape !== currentY.shape) { 
                    currentY.shape = predShape; 
                    createShape(predShape); 
                    updateShapeNameDisplay();
                }
            }
        });
    }

    for(let k of ['y1', 'y2', 'y3', 'y4']) {
        currentY[k] += (targetY[k] - currentY[k]) * 0.1;
        shaderUniforms[`u${k.toUpperCase()}`].value = currentY[k];
    }
    
    if (currentMesh) currentMesh.rotation.y += 0.005;
    if (renderer) renderer.render(scene, camera);
}

function analyzeAudio() {
    const data = new Uint8Array(analyser.frequencyBinCount);
    const time = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(data); 
    analyser.getByteTimeDomainData(time);
    
    if (state === 'REVIEWING' && !isPlaying) {
        currentX = {...recordedX};
    } else {
        let sum = 0; 
        for(let v of time) { let n=(v-128)/128; sum+=n*n; }
        currentX.loudness = Math.sqrt(sum/time.length) * 10.0;
        
        let te=0, we=0; 
        for(let i=0; i<data.length; i++) { we+=i*data[i]; te+=data[i]; }
        currentX.pitch = te>0 ? (we/te)/50.0 : 0;
        currentX.brightness = currentX.pitch * 1.2;
        
        let zcr=0; 
        for(let i=1; i<time.length; i++) if(time[i]>128 && time[i-1]<=128) zcr++;
        currentX.roughness = zcr/40.0;
        
        if (state === 'RECORDING') {
            recordedX.loudness += currentX.loudness; 
            recordedX.pitch += currentX.pitch;
            recordedX.brightness += currentX.brightness; 
            recordedX.roughness += currentX.roughness;
            recordedX.count++;
        }
    }
    const l = document.getElementById('val-loud'); if(l) l.innerText = currentX.loudness.toFixed(2);
    const p = document.getElementById('val-pitch'); if(p) p.innerText = currentX.pitch.toFixed(2);
    const b = document.getElementById('val-bright'); if(b) b.innerText = currentX.brightness.toFixed(2);
    const r = document.getElementById('val-rough'); if(r) r.innerText = currentX.roughness.toFixed(2);
}

function confirmTrainingWrapper() {
    const labels = [
        parseFloat(document.getElementById('y1').value), 
        parseFloat(document.getElementById('y2').value), 
        parseFloat(document.getElementById('y3').value), 
        parseFloat(document.getElementById('y4').value), 
        parseInt(document.getElementById('shape-selector').value) / 5.0
    ];
    brain.addData([recordedX.loudness, recordedX.pitch, recordedX.brightness, recordedX.roughness], labels);
    customTrainingData.push({ x: {...recordedX}, y: labels });
    saveTrainingData();
    
    updateStatus("학습 중...", "status-recording");
    brain.normalizeData();
    brain.train({ epochs: 50 }, () => {
        alert(currentLang === 'KR' ? "학습 완료! 실시간 모드" : "Training Done! Real-time mode.");
        const dc = document.getElementById('data-count'); if(dc) dc.innerText = customTrainingData.length;
        state = 'IDLE'; 
        document.getElementById('labeling-zone').style.display = 'none';
        document.getElementById('btn-play').style.display = 'none';
        document.getElementById('btn-confirm').style.display = 'none';
        updateStatus(translations[currentLang].statusReady, 'status-idle');
        updateAllUIText();
    });
}

function saveTrainingData() {
    try {
        const dataToSave = JSON.stringify({ data: customTrainingData });
        localStorage.setItem('soundTo3D_data', dataToSave);
        console.log('Training data saved:', customTrainingData.length, 'samples');
    } catch (e) {
        console.error('Failed to save training data:', e);
        alert('Failed to save data: ' + e.message);
    }
}
function loadTrainingData() {
    const saved = localStorage.getItem('soundTo3D_data');
    if (!saved) {
        console.log('No saved training data found');
        return;
    }
    try {
        const obj = JSON.parse(saved);
        customTrainingData = obj.data || [];
        console.log('Training data loaded:', customTrainingData.length, 'samples');

        const dc = document.getElementById('data-count');
        if (dc) dc.innerText = customTrainingData.length;

        if (brain && customTrainingData.length > 0) {
            customTrainingData.forEach(i => brain.addData([i.x.loudness, i.x.pitch, i.x.brightness, i.x.roughness], i.y));
            brain.normalizeData();
            brain.train({ epochs: 10 }, () => {
                console.log('Model trained with loaded data');
            });
        }
    } catch (e) {
        console.error('Failed to load training data:', e);
        alert('Failed to load data: ' + e.message);
    }
}

function clearAllData() {
    if(confirm("데이터 삭제?")) { localStorage.removeItem('soundTo3D_data'); location.reload(); }
}

// --- UI & Translation ---
const translations = {
    KR: {
        title: "IML Research", btnEngine: "오디오 엔진 가동", btnRecord: "녹음 시작", btnStop: "중단", btnReRecord: "다시 녹음",
        btnPlay: "소리 재생", btnPause: "재생 중지", btnConfirm: "데이터 확정 및 학습", btnExport: "CSV 추출",
        statusReady: "준비 완료", statusRecording: "녹음 중...", statusReviewing: "검토 및 라벨링",
        labelLoud: "음량", labelPitch: "음높이", labelBright: "밝기", labelRough: "거칠기",
        y1Label: "y1: 각짐", y1Left: "둥근", y1Right: "각진",
        y2Label: "y2: 뾰족함", y2Left: "부드러운", y2Right: "뾰족한",
        y3Label: "y3: 거칠기", y3Left: "매끈한", y3Right: "거친",
        y4Label: "y4: 복잡도", y4Left: "단순", y4Right: "복잡",
        shapeLabel: "기본 형태", dataLabel: "학습 데이터:", samplesLabel: "개",
        shapeNames: ['구', '정육면체', '토러스', '원뿔', '원기둥', '팔면체']
    },
    EN: {
        title: "IML Research", btnEngine: "Start Engine", btnRecord: "Record", btnStop: "Stop", btnReRecord: "Re-record",
        btnPlay: "Play", btnPause: "Pause", btnConfirm: "Confirm & Train", btnExport: "Export CSV",
        statusReady: "Ready", statusRecording: "Recording...", statusReviewing: "Reviewing...",
        labelLoud: "Loudness", labelPitch: "Pitch", labelBright: "Brightness", labelRough: "Roughness",
        y1Label: "y1: Angularity", y1Left: "Round", y1Right: "Angular",
        y2Label: "y2: Spikiness", y2Left: "Smooth", y2Right: "Spiky",
        y3Label: "y3: Texture", y3Left: "Sleek", y3Right: "Rough",
        y4Label: "y4: Density", y4Left: "Simple", y4Right: "Complex",
        shapeLabel: "Base Shape", dataLabel: "Data:", samplesLabel: "samples",
        shapeNames: ['Sphere', 'Cube', 'Torus', 'Cone', 'Cylinder', 'Octahedron']
    }
};

function toggleLanguage() { 
    currentLang = currentLang === 'KR' ? 'EN' : 'KR'; 
    updateAllUIText(); 
}

function updateAllUIText() {
    const t = translations[currentLang];
    const langBtn = document.getElementById('lang-toggle'); if(langBtn) langBtn.innerText = currentLang === 'KR' ? 'EN' : 'KR';

    const mapping = {
        'title': t.title, 'btn-engine': t.btnEngine, 'btn-confirm': t.btnConfirm,
        'btn-play': isPlaying ? t.btnPause : t.btnPlay, // Fix: Added Play button text
        'label-loud': t.labelLoud, 'label-pitch': t.labelPitch, 'label-bright': t.labelBright, 'label-rough': t.labelRough,
        'y1-label': t.y1Label, 'y1-left': t.y1Left, 'y1-right': t.y1Right,
        'y2-label': t.y2Label, 'y2-left': t.y2Left, 'y2-right': t.y2Right,
        'y3-label': t.y3Label, 'y3-left': t.y3Left, 'y3-right': t.y3Right,
        'y4-label': t.y4Label, 'y4-left': t.y4Left, 'y4-right': t.y4Right,
        'shape-label': t.shapeLabel, 'data-label': t.dataLabel, 'samples-label': t.samplesLabel
    };

    for (let id in mapping) {
        const el = document.getElementById(id);
        if (el) el.innerText = mapping[id];
    }

    const mainBtn = document.getElementById('btn-main');
    if (mainBtn) {
        if (state === 'IDLE') mainBtn.innerText = t.btnRecord;
        else if (state === 'RECORDING') mainBtn.innerText = t.btnStop;
        else if (state === 'REVIEWING') mainBtn.innerText = t.btnReRecord;
    }
    updateShapeNameDisplay();
}

function updateShapeNameDisplay() {
    const t = translations[currentLang];
    const s = document.getElementById('shape-selector'); if(!s) return;
    const n = document.getElementById('shape-name'); if(n) n.innerText = t.shapeNames[parseInt(s.value)];
}

function updateStatus(msg, cls) { 
    const el = document.getElementById('status'); 
    if(el) { el.innerText = msg; el.className = 'status-badge ' + cls; }
}

function changeShape(val) { updateShapeNameDisplay(); createShape(val); }
function exportCSV() {
    let csv = "loudness,pitch,brightness,roughness,y1,y2,y3,y4,shape\n";
    customTrainingData.forEach(d => { csv += `${d.x.loudness},${d.x.pitch},${d.x.brightness},${d.x.roughness},${d.y.join(',')}\n`; });
    const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    a.download = `IML_Data.csv`; a.click();
}

window.addEventListener('DOMContentLoaded', () => { updateAllUIText(); });
