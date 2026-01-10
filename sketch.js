let audioCtx, analyser, microphone, mediaRecorder, sourceNode;
let audioChunks = [];
let audioTag = null;
let brain;
let scene, camera, renderer, currentMesh;
let state = 'IDLE'; // IDLE, RECORDING, REVIEWING
let isPlaying = false; // Track if audio is playing
let recordedX = { loudness: 0, pitch: 0, brightness: 0, roughness: 0, count: 0 };
let currentX = { loudness: 0, pitch: 0, brightness: 0, roughness: 0 };
let targetY = { y1: 0.5, y2: 0.5, y3: 0.5, y4: 0.5, shape: 0 };
let currentY = { y1: 0.5, y2: 0.5, y3: 0.5, y4: 0.5, shape: 0 };
let customTrainingData = [];
let currentLang = 'KR';
let micStream = null;

// --- GPU Shader Code ---
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
        return mix(mix(mix(hash(n+0.0), hash(n+1.0),f.x), mix(hash(n+57.0), hash(n+58.0),f.x),f.y),
                   mix(mix(mix(hash(n+113.0),hash(n+114.0),f.x), mix(hash(n+170.0),hash(n+171.0),f.x),f.y),f.z);
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

// Audio constants
const LOUDNESS_NORMALIZER = 2.0;
const PITCH_NORMALIZER = 1000.0;
const BRIGHTNESS_NORMALIZER = 10000.0;
const ROUGHNESS_NORMALIZER = 1.0;

function autoClassifyShape(audioFeatures) {
    const scores = { sphere: 0, cube: 0, torus: 0, cone: 0, cylinder: 0, octahedron: 0 };
    let bestShape = 0;
    let bestScore = -1;
    const shapeNames = ['sphere', 'cube', 'torus', 'cone', 'cylinder', 'octahedron'];
    shapeNames.forEach((name, idx) => { if (scores[name] > bestScore) { bestScore = scores[name]; bestShape = idx; } });
    return bestShape;
}

function autoSuggestParameters(audioFeatures) {
    const loud = Math.min(audioFeatures.loudness / LOUDNESS_NORMALIZER, 1.0);
    const pitch = Math.min(audioFeatures.pitch / PITCH_NORMALIZER, 1.0);
    const bright = Math.min(audioFeatures.brightness / BRIGHTNESS_NORMALIZER, 1.0);
    const rough = Math.min(audioFeatures.roughness / ROUGHNESS_NORMALIZER, 1.0);
    return { y1: pitch * 0.6 + bright * 0.4, y2: bright * 0.5 + rough * 0.5, y3: rough * 0.7 + (1 - loud) * 0.3, y4: bright * 0.5 + loud * 0.5 };
}

// --- Initialization ---
function initThree() {
    const container = document.getElementById('three-container');
    scene = new THREE.Scene();
    const rightPanelWidth = window.innerWidth - 320;
    camera = new THREE.PerspectiveCamera(75, rightPanelWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 3.5;
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(rightPanelWidth, window.innerHeight);
    container.appendChild(renderer.domElement);
    createShape(0);
    animate();
}

function createConnectedCube(size, subdivisions) {
    const geo = new THREE.BoxGeometry(size, size, size, subdivisions, subdivisions, subdivisions);
    return geo;
}

function createShape(type) {
    if (currentMesh) { scene.remove(currentMesh); currentMesh.geometry.dispose(); }
    let geo;
    if (type == 0) geo = new THREE.SphereGeometry(1, 128, 128);
    else if (type == 1) geo = new THREE.BoxGeometry(1.4, 1.4, 1.4, 32, 32, 32);
    else if (type == 2) geo = new THREE.TorusGeometry(0.8, 0.4, 64, 128);
    else if (type == 3) geo = new THREE.ConeGeometry(1, 2, 64, 64);
    else if (type == 4) geo = new THREE.CylinderGeometry(0.8, 0.8, 2, 64, 64);
    else geo = new THREE.OctahedronGeometry(1.2, 32);

    const mat = new THREE.ShaderMaterial({ uniforms: shaderUniforms, vertexShader, fragmentShader, wireframe: true, transparent: true });
    currentMesh = new THREE.Mesh(geo, mat);
    scene.add(currentMesh);
}

async function initEngine() {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
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
    updateAllUIText();
    initThree();
    loadTrainingData();
}

// --- Workflow ---
async function handleRecord() {
    const t = translations[currentLang];
    if (state === 'IDLE' || state === 'REVIEWING') {
        state = 'RECORDING'; audioChunks = [];
        recordedX = { loudness: 0, pitch: 0, brightness: 0, roughness: 0, count: 0 };
        if(audioTag) audioTag.pause();
        isPlaying = false;
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
        } catch (err) { alert('마이크 접근 권한이 필요합니다.'); state = 'IDLE'; }
    } else {
        mediaRecorder.stop();
        state = 'REVIEWING';
        if (microphone) microphone.disconnect();
        document.getElementById('labeling-zone').style.display = 'block';
        document.getElementById('btn-play').style.display = 'block';
        document.getElementById('btn-main').innerText = t.btnReRecord;
        updateStatus(t.statusReviewing, 'status-review');
    }
}

function saveRecording() {
    const blob = new Blob(audioChunks, { type: 'audio/wav' });
    audioTag = new Audio(URL.createObjectURL(blob));
    audioTag.loop = true;
    recordedX.loudness /= recordedX.count; recordedX.pitch /= recordedX.count;
    recordedX.brightness /= recordedX.count; recordedX.roughness /= recordedX.count;
}

function togglePlayback() {
    const playBtn = document.getElementById('btn-play');
    const t = translations[currentLang];
    if (!audioTag) return;
    if (audioTag.paused) {
        if (!sourceNode) {
            sourceNode = audioCtx.createMediaElementSource(audioTag);
            sourceNode.connect(analyser); analyser.connect(audioCtx.destination);
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
    } else if (state === 'IDLE' && customTrainingData.length >= 3) {
        brain.predict([currentX.loudness, currentX.pitch, currentX.brightness, currentX.roughness], (err, res) => {
            if(!err) {
                targetY.y1 = res[0].value; targetY.y2 = res[1].value;
                targetY.y3 = res[2].value; targetY.y4 = res[3].value;
                let predShape = Math.round(res[4].value * 5);
                if(predShape !== currentY.shape) { currentY.shape = predShape; createShape(predShape); }
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
    if (state === 'REVIEWING' && !isPlaying) {
        currentX = {...recordedX};
    } else {
        const data = new Uint8Array(analyser.frequencyBinCount);
        const time = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(data); analyser.getByteTimeDomainData(time);
        let sum = 0; for(let v of time) { let n=(v-128)/128; sum+=n*n; }
        currentX.loudness = Math.sqrt(sum/time.length) * 10.0;
        let te=0, we=0; for(let i=0; i<data.length; i++) { we+=i*data[i]; te+=data[i]; }
        currentX.pitch = te>0 ? (we/te)/50.0 : 0;
        currentX.brightness = currentX.pitch * 1.2;
        let zcr=0; for(let i=1; i<time.length; i++) if(time[i]>128 && time[i-1]<=128) zcr++;
        currentX.roughness = zcr/40.0;
        if (state === 'RECORDING') {
            recordedX.loudness += currentX.loudness; recordedX.pitch += currentX.pitch;
            recordedX.brightness += currentX.brightness; recordedX.roughness += currentX.roughness;
            recordedX.count++;
        }
    }
    document.getElementById('val-loud').innerText = currentX.loudness.toFixed(2);
    document.getElementById('val-pitch').innerText = currentX.pitch.toFixed(2);
    document.getElementById('val-bright').innerText = currentX.brightness.toFixed(2);
    document.getElementById('val-rough').innerText = currentX.roughness.toFixed(2);
}

function confirmTrainingWrapper() {
    const labels = [targetY.y1, targetY.y2, targetY.y3, targetY.y4, targetY.shape / 5.0];
    brain.addData([recordedX.loudness, recordedX.pitch, recordedX.brightness, recordedX.roughness], labels);
    customTrainingData.push({ x: {...recordedX}, y: labels });
    saveTrainingData();
    brain.normalizeData();
    brain.train({ epochs: 32 }, () => {
        alert(currentLang === 'KR' ? "학습 완료!" : "Training complete!");
        document.getElementById('data-count').innerText = customTrainingData.length;
        state = 'IDLE'; 
        document.getElementById('labeling-zone').style.display = 'none';
        document.getElementById('btn-play').style.display = 'none';
        updateStatus(translations[currentLang].statusReady, 'status-idle');
    });
}

function saveTrainingData() {
    localStorage.setItem('soundTo3D_trainingData', JSON.stringify({ data: customTrainingData }));
}

function loadTrainingData() {
    const saved = localStorage.getItem('soundTo3D_trainingData');
    if (!saved) return;
    try {
        const saveObj = JSON.parse(saved);
        customTrainingData = saveObj.data || [];
        document.getElementById('data-count').innerText = customTrainingData.length;
        if (brain && customTrainingData.length > 0) {
            customTrainingData.forEach(item => brain.addData([item.x.loudness, item.x.pitch, item.x.brightness, item.x.roughness], item.y));
            if (customTrainingData.length >= 3) {
                brain.normalizeData();
                brain.train({ epochs: 10 }, () => console.log('Loaded & Trained'));
            }
        }
    } catch (e) { console.error(e); }
}

// --- UI & Translation ---
const translations = {
    KR: {
        title: "IML Research", btnEngine: "오디오 엔진 가동", btnRecord: "녹음 시작", btnStop: "중단", btnReRecord: "다시 녹음",
        btnPlay: "소리 재생", btnPause: "재생 중지", btnConfirm: "데이터 확정 및 학습", btnExport: "CSV 추출",
        statusReady: "준비 완료", statusRecording: "녹음 중...", statusReviewing: "검토 중...",
        labelLoud: "음량", labelPitch: "음높이", labelBright: "밝기", labelRough: "거칠기",
        y1Label: "y1: 각짐", y1Left: "둥근", y1Right: "각진",
        y2Label: "y2: 뾰족함", y2Left: "부드러운", y2Right: "뾰족한",
        y3Label: "y3: 거칠기", y3Left: "매끈한", y3Right: "거친",
        y4Label: "y4: 복잡도", y4Left: "단순", y4Right: "복잡",
        shapeLabel: "기본 형태", dataLabel: "학습 데이터:", samplesLabel: "개", labelingInstruction: "소리에 어울리는 형태를 조절하세요",
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
        shapeLabel: "Base Shape", dataLabel: "Data:", samplesLabel: "samples", labelingInstruction: "Adjust visual to match sound",
        shapeNames: ['Sphere', 'Cube', 'Torus', 'Cone', 'Cylinder', 'Octahedron']
    }
};

function toggleLanguage() { currentLang = currentLang === 'KR' ? 'EN' : 'KR'; updateAllUIText(); }

function updateAllUIText() {
    const t = translations[currentLang];
    document.getElementById('lang-toggle').innerText = currentLang === 'KR' ? 'EN' : 'KR';

    // ID mapping logic: label-loud -> labelLoud
    const ids = ['title', 'btn-engine', 'btn-confirm', 'btn-export', 'label-loud', 'label-pitch', 'label-bright', 'label-rough', 
                 'y1-label', 'y1-left', 'y1-right', 'y2-label', 'y2-left', 'y2-right', 
                 'y3-label', 'y3-left', 'y3-right', 'y4-label', 'y4-left', 'y4-right', 
                 'shape-label', 'data-label', 'samples-label', 'labeling-instruction'];

    ids.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            // Convert 'label-loud' to 'labelLoud'
            const key = id.split('-').map((word, i) => i === 0 ? word : word.charAt(0).toUpperCase() + word.slice(1)).join('');
            if (t[key]) el.innerText = t[key];
        }
    });

    const mainBtn = document.getElementById('btn-main');
    if (mainBtn) {
        if (state === 'IDLE') mainBtn.innerText = t.btnRecord;
        else if (state === 'RECORDING') mainBtn.innerText = t.btnStop;
        else if (state === 'REVIEWING') mainBtn.innerText = t.btnReRecord;
    }
    
    const shapeSelector = document.getElementById('shape-selector');
    if (shapeSelector) document.getElementById('shape-name').innerText = t.shapeNames[parseInt(shapeSelector.value)];
}

function updateStatus(msg, cls) { 
    const el = document.getElementById('status'); 
    if(el) { el.innerText = msg; el.className = 'status-badge ' + cls; }
}

function changeShape(val) { 
    if(translations[currentLang]) {
        document.getElementById('shape-name').innerText = translations[currentLang].shapeNames[val]; 
        createShape(val); 
    }
}

function exportCSV() {
    let csv = "loudness,pitch,brightness,roughness,y1,y2,y3,y4,shape\n";
    customTrainingData.forEach(d => { csv += `${d.x.loudness},${d.x.pitch},${d.x.brightness},${d.x.roughness},${d.y.join(',')}\n`; });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    a.download = `IML_Data.csv`; a.click();
}

window.addEventListener('DOMContentLoaded', () => {
    updateAllUIText();
    const saved = localStorage.getItem('soundTo3D_trainingData');
    if (saved) {
        try {
            const saveObj = JSON.parse(saved);
            const countEl = document.getElementById('data-count');
            if (countEl && saveObj.data) countEl.innerText = saveObj.data.length;
        } catch(e) {}
    }
});
