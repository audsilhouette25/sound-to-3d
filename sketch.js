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
let micStream = null; // Store microphone stream

// CPU-based rendering variables
let originalVertices = [];
let tempVec = new THREE.Vector3();
let animTime = 0;

// Audio constants for normalization (from main branch)
const LOUDNESS_NORMALIZER = 2.0;
const PITCH_NORMALIZER = 1000.0;
const BRIGHTNESS_NORMALIZER = 10000.0;
const ROUGHNESS_NORMALIZER = 1.0;

// Rule-based auto-classification (from main branch)
function autoClassifyShape(audioFeatures) {
    // Normalize audio features
    const loud = Math.min(audioFeatures.loudness / LOUDNESS_NORMALIZER, 1.0);
    const pitch = Math.min(audioFeatures.pitch / PITCH_NORMALIZER, 1.0);
    const bright = Math.min(audioFeatures.brightness / BRIGHTNESS_NORMALIZER, 1.0);
    const rough = Math.min(audioFeatures.roughness / ROUGHNESS_NORMALIZER, 1.0);

    // Define shape scores
    const scores = {
        sphere: 0,    // 0
        cube: 0,      // 1
        torus: 0,     // 2
        cone: 0,      // 3
        cylinder: 0,  // 4
        octahedron: 0 // 5
    };

    // Sphere: soft, balanced sounds (low roughness, mid loudness)
    scores.sphere = (1 - rough) * 0.4 + (1 - Math.abs(loud - 0.5)) * 0.3 + (1 - Math.abs(pitch - 0.5)) * 0.3;

    // Cube: strong, stable sounds (high loudness, low pitch variation)
    scores.cube = loud * 0.4 + (1 - pitch) * 0.3 + rough * 0.3;

    // Torus: rhythmic, circular sounds (mid-high brightness, balanced)
    scores.torus = bright * 0.4 + (1 - Math.abs(rough - 0.5)) * 0.3 + (1 - Math.abs(loud - 0.6)) * 0.3;

    // Cone: sharp, directional sounds (high pitch, high brightness)
    scores.cone = pitch * 0.4 + bright * 0.3 + (1 - loud) * 0.3;

    // Cylinder: sustained, stable sounds (low roughness, consistent loudness)
    scores.cylinder = (1 - rough) * 0.4 + (1 - Math.abs(loud - 0.7)) * 0.4 + (1 - Math.abs(pitch - 0.3)) * 0.2;

    // Octahedron: complex, textured sounds (high roughness, high brightness)
    scores.octahedron = rough * 0.4 + bright * 0.3 + loud * 0.3;

    // Find shape with highest score
    let bestShape = 0;
    let bestScore = scores.sphere;
    const shapeNames = ['sphere', 'cube', 'torus', 'cone', 'cylinder', 'octahedron'];

    shapeNames.forEach((name, idx) => {
        if (scores[name] > bestScore) {
            bestScore = scores[name];
            bestShape = idx;
        }
    });

    return bestShape;
}

function autoSuggestParameters(audioFeatures) {
    // Normalize audio features
    const loud = Math.min(audioFeatures.loudness / LOUDNESS_NORMALIZER, 1.0);
    const pitch = Math.min(audioFeatures.pitch / PITCH_NORMALIZER, 1.0);
    const bright = Math.min(audioFeatures.brightness / BRIGHTNESS_NORMALIZER, 1.0);
    const rough = Math.min(audioFeatures.roughness / ROUGHNESS_NORMALIZER, 1.0);

    // Map audio features to visual parameters
    // y1: Angularity (higher pitch = more angular)
    const y1 = pitch * 0.6 + bright * 0.4;

    // y2: Spikiness (higher brightness + roughness = more spiky)
    const y2 = bright * 0.5 + rough * 0.5;

    // y3: Texture roughness (directly from audio roughness)
    const y3 = rough * 0.7 + (1 - loud) * 0.3;

    // y4: Density/Complexity (higher brightness + loudness = more complex)
    const y4 = bright * 0.5 + loud * 0.5;

    return { y1, y2, y3, y4 };
}

// Noise functions (CPU version of GPU shader noise)
function hash(n) {
    const x = Math.sin(n) * 43758.5453123;
    return x - Math.floor(x);
}

function noise3D(x, y, z) {
    const X = Math.floor(x), Y = Math.floor(y), Z = Math.floor(z);
    const fx = x - X, fy = y - Y, fz = z - Z;
    const u = fx * fx * (3 - 2 * fx);
    const v = fy * fy * (3 - 2 * fy);
    const w = fz * fz * (3 - 2 * fz);

    const n = X + Y * 57.0 + Z * 113.0;
    const a = hash(n + 0.0), b = hash(n + 1.0);
    const c = hash(n + 57.0), d = hash(n + 58.0);
    const e = hash(n + 113.0), f = hash(n + 114.0);
    const g = hash(n + 170.0), h = hash(n + 171.0);

    return (1 - w) * ((1 - v) * (a * (1 - u) + b * u) + v * (c * (1 - u) + d * u)) +
           w * ((1 - v) * (e * (1 - u) + f * u) + v * (g * (1 - u) + h * u));
}

// --- Initialization ---
function initThree() {
    const container = document.getElementById('three-container');
    scene = new THREE.Scene();

    // Calculate right panel width (viewport width - left panel width)
    const rightPanelWidth = window.innerWidth - 320;
    camera = new THREE.PerspectiveCamera(75, rightPanelWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 3.5;

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(rightPanelWidth, window.innerHeight);
    container.appendChild(renderer.domElement);

    createShape(0);
    animate();
}

function createShape(type) {
    if (currentMesh) {
        scene.remove(currentMesh);
        currentMesh.geometry.dispose();
    }

    let geo;
    if (type == 0) geo = new THREE.SphereGeometry(1, 128, 128); // High resolution like 0108수정(지원)
    else if (type == 1) geo = new THREE.BoxGeometry(1.4, 1.4, 1.4, 64, 64, 64); // Same as 0108수정(지원)
    else if (type == 2) geo = new THREE.TorusGeometry(0.8, 0.4, 64, 128);
    else if (type == 3) geo = new THREE.ConeGeometry(1, 2, 64, 64);
    else if (type == 4) geo = new THREE.CylinderGeometry(0.8, 0.8, 2, 64, 64);
    else geo = new THREE.OctahedronGeometry(1.2, 32);

    // CPU-based material with vertex colors for dynamic coloring
    const mat = new THREE.MeshBasicMaterial({
        vertexColors: true,
        wireframe: true,
        transparent: true,
        opacity: 0.9
    });

    currentMesh = new THREE.Mesh(geo, mat);
    scene.add(currentMesh);

    // Store original vertex positions and normals
    originalVertices = [];
    const pos = currentMesh.geometry.attributes.position;
    const norm = currentMesh.geometry.attributes.normal;

    for (let i = 0; i < pos.count; i++) {
        originalVertices.push({
            pos: new THREE.Vector3(pos.getX(i), pos.getY(i), pos.getZ(i)),
            normal: new THREE.Vector3(norm.getX(i), norm.getY(i), norm.getZ(i))
        });
    }

    // Add color attribute
    const colors = new Float32Array(pos.count * 3);
    currentMesh.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
}

async function initEngine() {
    // Initialize audio context and analyser only - don't get microphone yet
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 512;

    brain = ml5.neuralNetwork({
        inputs: ['loudness', 'pitch', 'brightness', 'roughness'],
        outputs: ['y1', 'y2', 'y3', 'y4', 'shape'],
        task: 'regression', debug: false
    });

    document.getElementById('btn-engine').style.display = 'none';
    document.getElementById('btn-main').style.display = 'block';
    document.getElementById('save-load-zone').style.display = 'block';

    // Update all UI text with current language
    updateAllUIText();

    initThree();

    // localStorage에서 저장된 학습 데이터 즉시 불러오기 (지연 없음)
    loadTrainingData();
}

// --- Workflow ---
async function handleRecord() {
    const t = translations[currentLang];
    if (state === 'IDLE' || state === 'REVIEWING') {
        // Start recording - get microphone permission and turn on
        state = 'RECORDING'; audioChunks = [];
        recordedX = { loudness: 0, pitch: 0, brightness: 0, roughness: 0, count: 0 };
        if(audioTag) audioTag.pause();
        isPlaying = false; // Reset playing state
        sourceNode = null; // Reset sourceNode for new recording

        try {
            // Request microphone access only when recording starts
            if (!micStream) {
                micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                microphone = audioCtx.createMediaStreamSource(micStream);
                mediaRecorder = new MediaRecorder(micStream);
                mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
                mediaRecorder.onstop = saveRecording;
            }

            // Connect microphone to analyser
            microphone.connect(analyser);
            mediaRecorder.start();

            document.getElementById('btn-main').innerText = t.btnStop;
            updateStatus(t.statusRecording, 'status-recording');
        } catch (err) {
            console.error('Microphone access denied:', err);
            alert('마이크 접근 권한이 필요합니다.');
            state = 'IDLE';
        }
    } else {
        // Stop recording - turn off microphone completely
        mediaRecorder.stop();
        state = 'REVIEWING';

        // Disconnect microphone from analyser
        if (microphone) {
            microphone.disconnect();
        }

        // Stop all media stream tracks to actually turn off microphone
        if (micStream) {
            micStream.getTracks().forEach(track => track.stop());
            micStream = null;
            microphone = null;
            mediaRecorder = null;
        }

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

    if (audioTag.paused) {
        // Only create sourceNode once - reuse it if it exists
        if (!sourceNode) {
            sourceNode = audioCtx.createMediaElementSource(audioTag);
            sourceNode.connect(analyser);
            analyser.connect(audioCtx.destination);
        }
        audioTag.play();
        isPlaying = true;
        playBtn.classList.add('playing');
        playBtn.innerText = t.btnPause || '재생 중지';
    } else {
        audioTag.pause();
        isPlaying = false;
        playBtn.classList.remove('playing');
        playBtn.innerText = t.btnPlay || '소리 반복 재생';
    }
}

function animate() {
    requestAnimationFrame(animate);
    if (!analyser) return;

    analyzeAudio();
    animTime += 0.05;

    if (state === 'REVIEWING') {
        targetY.y1 = parseFloat(document.getElementById('y1').value);
        targetY.y2 = parseFloat(document.getElementById('y2').value);
        targetY.y3 = parseFloat(document.getElementById('y3').value);
        targetY.y4 = parseFloat(document.getElementById('y4').value);
        targetY.shape = parseInt(document.getElementById('shape-selector').value);
    } else if (state === 'RECORDING') {
        // During recording, keep sphere shape and use rule-based parameters
        if(currentY.shape !== 0) {
            currentY.shape = 0;
            createShape(0);
        }
        const suggestedParams = autoSuggestParameters(currentX);
        targetY.y1 = suggestedParams.y1;
        targetY.y2 = suggestedParams.y2;
        targetY.y3 = suggestedParams.y3;
        targetY.y4 = suggestedParams.y4;
    } else if (customTrainingData.length >= 3) {
        // AI prediction mode (when training data exists)
        brain.predict([currentX.loudness, currentX.pitch, currentX.brightness, currentX.roughness], (err, res) => {
            if(!err) {
                targetY.y1 = res[0].value; targetY.y2 = res[1].value;
                targetY.y3 = res[2].value; targetY.y4 = res[3].value;
                let predShape = Math.round(res[4].value * 5);
                if(predShape !== currentY.shape) { currentY.shape = predShape; createShape(predShape); }
            }
        });
    } else {
        // Rule-based auto-classification mode (when no training data)
        const suggestedShape = autoClassifyShape(currentX);
        if(suggestedShape !== currentY.shape) {
            currentY.shape = suggestedShape;
            createShape(suggestedShape);
        }

        const suggestedParams = autoSuggestParameters(currentX);
        targetY.y1 = suggestedParams.y1;
        targetY.y2 = suggestedParams.y2;
        targetY.y3 = suggestedParams.y3;
        targetY.y4 = suggestedParams.y4;
    }

    for(let k of ['y1', 'y2', 'y3', 'y4']) {
        currentY[k] += (targetY[k] - currentY[k]) * 0.1;
    }

    // CPU-based vertex displacement (mimicking GPU shader)
    updateVisuals();

    currentMesh.rotation.y += 0.005;
    renderer.render(scene, camera);
}

function updateVisuals() {
    const pos = currentMesh.geometry.attributes.position;
    const colors = currentMesh.geometry.attributes.color;

    // Reset scale for all shapes
    currentMesh.scale.set(1, 1, 1);

    // Apply vertex displacement to all shapes uniformly
    for (let i = 0; i < originalVertices.length; i++) {
        const orig = originalVertices[i];
        const p = orig.pos;
        const n = orig.normal;

        // Replicate GPU shader noise calculation
        const noiseScale = 2.0 + currentY.y4 * 8.0;
        const noiseVal = noise3D(p.x * noiseScale + animTime * 0.4,
                                 p.y * noiseScale + animTime * 0.4,
                                 p.z * noiseScale + animTime * 0.4);

        // Angular quantization effect
        const steps = 1.0 + (1.0 - currentY.y1) * 12.0;
        const angular = Math.floor(noiseVal * steps) / steps;
        const finalNoise = noiseVal * (1 - currentY.y1) + angular * currentY.y1;

        // Wave effect
        const wave = Math.sin(p.x * 12.0 + animTime) * currentY.y2 * 0.45;

        // Total displacement
        const displacement = (finalNoise * currentY.y3 * 0.7) + (currentX.loudness * 0.6) + wave;

        // Apply displacement along normal
        pos.setXYZ(i,
            p.x + n.x * displacement,
            p.y + n.y * displacement,
            p.z + n.z * displacement
        );

        // Color based on displacement
        const colorA = { r: 0.0, g: 1.0, b: 0.7 }; // Cyan
        const colorB = { r: 0.1, g: 0.05, b: 0.4 }; // Dark Blue
        const t = Math.max(0, Math.min(1, displacement + 0.25));

        colors.setXYZ(i,
            colorB.r + (colorA.r - colorB.r) * t,
            colorB.g + (colorA.g - colorB.g) * t,
            colorB.b + (colorA.b - colorB.b) * t
        );
    }

    pos.needsUpdate = true;
    colors.needsUpdate = true;
}

function analyzeAudio() {
    // If in REVIEWING state and NOT playing, use recorded averages
    if (state === 'REVIEWING' && !isPlaying) {
        currentX.loudness = recordedX.loudness;
        currentX.pitch = recordedX.pitch;
        currentX.brightness = recordedX.brightness;
        currentX.roughness = recordedX.roughness;

        document.getElementById('val-loud').innerText = currentX.loudness.toFixed(2);
        document.getElementById('val-pitch').innerText = currentX.pitch.toFixed(2);
        document.getElementById('val-bright').innerText = currentX.brightness.toFixed(2);
        document.getElementById('val-rough').innerText = currentX.roughness.toFixed(2);
        return;
    }

    // Analyze live audio when RECORDING or when REVIEWING and playing
    const data = new Uint8Array(analyser.frequencyBinCount);
    const time = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(data);
    analyser.getByteTimeDomainData(time);

    let sum = 0; for(let v of time) { let n=(v-128)/128; sum+=n*n; }
    currentX.loudness = Math.sqrt(sum/time.length) * 10.0;

    let te=0, we=0; for(let i=0; i<data.length; i++) { we+=i*data[i]; te+=data[i]; }
    currentX.pitch = te>0 ? (we/te)/50.0 : 0;
    currentX.brightness = currentX.pitch * 1.2;

    let zcr=0; for(let i=1; i<time.length; i++) if(time[i]>128 && time[i-1]<=128) zcr++;
    currentX.roughness = zcr/40.0;

    document.getElementById('val-loud').innerText = currentX.loudness.toFixed(2);
    document.getElementById('val-pitch').innerText = currentX.pitch.toFixed(2);
    document.getElementById('val-bright').innerText = currentX.brightness.toFixed(2);
    document.getElementById('val-rough').innerText = currentX.roughness.toFixed(2);

    if (state === 'RECORDING') {
        recordedX.loudness += currentX.loudness; recordedX.pitch += currentX.pitch;
        recordedX.brightness += currentX.brightness; recordedX.roughness += currentX.roughness;
        recordedX.count++;
    }
}

function confirmTrainingWrapper() {
    const labels = [
        targetY.y1, targetY.y2, targetY.y3, targetY.y4, targetY.shape / 5.0
    ];
    brain.addData([recordedX.loudness, recordedX.pitch, recordedX.brightness, recordedX.roughness], labels);
    customTrainingData.push({ x: {...recordedX}, y: labels });

    // 학습 데이터를 localStorage에 저장
    saveTrainingData();

    brain.normalizeData();
    brain.train({ epochs: 32 }, () => {
        const t = translations[currentLang];
        alert(currentLang === 'KR' ? "학습 완료!" : "Training complete!");
        document.getElementById('data-count').innerText = customTrainingData.length;
        state = 'IDLE'; if(audioTag) audioTag.pause();
        document.getElementById('labeling-zone').style.display = 'none';
        document.getElementById('btn-play').style.display = 'none';
        updateStatus(t.statusIdle, 'status-idle');
    });
}

// localStorage에 학습 데이터 저장
function saveTrainingData() {
    try {
        const saveObj = {
            version: 1,
            count: customTrainingData.length,
            data: customTrainingData,
            timestamp: Date.now()
        };
        localStorage.setItem('soundTo3D_trainingData', JSON.stringify(saveObj));
        console.log(`✓ Saved ${customTrainingData.length} samples to localStorage`);
    } catch (e) {
        console.error('Save failed:', e);
        alert('데이터 저장 실패: ' + e.message);
    }
}

// localStorage에서 학습 데이터 불러오기
function loadTrainingData() {
    const saved = localStorage.getItem('soundTo3D_trainingData');

    // 데이터가 없으면 0으로 표시
    if (!saved) {
        console.log('No saved training data');
        const countEl = document.getElementById('data-count');
        if (countEl) countEl.innerText = '0';
        return;
    }

    try {
        const saveObj = JSON.parse(saved);
        if (!saveObj || !Array.isArray(saveObj.data)) {
            console.warn('Invalid data format, clearing');
            localStorage.removeItem('soundTo3D_trainingData');
            const countEl = document.getElementById('data-count');
            if (countEl) countEl.innerText = '0';
            return;
        }

        customTrainingData = saveObj.data;
        console.log(`✓ Loaded ${customTrainingData.length} samples from localStorage`);

        // 카운트 즉시 업데이트 (학습 전에도 표시)
        const countEl = document.getElementById('data-count');
        if (countEl) countEl.innerText = customTrainingData.length;

        // brain에 불러온 데이터 추가
        customTrainingData.forEach(item => {
            brain.addData(
                [item.x.loudness, item.x.pitch, item.x.brightness, item.x.roughness],
                item.y
            );
        });

        // 데이터가 있으면 자동으로 재학습
        if (customTrainingData.length >= 3) {
            console.log('Auto-retraining with loaded data...');
            brain.normalizeData();
            brain.train({ epochs: 32 }, () => {
                console.log(`✓ Auto-training complete with ${customTrainingData.length} samples`);
            });
        }
    } catch (e) {
        console.error('Load failed:', e);
        localStorage.removeItem('soundTo3D_trainingData');
        const countEl = document.getElementById('data-count');
        if (countEl) countEl.innerText = '0';
    }
}

const translations = {
    KR: {
        btnRecord: "녹음 시작",
        btnStop: "중단",
        btnReRecord: "다시 녹음",
        btnPlay: "소리 반복 재생",
        btnPause: "재생 중지",
        title: "IML Research",
        engineBtn: "오디오 엔진 가동",
        statusReady: "엔진 준비됨",
        statusRecording: "녹음 중...",
        statusReviewing: "검토 중...",
        statusIdle: "준비 완료",
        dataLabel: "학습 데이터:",
        samplesLabel: "개",
        confirmBtn: "데이터 확정 및 학습",
        exportBtn: "데이터 추출 (.CSV)",
        labelLoud: "음량",
        labelPitch: "음높이",
        labelBright: "밝기",
        labelRough: "거칠기",
        labelingInstruction: "소리에 맞는 시각적 파라미터를 설정하세요",
        y1Label: "y1: 각짐",
        y1Left: "둥근",
        y1Right: "각진",
        y2Label: "y2: 뾰족함",
        y2Left: "부드러운",
        y2Right: "뾰족한",
        y3Label: "y3: 거칠기",
        y3Left: "매끈한",
        y3Right: "거친",
        y4Label: "y4: 복잡도",
        y4Left: "단순",
        y4Right: "복잡",
        shapeLabel: "기본 형태"
    },
    EN: {
        btnRecord: "Record",
        btnStop: "Stop",
        btnReRecord: "Re-record",
        btnPlay: "Play Loop",
        btnPause: "Stop Playing",
        title: "IML Research",
        engineBtn: "Start Audio Engine",
        statusReady: "Engine Ready",
        statusRecording: "Recording...",
        statusReviewing: "Reviewing...",
        statusIdle: "Ready",
        dataLabel: "Training Data:",
        samplesLabel: "samples",
        confirmBtn: "Confirm & Train",
        exportBtn: "Export Data (.CSV)",
        labelLoud: "Loudness",
        labelPitch: "Pitch",
        labelBright: "Brightness",
        labelRough: "Roughness",
        labelingInstruction: "Set visual parameters for the sound",
        y1Label: "y1: Angularity",
        y1Left: "Round",
        y1Right: "Angular",
        y2Label: "y2: Spikiness",
        y2Left: "Smooth",
        y2Right: "Spiky",
        y3Label: "y3: Texture",
        y3Left: "Sleek",
        y3Right: "Rough",
        y4Label: "y4: Density",
        y4Left: "Simple",
        y4Right: "Complex",
        shapeLabel: "Base Shape"
    }
};

function toggleLanguage() {
    currentLang = currentLang === 'KR' ? 'EN' : 'KR';
    updateAllUIText();
}

function updateAllUIText() {
    const t = translations[currentLang];

    // Update toggle button
    document.getElementById('lang-toggle').innerText = currentLang === 'KR' ? 'EN' : 'KR';

    // Update title
    const titleEl = document.getElementById('title');
    if (titleEl) titleEl.innerText = t.title;

    // Update buttons
    const engineBtn = document.getElementById('btn-engine');
    if (engineBtn) engineBtn.innerText = t.engineBtn;

    const mainBtn = document.getElementById('btn-main');
    if (mainBtn) {
        if (state === 'IDLE') mainBtn.innerText = t.btnRecord;
        else if (state === 'RECORDING') mainBtn.innerText = t.btnStop;
        else if (state === 'REVIEWING') mainBtn.innerText = t.btnReRecord;
    }

    const playBtn = document.getElementById('btn-play');
    if (playBtn) {
        playBtn.innerText = audioTag && !audioTag.paused ? t.btnPause : t.btnPlay;
    }

    const confirmBtn = document.getElementById('btn-confirm');
    if (confirmBtn) confirmBtn.innerText = t.confirmBtn;

    const exportBtn = document.getElementById('btn-export');
    if (exportBtn) exportBtn.innerText = t.exportBtn;

    // Update status
    const statusEl = document.getElementById('status');
    if (statusEl) {
        if (state === 'IDLE') statusEl.innerText = t.statusReady;
        else if (state === 'RECORDING') statusEl.innerText = t.statusRecording;
        else if (state === 'REVIEWING') statusEl.innerText = t.statusReviewing;
    }

    // Update audio feature labels
    const labelLoud = document.getElementById('label-loud');
    if (labelLoud) labelLoud.innerText = t.labelLoud;

    const labelPitch = document.getElementById('label-pitch');
    if (labelPitch) labelPitch.innerText = t.labelPitch;

    const labelBright = document.getElementById('label-bright');
    if (labelBright) labelBright.innerText = t.labelBright;

    const labelRough = document.getElementById('label-rough');
    if (labelRough) labelRough.innerText = t.labelRough;

    // Update labeling instruction
    const labelingInstruction = document.getElementById('labeling-instruction');
    if (labelingInstruction) labelingInstruction.innerText = t.labelingInstruction;

    // Update slider labels
    const y1Label = document.getElementById('y1-label');
    if (y1Label) y1Label.innerText = t.y1Label;
    const y1Left = document.getElementById('y1-left');
    if (y1Left) y1Left.innerText = t.y1Left;
    const y1Right = document.getElementById('y1-right');
    if (y1Right) y1Right.innerText = t.y1Right;

    const y2Label = document.getElementById('y2-label');
    if (y2Label) y2Label.innerText = t.y2Label;
    const y2Left = document.getElementById('y2-left');
    if (y2Left) y2Left.innerText = t.y2Left;
    const y2Right = document.getElementById('y2-right');
    if (y2Right) y2Right.innerText = t.y2Right;

    const y3Label = document.getElementById('y3-label');
    if (y3Label) y3Label.innerText = t.y3Label;
    const y3Left = document.getElementById('y3-left');
    if (y3Left) y3Left.innerText = t.y3Left;
    const y3Right = document.getElementById('y3-right');
    if (y3Right) y3Right.innerText = t.y3Right;

    const y4Label = document.getElementById('y4-label');
    if (y4Label) y4Label.innerText = t.y4Label;
    const y4Left = document.getElementById('y4-left');
    if (y4Left) y4Left.innerText = t.y4Left;
    const y4Right = document.getElementById('y4-right');
    if (y4Right) y4Right.innerText = t.y4Right;

    const shapeLabel = document.getElementById('shape-label');
    if (shapeLabel) shapeLabel.innerText = t.shapeLabel;

    // Update data section
    const dataLabel = document.getElementById('data-label');
    if (dataLabel) dataLabel.innerText = t.dataLabel;

    const samplesLabel = document.getElementById('samples-label');
    if (samplesLabel) samplesLabel.innerText = t.samplesLabel;
}

function updateStatus(msg, cls) {
    const el = document.getElementById('status');
    el.innerText = msg; el.className = 'status-badge ' + cls;
}

function changeShape(val) {
    const names = ['Sphere', 'Cube', 'Torus', 'Cone', 'Cylinder', 'Octahedron'];
    document.getElementById('shape-name').innerText = names[val];
    createShape(val);
}

function exportCSV() {
    let csv = "loudness,pitch,brightness,roughness,y1,y2,y3,y4,shape\n";
    customTrainingData.forEach(d => { csv += `${d.x.loudness},${d.x.pitch},${d.x.brightness},${d.x.roughness},${d.y.join(',')}\n`; });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    a.download = `IML_Research_Data_${Date.now()}.csv`; a.click();
}

// Initialize UI text on page load
window.addEventListener('DOMContentLoaded', () => {
    updateAllUIText();
});
