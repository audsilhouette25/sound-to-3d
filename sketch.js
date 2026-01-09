let audioCtx, analyser, microphone, mediaRecorder, sourceNode;
let audioChunks = [];
let audioTag = null;
let brain;
let scene, camera, renderer, currentMesh;
let state = 'IDLE'; // IDLE, RECORDING, REVIEWING
let recordedX = { loudness: 0, pitch: 0, brightness: 0, roughness: 0, count: 0 };
let currentX = { loudness: 0, pitch: 0, brightness: 0, roughness: 0 };
let targetY = { y1: 0.5, y2: 0.5, y3: 0.5, y4: 0.5, shape: 0 };
let currentY = { y1: 0.5, y2: 0.5, y3: 0.5, y4: 0.5, shape: 0 };
let customTrainingData = [];
let currentLang = 'KR';

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
                   mix(mix(hash(n+113.0),hash(n+114.0),f.x), mix(hash(n+170.0),hash(n+171.0),f.x),f.y),f.z);
    }

    void main() {
        vNormal = normal;
        vec3 pos = position;
        float noiseVal = noise(pos * (2.0 + uY4 * 8.0) + uTime * 0.4);
        float angular = floor(noiseVal * (1.0 + (1.0-uY1)*12.0)) / (1.0 + (1.0-uY1)*12.0);
        float finalNoise = mix(noiseVal, angular, uY1);
        float wave = sin(pos.x * 12.0 + uTime) * uY2 * 0.45;

        // 변형 강도를 줄여서 형상이 터지지 않게 함
        float displacement = (finalNoise * uY3 * 0.35) + (uLoudness * 0.3) + (wave * 0.5);
        vDisplacement = displacement;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(pos + normal * displacement, 1.0);
    }
`;

const fragmentShader = `
    varying float vDisplacement;
    void main() {
        vec3 colorA = vec3(0.0, 1.0, 0.7); // Cyan
        vec3 colorB = vec3(0.1, 0.05, 0.4); // Dark Blue
        vec3 finalColor = mix(colorB, colorA, vDisplacement + 0.25);
        gl_FragColor = vec4(finalColor, 0.9);
    }
`;

let shaderUniforms = {
    uTime: { value: 0 }, uLoudness: { value: 0 },
    uY1: { value: 0.5 }, uY2: { value: 0.5 }, uY3: { value: 0.5 }, uY4: { value: 0.5 }
};

// --- Initialization ---
function initThree() {
    const container = document.getElementById('three-container');
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 3.5;

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    container.appendChild(renderer.domElement);

    createShape(0);
    animate();
}

function createShape(type) {
    if (currentMesh) { scene.remove(currentMesh); currentMesh.geometry.dispose(); }
    let geo;
    const res = 128; // 고해상도 (GPU 덕분에 가능)
    if (type == 0) geo = new THREE.SphereGeometry(1, res, res);
    else if (type == 1) geo = new THREE.BoxGeometry(1.4, 1.4, 1.4, 16, 16, 16); // 낮은 해상도로 터짐 방지
    else if (type == 2) geo = new THREE.TorusGeometry(0.8, 0.4, 64, 128);
    else if (type == 3) geo = new THREE.ConeGeometry(1, 2, 64, 64);
    else if (type == 4) geo = new THREE.CylinderGeometry(0.8, 0.8, 2, 64, 64);
    else geo = new THREE.OctahedronGeometry(1.2, 32);

    const mat = new THREE.ShaderMaterial({
        uniforms: shaderUniforms, vertexShader, fragmentShader, wireframe: true, transparent: true
    });
    currentMesh = new THREE.Mesh(geo, mat);
    scene.add(currentMesh);
}

async function initEngine() {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    microphone = audioCtx.createMediaStreamSource(stream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 512;
    microphone.connect(analyser);

    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
    mediaRecorder.onstop = saveRecording;

    brain = ml5.neuralNetwork({
        inputs: ['loudness', 'pitch', 'brightness', 'roughness'],
        outputs: ['y1', 'y2', 'y3', 'y4', 'shape'],
        task: 'regression', debug: false
    });

    document.getElementById('btn-engine').style.display = 'none';
    document.getElementById('btn-main').style.display = 'block';
    document.getElementById('save-load-zone').style.display = 'block';
    initThree();

    // localStorage에서 저장된 학습 데이터 즉시 불러오기 (지연 없음)
    loadTrainingData();
}

// --- Workflow ---
function handleRecord() {
    const t = translations[currentLang];
    if (state === 'IDLE' || state === 'REVIEWING') {
        state = 'RECORDING'; audioChunks = [];
        recordedX = { loudness: 0, pitch: 0, brightness: 0, roughness: 0, count: 0 };
        if(audioTag) audioTag.pause();
        mediaRecorder.start();
        document.getElementById('btn-main').innerText = t.btnStop;
        updateStatus('Recording...', 'status-recording');
    } else {
        mediaRecorder.stop();
        state = 'REVIEWING';
        document.getElementById('labeling-zone').style.display = 'block';
        document.getElementById('btn-play').style.display = 'block';
        document.getElementById('btn-main').innerText = t.btnReRecord;
        updateStatus('Reviewing...', 'status-review');
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
    if (audioTag.paused) {
        if (sourceNode) sourceNode.disconnect();
        sourceNode = audioCtx.createMediaElementSource(audioTag);
        sourceNode.connect(analyser); analyser.connect(audioCtx.destination);
        audioTag.play();
    } else { audioTag.pause(); }
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
    } else if (customTrainingData.length >= 3) {
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
    currentMesh.rotation.y += 0.005;
    renderer.render(scene, camera);
}

function analyzeAudio() {
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
        alert("학습 완료!");
        document.getElementById('data-count').innerText = customTrainingData.length;
        state = 'IDLE'; if(audioTag) audioTag.pause();
        document.getElementById('labeling-zone').style.display = 'none';
        document.getElementById('btn-play').style.display = 'none';
        updateStatus('Ready', 'status-idle');
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
    KR: { btnRecord: "녹음 시작", btnStop: "중단", btnReRecord: "다시 녹음" },
    EN: { btnRecord: "Record", btnStop: "Stop", btnReRecord: "Re-record" }
};

function toggleLanguage() {
    currentLang = currentLang === 'KR' ? 'EN' : 'KR';
    document.getElementById('lang-toggle').innerText = currentLang === 'KR' ? 'EN' : 'KR';
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
