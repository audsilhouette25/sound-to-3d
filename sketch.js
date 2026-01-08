let audioCtx, analyser, microphone, mediaRecorder, sourceNode;
let audioChunks = [];
let audioTag = null;
let brain;
let scene, camera, renderer, currentMesh;
let state = 'IDLE';
let recordedX = { loudness: 0, pitch: 0, brightness: 0, roughness: 0, count: 0 };
let currentX = { loudness: 0, pitch: 0, brightness: 0, roughness: 0 };
let targetY = { y1: 0.5, y2: 0.5, y3: 0.5, y4: 0.5, shape: 0 };
let currentY = { y1: 0.5, y2: 0.5, y3: 0.5, y4: 0.5, shape: 0 };
let customTrainingData = [];
let currentLang = 'KR';

// --- GPU 셰이더 정의 (Vertex Shader: 형태 변형 담당) ---
const vertexShader = `
    varying float vDisplacement;
    varying vec3 vNormal;
    uniform float uTime;
    uniform float uLoudness;
    uniform float uY1, uY2, uY3, uY4;

    // GPU용 노이즈 함수
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
        
        // y4: Density (노이즈의 밀도 조절)
        float noiseVal = noise(pos * (2.0 + uY4 * 10.0) + uTime * 0.5);
        
        // y1: Angularity (노이즈를 계단식으로 깎아 각진 형태 생성)
        float angular = floor(noiseVal * (1.0 + (1.0-uY1)*10.0)) / (1.0 + (1.0-uY1)*10.0);
        float finalNoise = mix(noiseVal, angular, uY1);
        
        // y2: Spikiness (삼각함수를 이용한 뾰족한 가시 생성)
        float wave = sin(pos.x * 15.0 + uTime) * uY2 * 0.4;
        
        // y3: Texture (거칠기 조절)
        float displacement = (finalNoise * uY3 * 0.6) + (uLoudness * 0.5) + wave;
        
        vDisplacement = displacement;
        vec3 newPos = pos + normal * displacement;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(newPos, 1.0);
    }
`;

// --- GPU 셰이더 정의 (Fragment Shader: 색상 담당) ---
const fragmentShader = `
    varying float vDisplacement;
    varying vec3 vNormal;
    void main() {
        vec3 colorA = vec3(0.0, 1.0, 0.8); // Cyan
        vec3 colorB = vec3(0.1, 0.05, 0.3); // Deep Purple
        vec3 finalColor = mix(colorB, colorA, vDisplacement + 0.3);
        gl_FragColor = vec4(finalColor, 0.85);
    }
`;

let shaderUniforms = {
    uTime: { value: 0 }, uLoudness: { value: 0 },
    uY1: { value: 0.5 }, uY2: { value: 0.5 }, uY3: { value: 0.5 }, uY4: { value: 0.5 }
};

// --- 초기화 및 엔진 로직 ---
function initThree() {
    const container = document.getElementById('three-container');
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.z = 3.5;

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
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
    if (type == 0) geo = new THREE.SphereGeometry(1, 128, 128);
    else if (type == 1) geo = new THREE.BoxGeometry(1.5, 1.5, 1.5, 64, 64, 64);
    else if (type == 2) geo = new THREE.TorusGeometry(0.8, 0.4, 64, 128);
    else if (type == 3) geo = new THREE.ConeGeometry(1, 2, 64, 64);
    else if (type == 4) geo = new THREE.CylinderGeometry(0.8, 0.8, 2, 64, 64);
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
}

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
        updateStatus('Reviewing & Labeling', 'status-review');
    }
}

function saveRecording() {
    const blob = new Blob(audioChunks, { type: 'audio/wav' });
    const url = URL.createObjectURL(blob);
    audioTag = new Audio(url);
    audioTag.loop = true;
    
    recordedX.loudness /= recordedX.count;
    recordedX.pitch /= recordedX.count;
    recordedX.brightness /= recordedX.count;
    recordedX.roughness /= recordedX.count;
}

function togglePlayback() {
    const t = translations[currentLang];
    if (audioTag.paused) {
        if (sourceNode) sourceNode.disconnect();
        sourceNode = audioCtx.createMediaElementSource(audioTag);
        sourceNode.connect(analyser);
        analyser.connect(audioCtx.destination);
        audioTag.play();
        document.getElementById('btn-play').innerText = t.btnPause || "Pause";
    } else {
        audioTag.pause();
        document.getElementById('btn-play').innerText = t.btnPlay || "Play Loop";
    }
}

function animate() {
    requestAnimationFrame(animate);
    if (!analyser) return;

    // 1. 오디오 분석
    analyzeAudio();
    
    // 2. 셰이더 유니폼 업데이트
    shaderUniforms.uTime.value += 0.05;
    shaderUniforms.uLoudness.value = currentX.loudness;

    // 3. 상태별 로직 (리뷰 중엔 슬라이더 값, 그 외엔 AI 예측값 사용)
    if (state === 'REVIEWING') {
        targetY.y1 = parseFloat(document.getElementById('y1').value);
        targetY.y2 = parseFloat(document.getElementById('y2').value);
        targetY.y3 = parseFloat(document.getElementById('y3').value);
        targetY.y4 = parseFloat(document.getElementById('y4').value);
    } else if (customTrainingData.length >= 3) {
        brain.predict(currentX, (err, res) => {
            if(!err) {
                targetY.y1 = res[0].value; targetY.y2 = res[1].value;
                targetY.y3 = res[2].value; targetY.y4 = res[3].value;
                const nextShape = Math.round(res[4].value * 5);
                if(nextShape !== targetY.shape) {
                    targetY.shape = nextShape;
                    createShape(nextShape);
                }
            }
        });
    }

    // 4. 부드러운 보간(Lerp) 적용하여 유니폼에 전송
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
        recordedX.loudness += currentX.loudness;
        recordedX.pitch += currentX.pitch;
        recordedX.brightness += currentX.brightness;
        recordedX.roughness += currentX.roughness;
        recordedX.count++;
    }
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
    
    brain.normalizeData();
    brain.train({ epochs: 30 }, () => {
        alert(currentLang === 'KR' ? "학습 완료!" : "Training Complete!");
        document.getElementById('data-count').innerText = customTrainingData.length;
        state = 'IDLE';
        if(audioTag) audioTag.pause();
        document.getElementById('labeling-zone').style.display = 'none';
        document.getElementById('btn-play').style.display = 'none';
        updateStatus('Ready', 'status-idle');
    });
}

const translations = {
    KR: { btnRecord: "녹음 시작", btnStop: "중단", btnReRecord: "다시 녹음", btnPlay: "소리 듣기", btnPause: "일시정지" },
    EN: { btnRecord: "Record", btnStop: "Stop", btnReRecord: "Re-record", btnPlay: "Play Loop", btnPause: "Pause" }
};

function toggleLanguage() {
    currentLang = currentLang === 'KR' ? 'EN' : 'KR';
    document.getElementById('lang-toggle').innerText = currentLang === 'KR' ? 'EN' : 'KR';
}

function updateStatus(msg, cls) {
    const el = document.getElementById('status');
    el.innerText = msg;
    el.className = 'status-badge ' + cls;
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
    a.download = `IML_Shader_Data_${Date.now()}.csv`;
    a.click();
}
