// --- 전역 변수 및 상태 관리 (제공해주신 로직 유지) ---
let audioCtx, analyser, microphone, mediaRecorder, sourceNode;
let audioChunks = [];
let brain;
let scene, camera, renderer, currentMesh;
let state = 'IDLE';
let recordedX = { loudness: 0, pitch: 0, brightness: 0, roughness: 0, count: 0 };
let currentX = { loudness: 0, pitch: 0, brightness: 0, roughness: 0 };
let targetY = { y1: 0.2, y2: 0.2, y3: 0.2, y4: 0.2 };
let currentY = { y1: 0.2, y2: 0.2, y3: 0.2, y4: 0.2 };
let currentLang = 'KR';
let customTrainingData = [];

// --- [GPU 셰이더 코드] 정점 변형을 처리하는 GLSL ---
const vertexShader = `
    varying float vDisplacement;
    uniform float uTime;
    uniform float uLoudness;
    uniform float uY1; // Angularity
    uniform float uY2; // Spikiness
    uniform float uY3; // Roughness
    uniform float uY4; // Density

    // 클래식 노이즈 함수
    float hash(float n) { return fract(sin(n) * 43758.5453123); }
    float noise(vec3 x) {
        vec3 p = floor(x); vec3 f = fract(x);
        f = f*f*(3.0-2.0*f);
        float n = p.x + p.y*57.0 + 113.0*p.z;
        return mix(mix(mix(hash(n+0.0), hash(n+1.0),f.x), mix(hash(n+57.0), hash(n+58.0),f.x),f.y),
                   mix(mix(hash(n+113.0),hash(n+114.0),f.x), mix(hash(n+170.0),hash(n+171.0),f.x),f.y),f.z);
    }

    void main() {
        vec3 pos = position;
        
        // 1. 기본 노이즈 (Roughness + Density)
        float noiseVal = noise(pos * (2.0 + uY4 * 15.0) + uTime);
        
        // 2. 파동 변형 (Spikiness)
        float wave = sin(pos.x * 5.0 + uTime * 2.0) * uY2 * 0.4;
        
        // 3. 각짐 표현 (Angularity) - 노이즈를 계단식으로 처리
        float angularNoise = floor(noiseVal * (2.0 + (1.0 - uY1) * 10.0)) / (2.0 + (1.0 - uY1) * 10.0);
        float finalNoise = mix(noiseVal, angularNoise, uY1);
        
        float displacement = (finalNoise * uY3 * 0.6) + (uLoudness * 0.5) + wave;
        vDisplacement = displacement;

        vec3 newPosition = pos + normal * displacement;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(newPosition, 1.0);
    }
`;

const fragmentShader = `
    varying float vDisplacement;
    void main() {
        // 변형 정도에 따른 색상 변화 (연구용 시각 피드백)
        vec3 colorA = vec3(0.0, 1.0, 0.8); // Neon Green
        vec3 colorB = vec3(0.1, 0.2, 0.8); // Deep Blue
        vec3 finalColor = mix(colorB, colorA, vDisplacement);
        gl_FragColor = vec4(finalColor, 0.9);
    }
`;

let shaderUniforms = {
    uTime: { value: 0 },
    uLoudness: { value: 0 },
    uY1: { value: 0.5 },
    uY2: { value: 0.5 },
    uY3: { value: 0.5 },
    uY4: { value: 0.5 }
};

// --- [초기화 및 렌더링] ---
function initThree() {
    const container = document.getElementById('three-container');
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 3;

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    container.appendChild(renderer.domElement);

    // 고해상도 구체 생성 (셰이더 덕분에 가능)
    const geometry = new THREE.SphereGeometry(1, 128, 128);
    const material = new THREE.ShaderMaterial({
        uniforms: shaderUniforms,
        vertexShader: vertexShader,
        fragmentShader: fragmentShader,
        wireframe: true,
        transparent: true
    });
    currentMesh = new THREE.Mesh(geometry, material);
    scene.add(currentMesh);

    scene.add(new THREE.AmbientLight(0xffffff, 0.5));
    animate();
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
        outputs: ['y1', 'y2', 'y3', 'y4'],
        task: 'regression', debug: false
    });

    document.getElementById('btn-start').style.display = 'none';
    document.getElementById('btn-main').style.display = 'block';
    document.getElementById('save-load-zone').style.display = 'block';
    initThree();
}

// --- [오디오 및 상태 관리 (기존 로직 유지)] ---
function handleRecord() {
    if (state === 'IDLE' || state === 'REVIEWING') startRecording();
    else if (state === 'RECORDING') stopRecording();
}

function startRecording() {
    state = 'RECORDING'; audioChunks = [];
    recordedX = { loudness: 0, pitch: 0, brightness: 0, roughness: 0, count: 0 };
    if(audioTag) audioTag.pause();
    mediaRecorder.start();
    updateUIForState();
}

function stopRecording() {
    mediaRecorder.stop();
    state = 'REVIEWING';
    updateUIForState();
    document.getElementById('labeling-zone').style.display = 'block';
}

function saveRecording() {
    const blob = new Blob(audioChunks, { type: 'audio/wav' });
    const url = URL.createObjectURL(blob);
    audioTag = new Audio(url);
    audioTag.loop = true;
    if (sourceNode) sourceNode.disconnect();
    sourceNode = audioCtx.createMediaElementSource(audioTag);
    sourceNode.connect(analyser);
    analyser.connect(audioCtx.destination);
    audioTag.play();

    recordedX.loudness /= recordedX.count;
    recordedX.pitch /= recordedX.count;
    recordedX.brightness /= recordedX.count;
    recordedX.roughness /= recordedX.count;
}

// --- [메인 루프: GPU 데이터 전송] ---
function animate() {
    requestAnimationFrame(animate);
    if (!analyser) return;

    analyzeAudio();
    
    // 1. 시간 업데이트
    shaderUniforms.uTime.value += 0.05;
    
    // 2. 실시간 소리 크기 업데이트
    shaderUniforms.uLoudness.value = currentX.loudness;

    // 3. 형태 파라미터 업데이트
    if (state === 'REVIEWING') {
        // 리뷰 중에는 사용자가 조절하는 슬라이더 값을 직접 셰이더에 전송
        targetY.y1 = parseFloat(document.getElementById('y1').value);
        targetY.y2 = parseFloat(document.getElementById('y2').value);
        targetY.y3 = parseFloat(document.getElementById('y3').value);
        targetY.y4 = parseFloat(document.getElementById('y4').value);
    } else if (customTrainingData.length >= 5) {
        // 학습 후에는 AI가 예측한 값을 전송
        brain.predict(currentX, (err, res) => {
            if(!err) {
                targetY.y1 = res[0].value; targetY.y2 = res[1].value;
                targetY.y3 = res[2].value; targetY.y4 = res[3].value;
            }
        });
    }

    // 부드러운 전이를 위한 보간
    for(let k in currentY) {
        currentY[k] += (targetY[k] - currentY[k]) * 0.1;
        shaderUniforms[`u${k.toUpperCase()}`].value = currentY[k];
    }

    currentMesh.rotation.y += 0.005 + (currentY.y1 * 0.05);
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
    
    let z=0; for(let i=1; i<time.length; i++) if(time[i]>128 && time[i-1]<=128) z++;
    currentX.roughness = z/40.0;

    // UI 모니터링 업데이트
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

// --- [학습 데이터 확정] ---
function confirmTrainingWrapper() {
    const labels = {
        y1: parseFloat(document.getElementById('y1').value),
        y2: parseFloat(document.getElementById('y2').value),
        y3: parseFloat(document.getElementById('y3').value),
        y4: parseFloat(document.getElementById('y4').value)
    };
    
    brain.addData([recordedX.loudness, recordedX.pitch, recordedX.brightness, recordedX.roughness], [labels.y1, labels.y2, labels.y3, labels.y4]);
    customTrainingData.push({ x: {...recordedX}, y: labels });
    
    brain.normalizeData();
    brain.train({ epochs: 30 }, () => {
        alert("Training Complete!");
        document.getElementById('data-count').innerText = customTrainingData.length;
        state = 'IDLE';
        if(audioTag) audioTag.pause();
        document.getElementById('labeling-zone').style.display = 'none';
        updateUIForState();
    });
}

function updateUIForState() {
    const btn = document.getElementById('btn-main');
    const status = document.getElementById('status');
    if (state === 'RECORDING') { btn.innerText = "Stop"; status.innerText = "Recording..."; status.className = "status-recording"; }
    else if (state === 'REVIEWING') { btn.innerText = "Re-record"; status.innerText = "Labeling Mode"; status.className = "status-review"; }
    else { btn.innerText = "Record"; status.innerText = "Ready"; status.className = "status-idle"; }
}

function exportCSV() {
    let csv = "loudness,pitch,brightness,roughness,y1,y2,y3,y4\n";
    customTrainingData.forEach(d => {
        csv += `${d.x.loudness},${d.x.pitch},${d.x.brightness},${d.x.roughness},${d.y.y1},${d.y.y2},${d.y.y3},${d.y.y4}\n`;
    });
    const blob = new Blob([csv], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `IML_GPU_Data_${Date.now()}.csv`;
    a.click();
}
