let audioCtx, analyser, microphone, mediaRecorder, sourceNode;
let audioChunks = [];
let brain;
let scene, camera, renderer, mesh, material;

// 상태 및 데이터
let state = 'IDLE'; // IDLE, RECORDING, REVIEWING
let currentX = { loudness: 0, pitch: 0, bright: 0, rough: 0 };
let recordedX = { loudness: 0, pitch: 0, bright: 0, rough: 0, count: 0 };
let uniforms = {
    uTime: { value: 0 },
    uLoudness: { value: 0 },
    uAngularity: { value: 0.2 },
    uSpikiness: { value: 0.2 },
    uRoughness: { value: 0.2 },
    uComplexity: { value: 0.2 }
};

// --- [GPU 셰이더 정의] 성능 극대화를 위해 계산을 GPU로 이전 ---
const vertexShader = `
    varying vec2 vUv;
    varying float vDisplacement;
    uniform float uTime;
    uniform float uLoudness;
    uniform float uAngularity;
    uniform float uSpikiness;
    uniform float uRoughness;
    uniform float uComplexity;

    // 노이즈 함수 (각짐과 질감 표현용)
    float hash(float n) { return fract(sin(n) * 43758.5453123); }
    float noise(vec3 x) {
        vec3 p = floor(x); vec3 f = fract(x);
        f = f*f*(3.0-2.0*f);
        float n = p.x + p.y*57.0 + 113.0*p.z;
        return mix(mix(mix(hash(n+0.0), hash(n+1.0),f.x), mix(hash(n+57.0), hash(n+58.0),f.x),f.y),
                   mix(mix(hash(n+113.0),hash(n+114.0),f.x), mix(hash(n+170.0),hash(n+171.0),f.x),f.y),f.z);
    }

    void main() {
        vUv = uv;
        vec3 pos = position;
        float b = noise(pos * (2.0 + uComplexity * 10.0) + uTime);
        
        // 변형 계산 로직
        float wave = sin(pos.x * 10.0 + uTime) * uSpikiness * 0.3;
        float displace = (b * uRoughness * 0.5) + (uLoudness * 0.5) + wave;
        
        vDisplacement = displace;
        vec3 newPos = pos + normal * displace;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(newPos, 1.0);
    }
`;

const fragmentShader = `
    varying float vDisplacement;
    void main() {
        vec3 color = mix(vec3(0.0, 1.0, 0.8), vec3(0.1, 0.2, 0.5), vDisplacement);
        gl_FragColor = vec4(color, 0.8);
    }
`;

// --- [초기화] ---
function initThree() {
    const container = document.getElementById('canvas-container');
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
    camera.position.z = 3;

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    container.appendChild(renderer.domElement);

    // 고해상도 구체 (GPU 셰이더 덕분에 가능)
    const geometry = new THREE.SphereGeometry(1, 128, 128);
    material = new THREE.ShaderMaterial({
        uniforms: uniforms,
        vertexShader: vertexShader,
        fragmentShader: fragmentShader,
        wireframe: true,
        transparent: true
    });
    mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);
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
    mediaRecorder.onstop = playReview;

    brain = ml5.neuralNetwork({
        inputs: ['loudness', 'pitch', 'bright', 'rough'],
        outputs: ['y1', 'y2', 'y3', 'y4'],
        task: 'regression', debug: false
    });

    document.getElementById('btn-start').classList.add('hidden');
    document.getElementById('btn-record').classList.remove('hidden');
    document.getElementById('monitor-section').classList.remove('hidden');
    state = 'IDLE';
    initThree();
}

// --- [오디오 분석 및 워크플로우] ---
function handleRecord() {
    if (state !== 'RECORDING') {
        state = 'RECORDING';
        audioChunks = [];
        recordedX = { loudness: 0, pitch: 0, bright: 0, rough: 0, count: 0 };
        mediaRecorder.start();
        document.getElementById('btn-record').innerText = "녹음 중단";
        document.getElementById('status').className = "status-recording";
        document.getElementById('status').innerText = "녹음 중...";
    } else {
        mediaRecorder.stop();
        state = 'REVIEWING';
        document.getElementById('btn-record').innerText = "다시 녹음하기";
        document.getElementById('status').className = "status-review";
        document.getElementById('status').innerText = "리뷰 및 라벨링";
        document.getElementById('label-section').classList.remove('hidden');
    }
}

function playReview() {
    const blob = new Blob(audioChunks, { type: 'audio/wav' });
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.loop = true;
    
    if (sourceNode) sourceNode.disconnect();
    sourceNode = audioCtx.createMediaElementSource(audio);
    sourceNode.connect(analyser);
    analyser.connect(audioCtx.destination);
    audio.play();

    // 평균값 확정 (학습용)
    recordedX.loudness /= recordedX.count;
    recordedX.pitch /= recordedX.count;
    recordedX.bright /= recordedX.count;
    recordedX.rough /= recordedX.count;
}

function analyzeAudio() {
    const data = new Uint8Array(analyser.frequencyBinCount);
    const time = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(data);
    analyser.getByteTimeDomainData(time);

    // 1. Loudness (정규화 반영)
    let sum = 0;
    for(let v of time) { let n = (v-128)/128; sum += n*n; }
    currentX.loudness = Math.sqrt(sum/time.length) * 5.0;

    // 2. Pitch / Centroid
    let total = 0, weight = 0;
    for(let i=0; i<data.length; i++) { weight += i * data[i]; total += data[i]; }
    currentX.pitch = total > 0 ? (weight / total) / 50.0 : 0;
    currentX.bright = currentX.pitch * 1.2;

    // 3. Roughness (ZCR)
    let zcr = 0;
    for(let i=1; i<time.length; i++) if(time[i]>128 && time[i-1]<=128) zcr++;
    currentX.rough = zcr / 40.0;

    // UI 업데이트
    document.getElementById('val-loud').innerText = currentX.loudness.toFixed(2);
    document.getElementById('val-pitch').innerText = currentX.pitch.toFixed(2);
    document.getElementById('val-bright').innerText = currentX.bright.toFixed(2);
    document.getElementById('val-rough').innerText = currentX.rough.toFixed(2);

    if (state === 'RECORDING') {
        recordedX.loudness += currentX.loudness;
        recordedX.pitch += currentX.pitch;
        recordedX.bright += currentX.bright;
        recordedX.rough += currentX.rough;
        recordedX.count++;
    }
}

function animate() {
    requestAnimationFrame(animate);
    analyzeAudio();

    uniforms.uTime.value += 0.05;
    uniforms.uLoudness.value = currentX.loudness;

    if (state === 'REVIEWING') {
        // 리뷰 중에는 슬라이더 값을 셰이더에 즉시 반영
        uniforms.uAngularity.value = parseFloat(document.getElementById('y1').value);
        uniforms.uSpikiness.value = parseFloat(document.getElementById('y2').value);
        uniforms.uRoughness.value = parseFloat(document.getElementById('y3').value);
        uniforms.uComplexity.value = parseFloat(document.getElementById('y4').value);
    } else if (brain.data.list.length >= 5) {
        // 학습 후에는 AI가 예측한 값을 셰이더에 반영
        brain.predict(currentX, (err, res) => {
            if(!err) {
                uniforms.uAngularity.value = res[0].value;
                uniforms.uSpikiness.value = res[1].value;
                uniforms.uRoughness.value = res[2].value;
                uniforms.uComplexity.value = res[3].value;
            }
        });
    }

    renderer.render(scene, camera);
}

// --- [데이터 관리] ---
function confirmTraining() {
    const labels = [
        parseFloat(document.getElementById('y1').value),
        parseFloat(document.getElementById('y2').value),
        parseFloat(document.getElementById('y3').value),
        parseFloat(document.getElementById('y4').value)
    ];

    brain.addData(
        [recordedX.loudness, recordedX.pitch, recordedX.bright, recordedX.rough],
        labels
    );

    brain.normalizeData();
    brain.train({ epochs: 30 }, () => {
        alert("데이터 수집 완료!");
        document.getElementById('count').innerText = brain.data.list.length;
        state = 'IDLE';
    });
}

function exportCSV() {
    brain.saveData('IML_Research_Data');
}
