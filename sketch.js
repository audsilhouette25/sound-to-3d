let audioCtx, analyser, microphone, mediaRecorder, sourceNode;
let audioChunks = [];
let audioTag = null;
let brain;
let scene, camera, renderer, currentMesh;
let state = 'IDLE';
let recordedX = { loudness: 0, pitch: 0, brightness: 0, roughness: 0, count: 0 };
let currentX = { loudness: 0, pitch: 0, brightness: 0, roughness: 0 };
let targetY = { y1: 0.5, y2: 0.5, y3: 0.5, y4: 0.5 };
let currentY = { y1: 0.5, y2: 0.5, y3: 0.5, y4: 0.5 };
let currentLang = 'KR';
let customTrainingData = [];

// --- GPU 셰이더 정의 (Vertex & Fragment) ---
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
        float noiseVal = noise(pos * (2.0 + uY4 * 10.0) + uTime * 0.5);
        float wave = sin(pos.x * 10.0 + uTime) * uY2 * 0.3;
        float angular = floor(noiseVal * (1.0 + (1.0-uY1)*10.0)) / (1.0 + (1.0-uY1)*10.0);
        float finalNoise = mix(noiseVal, angular, uY1);
        
        float displacement = (finalNoise * uY3 * 0.5) + (uLoudness * 0.4) + wave;
        vDisplacement = displacement;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(pos + normal * displacement, 1.0);
    }
`;

const fragmentShader = `
    varying float vDisplacement;
    varying vec3 vNormal;
    void main() {
        vec3 color1 = vec3(0.0, 1.0, 0.8); // Cyan
        vec3 color2 = vec3(0.1, 0.1, 0.4); // Deep Blue
        vec3 finalColor = mix(color2, color1, vDisplacement + 0.2);
        gl_FragColor = vec4(finalColor, 0.9);
    }
`;

let shaderUniforms = {
    uTime: { value: 0 }, uLoudness: { value: 0 },
    uY1: { value: 0.5 }, uY2: { value: 0.5 }, uY3: { value: 0.5 }, uY4: { value: 0.5 }
};

// --- 초기화 로직 ---
function initThree() {
    const container = document.getElementById('three-container');
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 3;

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    container.appendChild(renderer.domElement);

    const geometry = new THREE.SphereGeometry(1, 128, 128);
    const material = new THREE.ShaderMaterial({
        uniforms: shaderUniforms, vertexShader, fragmentShader, wireframe: true, transparent: true
    });
    currentMesh = new THREE.Mesh(geometry, material);
    scene.add(currentMesh);
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

// --- 워크플로우 제어 ---
function handleRecord() {
    if (state === 'IDLE' || state === 'REVIEWING') {
        state = 'RECORDING'; audioChunks = [];
        recordedX = { loudness: 0, pitch: 0, brightness: 0, roughness: 0, count: 0 };
        if(audioTag) audioTag.pause();
        mediaRecorder.start();
    } else {
        mediaRecorder.stop();
        state = 'REVIEWING';
        document.getElementById('labeling-zone').style.display = 'block';
    }
    updateUI();
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
    } else if (customTrainingData.length >= 5) {
        brain.predict(currentX, (err, res) => {
            if(!err) {
                targetY.y1 = res[0].value; targetY.y2 = res[1].value;
                targetY.y3 = res[2].value; targetY.y4 = res[3].value;
            }
        });
    }

    for(let k in currentY) {
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
        parseFloat(document.getElementById('y4').value)
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
        updateUI();
    });
}

function updateUI() {
    const btn = document.getElementById('btn-main');
    const status = document.getElementById('status');
    const t = translations[currentLang];
    
    if (state === 'RECORDING') {
        btn.innerText = t.btnStop; status.innerText = t.statusRecording; status.className = "status-recording";
    } else if (state === 'REVIEWING') {
        btn.innerText = t.btnReRecord; status.innerText = t.statusReview; status.className = "status-review";
    } else {
        btn.innerText = t.btnRecord; status.innerText = t.statusReady; status.className = "status-idle";
    }
}

const translations = {
    KR: { btnRecord: "녹음 시작", btnStop: "녹음 중단", btnReRecord: "다시 녹음하기", statusReady: "엔진 준비됨", statusRecording: "녹음 중...", statusReview: "라벨링 모드" },
    EN: { btnRecord: "Record", btnStop: "Stop", btnReRecord: "Re-record", statusReady: "Ready", statusRecording: "Recording...", statusReview: "Labeling" }
};

function toggleLang() {
    currentLang = currentLang === 'KR' ? 'EN' : 'KR';
    document.getElementById('lang-toggle').innerText = currentLang === 'KR' ? 'EN' : 'KR';
    updateUI();
}

function exportCSV() {
    let csv = "loudness,pitch,brightness,roughness,y1,y2,y3,y4\n";
    customTrainingData.forEach(d => { csv += `${d.x.loudness},${d.x.pitch},${d.x.brightness},${d.x.roughness},${d.y.join(',')}\n`; });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    a.download = `IML_Data_${Date.now()}.csv`;
    a.click();
}
