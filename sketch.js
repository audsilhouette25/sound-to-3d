let audioCtx, analyser, microphone, mediaRecorder;
let audioChunks = [];
let audioBlob, audioUrl, audioTag, sourceNode;
let brain;
let scene, camera, renderer, currentMesh, originalVertices;

// 상태 관리
let state = 'IDLE';
let recordedX = { loudness: 0, pitch: 0, brightness: 0, roughness: 0 };
let currentX = { loudness: 0, pitch: 0, brightness: 0, roughness: 0 };
let targetY = { y1: 0.5, y2: 0.5, y3: 0.5, y4: 0.5, shape: 0 };
let currentY = { y1: 0.5, y2: 0.5, y3: 0.5, y4: 0.5, shape: 0 };

const tempVec = new THREE.Vector3();

// 다양한 기본 형태 정의
const SHAPES = {
    SPHERE: 0,
    CUBE: 1,
    TORUS: 2,
    CONE: 3,
    CYLINDER: 4,
    OCTAHEDRON: 5
};

const SHAPE_NAMES = ['구체', '정육면체', '토러스', '원뿔', '원기둥', '팔면체'];

window.onload = () => { initThree(); };

function initThree() {
    const container = document.getElementById('three-container');
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 3.5;
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    container.appendChild(renderer.domElement);

    // 기본 형태로 구체 생성
    createShape(SHAPES.SPHERE);

    scene.add(new THREE.DirectionalLight(0xffffff, 1), new THREE.AmbientLight(0x222222));
    animate();
}

// 형태 생성 함수
function createShape(shapeType) {
    // 기존 메쉬 제거
    if (currentMesh) {
        scene.remove(currentMesh);
        currentMesh.geometry.dispose();
        currentMesh.material.dispose();
    }

    let geometry;
    switch(shapeType) {
        case SHAPES.SPHERE:
            geometry = new THREE.SphereGeometry(1, 48, 48);
            break;
        case SHAPES.CUBE:
            geometry = new THREE.BoxGeometry(1.5, 1.5, 1.5, 32, 32, 32);
            break;
        case SHAPES.TORUS:
            geometry = new THREE.TorusGeometry(0.8, 0.4, 32, 64);
            break;
        case SHAPES.CONE:
            geometry = new THREE.ConeGeometry(1, 2, 48, 32);
            break;
        case SHAPES.CYLINDER:
            geometry = new THREE.CylinderGeometry(0.8, 0.8, 2, 48, 32);
            break;
        case SHAPES.OCTAHEDRON:
            geometry = new THREE.OctahedronGeometry(1.2, 4);
            break;
        default:
            geometry = new THREE.SphereGeometry(1, 48, 48);
    }

    const material = new THREE.MeshStandardMaterial({
        color: 0x00ffcc,
        wireframe: true,
        metalness: 0.3,
        roughness: 0.4
    });

    currentMesh = new THREE.Mesh(geometry, material);
    scene.add(currentMesh);
    originalVertices = currentMesh.geometry.attributes.position.array.slice();
}

async function initEngine() {
    updateStatus('엔진 초기화 중...', 'status-idle');

    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    microphone = audioCtx.createMediaStreamSource(stream);

    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 512;
    microphone.connect(analyser); // 기본적으로 마이크를 분석기에 연결

    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
    mediaRecorder.onstop = saveRecording;

    brain = ml5.neuralNetwork({
        inputs: ['loudness', 'pitch', 'brightness', 'roughness'],
        outputs: ['y1', 'y2', 'y3', 'y4', 'shape'],
        task: 'regression', debug: true
    });

    // 저장된 학습 데이터 불러오기
    loadTrainingData();

    document.getElementById('btn-engine').style.display = 'none';
    document.getElementById('btn-main').style.display = 'block';
    document.getElementById('save-load-zone').style.display = 'block';
    updateDataCount();

    updateStatus('대기 중 (녹음 가능)', 'status-idle');
}

// 상태 업데이트 함수
function updateStatus(message, className) {
    const statusEl = document.getElementById('status');
    statusEl.innerText = message;
    statusEl.className = 'status-badge ' + className;
}

function handleRecord() {
    if (state === 'IDLE' || state === 'REVIEWING') startRecording();
    else if (state === 'RECORDING') stopRecording();
}

function startRecording() {
    state = 'RECORDING';
    audioChunks = [];
    recordedX = { loudness: 0, pitch: 0, brightness: 0, roughness: 0, count: 0 };

    // 이전 녹음 데이터 삭제
    if(audioTag) {
        audioTag.pause();
        audioTag = null;
    }
    if(audioUrl) {
        URL.revokeObjectURL(audioUrl);
        audioUrl = null;
    }
    if(sourceNode) {
        sourceNode.disconnect();
        sourceNode = null;
    }

    // 분석기를 다시 마이크에 연결
    microphone.connect(analyser);

    mediaRecorder.start();
    document.getElementById('btn-main').innerText = "녹음 중단 (Stop)";
    document.getElementById('labeling-zone').style.display = "none";
    document.getElementById('btn-play').style.display = "none";

    updateStatus('🔴 녹음 중...', 'status-recording');
}

function stopRecording() {
    mediaRecorder.stop();
    state = 'REVIEWING';
    document.getElementById('btn-main').innerText = "다시 녹음하기";
    document.getElementById('labeling-zone').style.display = "block";
    document.getElementById('btn-confirm').style.display = "block";
    document.getElementById('btn-play').style.display = "inline-block";
    document.getElementById('btn-play').innerText = "▶ 녹음 재생";

    updateStatus('✏️ 리뷰 중 (라벨링 대기)', 'status-review');
}

function saveRecording() {
    audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    audioUrl = URL.createObjectURL(audioBlob);

    // 오디오 태그 준비만 하고 자동 재생하지 않음
    audioTag = new Audio(audioUrl);
    audioTag.loop = true;

    // 녹음된 평균값 저장 (학습용)
    recordedX.loudness /= recordedX.count;
    recordedX.pitch /= recordedX.count;
    recordedX.brightness /= recordedX.count;
    recordedX.roughness /= recordedX.count;
}

// 녹음 재생/일시정지 토글
function togglePlayback() {
    if (!audioTag) return;

    if (audioTag.paused) {
        // 재생 시작
        if (sourceNode) sourceNode.disconnect();
        sourceNode = audioCtx.createMediaElementSource(audioTag);
        sourceNode.connect(analyser);
        analyser.connect(audioCtx.destination);

        audioTag.play();
        document.getElementById('btn-play').innerText = "⏸ 일시정지";
    } else {
        // 일시정지
        audioTag.pause();
        document.getElementById('btn-play').innerText = "▶ 녹음 재생";
    }
}

function animate() {
    requestAnimationFrame(animate);
    
    if (analyser) {
        analyzeAudio();

        // [핵심 수정] 리뷰 모드일 때는 슬라이더 값을 즉시 targetY에 반영
        if (state === 'REVIEWING') {
            targetY.y1 = parseFloat(document.getElementById('y1').value);
            targetY.y2 = parseFloat(document.getElementById('y2').value);
            targetY.y3 = parseFloat(document.getElementById('y3').value);
            targetY.y4 = parseFloat(document.getElementById('y4').value);
            targetY.shape = parseFloat(document.getElementById('shape-selector').value);
        } else if (brain && brain.data && brain.data.list.length >= 5) {
            // 평상시에는 AI가 예측
            brain.predict(currentX, (err, res) => {
                if(!err) {
                    targetY.y1 = res[0].value;
                    targetY.y2 = res[1].value;
                    targetY.y3 = res[2].value;
                    targetY.y4 = res[3].value;
                    targetY.shape = res[4].value;
                }
            });
        }

        // 시각화 수치 부드럽게 전이
        currentY.y1 += (targetY.y1 - currentY.y1) * 0.1;
        currentY.y2 += (targetY.y2 - currentY.y2) * 0.1;
        currentY.y3 += (targetY.y3 - currentY.y3) * 0.1;
        currentY.y4 += (targetY.y4 - currentY.y4) * 0.1;
        currentY.shape += (targetY.shape - currentY.shape) * 0.1;

        // 형태 변경 (임계값 도달 시)
        const roundedShape = Math.round(currentY.shape);
        if (roundedShape !== Math.round(currentY.shape - (targetY.shape - currentY.shape) * 0.1)) {
            createShape(Math.max(0, Math.min(5, roundedShape)));
        }

        // 시각화는 항상 실시간 소리(currentX.loudness)에 반응하게 함
        updateVisuals(currentX.loudness);
    }
    renderer.render(scene, camera);
}

function analyzeAudio() {
    const data = new Uint8Array(analyser.frequencyBinCount);
    const time = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(data);
    analyser.getByteTimeDomainData(time);

    let s = 0; for(let v of time) { let n=(v-128)/128; s+=n*n; }
    currentX.loudness = Math.sqrt(s/time.length) * 10;
    
    let te=0, we=0; for(let i=0; i<data.length; i++) { we+=i*data[i]; te+=data[i]; }
    currentX.pitch = currentX.brightness = te>0 ? (we/te)/40 : 0;
    
    let z=0; for(let i=1; i<time.length; i++) if(time[i]>128 && time[i-1]<=128) z++;
    currentX.roughness = z/30;

    if (state === 'RECORDING') {
        recordedX.loudness += currentX.loudness;
        recordedX.pitch += currentX.pitch;
        recordedX.brightness += currentX.brightness;
        recordedX.roughness += currentX.roughness;
        recordedX.count++;
    }
}

function updateVisuals(loudness) {
    if (!currentMesh) return;

    const pos = currentMesh.geometry.attributes.position;
    const t = Date.now() * 0.001;
    const shapeType = Math.round(currentY.shape);

    for (let i = 0; i < pos.count; i++) {
        const i3 = i * 3;
        const ox = originalVertices[i3];
        const oy = originalVertices[i3 + 1];
        const oz = originalVertices[i3 + 2];

        tempVec.set(ox, oy, oz);

        // 형태별 고유한 변형 로직
        let displacement = 0;

        switch(shapeType) {
            case SHAPES.SPHERE:
                // 구: 방사형 파동
                tempVec.normalize();
                const sphereWave = Math.sin(tempVec.x * 3 + tempVec.y * 2 + t * (1 + currentY.y4 * 3)) * currentY.y2;
                const sphereRough = (Math.random() - 0.5) * currentY.y3 * 0.1;
                displacement = 1 + sphereWave * 0.3 + sphereRough + loudness * 0.3;
                tempVec.multiplyScalar(displacement);
                break;

            case SHAPES.CUBE:
                // 정육면체: 면 단위 펄스
                const cubeWave = Math.sin((Math.abs(ox) + Math.abs(oy) + Math.abs(oz)) * 2 + t * 2) * currentY.y2;
                const faceNoise = (Math.sin(ox * 10 + t) * Math.cos(oy * 10 + t)) * currentY.y3 * 0.1;
                displacement = 1 + cubeWave * 0.2 + faceNoise + loudness * 0.25;
                tempVec.multiplyScalar(displacement);
                break;

            case SHAPES.TORUS:
                // 토러스: 회전 나선 파동
                const angle = Math.atan2(oz, ox);
                const torusWave = Math.sin(angle * (3 + currentY.y4 * 10) + t * 2) * currentY.y2;
                const radialPulse = Math.sin(oy * 5 + t * 3) * currentY.y3 * 0.15;
                const scale = 1 + (torusWave * 0.2 + radialPulse + loudness * 0.2);
                tempVec.x = ox * scale;
                tempVec.z = oz * scale;
                tempVec.y = oy * (1 + torusWave * 0.3 + loudness * 0.15);
                break;

            case SHAPES.CONE:
                // 원뿔: 높이에 따른 차등 변형
                const heightFactor = (oy + 1) / 2; // 0~1 정규화
                const coneWave = Math.sin(Math.atan2(oz, ox) * (4 + currentY.y4 * 8) + t) * currentY.y2;
                const heightWave = Math.sin(oy * 3 + t * 2) * currentY.y3 * 0.2;
                const coneScale = 1 + (coneWave * 0.25 + heightWave) * heightFactor + loudness * 0.3;
                tempVec.x = ox * coneScale;
                tempVec.z = oz * coneScale;
                tempVec.y = oy * (1 + Math.sin(t) * currentY.y2 * 0.1 + loudness * 0.2);
                break;

            case SHAPES.CYLINDER:
                // 원기둥: 세로 파동 + 회전 왜곡
                const cylAngle = Math.atan2(oz, ox);
                const cylWave = Math.sin(cylAngle * (5 + currentY.y4 * 10) + oy * 2 + t * 2) * currentY.y2;
                const verticalWave = Math.sin(oy * 4 + t * 3) * currentY.y3 * 0.15;
                const cylScale = 1 + cylWave * 0.25 + verticalWave + loudness * 0.25;
                tempVec.x = ox * cylScale;
                tempVec.z = oz * cylScale;
                break;

            case SHAPES.OCTAHEDRON:
                // 팔면체: 꼭지점 기반 복잡한 변형
                tempVec.normalize();
                const octWave1 = Math.sin(tempVec.x * 5 + t) * Math.cos(tempVec.y * 5 + t);
                const octWave2 = Math.sin(tempVec.z * 5 + t * 1.5) * currentY.y4;
                const octRough = (Math.sin(t * 15) * 0.05) * currentY.y3;
                displacement = 1.2 + octWave1 * currentY.y2 * 0.4 + octWave2 * 0.3 + octRough + loudness * 0.35;
                tempVec.set(ox, oy, oz).normalize().multiplyScalar(displacement);
                break;
        }

        pos.setXYZ(i, tempVec.x, tempVec.y, tempVec.z);
    }

    // 회전 속도도 형태에 따라 다르게
    const rotationSpeed = 0.005 + (currentY.y1 * 0.05);
    currentMesh.rotation.y += rotationSpeed;

    if (shapeType === SHAPES.TORUS || shapeType === SHAPES.CYLINDER) {
        currentMesh.rotation.x += rotationSpeed * 0.3;
    }

    pos.needsUpdate = true;
}

function confirmTraining() {
    const labels = {
        y1: parseFloat(document.getElementById('y1').value),
        y2: parseFloat(document.getElementById('y2').value),
        y3: parseFloat(document.getElementById('y3').value),
        y4: parseFloat(document.getElementById('y4').value),
        shape: parseFloat(document.getElementById('shape-selector').value)
    };

    brain.addData({
        loudness: recordedX.loudness,
        pitch: recordedX.pitch,
        brightness: recordedX.brightness,
        roughness: recordedX.roughness
    }, labels);

    // 학습 데이터 자동 저장
    saveTrainingData();
    updateDataCount();

    brain.normalizeData();

    updateStatus('🧠 AI 학습 중...', 'status-recording');

    brain.train({ epochs: 20 }, () => {
        alert("학습 완료! 데이터가 자동 저장되었습니다.");
        state = 'IDLE';

        // 재생 중지
        if(audioTag) audioTag.pause();

        document.getElementById('labeling-zone').style.display = "none";
        document.getElementById('btn-main').innerText = "녹음 시작";
        document.getElementById('btn-play').style.display = "none";

        updateStatus('대기 중 (녹음 가능)', 'status-idle');
    });
}

// 학습 데이터를 localStorage에 저장
function saveTrainingData() {
    if (!brain || !brain.data) return;

    const trainingData = {
        data: brain.data.data.raw,
        timestamp: Date.now(),
        version: 1
    };

    localStorage.setItem('soundTo3D_trainingData', JSON.stringify(trainingData));
    console.log('학습 데이터 저장됨:', brain.data.data.raw.length, '개');
}

// localStorage에서 학습 데이터 불러오기
function loadTrainingData() {
    const saved = localStorage.getItem('soundTo3D_trainingData');
    if (!saved) {
        console.log('저장된 학습 데이터 없음');
        return;
    }

    try {
        const trainingData = JSON.parse(saved);
        console.log('학습 데이터 불러오는 중:', trainingData.data.length, '개');

        // 기존 데이터에 추가
        trainingData.data.forEach(item => {
            brain.addData(item.xs, item.ys);
        });

        // 데이터가 충분하면 정규화 및 학습
        if (brain.data.data.raw.length >= 5) {
            brain.normalizeData();
            brain.train({ epochs: 20 }, () => {
                console.log('기존 학습 데이터로 재학습 완료!');
                alert(`이전 학습 데이터 ${trainingData.data.length}개를 불러왔습니다!\n계속 학습하면 더 똑똑해집니다.`);
            });
        }
    } catch (e) {
        console.error('데이터 불러오기 실패:', e);
    }
}

// 학습 데이터 개수 업데이트
function updateDataCount() {
    const count = brain && brain.data ? brain.data.data.raw.length : 0;
    const countEl = document.getElementById('data-count');
    if (countEl) {
        countEl.innerText = count;
    }
}

// 모든 학습 데이터 삭제
function clearAllData() {
    if (confirm('모든 학습 데이터를 삭제하시겠습니까?\n이 작업은 되돌릴 수 없습니다.')) {
        localStorage.removeItem('soundTo3D_trainingData');

        // brain 재생성
        brain = ml5.neuralNetwork({
            inputs: ['loudness', 'pitch', 'brightness', 'roughness'],
            outputs: ['y1', 'y2', 'y3', 'y4', 'shape'],
            task: 'regression', debug: true
        });

        updateDataCount();
        alert('모든 학습 데이터가 삭제되었습니다.');
    }
}

// CSV로 데이터 내보내기 (기존 함수 개선)
function exportCSV() {
    if (!brain || !brain.data || brain.data.data.raw.length === 0) {
        alert('내보낼 데이터가 없습니다.');
        return;
    }

    let csv = 'loudness,pitch,brightness,roughness,y1,y2,y3,y4,shape\n';
    brain.data.data.raw.forEach(item => {
        csv += `${item.xs.loudness},${item.xs.pitch},${item.xs.brightness},${item.xs.roughness},`;
        csv += `${item.ys.y1},${item.ys.y2},${item.ys.y3},${item.ys.y4},${item.ys.shape}\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `soundTo3D_data_${Date.now()}.csv`;
    a.click();
}

// 형태 선택기 변경 시 실시간 미리보기
function onShapeChange() {
    if (state === 'REVIEWING') {
        const shapeValue = parseInt(document.getElementById('shape-selector').value);
        createShape(shapeValue);
        document.getElementById('shape-name').innerText = SHAPE_NAMES[shapeValue];
    }
}