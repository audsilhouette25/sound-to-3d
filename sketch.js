let audioCtx, analyser, microphone, mediaRecorder;
let audioChunks = [];
let audioBlob, audioUrl, audioTag, sourceNode;
let brain;
let scene, camera, renderer, currentMesh, originalVertices;
let microphoneStream; // 마이크 스트림 저장

// 상태 관리
let state = 'IDLE';
let recordedX = { loudness: 0, pitch: 0, brightness: 0, roughness: 0 };
let currentX = { loudness: 0, pitch: 0, brightness: 0, roughness: 0 };
let targetY = { y1: 0.5, y2: 0.5, y3: 0.5, y4: 0.5, shape: 0 };
let currentY = { y1: 0.5, y2: 0.5, y3: 0.5, y4: 0.5, shape: 0 };

// 우리만의 데이터 저장소 (ml5.js 우회)
let customTrainingData = [];
let isModelTrained = false; // 모델이 학습되었는지 추적

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

const SHAPE_NAMES = ['Sphere', 'Cube', 'Torus', 'Cone', 'Cylinder', 'Octahedron'];

// 소리 특성에 따라 자동으로 형태 분류
function autoClassifyShape(loudness, pitch, brightness, roughness) {
    // 정규화된 값들로 분류 (0-1 범위 가정)
    const normalizedLoudness = Math.min(1, loudness / 5); // loudness는 보통 0-5 범위
    const normalizedPitch = Math.min(1, pitch);
    const normalizedBrightness = Math.min(1, brightness);
    const normalizedRoughness = Math.min(1, roughness);

    console.log('🎵 Audio features:', {
        loudness: loudness.toFixed(3),
        pitch: pitch.toFixed(3),
        brightness: brightness.toFixed(3),
        roughness: roughness.toFixed(3),
        normalized: {
            loudness: normalizedLoudness.toFixed(3),
            pitch: normalizedPitch.toFixed(3),
            brightness: normalizedBrightness.toFixed(3),
            roughness: normalizedRoughness.toFixed(3)
        }
    });

    // 분류 로직:
    // - Sphere (0): 부드럽고 균일한 소리 (낮은 roughness, 중간 pitch)
    // - Cube (1): 각진, 명확한 소리 (높은 brightness, 중간 roughness)
    // - Torus (2): 회전하는 느낌의 소리 (중간-높은 pitch, 변화가 있는)
    // - Cone (3): 뾰족하고 날카로운 소리 (높은 pitch, 높은 brightness)
    // - Cylinder (4): 일정하고 연속적인 소리 (낮은 roughness, 일정한 pitch)
    // - Octahedron (5): 복잡하고 불규칙한 소리 (높은 roughness, 변화 많음)

    const scores = [0, 0, 0, 0, 0, 0];

    // Sphere: 부드럽고 중간 범위
    scores[0] = (1 - normalizedRoughness) * 0.4 +
                (normalizedPitch > 0.3 && normalizedPitch < 0.7 ? 0.6 : 0);

    // Cube: 밝고 적당히 거친
    scores[1] = normalizedBrightness * 0.5 +
                (normalizedRoughness > 0.3 && normalizedRoughness < 0.7 ? 0.5 : 0);

    // Torus: 중간-높은 pitch, 회전감
    scores[2] = (normalizedPitch > 0.5 ? 0.6 : 0.2) +
                normalizedLoudness * 0.4;

    // Cone: 높고 날카로운
    scores[3] = (normalizedPitch > 0.6 ? 0.5 : 0) +
                (normalizedBrightness > 0.6 ? 0.5 : 0);

    // Cylinder: 일정하고 연속적
    scores[4] = (1 - normalizedRoughness) * 0.5 +
                (normalizedLoudness > 0.3 ? 0.5 : 0);

    // Octahedron: 복잡하고 거친
    scores[5] = normalizedRoughness * 0.6 +
                (normalizedBrightness > 0.5 ? 0.4 : 0.2);

    console.log('📊 Shape scores:', scores.map((s, i) => `${SHAPE_NAMES[i]}: ${s.toFixed(3)}`).join(', '));

    // 가장 높은 점수의 형태 반환
    let maxScore = -1;
    let bestShape = 0;
    for (let i = 0; i < 6; i++) {
        if (scores[i] > maxScore) {
            maxScore = scores[i];
            bestShape = i;
        }
    }

    console.log(`✅ Selected: ${SHAPE_NAMES[bestShape]} (score: ${maxScore.toFixed(3)})`);
    return bestShape;
}

window.onload = () => { initThree(); };

function initThree() {
    const container = document.getElementById('three-container');
    scene = new THREE.Scene();

    // 컨테이너 크기 기준으로 카메라 설정
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    camera = new THREE.PerspectiveCamera(75, containerWidth / containerHeight, 0.1, 1000);

    updateCameraPosition();

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(containerWidth, containerHeight);
    container.appendChild(renderer.domElement);

    // 기본 형태로 구체 생성
    createShape(SHAPES.SPHERE);

    scene.add(new THREE.DirectionalLight(0xffffff, 1), new THREE.AmbientLight(0x222222));

    // 창 크기 변경 시 컨테이너 크기에 맞춰 업데이트
    window.addEventListener('resize', () => {
        const containerWidth = container.clientWidth;
        const containerHeight = container.clientHeight;
        camera.aspect = containerWidth / containerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(containerWidth, containerHeight);
        updateCameraPosition();
    });

    animate();
}

// 카메라 위치 업데이트 함수
function updateCameraPosition() {
    // 카메라를 화면 중앙에 배치
    camera.position.set(0, 0, 3.5);
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
    // 오브젝트는 중앙에 배치
    scene.add(currentMesh);
    originalVertices = currentMesh.geometry.attributes.position.array.slice();
}

async function initEngine() {
    updateStatus('statusInit', 'status-idle');

    audioCtx = new (window.AudioContext || window.webkitAudioContext)();

    // 마이크는 녹음 시작할 때만 켜도록 변경 (사용자 요청)
    // microphoneStream은 null로 시작

    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 512;

    brain = ml5.neuralNetwork({
        inputs: 4,
        outputs: 5,
        task: 'regression',
        debug: false
    });

    console.log('Brain created, waiting for initialization...');

    // ml5.js neuralNetwork는 생성 직후에는 brain.data가 undefined일 수 있음
    // brain.data.training이 실제로 존재할 때까지 대기 (최대 10초)
    let retryCount = 0;
    const maxRetries = 200; // 50ms * 200 = 10초

    const waitForBrainReady = () => {
        retryCount++;

        if (brain.data && Array.isArray(brain.data.training)) {
            console.log('✓ Brain initialized successfully');
            console.log('brain.data.training length:', brain.data.training.length);

            // 저장된 학습 데이터 불러오기
            loadTrainingData();

            // 데이터 카운트 업데이트
            updateDataCount();

            const finalCount = brain.data.training ? brain.data.training.length : 0;
            console.log(`Initialization complete. Loaded ${finalCount} training samples.`);
        } else if (retryCount >= maxRetries) {
            console.error('CRITICAL: Brain initialization timeout after 10 seconds');
            alert('Failed to initialize neural network. Please refresh the page.');
        } else {
            // 아직 초기화 안됨, 계속 대기
            setTimeout(waitForBrainReady, 50);
        }
    };

    // 초기화 대기 시작
    setTimeout(waitForBrainReady, 100);

    document.getElementById('btn-engine').style.display = 'none';
    document.getElementById('btn-main').style.display = 'block';
    document.getElementById('save-load-zone').style.display = 'block';

    updateStatus('statusActive', 'status-idle');
}

// 상태 업데이트 함수
function updateStatus(messageKey, className) {
    const statusEl = document.getElementById('status');
    const t = translations[currentLang];

    // messageKey가 translations에 있으면 번역된 텍스트 사용
    const message = t[messageKey] || messageKey;

    statusEl.innerText = message;
    statusEl.className = 'status-badge ' + className;
}

async function handleRecord() {
    if (state === 'IDLE' || state === 'REVIEWING') await startRecording();
    else if (state === 'RECORDING') stopRecording();
}

async function startRecording() {
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

    // 마이크가 없으면 새로 요청
    if (!microphoneStream) {
        console.log('마이크 다시 켜기...');
        microphoneStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        microphone = audioCtx.createMediaStreamSource(microphoneStream);
        microphone.connect(analyser);
    }

    // 새 MediaRecorder 생성
    mediaRecorder = new MediaRecorder(microphoneStream);
    mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
    mediaRecorder.onstop = saveRecording;

    mediaRecorder.start();
    const t = translations[currentLang];
    document.getElementById('btn-main').innerText = t.btnStop;
    document.getElementById('labeling-zone').style.display = "none";
    document.getElementById('btn-play').style.display = "none";

    updateStatus('statusRecording', 'status-recording');
}

function stopRecording() {
    mediaRecorder.stop();
    state = 'REVIEWING';

    // 녹음 종료 시 마이크 중단 (사용자 요청)
    // 단, analyser는 연결 유지하여 재생 시 분석 가능하게 함
    if (microphoneStream) {
        microphoneStream.getTracks().forEach(track => track.stop());
        microphoneStream = null;
        console.log('마이크 꺼짐');
    }
    if (microphone) {
        microphone.disconnect();
        microphone = null;
    }

    const t = translations[currentLang];
    document.getElementById('btn-main').innerText = t.btnReRecord;
    document.getElementById('labeling-zone').style.display = "block";
    document.getElementById('btn-confirm').style.display = "block";
    document.getElementById('btn-play').style.display = "inline-block";
    document.getElementById('btn-play').innerText = t.btnPlay;

    updateStatus('statusReview', 'status-review');

    // 자동 분류가 켜져 있으면 즉시 실행
    if (document.getElementById('auto-shape').checked) {
        setTimeout(() => {
            if (recordedX && recordedX.count > 0) {
                // 학습된 모델이 있으면 예측, 없으면 규칙 기반
                if (isModelTrained && brain) {
                    brain.predict([recordedX.loudness, recordedX.pitch, recordedX.brightness, recordedX.roughness], (err, res) => {
                        if (!err && res && res.length >= 5) {
                            const predictedShape = Math.round(Math.max(0, Math.min(5, res[4].value)));
                            document.getElementById('shape-selector').value = predictedShape;
                            document.getElementById('shape-name').innerText = SHAPE_NAMES[predictedShape];
                            createShape(predictedShape);
                            console.log(`🤖 AI-predicted shape: ${SHAPE_NAMES[predictedShape]}`);
                        }
                    });
                } else {
                    // 학습 데이터 없으면 규칙 기반 분류
                    const autoShape = autoClassifyShape(
                        recordedX.loudness,
                        recordedX.pitch,
                        recordedX.brightness,
                        recordedX.roughness
                    );
                    document.getElementById('shape-selector').value = autoShape;
                    document.getElementById('shape-name').innerText = SHAPE_NAMES[autoShape];
                    createShape(autoShape);
                    console.log(`📏 Rule-based shape: ${SHAPE_NAMES[autoShape]}`);
                }
            }
        }, 100); // 약간의 딜레이를 주어 recordedX가 완전히 업데이트되도록 함
    }
}

function saveRecording() {
    audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    audioUrl = URL.createObjectURL(audioBlob);

    // 오디오 태그 준비만 하고 자동 재생하지 않음
    audioTag = new Audio(audioUrl);
    audioTag.loop = true;

    // 녹음된 평균값 저장 (학습용)
    if (recordedX.count > 0) {
        recordedX.loudness /= recordedX.count;
        recordedX.pitch /= recordedX.count;
        recordedX.brightness /= recordedX.count;
        recordedX.roughness /= recordedX.count;
    }

    console.log('saveRecording - recordedX after processing:', recordedX);
}

// 녹음 재생/일시정지 토글
function togglePlayback() {
    if (!audioTag) return;

    const t = translations[currentLang];

    if (audioTag.paused) {
        // 재생 시작
        if (sourceNode) sourceNode.disconnect();
        sourceNode = audioCtx.createMediaElementSource(audioTag);
        sourceNode.connect(analyser);
        analyser.connect(audioCtx.destination);

        audioTag.play();
        document.getElementById('btn-play').innerText = t.btnPause;
    } else {
        // 일시정지
        audioTag.pause();
        document.getElementById('btn-play').innerText = t.btnPlay;
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
        } else if (brain && customTrainingData.length >= 5) {
            // 평상시에는 AI가 예측 (customTrainingData 기준으로 체크)
            brain.predict([currentX.loudness, currentX.pitch, currentX.brightness, currentX.roughness], (err, res) => {
                if(!err && res && res.length >= 5) {
                    targetY.y1 = res[0].value;
                    targetY.y2 = res[1].value;
                    targetY.y3 = res[2].value;
                    targetY.y4 = res[3].value;
                    targetY.shape = res[4].value;
                }
            });
        }

        // 시각화 수치 부드럽게 전이 (리뷰 모드에서는 더 빠르게)
        const lerpSpeed = (state === 'REVIEWING') ? 0.3 : 0.1;
        currentY.y1 += (targetY.y1 - currentY.y1) * lerpSpeed;
        currentY.y2 += (targetY.y2 - currentY.y2) * lerpSpeed;
        currentY.y3 += (targetY.y3 - currentY.y3) * lerpSpeed;
        currentY.y4 += (targetY.y4 - currentY.y4) * lerpSpeed;
        currentY.shape += (targetY.shape - currentY.shape) * lerpSpeed;

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
                const torusWave = Math.sin(angle * (3 + currentY.y4 * 3) + t * 2) * currentY.y2;
                const radialPulse = Math.sin(oy * 5 + t * 3) * currentY.y3 * 0.15;
                const scale = 1 + (torusWave * 0.2 + radialPulse + loudness * 0.2);
                tempVec.x = ox * scale;
                tempVec.z = oz * scale;
                tempVec.y = oy * (1 + torusWave * 0.3 + loudness * 0.15);
                break;

            case SHAPES.CONE:
                // 원뿔: 높이에 따른 차등 변형
                const heightFactor = (oy + 1) / 2; // 0~1 정규화
                const coneWave = Math.sin(Math.atan2(oz, ox) * (4 + currentY.y4 * 3) + t) * currentY.y2;
                const heightWave = Math.sin(oy * 3 + t * 2) * currentY.y3 * 0.2;
                const coneScale = 1 + (coneWave * 0.25 + heightWave) * heightFactor + loudness * 0.3;
                tempVec.x = ox * coneScale;
                tempVec.z = oz * coneScale;
                tempVec.y = oy * (1 + Math.sin(t) * currentY.y2 * 0.1 + loudness * 0.2);
                break;

            case SHAPES.CYLINDER:
                // 원기둥: 세로 파동 + 회전 왜곡
                const cylAngle = Math.atan2(oz, ox);
                const cylWave = Math.sin(cylAngle * (5 + currentY.y4 * 3) + oy * 2 + t * 2) * currentY.y2;
                const verticalWave = Math.sin(oy * 4 + t * 3) * currentY.y3 * 0.15;
                const cylScale = 1 + cylWave * 0.25 + verticalWave + loudness * 0.25;
                tempVec.x = ox * cylScale;
                tempVec.z = oz * cylScale;
                break;

            case SHAPES.OCTAHEDRON:
                // 팔면체: 꼭지점 기반 복잡한 변형
                tempVec.normalize();
                const octWave1 = Math.sin(tempVec.x * 5 + t) * Math.cos(tempVec.y * 5 + t);
                const octWave2 = Math.sin(tempVec.z * 5 + t * 1.5) * currentY.y4 * 0.3;
                const octRough = (Math.sin(t * 15) * 0.05) * currentY.y3;
                displacement = 1.2 + octWave1 * currentY.y2 * 0.4 + octWave2 + octRough + loudness * 0.35;
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

function confirmTraining(useAutoShape = false) {
    console.log('=== Confirming training data ===');

    // recordedX 검증
    if (!recordedX || recordedX.count === 0 ||
        isNaN(recordedX.loudness) ||
        isNaN(recordedX.pitch) ||
        isNaN(recordedX.brightness) ||
        isNaN(recordedX.roughness)) {
        alert('Recording data is invalid. Please record again.');
        console.error('Invalid recordedX:', recordedX);
        return;
    }

    // brain 상태 확인
    if (!brain || !brain.data || !Array.isArray(brain.data.training)) {
        console.error('CRITICAL: Brain not initialized properly');
        alert('Neural network not ready. Please refresh the page and try again.');
        return;
    }

    // 형태 값 결정: Auto 모드에서는 학습된 모델 또는 규칙 기반, 수동 모드에서는 슬라이더 값
    let shapeValue;
    if (useAutoShape) {
        // 이미 UI에 표시된 값 사용 (stopRecording에서 이미 분류됨)
        shapeValue = parseFloat(document.getElementById('shape-selector').value);
        console.log(`Using auto-classified shape: ${SHAPE_NAMES[shapeValue]} (${shapeValue})`);
    } else {
        shapeValue = parseFloat(document.getElementById('shape-selector').value);
    }

    const labels = {
        y1: parseFloat(document.getElementById('y1').value),
        y2: parseFloat(document.getElementById('y2').value),
        y3: parseFloat(document.getElementById('y3').value),
        y4: parseFloat(document.getElementById('y4').value),
        shape: shapeValue
    };

    const inputArray = [recordedX.loudness, recordedX.pitch, recordedX.brightness, recordedX.roughness];
    const outputArray = [labels.y1, labels.y2, labels.y3, labels.y4, labels.shape];

    // 데이터 검증
    if (inputArray.length !== 4 || outputArray.length !== 5) {
        console.error('ERROR: Wrong array dimensions!');
        alert('Data dimension error. Please refresh and try again.');
        return;
    }

    if (inputArray.some(v => typeof v !== 'number' || isNaN(v)) ||
        outputArray.some(v => typeof v !== 'number' || isNaN(v))) {
        console.error('ERROR: Data contains non-numbers or NaN!');
        alert('Data validation error. Please refresh and try again.');
        return;
    }

    // customTrainingData에 저장
    const dataItem = {
        xs: [...inputArray],
        ys: [...outputArray]
    };

    customTrainingData.push(dataItem);
    console.log(`✓ Added to customTrainingData (${customTrainingData.length} total)`);

    // 학습 데이터 자동 저장
    saveTrainingData();
    updateDataCount();

    const actualCount = customTrainingData.length;
    console.log(`Successfully saved! Total samples: ${actualCount}`);

    // [개선됨] 기존 brain에 새 데이터만 추가하고 증분 학습
    console.log('Adding new data to existing brain...');

    // brain에 새 데이터 추가
    brain.addData(inputArray, outputArray);

    // 정규화 및 학습
    if (customTrainingData.length >= 2) {
        brain.normalizeData();
    }

    updateStatus('statusTraining', 'status-recording');

    // 증분 학습 (epochs 수를 줄여서 빠르게)
    const epochs = customTrainingData.length < 10 ? 20 : 10;
    brain.train({ epochs: epochs }, () => {
        console.log('Training complete!');
        isModelTrained = true;

        // 학습된 모델 저장
        saveModel();

        alert(`✓ Training Complete!\n\nSaved ${actualCount} sample(s) to storage.\nModel is ready for predictions.`);
        state = 'IDLE';

        if(audioTag) audioTag.pause();

        const t = translations[currentLang];
        document.getElementById('labeling-zone').style.display = "none";
        document.getElementById('btn-main').innerText = t.btnRecord;
        document.getElementById('btn-play').style.display = "none";

        updateStatus('statusActive', 'status-idle');
    });
}

// 학습된 모델 저장 (ml5.js model serialization)
function saveModel() {
    if (!brain || !isModelTrained) {
        console.log('No trained model to save');
        return;
    }

    try {
        brain.save('soundTo3D_model', () => {
            console.log('✓ Model saved to browser storage');
        });
    } catch (e) {
        console.error('Model save failed:', e);
    }
}

// 저장된 모델 불러오기
function loadModel() {
    // ml5.js는 파일 시스템에서 모델을 로드하므로,
    // 브라우저 환경에서는 IndexedDB 같은 방법이 필요
    // 대신 우리는 customTrainingData가 있으면 재학습하는 방식 사용
    console.log('Model loading from localStorage not directly supported by ml5.js in browser');
    console.log('Will retrain from customTrainingData if needed');
}

// 학습 데이터를 localStorage에 저장 (customTrainingData 사용)
function saveTrainingData() {
    try {
        const saveObj = {
            version: 3, // 새 버전 (customTrainingData 사용)
            count: customTrainingData.length,
            data: customTrainingData,
            timestamp: Date.now()
        };

        localStorage.setItem('soundTo3D_trainingData', JSON.stringify(saveObj));
        console.log(`✓ Saved ${customTrainingData.length} samples to localStorage`);
    } catch (e) {
        console.error('Save failed:', e);
    }
}

// localStorage에서 학습 데이터 불러오기 및 자동 재학습
function loadTrainingData() {
    const saved = localStorage.getItem('soundTo3D_trainingData');
    if (!saved) {
        console.log('No saved training data');
        return;
    }

    try {
        const saveObj = JSON.parse(saved);

        if (!saveObj || !Array.isArray(saveObj.data)) {
            console.warn('Invalid data, clearing');
            localStorage.removeItem('soundTo3D_trainingData');
            return;
        }

        // customTrainingData에 로드
        customTrainingData = [];

        for (let i = 0; i < saveObj.data.length; i++) {
            const item = saveObj.data[i];

            if (!item || !Array.isArray(item.xs) || !Array.isArray(item.ys)) continue;
            if (item.xs.length !== 4 || item.ys.length !== 5) continue;

            // 유효성 체크
            let valid = true;
            for (let j = 0; j < 4; j++) {
                if (typeof item.xs[j] !== 'number' || isNaN(item.xs[j])) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                for (let j = 0; j < 5; j++) {
                    if (typeof item.ys[j] !== 'number' || isNaN(item.ys[j])) {
                        valid = false;
                        break;
                    }
                }
            }

            if (valid) {
                customTrainingData.push(item);
            }
        }

        console.log(`✓ Loaded ${customTrainingData.length} samples into customTrainingData`);

        // [개선됨] 데이터가 있으면 자동으로 brain 재학습
        if (customTrainingData.length >= 2) {
            console.log('Auto-retraining brain with loaded data...');

            // brain에 모든 데이터 추가
            for (let i = 0; i < customTrainingData.length; i++) {
                brain.addData(customTrainingData[i].xs, customTrainingData[i].ys);
            }

            // 정규화
            brain.normalizeData();

            // 백그라운드 학습 (alert 없이)
            const epochs = Math.max(10, Math.min(50, customTrainingData.length * 2));
            brain.train({ epochs: epochs }, () => {
                isModelTrained = true;
                console.log(`✓ Auto-training complete with ${customTrainingData.length} samples`);
            });
        }
    } catch (e) {
        console.error('Load failed:', e);
        localStorage.removeItem('soundTo3D_trainingData');
    }
}

// 학습 데이터 개수 업데이트
function updateDataCount() {
    const countEl = document.getElementById('data-count');
    if (countEl) {
        countEl.innerText = customTrainingData.length;
    }
}

// 모든 학습 데이터 삭제
function clearAllData() {
    if (!confirm('Delete all training data?\nThis action cannot be undone.')) {
        return;
    }

    console.log('=== Clearing all training data ===');

    // localStorage 삭제
    localStorage.removeItem('soundTo3D_trainingData');
    console.log('✓ localStorage cleared');

    // customTrainingData 초기화
    const oldLength = customTrainingData.length;
    customTrainingData = [];
    console.log(`✓ Custom training data reset (${oldLength} → 0)`);

    updateDataCount();
    alert('✓ All training data deleted successfully.');
}

// 긴급 복구: 완전 초기화 (브라우저 콘솔에서 사용)
function emergencyReset() {
    console.log('=== EMERGENCY RESET ===');

    // localStorage 삭제
    localStorage.clear();
    console.log('✓ All localStorage cleared');

    // customTrainingData 초기화
    customTrainingData = [];
    console.log('✓ customTrainingData cleared');

    // brain 재생성
    if (typeof ml5 !== 'undefined') {
        brain = ml5.neuralNetwork({
            inputs: 4,
            outputs: 5,
            task: 'regression',
            debug: false
        });
        console.log('✓ Brain recreated');

        const waitAndUpdate = () => {
            if (brain.data && Array.isArray(brain.data.training)) {
                updateDataCount();
                console.log('✓ Emergency reset complete');
                alert('Emergency reset complete. Everything has been reset.');
            } else {
                setTimeout(waitAndUpdate, 100);
            }
        };
        setTimeout(waitAndUpdate, 100);
    } else {
        updateDataCount();
        alert('Emergency reset complete. Please reload the page.');
    }
}

// CSV로 데이터 내보내기 (customTrainingData 사용)
function exportCSV() {
    if (customTrainingData.length === 0) {
        alert('No data to export.');
        return;
    }

    let csv = 'loudness,pitch,brightness,roughness,y1,y2,y3,y4,shape\n';

    for (let i = 0; i < customTrainingData.length; i++) {
        const item = customTrainingData[i];
        if (!item || !item.xs || !item.ys) continue;

        const xs = item.xs;
        const ys = item.ys;

        if (xs.length === 4 && ys.length === 5) {
            csv += `${xs[0]},${xs[1]},${xs[2]},${xs[3]},`;
            csv += `${ys[0]},${ys[1]},${ys[2]},${ys[3]},${ys[4]}\n`;
        }
    }

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `soundTo3D_data_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
}

// 형태 선택기 변경 시 실시간 미리보기
function onShapeChange() {
    if (state === 'REVIEWING') {
        const shapeValue = parseInt(document.getElementById('shape-selector').value);
        createShape(shapeValue);
        document.getElementById('shape-name').innerText = SHAPE_NAMES[shapeValue];
    }
}

// 자동 형태 분류 토글
function onAutoShapeToggle() {
    const isAutoOn = document.getElementById('auto-shape').checked;
    const shapeSelector = document.getElementById('shape-selector');

    if (isAutoOn) {
        // 자동 모드: 슬라이더 비활성화
        shapeSelector.disabled = true;
        shapeSelector.style.opacity = '0.5';

        // 현재 녹음된 소리로 자동 분류
        if (state === 'REVIEWING' && recordedX && recordedX.count > 0) {
            // 학습된 모델이 있으면 예측, 없으면 규칙 기반
            if (isModelTrained && brain) {
                brain.predict([recordedX.loudness, recordedX.pitch, recordedX.brightness, recordedX.roughness], (err, res) => {
                    if (!err && res && res.length >= 5) {
                        const predictedShape = Math.round(Math.max(0, Math.min(5, res[4].value)));
                        shapeSelector.value = predictedShape;
                        document.getElementById('shape-name').innerText = SHAPE_NAMES[predictedShape];
                        createShape(predictedShape);
                        console.log(`🤖 AI-predicted: ${SHAPE_NAMES[predictedShape]}`);
                    }
                });
            } else {
                const autoShape = autoClassifyShape(
                    recordedX.loudness,
                    recordedX.pitch,
                    recordedX.brightness,
                    recordedX.roughness
                );
                shapeSelector.value = autoShape;
                document.getElementById('shape-name').innerText = SHAPE_NAMES[autoShape];
                createShape(autoShape);
                console.log(`📏 Rule-based: ${SHAPE_NAMES[autoShape]}`);
            }
        }
    } else {
        // 수동 모드: 슬라이더 활성화
        shapeSelector.disabled = false;
        shapeSelector.style.opacity = '1';
    }
}

// confirmTraining 호출 시 자동 분류 옵션 확인
window.confirmTrainingWrapper = function() {
    const useAutoShape = document.getElementById('auto-shape').checked;
    confirmTraining(useAutoShape);
}

// 언어 전환
let currentLang = 'en';

const translations = {
    en: {
        langBtn: '한국어',
        title: 'IML Experiment Panel',
        btnEngine: 'Initialize Audio Engine',
        btnRecord: 'Start Recording',
        btnStop: 'Stop Recording',
        btnReRecord: 'Re-record',
        btnPlay: '▶ Play Recording',
        btnPause: '⏸ Pause',
        labelInstruction: 'Define visual characteristics of the recorded sound',
        y1Left: 'Smooth', y1Right: 'Angular',
        y2Left: 'Flat', y2Right: 'Sharp',
        y3Left: 'Smooth', y3Right: 'Rough',
        y4Left: 'Simple', y4Right: 'Complex',
        btnConfirm: 'Confirm Training Data',
        dataLabel: 'Training Data:',
        samplesLabel: 'samples',
        btnExport: 'Export Data (CSV)',
        btnClear: 'Clear All Training Data',
        statusReady: 'Ready - Click to Initialize Audio Engine',
        statusInit: 'Initializing Audio Engine...',
        statusActive: 'Ready - Microphone Active',
        statusRecording: 'Recording...',
        statusReview: 'Review - Awaiting Labels',
        statusTraining: 'Training Neural Network...'
    },
    ko: {
        langBtn: 'English',
        title: 'IML 실험 패널',
        btnEngine: '오디오 엔진 가동',
        btnRecord: '녹음 시작',
        btnStop: '녹음 중단 (Stop)',
        btnReRecord: '다시 녹음하기',
        btnPlay: '▶ 녹음 재생',
        btnPause: '⏸ 일시정지',
        labelInstruction: '방금 소리의 시각적 형질을 결정하세요',
        y1Left: '둥근', y1Right: '각진',
        y2Left: '평평', y2Right: '뾰족',
        y3Left: '매끈', y3Right: '거침',
        y4Left: '단순', y4Right: '복잡',
        btnConfirm: '학습 데이터로 확정',
        dataLabel: '학습 데이터:',
        samplesLabel: '개',
        btnExport: '데이터 내보내기 (CSV)',
        btnClear: '모든 학습 데이터 삭제',
        statusReady: '준비됨 - 엔진 가동 클릭',
        statusInit: '엔진 초기화 중...',
        statusActive: '대기 중 (녹음 가능)',
        statusRecording: '녹음 중...',
        statusReview: '리뷰 중 (라벨링 대기)',
        statusTraining: 'AI 학습 중...'
    }
};

function toggleLanguage() {
    currentLang = currentLang === 'en' ? 'ko' : 'en';
    const t = translations[currentLang];

    document.getElementById('lang-toggle').innerText = t.langBtn;
    document.getElementById('title').innerText = t.title;
    document.getElementById('btn-engine').innerText = t.btnEngine;
    document.getElementById('label-instruction').innerText = t.labelInstruction;

    document.querySelector('.y1-left').innerText = t.y1Left;
    document.querySelector('.y1-right').innerText = t.y1Right;
    document.querySelector('.y2-left').innerText = t.y2Left;
    document.querySelector('.y2-right').innerText = t.y2Right;
    document.querySelector('.y3-left').innerText = t.y3Left;
    document.querySelector('.y3-right').innerText = t.y3Right;
    document.querySelector('.y4-left').innerText = t.y4Left;
    document.querySelector('.y4-right').innerText = t.y4Right;

    document.getElementById('btn-confirm').innerText = t.btnConfirm;
    document.getElementById('data-label').innerText = t.dataLabel;
    document.getElementById('samples-label').innerText = t.samplesLabel;
    document.getElementById('btn-export').innerText = t.btnExport;
    document.getElementById('btn-clear').innerText = t.btnClear;

    // 상태에 따라 버튼 텍스트 업데이트
    if (state === 'IDLE') {
        document.getElementById('btn-main').innerText = t.btnRecord;
    } else if (state === 'RECORDING') {
        document.getElementById('btn-main').innerText = t.btnStop;
    } else if (state === 'REVIEWING') {
        document.getElementById('btn-main').innerText = t.btnReRecord;
    }

    // 현재 상태 메시지 업데이트
    updateStatusText();
}

function updateStatusText() {
    const t = translations[currentLang];
    const statusEl = document.getElementById('status');
    const currentClass = statusEl.className;

    if (currentClass.includes('status-idle')) {
        if (brain) {
            statusEl.innerText = t.statusActive;
        } else {
            statusEl.innerText = t.statusReady;
        }
    } else if (currentClass.includes('status-recording')) {
        if (statusEl.innerText.includes('Training') || statusEl.innerText.includes('학습')) {
            statusEl.innerText = t.statusTraining;
        } else {
            statusEl.innerText = t.statusRecording;
        }
    } else if (currentClass.includes('status-review')) {
        statusEl.innerText = t.statusReview;
    }
}