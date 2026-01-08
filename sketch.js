let audioCtx, analyser, microphone, mediaRecorder;
let audioChunks = [];
let audioBlob, audioUrl, audioTag, sourceNode;
let brain;
let scene, camera, renderer, currentMesh, originalVertices;
let microphoneStream; // 마이크 스트림 저장

// 상태 관리
let state = 'IDLE';
let recordedX = { loudness: 0, pitch: 0, brightness: 0, roughness: 0, count: 0 };
let currentX = { loudness: 0, pitch: 0, brightness: 0, roughness: 0 };
let targetY = { y1: 0.5, y2: 0.5, y3: 0.5, y4: 0.5, shape: 0 };
let currentY = { y1: 0.5, y2: 0.5, y3: 0.5, y4: 0.5, shape: 0 };

// 현재 리뷰 중인 소리의 자동 분류 결과 캐시
let cachedAutoShape = null;

// 우리만의 데이터 저장소 (ml5.js 우회)
let customTrainingData = [];
let isModelTrained = false; // 모델이 학습되었는지 추적

const tempVec = new THREE.Vector3();

// DOM 요소 캐시 (성능 최적화)
let cachedDOMElements = null;

// 예측 throttle 및 race condition 방지
let predictionFrameCounter = 0;
let activePredictionId = 0;
const PREDICTION_INTERVAL = 5; // 5프레임마다 1번 예측 (60fps → 12 predictions/sec)

// 형태 변경 추적
let previousShape = -1;

// 리사이즈 이벤트 핸들러 참조
let resizeHandler = null;

// 디바운스 타이머
let shapeChangeTimer = null;

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

// [추가됨] 상수 정의 (매직 넘버 제거)
const AUDIO_CONSTANTS = {
    LOUDNESS_NORMALIZER: 5,
    LOUDNESS_MULTIPLIER: 10,
    PITCH_NORMALIZER: 40,
    ROUGHNESS_NORMALIZER: 30,
    MIN_TRAINING_SAMPLES: 0,  // AI가 처음부터 예측, 데이터 추가시 점진적 개선
    MIN_PREDICTION_SAMPLES: 0  // 학습 데이터 없어도 예측 허용
};

// 소리 특성에 따라 자동으로 형태 분류
function autoClassifyShape(loudness, pitch, brightness, roughness) {
    // [추가됨] 입력 검증
    if (typeof loudness !== 'number' || isNaN(loudness) ||
        typeof pitch !== 'number' || isNaN(pitch) ||
        typeof brightness !== 'number' || isNaN(brightness) ||
        typeof roughness !== 'number' || isNaN(roughness)) {
        console.error('Invalid input to autoClassifyShape:', { loudness, pitch, brightness, roughness });
        return SHAPES.SPHERE; // 기본값 반환
    }

    // 정규화된 값들로 분류 (0-1 범위 가정)
    const normalizedLoudness = Math.min(1, Math.max(0, loudness / AUDIO_CONSTANTS.LOUDNESS_NORMALIZER));
    const normalizedPitch = Math.min(1, Math.max(0, pitch));
    const normalizedBrightness = Math.min(1, Math.max(0, brightness));
    const normalizedRoughness = Math.min(1, Math.max(0, roughness));

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

// DOM 요소 캐싱 함수 (성능 최적화)
function cacheDOMElements() {
    cachedDOMElements = {
        autoShape: document.getElementById('auto-shape'),
        y1: document.getElementById('y1'),
        y2: document.getElementById('y2'),
        y3: document.getElementById('y3'),
        y4: document.getElementById('y4'),
        shapeSelector: document.getElementById('shape-selector'),
        shapeName: document.getElementById('shape-name'),
        btnMain: document.getElementById('btn-main'),
        btnPlay: document.getElementById('btn-play'),
        btnConfirm: document.getElementById('btn-confirm'),
        labelingZone: document.getElementById('labeling-zone'),
        status: document.getElementById('status')
    };
}

// [추가됨] 중복 제거: 자동 분류 로직을 공통 함수로 추출
function performAutoClassification() {
    if (!recordedX || recordedX.count === 0) {
        console.warn('No recorded audio data for auto-classification');
        return;
    }

    // [수정됨] brain이 있으면 항상 AI 예측 시도 (학습 여부 무관)
    // 학습 데이터가 없으면 랜덤 초기 weights로 예측 → 점진적 개선
    if (brain) {
        // AI 예측 모드 (학습 데이터 0개여도 가능)
        brain.predict([recordedX.loudness, recordedX.pitch, recordedX.brightness, recordedX.roughness], (err, res) => {
            if (!err && res && res.length >= 5) {
                // [수정됨] 모든 예측값에 대한 NaN 체크
                const y1 = res[0].value;
                const y2 = res[1].value;
                const y3 = res[2].value;
                const y4 = res[3].value;
                const rawShapeValue = res[4].value;

                // NaN이 하나라도 있으면 fallback 사용
                if (isNaN(y1) || isNaN(y2) || isNaN(y3) || isNaN(y4) || isNaN(rawShapeValue)) {
                    console.warn('⚠️ AI prediction returned NaN values. Using rule-based classification as fallback.');
                    console.log('  Predicted values:', { y1, y2, y3, y4, shape: rawShapeValue });

                    const fallbackShape = autoClassifyShape(
                        recordedX.loudness,
                        recordedX.pitch,
                        recordedX.brightness,
                        recordedX.roughness
                    );

                    // Fallback: 기본값 사용
                    targetY.y1 = 0.5;
                    targetY.y2 = 0.5;
                    targetY.y3 = 0.5;
                    targetY.y4 = 0.5;
                    targetY.shape = fallbackShape;
                    cachedAutoShape = fallbackShape;
                    document.getElementById('shape-selector').value = fallbackShape;
                    document.getElementById('shape-name').innerText = SHAPE_NAMES[fallbackShape];
                    createShape(fallbackShape);
                    console.log(`📏 Fallback to rule-based: shape=${SHAPE_NAMES[fallbackShape]}, y1-y4=0.5`);
                    return;
                }

                // AI 예측값으로 y1~y4, shape 모두 설정
                targetY.y1 = y1;
                targetY.y2 = y2;
                targetY.y3 = y3;
                targetY.y4 = y4;

                const predictedShape = Math.round(Math.max(0, Math.min(5, rawShapeValue)));
                targetY.shape = predictedShape;
                cachedAutoShape = predictedShape;
                document.getElementById('shape-selector').value = predictedShape;
                document.getElementById('shape-name').innerText = SHAPE_NAMES[predictedShape];
                createShape(predictedShape);
                console.log(`🤖 AI-predicted shape: ${SHAPE_NAMES[predictedShape]} (raw: ${rawShapeValue.toFixed(3)})`);
            }
        });
    } else {
        // 규칙 기반 분류 모드
        const autoShape = autoClassifyShape(
            recordedX.loudness,
            recordedX.pitch,
            recordedX.brightness,
            recordedX.roughness
        );
        targetY.shape = autoShape;
        cachedAutoShape = autoShape;
        document.getElementById('shape-selector').value = autoShape;
        document.getElementById('shape-name').innerText = SHAPE_NAMES[autoShape];
        createShape(autoShape);
        console.log(`📏 Rule-based shape: ${SHAPE_NAMES[autoShape]}`);
    }
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

    // 창 크기 변경 시 컨테이너 크기에 맞춰 업데이트 (메모리 누수 방지)
    if (resizeHandler) {
        window.removeEventListener('resize', resizeHandler);
    }

    resizeHandler = () => {
        const containerWidth = container.clientWidth;
        const containerHeight = container.clientHeight;
        camera.aspect = containerWidth / containerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(containerWidth, containerHeight);
        updateCameraPosition();
    };

    window.addEventListener('resize', resizeHandler);

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
        debug: false,
        hiddenUnits: 8,           // 더 작은 hidden layer (과적합 방지)
        learningRate: 0.01,       // 낮은 learning rate (안정적 학습)
        activationHidden: 'relu', // ReLU activation (학습 안정성)
        activationOutput: 'sigmoid' // 0-1 범위 출력 보장
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

    // DOM 요소 캐싱 (성능 최적화)
    cacheDOMElements();

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
    console.log('=== START RECORDING ===');
    state = 'RECORDING';
    audioChunks = [];
    recordedX = { loudness: 0, pitch: 0, brightness: 0, roughness: 0, count: 0 };
    cachedAutoShape = null; // 새 녹음 시작하면 캐시 초기화

    console.log('Initial recordedX:', recordedX);
    console.log('analyser exists:', !!analyser);
    console.log('audioCtx state:', audioCtx ? audioCtx.state : 'no audioCtx');

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

    // [수정] 매번 새로운 마이크 스트림 요청 (stopRecording에서 끊었으므로)
    console.log('마이크 새로 켜기...');
    try {
        microphoneStream = await navigator.mediaDevices.getUserMedia({ audio: true });

        // AudioContext가 suspended 상태면 resume
        if (audioCtx.state === 'suspended') {
            await audioCtx.resume();
            console.log('AudioContext resumed');
        }

        microphone = audioCtx.createMediaStreamSource(microphoneStream);
        microphone.connect(analyser);
        console.log('✓ Microphone connected to analyser');
        console.log('✓ microphone.mediaStream active:', microphoneStream.active);
        console.log('✓ microphone.mediaStream tracks:', microphoneStream.getTracks().map(t => t.enabled));
    } catch (err) {
        console.error('Microphone access error:', err);
        alert('Failed to access microphone. Please check permissions and try again.');
        state = 'IDLE';
        updateStatus('statusActive', 'status-idle');
        return;
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
    console.log(`Stopping recording... recordedX.count so far: ${recordedX.count}`);

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

    // [수정됨] shape-name을 초기 상태로 설정 (undefined 방지)
    if (!document.getElementById('auto-shape').checked) {
        // 수동 모드면 현재 슬라이더 값으로 표시
        const currentShape = parseInt(document.getElementById('shape-selector').value);
        document.getElementById('shape-name').innerText = SHAPE_NAMES[currentShape];
    }
    // Auto 모드일 때는 saveRecording 콜백에서 처리
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

    // [추가됨] 자동 분류가 켜져 있으면 평균 계산 후 실행
    if (document.getElementById('auto-shape').checked) {
        setTimeout(() => {
            performAutoClassification();
        }, 50); // recordedX 평균 계산 완료 후 실행
    }
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

        // [최적화] 리뷰 모드일 때는 슬라이더 값을 즉시 targetY에 반영
        // 단, Auto-classify가 켜져있으면 슬라이더 값 무시
        // DOM 쿼리를 캐시된 요소로 대체하여 성능 향상 (360 queries/sec → 6 queries/sec)
        if (state === 'REVIEWING') {
            if (cachedDOMElements && !cachedDOMElements.autoShape.checked) {
                // 수동 모드: 슬라이더 값 사용
                targetY.y1 = parseFloat(cachedDOMElements.y1.value);
                targetY.y2 = parseFloat(cachedDOMElements.y2.value);
                targetY.y3 = parseFloat(cachedDOMElements.y3.value);
                targetY.y4 = parseFloat(cachedDOMElements.y4.value);
                targetY.shape = parseFloat(cachedDOMElements.shapeSelector.value);
            }
            // Auto 모드일 때는 stopRecording()에서 설정한 값 유지
        } else if (brain) {
            // [수정됨] brain이 있으면 항상 예측 (학습 데이터 없어도 가능)
            // [최적화] AI 예측 throttle: 5프레임마다 1번만 실행 (60fps → 12 predictions/sec)
            predictionFrameCounter++;
            if (predictionFrameCounter >= PREDICTION_INTERVAL) {
                predictionFrameCounter = 0;

                // [최적화] Race condition 방지: 예측 ID로 오래된 결과 무시
                const currentPredictionId = ++activePredictionId;
                const features = [currentX.loudness, currentX.pitch, currentX.brightness, currentX.roughness];

                brain.predict(features, (err, res) => {
                    // 새로운 예측이 이미 시작되었으면 이 결과 무시
                    if (currentPredictionId !== activePredictionId) return;

                    if(!err && res && res.length >= 5) {
                        // [추가됨] NaN 검증: 실시간 예측에서도 NaN 방지
                        const y1 = res[0].value;
                        const y2 = res[1].value;
                        const y3 = res[2].value;
                        const y4 = res[3].value;
                        const shape = res[4].value;

                        // 유효한 값만 적용
                        if (!isNaN(y1)) targetY.y1 = y1;
                        if (!isNaN(y2)) targetY.y2 = y2;
                        if (!isNaN(y3)) targetY.y3 = y3;
                        if (!isNaN(y4)) targetY.y4 = y4;
                        if (!isNaN(shape)) targetY.shape = shape;
                    }
                });
            }
        }

        // 시각화 수치 부드럽게 전이 (리뷰 모드에서는 더 빠르게)
        const lerpSpeed = (state === 'REVIEWING') ? 0.3 : 0.1;
        currentY.y1 += (targetY.y1 - currentY.y1) * lerpSpeed;
        currentY.y2 += (targetY.y2 - currentY.y2) * lerpSpeed;
        currentY.y3 += (targetY.y3 - currentY.y3) * lerpSpeed;
        currentY.y4 += (targetY.y4 - currentY.y4) * lerpSpeed;
        currentY.shape += (targetY.shape - currentY.shape) * lerpSpeed;

        // [버그 수정] 형태 변경 감지 로직 개선
        const roundedShape = Math.round(currentY.shape);
        if (roundedShape !== previousShape && roundedShape >= 0 && roundedShape <= 5) {
            previousShape = roundedShape;
            createShape(roundedShape);
        }

        // [수정] 시각화:
        // - REVIEWING 상태이고 오디오를 재생 중이면: 재생되는 오디오의 실시간 분석값 사용
        // - REVIEWING 상태이고 재생 안 하면: 녹음된 평균값 사용 (정적 표시)
        // - RECORDING 상태: 마이크 실시간 값 사용
        let visualLoudness = currentX.loudness;
        if (state === 'REVIEWING' && recordedX.count > 0) {
            // 재생 중이 아니면 평균값 사용
            if (!audioTag || audioTag.paused) {
                visualLoudness = recordedX.loudness;
            }
        }
        updateVisuals(visualLoudness);
    }
    renderer.render(scene, camera);
}

function analyzeAudio() {
    const data = new Uint8Array(analyser.frequencyBinCount);
    const time = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(data);
    analyser.getByteTimeDomainData(time);

    // 디버깅: 첫 녹음 프레임에서 raw data 확인
    if (state === 'RECORDING' && recordedX.count === 0) {
        console.log('First recording frame - raw audio data check:');
        console.log('  frequencyBinCount:', analyser.frequencyBinCount);
        console.log('  time data sample:', Array.from(time.slice(0, 10)));
        console.log('  freq data sample:', Array.from(data.slice(0, 10)));
    }

    // [개선됨] 볼륨/loudness 계산 (RMS)
    let s = 0;
    for(let v of time) {
        let n = (v - 128) / 128;
        s += n * n;
    }
    currentX.loudness = time.length > 0 ? Math.sqrt(s / time.length) * AUDIO_CONSTANTS.LOUDNESS_MULTIPLIER : 0;

    // [개선됨] 피치/밝기 계산 (가중 평균) - division by zero 방지
    let te = 0, we = 0;
    for(let i = 0; i < data.length; i++) {
        we += i * data[i];
        te += data[i];
    }
    currentX.pitch = currentX.brightness = te > 0 ? (we / te) / AUDIO_CONSTANTS.PITCH_NORMALIZER : 0;

    // [개선됨] 거칠기/roughness 계산 (영점 교차율)
    let z = 0;
    for(let i = 1; i < time.length; i++) {
        if(time[i] > 128 && time[i-1] <= 128) z++;
    }
    currentX.roughness = time.length > 0 ? z / AUDIO_CONSTANTS.ROUGHNESS_NORMALIZER : 0;

    if (state === 'RECORDING') {
        recordedX.loudness += currentX.loudness;
        recordedX.pitch += currentX.pitch;
        recordedX.brightness += currentX.brightness;
        recordedX.roughness += currentX.roughness;
        recordedX.count++;

        // 디버깅: 처음 몇 프레임만 로그
        if (recordedX.count <= 3) {
            console.log(`Recording frame ${recordedX.count}:`, {
                loudness: currentX.loudness.toFixed(3),
                pitch: currentX.pitch.toFixed(3),
                brightness: currentX.brightness.toFixed(3),
                roughness: currentX.roughness.toFixed(3)
            });
        }
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

    // [수정됨] 항상 정규화 (데이터 개수 무관)
    brain.normalizeData();

    updateStatus('statusTraining', 'status-recording');

    // [개선됨] 적응형 epochs: 데이터 수에 따라 조정 (더 많은 학습으로 안정성 확보)
    const epochs = customTrainingData.length < 10 ? 50 : 30;
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
        if (customTrainingData.length >= AUDIO_CONSTANTS.MIN_TRAINING_SAMPLES) {
            console.log('Auto-retraining brain with loaded data...');

            // brain에 모든 데이터 추가
            for (let i = 0; i < customTrainingData.length; i++) {
                brain.addData(customTrainingData[i].xs, customTrainingData[i].ys);
            }

            // 정규화
            brain.normalizeData();

            // [개선됨] 백그라운드 학습 - 충분한 epochs로 안정성 확보
            const epochs = customTrainingData.length < 10 ? 50 : 30;
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
            debug: false,
            hiddenUnits: 8,           // 더 작은 hidden layer (과적합 방지)
            learningRate: 0.01,       // 낮은 learning rate (안정적 학습)
            activationHidden: 'relu', // ReLU activation (학습 안정성)
            activationOutput: 'sigmoid' // 0-1 범위 출력 보장
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

// [개선됨] 형태 선택기 변경 시 실시간 미리보기 (디바운스 추가)
function onShapeChange() {
    if (state === 'REVIEWING') {
        // 디바운스: 빠른 연속 변경 시 마지막 값만 처리
        if (shapeChangeTimer) {
            clearTimeout(shapeChangeTimer);
        }

        const shapeValue = parseInt(document.getElementById('shape-selector').value);
        document.getElementById('shape-name').innerText = SHAPE_NAMES[shapeValue];

        shapeChangeTimer = setTimeout(() => {
            createShape(shapeValue);
            shapeChangeTimer = null;
        }, 100); // 100ms 디바운스
    }
}

// 자동 형태 분류 토글
function onAutoShapeToggle() {
    const isAutoOn = document.getElementById('auto-shape').checked;
    const shapeSelector = document.getElementById('shape-selector');
    const y1Slider = document.getElementById('y1');
    const y2Slider = document.getElementById('y2');
    const y3Slider = document.getElementById('y3');
    const y4Slider = document.getElementById('y4');

    if (isAutoOn) {
        // 자동 모드: 모든 슬라이더 비활성화
        shapeSelector.disabled = true;
        shapeSelector.style.opacity = '0.5';
        y1Slider.disabled = true;
        y1Slider.style.opacity = '0.5';
        y2Slider.disabled = true;
        y2Slider.style.opacity = '0.5';
        y3Slider.disabled = true;
        y3Slider.style.opacity = '0.5';
        y4Slider.disabled = true;
        y4Slider.style.opacity = '0.5';

        // [개선됨] 현재 녹음된 소리로 자동 분류 (캐시된 값 우선 사용, 공통 함수 활용)
        if (state === 'REVIEWING' && recordedX && recordedX.count > 0) {
            if (cachedAutoShape !== null) {
                // 이미 계산된 값이 있으면 캐시 사용
                shapeSelector.value = cachedAutoShape;
                document.getElementById('shape-name').innerText = SHAPE_NAMES[cachedAutoShape];
                createShape(cachedAutoShape);
                console.log(`📦 Using cached shape: ${SHAPE_NAMES[cachedAutoShape]}`);
            } else {
                // 캐시 없으면 새로 계산
                performAutoClassification();
            }
        }
    } else {
        // 수동 모드: 모든 슬라이더 활성화
        shapeSelector.disabled = false;
        shapeSelector.style.opacity = '1';
        y1Slider.disabled = false;
        y1Slider.style.opacity = '1';
        y2Slider.disabled = false;
        y2Slider.style.opacity = '1';
        y3Slider.disabled = false;
        y3Slider.style.opacity = '1';
        y4Slider.disabled = false;
        y4Slider.style.opacity = '1';
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