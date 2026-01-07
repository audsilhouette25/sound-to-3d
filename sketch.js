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

    console.log('Brain created, checking initialization...');
    console.log('Initial brain.data:', brain.data);

    // ml5.js neuralNetwork는 생성 직후에는 brain.data가 undefined일 수 있음
    // brain.data.training이 실제로 존재할 때까지 대기
    const waitForBrainReady = () => {
        if (brain.data && brain.data.training !== undefined) {
            console.log('Brain is ready! brain.data.training exists');
            console.log('brain.data.training type:', typeof brain.data.training);
            console.log('brain.data.training is Array?', Array.isArray(brain.data.training));
            console.log('Brain state before load:', brain.data);

            // 테스트: 더미 데이터 추가해서 addData가 작동하는지 확인
            const testLength = brain.data.training.length;
            brain.addData([1, 2, 3, 4], [0.5, 0.5, 0.5, 0.5, 0]);
            const afterTestLength = brain.data.training.length;
            console.log(`Test addData: before=${testLength}, after=${afterTestLength}, success=${afterTestLength > testLength}`);

            if (afterTestLength === testLength) {
                console.error('CRITICAL: brain.addData() is NOT working! Brain may be corrupted.');
                console.log('Recreating brain...');
                brain = ml5.neuralNetwork({
                    inputs: 4,
                    outputs: 5,
                    task: 'regression',
                    debug: false
                });
                setTimeout(waitForBrainReady, 100);
                return;
            }

            // 테스트 데이터 제거 (pop)
            if (brain.data.training.length > 0) {
                brain.data.training.pop();
                console.log('Test data removed');
            }

            loadTrainingData();
            console.log('Brain state after load:', brain.data);
            updateDataCount();
        } else {
            console.log('Brain not ready yet, waiting... brain.data:', brain.data);
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
        } else if (brain && brain.data) {
            // 평상시에는 AI가 예측
            const dataArray = brain.data.training || [];
            if (dataArray.length >= 5) {
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
    console.log('confirmTraining called');
    console.log('recordedX:', recordedX);

    // recordedX 검증
    if (recordedX.count === 0 ||
        isNaN(recordedX.loudness) ||
        isNaN(recordedX.pitch) ||
        isNaN(recordedX.brightness) ||
        isNaN(recordedX.roughness)) {
        alert('Recording data is invalid. Please record again.');
        console.error('Invalid recordedX data:', recordedX);
        return;
    }

    const labels = {
        y1: parseFloat(document.getElementById('y1').value),
        y2: parseFloat(document.getElementById('y2').value),
        y3: parseFloat(document.getElementById('y3').value),
        y4: parseFloat(document.getElementById('y4').value),
        shape: parseFloat(document.getElementById('shape-selector').value)
    };

    console.log('labels:', labels);

    // addData 호출 전 상태 확인
    console.log('BEFORE addData - brain:', brain);
    console.log('BEFORE addData - brain.data:', brain.data);
    console.log('BEFORE addData - brain.data.training:', brain.data ? brain.data.training : 'undefined');

    const inputArray = [recordedX.loudness, recordedX.pitch, recordedX.brightness, recordedX.roughness];
    const outputArray = [labels.y1, labels.y2, labels.y3, labels.y4, labels.shape];

    console.log('Input array:', inputArray);
    console.log('Output array:', outputArray);

    brain.addData(inputArray, outputArray);

    // addData 호출 후 상태 확인
    console.log('AFTER addData - brain.data:', brain.data);
    console.log('AFTER addData - brain.data.training:', brain.data.training);
    console.log('AFTER addData - Data count:', brain.data.training ? brain.data.training.length : 0);

    // 학습 데이터 자동 저장
    saveTrainingData();
    updateDataCount();

    const dataCount = brain.data.training.length;

    console.log(`Successfully saved! Total samples: ${dataCount}`);

    // 데이터가 2개 이상이면 정규화, 1개는 정규화 없이 학습
    if (dataCount >= 2) {
        brain.normalizeData();
    }

    updateStatus('statusTraining', 'status-recording');

    brain.train({ epochs: 20 }, () => {
        console.log('Training complete!');
        alert(`✓ Training Complete!\n\nSaved ${dataCount} sample(s) to storage.\nModel is ready for predictions.`);
        state = 'IDLE';

        // 재생 중지
        if(audioTag) audioTag.pause();

        const t = translations[currentLang];
        document.getElementById('labeling-zone').style.display = "none";
        document.getElementById('btn-main').innerText = t.btnRecord;
        document.getElementById('btn-play').style.display = "none";

        updateStatus('statusActive', 'status-idle');
    });
}

// 학습 데이터를 localStorage에 저장
function saveTrainingData() {
    if (!brain || !brain.data) return;

    console.log('brain.data structure:', brain.data);

    // ml5.js neuralNetwork는 brain.data.training이 실제 데이터 배열
    const dataArray = brain.data.training || [];

    const trainingData = {
        data: dataArray,
        timestamp: Date.now(),
        version: 1
    };

    localStorage.setItem('soundTo3D_trainingData', JSON.stringify(trainingData));
    console.log('학습 데이터 저장됨:', dataArray.length, '개');
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

        // 데이터 유효성 검사
        if (!trainingData || !trainingData.data || !Array.isArray(trainingData.data)) {
            console.warn('잘못된 데이터 형식, localStorage 초기화');
            localStorage.removeItem('soundTo3D_trainingData');
            return;
        }

        console.log('학습 데이터 불러오는 중:', trainingData.data.length, '개');
        console.log('brain.data before forEach:', brain.data);

        // 기존 데이터에 추가
        let addedCount = 0;
        trainingData.data.forEach((item, index) => {
            if (!item || !item.xs || !item.ys) {
                console.warn(`Skipping invalid item at index ${index}`);
                return;
            }

            const xs = Array.isArray(item.xs) ? item.xs : [item.xs.loudness, item.xs.pitch, item.xs.brightness, item.xs.roughness];
            const ys = Array.isArray(item.ys) ? item.ys : [item.ys.y1, item.ys.y2, item.ys.y3, item.ys.y4, item.ys.shape];

            console.log(`Adding item ${index}:`, { xs, ys });
            brain.addData(xs, ys);
            addedCount++;

            console.log(`After adding item ${index}, brain.data.training length:`, brain.data.training ? brain.data.training.length : 'undefined');
        });

        // 데이터 불러오기만 하고 자동 학습은 하지 않음
        const dataArray = brain.data.training || [];
        console.log(`Loaded ${dataArray.length} samples from localStorage (added: ${addedCount})`);

        if (dataArray.length > 0) {
            console.log('데이터 불러오기 완료. 새 데이터 추가 후 학습이 진행됩니다.');
        } else if (addedCount > 0) {
            console.error(`WARNING: Added ${addedCount} items but brain.data.training is still empty!`);
        }
    } catch (e) {
        console.error('데이터 불러오기 실패:', e);
        console.warn('손상된 데이터 제거');
        localStorage.removeItem('soundTo3D_trainingData');
    }
}

// 학습 데이터 개수 업데이트
function updateDataCount() {
    if (!brain || !brain.data) {
        document.getElementById('data-count').innerText = '0';
        return;
    }
    const dataArray = brain.data.training || [];
    const countEl = document.getElementById('data-count');
    if (countEl) {
        countEl.innerText = dataArray.length;
    }
}

// 모든 학습 데이터 삭제
function clearAllData() {
    if (confirm('Delete all training data?\nThis action cannot be undone.')) {
        localStorage.removeItem('soundTo3D_trainingData');

        // brain 재생성
        brain = ml5.neuralNetwork({
            inputs: 4,
            outputs: 5,
            task: 'regression',
            debug: false
        });

        updateDataCount();
        alert('All training data has been deleted.');
    }
}

// CSV로 데이터 내보내기 (기존 함수 개선)
function exportCSV() {
    if (!brain || !brain.data) {
        alert('No data to export.');
        return;
    }

    const dataArray = brain.data.training || [];
    if (dataArray.length === 0) {
        alert('No data to export.');
        return;
    }

    let csv = 'loudness,pitch,brightness,roughness,y1,y2,y3,y4,shape\n';
    dataArray.forEach(item => {
        const xs = Array.isArray(item.xs) ? item.xs : [item.xs.loudness, item.xs.pitch, item.xs.brightness, item.xs.roughness];
        const ys = Array.isArray(item.ys) ? item.ys : [item.ys.y1, item.ys.y2, item.ys.y3, item.ys.y4, item.ys.shape];
        csv += `${xs[0]},${xs[1]},${xs[2]},${xs[3]},`;
        csv += `${ys[0]},${ys[1]},${ys[2]},${ys[3]},${ys[4]}\n`;
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