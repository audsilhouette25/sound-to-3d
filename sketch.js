let audioCtx, analyser, microphone, mediaRecorder;
let audioChunks = [];
let audioBlob, audioUrl, audioTag, sourceNode; // sourceNode 추가
let brain;
let scene, camera, renderer, sphere, originalVertices;

// 상태 관리
let state = 'IDLE'; 
let recordedX = { loudness: 0, pitch: 0, brightness: 0, roughness: 0 };
let currentX = { loudness: 0, pitch: 0, brightness: 0, roughness: 0 };
let targetY = { y1: 0.5, y2: 0.5, y3: 0.5, y4: 0.5 };
let currentY = { y1: 0.5, y2: 0.5, y3: 0.5, y4: 0.5 };

const tempVec = new THREE.Vector3();

window.onload = () => { initThree(); };

function initThree() {
    const container = document.getElementById('three-container');
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 3.5;
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    container.appendChild(renderer.domElement);

    const geometry = new THREE.SphereGeometry(1, 48, 48);
    sphere = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: 0x00ffcc, wireframe: true }));
    scene.add(sphere);
    originalVertices = sphere.geometry.attributes.position.array.slice();
    scene.add(new THREE.DirectionalLight(0xffffff, 1), new THREE.AmbientLight(0x222222));
    animate();
}

async function initEngine() {
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
        outputs: ['y1', 'y2', 'y3', 'y4'],
        task: 'regression', debug: true
    });

    // 저장된 학습 데이터 불러오기
    loadTrainingData();

    document.getElementById('btn-engine').style.display = 'none';
    document.getElementById('btn-main').style.display = 'block';
    document.getElementById('save-load-zone').style.display = 'block';
    updateDataCount();
}

function handleRecord() {
    if (state === 'IDLE' || state === 'REVIEWING') startRecording();
    else if (state === 'RECORDING') stopRecording();
}

function startRecording() {
    state = 'RECORDING';
    audioChunks = [];
    recordedX = { loudness: 0, pitch: 0, brightness: 0, roughness: 0, count: 0 };
    if(audioTag) audioTag.pause();
    mediaRecorder.start();
    document.getElementById('btn-main').innerText = "녹음 중단 (Stop)";
    document.getElementById('labeling-zone').style.display = "none";
}

function stopRecording() {
    mediaRecorder.stop();
    state = 'REVIEWING';
    document.getElementById('btn-main').innerText = "다시 녹음하기";
    document.getElementById('labeling-zone').style.display = "block";
    document.getElementById('btn-confirm').style.display = "block";
}

function saveRecording() {
    audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    audioUrl = URL.createObjectURL(audioBlob);
    
    if (audioTag) audioTag.pause();
    audioTag = new Audio(audioUrl);
    audioTag.loop = true;

    // [핵심 수정] 재생되는 오디오를 분석기에 연결
    if (sourceNode) sourceNode.disconnect();
    sourceNode = audioCtx.createMediaElementSource(audioTag);
    sourceNode.connect(analyser);
    analyser.connect(audioCtx.destination); // 스피커로 출력

    audioTag.play();

    // 녹음된 평균값 저장 (학습용)
    recordedX.loudness /= recordedX.count;
    recordedX.pitch /= recordedX.count;
    recordedX.brightness /= recordedX.count;
    recordedX.roughness /= recordedX.count;
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
        } else if (brain && brain.data && brain.data.list.length >= 5) {
            // 평상시에는 AI가 예측
            brain.predict(currentX, (err, res) => {
                if(!err) {
                    targetY.y1 = res[0].value; targetY.y2 = res[1].value;
                    targetY.y3 = res[2].value; targetY.y4 = res[3].value;
                }
            });
        }

        // 시각화 수치 부드럽게 전이
        currentY.y1 += (targetY.y1 - currentY.y1) * 0.1;
        currentY.y2 += (targetY.y2 - currentY.y2) * 0.1;
        currentY.y3 += (targetY.y3 - currentY.y3) * 0.1;
        currentY.y4 += (targetY.y4 - currentY.y4) * 0.1;

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
    const pos = sphere.geometry.attributes.position;
    const t = Date.now() * 0.001;
    for (let i = 0; i < pos.count; i++) {
        const i3 = i * 3;
        tempVec.set(originalVertices[i3], originalVertices[i3+1], originalVertices[i3+2]).normalize();
        
        // 슬라이더(currentY)에 의한 형태 변형 + 실시간 소리(loudness)에 의한 진폭
        const wave = Math.sin(tempVec.x * (5 + currentY.y4 * 25) + t) * currentY.y2 * 0.5;
        const rough = (Math.sin(t * 10) * 0.1) * currentY.y3; // 거친 느낌 추가
        const dist = 1 + wave + rough + (loudness * 0.4);
        
        tempVec.multiplyScalar(dist);
        pos.setXYZ(i, tempVec.x, tempVec.y, tempVec.z);
    }
    sphere.rotation.y += 0.005 + (currentY.y1 * 0.05);
    pos.needsUpdate = true;
}

function confirmTraining() {
    const labels = {
        y1: parseFloat(document.getElementById('y1').value),
        y2: parseFloat(document.getElementById('y2').value),
        y3: parseFloat(document.getElementById('y3').value),
        y4: parseFloat(document.getElementById('y4').value)
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
    brain.train({ epochs: 20 }, () => {
        alert("학습 완료! 데이터가 자동 저장되었습니다.");
        state = 'IDLE';
        if(audioTag) audioTag.pause();
        document.getElementById('labeling-zone').style.display = "none";
        document.getElementById('btn-main').innerText = "녹음 시작";
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
            outputs: ['y1', 'y2', 'y3', 'y4'],
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

    let csv = 'loudness,pitch,brightness,roughness,y1,y2,y3,y4\n';
    brain.data.data.raw.forEach(item => {
        csv += `${item.xs.loudness},${item.xs.pitch},${item.xs.brightness},${item.xs.roughness},`;
        csv += `${item.ys.y1},${item.ys.y2},${item.ys.y3},${item.ys.y4}\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `soundTo3D_data_${Date.now()}.csv`;
    a.click();
}