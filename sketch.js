// ==================================================================
// Sound-to-3D Visualization with ML
// Refactored for better code quality, performance, and maintainability
// ==================================================================

// --- Constants ---
const API_URL = 'https://sound-to-3d-server.onrender.com';
const USE_SERVER = true; // Server enabled - data syncs across all browsers

const SHAPE_NAMES = ['Sphere', 'Cube', 'Torus', 'Cone', 'Cylinder'];

const CONSTANTS = {
    // Audio normalization
    LOUDNESS_NORMALIZER: 2.0,
    PITCH_NORMALIZER: 20.0,
    BRIGHTNESS_NORMALIZER: 24.0,
    ROUGHNESS_NORMALIZER: 4.0,  // Increased from 1.0 to handle peak values up to 3.4

    // Rendering
    CAMERA_FOV: 75,
    CAMERA_NEAR: 0.1,
    CAMERA_FAR: 1000,
    CAMERA_DISTANCE: 3.5,
    LERP_SPEED_REVIEWING: 0.3,
    LERP_SPEED_LIVE: 0.1,
    LERP_THRESHOLD: 0.001, // Skip LERP if delta below this
    SHAPE_CHANGE_THRESHOLD: 0.1,

    // Prediction optimization
    PREDICTION_INTERVAL: 5, // 5 frames per prediction

    // Audio recording
    SAMPLE_RATE: 48000,
    AUDIO_BITS_PER_SECOND: 128000,
    RECORDER_TIMESLICE: 100,

    // Audio analysis
    FFT_SIZE: 2048,
    PITCH_DIVISOR: 50.0,
    BRIGHTNESS_MULTIPLIER: 1.2,
    ROUGHNESS_DIVISOR: 40.0,
    LOUDNESS_MULTIPLIER: 10.0,

    // File upload analysis
    ANALYZE_DURATION: 3000, // ms
    ANALYZE_INTERVAL: 50, // ms

    // DOM update throttling
    DOM_UPDATE_INTERVAL: 100, // ms
    DOM_UPDATE_THRESHOLD: 0.01, // Only update if change > 1%

    // Training
    TRAINING_EPOCHS: 50,
    TRAINING_DELAY: 500, // ms
    RELOAD_DELAY: 500, // ms

    // Mesh subdivision
    CUBE_SUBDIVISIONS: 32,

    // Shader timing
    TIME_INCREMENT: 0.05,
    ROTATION_SPEED: 0.005
};

// --- Application State (Consolidated) ---
const appState = {
    audio: {
        ctx: null,
        analyser: null,
        microphone: null,
        stream: null,
        recorder: null,
        sourceNode: null,
        audioTag: null,
        chunks: [],
        isPlaying: false,
        // Current audio features (real-time)
        features: {
            loudness: 0,
            pitch: 0,
            brightness: 0,
            roughness: 0
        },
        // Recorded audio features (accumulated during recording)
        recorded: {
            loudness: 0,
            pitch: 0,
            brightness: 0,
            roughness: 0,
            count: 0
        }
    },
    visuals: {
        scene: null,
        camera: null,
        renderer: null,
        mesh: null,
        // Current visual state (interpolated)
        current: {
            y1: 0.5,
            y2: 0.5,
            y3: 0.5,
            y4: 0.5,
            shape: 0
        },
        // Target visual state
        target: {
            y1: 0.5,
            y2: 0.5,
            y3: 0.5,
            y4: 0.5,
            shape: 0
        },
        uniforms: {
            uTime: { value: 0 },
            uLoudness: { value: 0 },
            uY1: { value: 0.5 },
            uY2: { value: 0.5 },
            uY3: { value: 0.5 },
            uY4: { value: 0.5 }
        },
        previousShape: -1,
        predictionFrameCounter: 0
    },
    ml: {
        brain: null,
        isTrained: false,
        trainingData: []
    },
    ui: {
        state: 'IDLE', // 'IDLE', 'RECORDING', 'REVIEWING'
        lang: 'KR',
        // Cached DOM elements
        elements: {
            btnMain: null,
            btnPlay: null,
            btnConfirm: null,
            btnUpload: null,
            btnEngine: null,
            y1: null,
            y2: null,
            y3: null,
            y4: null,
            shapeSelector: null,
            valLoud: null,
            valPitch: null,
            valBright: null,
            valRough: null,
            dataCount: null,
            labelingZone: null,
            saveLoadZone: null,
            status: null,
            trainingOverlay: null,
            trainingMessage: null,
            trainingProgress: null
        },
        lastDOMUpdate: 0,
        lastAudioDisplayUpdate: 0,
        lastAudioValues: {
            loudness: 0,
            pitch: 0,
            brightness: 0,
            roughness: 0
        }
    },
    resources: {
        objectURLs: [],
        listeners: []
    }
};

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
        return mix(
            mix(mix(hash(n+0.0), hash(n+1.0), f.x), mix(hash(n+57.0), hash(n+58.0), f.x), f.y),
            mix(mix(hash(n+113.0), hash(n+114.0), f.x), mix(hash(n+170.0), hash(n+171.0), f.x), f.y),
            f.z
        );
    }

    void main() {
        vNormal = normal;
        vec3 pos = position;
        float noiseVal = noise(pos * (2.0 + uY4 * 8.0) + uTime * 0.4);
        float angular = floor(noiseVal * (1.0 + (1.0-uY1)*12.0)) / (1.0 + (1.0-uY1)*12.0);
        float finalNoise = mix(noiseVal, angular, uY1);
        float wave = sin(pos.x * 12.0 + uTime) * uY2 * 0.45;

        float displacement = (finalNoise * uY3 * 0.7) + (uLoudness * 0.6) + wave;
        vDisplacement = displacement;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(pos + normal * displacement, 1.0);
    }
`;

const fragmentShader = `
    varying float vDisplacement;
    void main() {
        vec3 colorA = vec3(0.0, 1.0, 0.7);
        vec3 colorB = vec3(0.1, 0.05, 0.4);
        vec3 finalColor = mix(colorB, colorA, vDisplacement + 0.25);
        gl_FragColor = vec4(finalColor, 0.9);
    }
`;

// Cube-specific vertex shader with mirror effect
const cubeVertexShader = `
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
        return mix(
            mix(mix(hash(n+0.0), hash(n+1.0), f.x), mix(hash(n+57.0), hash(n+58.0), f.x), f.y),
            mix(mix(hash(n+113.0), hash(n+114.0), f.x), mix(hash(n+170.0), hash(n+171.0), f.x), f.y),
            f.z
        );
    }

    void main() {
        vNormal = normal;
        vec3 pos = position;

        // Mirror effect: use absolute position for noise so mirrored points get same base value
        vec3 absPos = abs(pos);
        float noiseVal = noise(absPos * (2.0 + uY4 * 8.0) + uTime * 0.4);
        float angular = floor(noiseVal * (1.0 + (1.0-uY1)*12.0)) / (1.0 + (1.0-uY1)*12.0);
        float finalNoise = mix(noiseVal, angular, uY1);
        float wave = sin(absPos.x * 12.0 + uTime) * uY2 * 0.45;

        float baseDisplacement = (finalNoise * uY3 * 0.7) + (uLoudness * 0.6) + wave;

        // Determine which axis based on normal direction
        float mirror = 1.0;
        if (abs(normal.x) > 0.3) mirror = sign(pos.x);  // Left (-x) and Right (+x) faces
        else if (abs(normal.y) > 0.3) mirror = sign(pos.y);  // Bottom (-y) and Top (+y) faces
        else if (abs(normal.z) > 0.3) mirror = sign(pos.z);  // Back (-z) and Front (+z) faces

        // Apply mirror: +face gets +displacement, -face gets -displacement
        float displacement = baseDisplacement * mirror;
        // Use baseDisplacement for color to get proper gradient from inner (dark) to outer (bright)
        vDisplacement = baseDisplacement;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(pos + normal * displacement, 1.0);
    }
`;

// --- Utility Functions ---

/**
 * Calculate audio features from frequency and time domain data
 */
function calculateAudioFeatures(frequencyData, timeDomainData) {
    // Loudness calculation
    let sum = 0;
    for (let v of timeDomainData) {
        let n = (v - 128) / 128;
        sum += n * n;
    }
    const loudness = Math.sqrt(sum / timeDomainData.length) * CONSTANTS.LOUDNESS_MULTIPLIER;

    // Pitch calculation (spectral centroid)
    let totalEnergy = 0;
    let weightedEnergy = 0;
    for (let i = 0; i < frequencyData.length; i++) {
        weightedEnergy += i * frequencyData[i];
        totalEnergy += frequencyData[i];
    }
    const pitch = totalEnergy > 0 ? (weightedEnergy / totalEnergy) / CONSTANTS.PITCH_DIVISOR : 0;
    const brightness = pitch * CONSTANTS.BRIGHTNESS_MULTIPLIER;

    // Roughness calculation (zero-crossing rate)
    let zcr = 0;
    for (let i = 1; i < timeDomainData.length; i++) {
        if (timeDomainData[i] > 128 && timeDomainData[i - 1] <= 128) zcr++;
    }
    const roughness = zcr / CONSTANTS.ROUGHNESS_DIVISOR;

    return { loudness, pitch, brightness, roughness };
}

/**
 * Normalize audio features to 0-1 range
 */
function normalizeAudioFeatures(features) {
    return {
        loudness: Math.min(1, Math.max(0, features.loudness / CONSTANTS.LOUDNESS_NORMALIZER)),
        pitch: Math.min(1, Math.max(0, features.pitch / CONSTANTS.PITCH_NORMALIZER)),
        brightness: Math.min(1, Math.max(0, features.brightness / CONSTANTS.BRIGHTNESS_NORMALIZER)),
        roughness: Math.min(1, Math.max(0, features.roughness / CONSTANTS.ROUGHNESS_NORMALIZER))
    };
}

/**
 * Validate input features
 */
function validateAudioFeatures(features) {
    return typeof features.loudness === 'number' && !isNaN(features.loudness) &&
           typeof features.pitch === 'number' && !isNaN(features.pitch) &&
           typeof features.brightness === 'number' && !isNaN(features.brightness) &&
           typeof features.roughness === 'number' && !isNaN(features.roughness);
}

/**
 * Auto-classify shape based on audio features (rule-based)
 */
function autoClassifyShape(features) {
    if (!validateAudioFeatures(features)) {
        console.error('Invalid input to autoClassifyShape:', features);
        return 0; // Default to sphere
    }

    const normalized = normalizeAudioFeatures(features);

    console.log('🎵 Audio features:', {
        loudness: features.loudness.toFixed(3),
        pitch: features.pitch.toFixed(3),
        brightness: features.brightness.toFixed(3),
        roughness: features.roughness.toFixed(3),
        normalized: {
            loudness: normalized.loudness.toFixed(3),
            pitch: normalized.pitch.toFixed(3),
            brightness: normalized.brightness.toFixed(3),
            roughness: normalized.roughness.toFixed(3)
        }
    });

    const scores = [0, 0, 0, 0, 0];

    // Sphere: 부드럽고 조용한 (ONLY for very smooth and quiet sounds)
    scores[0] = (normalized.roughness < 0.3 ? (1 - normalized.roughness) * 0.5 : 0) +
                (normalized.loudness < 0.4 ? (1 - normalized.loudness) * 0.3 : 0) +
                (normalized.pitch > 0.3 && normalized.pitch < 0.7 ? 0.2 : 0);

    // Cube: 거칠고 밝은 타악기 소리 (percussion, claps, rough sounds)
    scores[1] = normalized.brightness * 0.6 +
                (normalized.roughness > 0.3 ? normalized.roughness * 0.9 : 0) +
                (normalized.loudness > 0.4 ? normalized.loudness * 0.5 : 0);

    // Torus: 중간-높은 pitch, 회전감
    scores[2] = (normalized.pitch > 0.5 ? normalized.pitch * 0.7 : 0.2) +
                normalized.loudness * 0.5;

    // Cone: 높고 날카로운
    scores[3] = (normalized.pitch > 0.6 ? normalized.pitch * 0.6 : 0) +
                (normalized.brightness > 0.6 ? normalized.brightness * 0.5 : 0);

    // Cylinder: 부드럽지만 큰 소리
    scores[4] = (normalized.roughness < 0.4 ? (1 - normalized.roughness) * 0.6 : 0) +
                (normalized.loudness > 0.5 ? normalized.loudness * 0.7 : 0);

    console.log('📊 Shape scores:', scores.map((s, i) => `${SHAPE_NAMES[i]}: ${s.toFixed(3)}`).join(', '));

    let maxScore = -1;
    let bestShape = 0;
    for (let i = 0; i < 5; i++) {
        if (scores[i] > maxScore) {
            maxScore = scores[i];
            bestShape = i;
        }
    }

    console.log(`✅ Selected: ${SHAPE_NAMES[bestShape]} (score: ${maxScore.toFixed(3)})`);
    return bestShape;
}

/**
 * Auto-suggest visual parameters based on audio features
 */
function autoSuggestParameters(features) {
    const normalized = normalizeAudioFeatures(features);

    return {
        y1: normalized.pitch * 0.6 + normalized.brightness * 0.4,
        y2: normalized.brightness * 0.5 + normalized.roughness * 0.5,
        y3: normalized.roughness * 0.7 + (1 - normalized.loudness) * 0.3,
        y4: normalized.brightness * 0.5 + normalized.loudness * 0.5
    };
}

/**
 * Perform AI prediction (consolidated)
 */
function performAIPrediction(features, callback) {
    if (!appState.ml.brain || !appState.ml.isTrained) {
        callback(null);
        return;
    }

    appState.ml.brain.predict(
        [features.loudness, features.pitch, features.brightness, features.roughness],
        (err, results) => {
            if (err || !results || results.length !== 5) {
                callback(null);
                return;
            }

            callback({
                y1: results[0],
                y2: results[1],
                y3: results[2],
                y4: results[3],
                shape: Math.min(4, Math.max(0, Math.round(results[4] * 4)))
            });
        }
    );
}

/**
 * Update target visual parameters based on current state
 */
function updateTargetVisuals() {
    const { state } = appState.ui;
    const { elements } = appState.ui;

    if (state === 'REVIEWING') {
        // REVIEWING mode: Use slider values directly
        appState.visuals.target.y1 = parseFloat(elements.y1.value);
        appState.visuals.target.y2 = parseFloat(elements.y2.value);
        appState.visuals.target.y3 = parseFloat(elements.y3.value);
        appState.visuals.target.y4 = parseFloat(elements.y4.value);
        appState.visuals.target.shape = parseInt(elements.shapeSelector.value);
    } else if (state === 'RECORDING' || state === 'IDLE') {
        // During recording/idle: Keep sphere shape and use rule-based parameters only
        if (appState.visuals.current.shape !== 0) {
            appState.visuals.current.shape = 0;
            appState.visuals.previousShape = -1;
            createShape(0);
        }

        const suggestedParams = autoSuggestParameters(appState.audio.features);
        appState.visuals.target.y1 = suggestedParams.y1;
        appState.visuals.target.y2 = suggestedParams.y2;
        appState.visuals.target.y3 = suggestedParams.y3;
        appState.visuals.target.y4 = suggestedParams.y4;
    }
}

/**
 * Check if shape should be changed
 */
function shouldChangeShape() {
    const roundedShape = Math.round(appState.visuals.current.shape);
    const delta = Math.abs(appState.visuals.current.shape - roundedShape);
    return delta > CONSTANTS.SHAPE_CHANGE_THRESHOLD &&
           roundedShape !== appState.visuals.previousShape &&
           roundedShape >= 0 && roundedShape <= 4;
}

// --- Resource Management ---

/**
 * Create and track an object URL
 */
function createTrackedObjectURL(blob) {
    const url = URL.createObjectURL(blob);
    appState.resources.objectURLs.push(url);
    return url;
}

/**
 * Add a tracked event listener
 */
function addTrackedListener(element, event, handler) {
    element.addEventListener(event, handler);
    appState.resources.listeners.push({ element, event, handler });
}

/**
 * Cleanup all resources
 */
function cleanupResources() {
    // Revoke all object URLs
    appState.resources.objectURLs.forEach(url => URL.revokeObjectURL(url));
    appState.resources.objectURLs = [];

    // Remove all event listeners
    appState.resources.listeners.forEach(({ element, event, handler }) => {
        element.removeEventListener(event, handler);
    });
    appState.resources.listeners = [];

    // Disconnect audio nodes
    if (appState.audio.microphone) {
        appState.audio.microphone.disconnect();
        appState.audio.microphone = null;
    }

    if (appState.audio.sourceNode) {
        appState.audio.sourceNode.disconnect();
        appState.audio.sourceNode = null;
    }

    // Stop media stream
    if (appState.audio.stream) {
        appState.audio.stream.getTracks().forEach(track => track.stop());
        appState.audio.stream = null;
    }

    // Close audio context
    if (appState.audio.ctx && appState.audio.ctx.state !== 'closed') {
        appState.audio.ctx.close();
    }
}

// --- 3D Visualization ---

/**
 * Initialize Three.js scene
 */
function initThree() {
    console.log('initThree() called');

    appState.visuals.scene = new THREE.Scene();
    appState.visuals.scene.background = new THREE.Color(0x050505);

    const rightPanelWidth = window.innerWidth - 320;
    appState.visuals.camera = new THREE.PerspectiveCamera(
        CONSTANTS.CAMERA_FOV,
        rightPanelWidth / window.innerHeight,
        CONSTANTS.CAMERA_NEAR,
        CONSTANTS.CAMERA_FAR
    );
    appState.visuals.camera.position.z = CONSTANTS.CAMERA_DISTANCE;

    appState.visuals.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    appState.visuals.renderer.setSize(rightPanelWidth, window.innerHeight);
    appState.visuals.renderer.domElement.style.position = 'absolute';
    appState.visuals.renderer.domElement.style.top = '0';
    appState.visuals.renderer.domElement.style.left = '320px';
    appState.visuals.renderer.domElement.style.zIndex = '1';
    document.body.appendChild(appState.visuals.renderer.domElement);

    console.log('Renderer created and appended to body');
    console.log('Canvas size:', rightPanelWidth, 'x', window.innerHeight);

    // Track window resize listener
    const resizeHandler = onWindowResize;
    addTrackedListener(window, 'resize', resizeHandler);

    createShape(0);
    console.log('Initial shape created');
    animate();
    console.log('Animation loop started');
}

/**
 * Handle window resize
 */
function onWindowResize() {
    const rightPanelWidth = window.innerWidth - 320;
    appState.visuals.camera.aspect = rightPanelWidth / window.innerHeight;
    appState.visuals.camera.updateProjectionMatrix();
    appState.visuals.renderer.setSize(rightPanelWidth, window.innerHeight);
}

/**
 * Create a cube with connected vertices (prevents tearing)
 * @param {number} width - Width (x-axis)
 * @param {number} height - Height (y-axis)
 * @param {number} depth - Depth (z-axis)
 * @param {number} subdivisions - Number of subdivisions per face
 */
function createConnectedCube(width, height, depth, subdivisions) {
    const geo = new THREE.BufferGeometry();
    const halfX = width / 2;
    const halfY = height / 2;
    const halfZ = depth / 2;
    const vertices = [];
    const indices = [];
    const seg = subdivisions;
    const vertexMap = new Map();

    function getVertexIndex(x, y, z) {
        const key = `${x.toFixed(6)},${y.toFixed(6)},${z.toFixed(6)}`;
        if (vertexMap.has(key)) {
            return vertexMap.get(key);
        }
        const index = vertices.length / 3;
        vertices.push(x, y, z);
        vertexMap.set(key, index);
        return index;
    }

    // Generate vertices for all faces
    // Front and Back faces (x-y plane)
    for (let i = 0; i <= seg; i++) {
        for (let j = 0; j <= seg; j++) {
            const x = -halfX + (i / seg) * width;
            const y = -halfY + (j / seg) * height;
            getVertexIndex(x, y, halfZ);  // Front
            getVertexIndex(x, y, -halfZ); // Back
        }
    }

    // Top and Bottom faces (x-z plane)
    for (let i = 0; i <= seg; i++) {
        for (let k = 0; k <= seg; k++) {
            const x = -halfX + (i / seg) * width;
            const z = -halfZ + (k / seg) * depth;
            getVertexIndex(x, halfY, z);  // Top
            getVertexIndex(x, -halfY, z); // Bottom
        }
    }

    // Right and Left faces (y-z plane)
    for (let j = 0; j <= seg; j++) {
        for (let k = 0; k <= seg; k++) {
            const y = -halfY + (j / seg) * height;
            const z = -halfZ + (k / seg) * depth;
            getVertexIndex(halfX, y, z);  // Right
            getVertexIndex(-halfX, y, z); // Left
        }
    }

    // Generate indices
    function addFaceIndices(getIndex) {
        for (let i = 0; i < seg; i++) {
            for (let j = 0; j < seg; j++) {
                const a = getIndex(i, j);
                const b = getIndex(i + 1, j);
                const c = getIndex(i + 1, j + 1);
                const d = getIndex(i, j + 1);
                indices.push(a, b, c);
                indices.push(a, c, d);
            }
        }
    }

    // Add indices for each face
    const faceConfigs = [
        (i, j) => vertexMap.get(`${(-halfX + (i / seg) * width).toFixed(6)},${(-halfY + (j / seg) * height).toFixed(6)},${halfZ.toFixed(6)}`), // Front
        (i, j) => vertexMap.get(`${(-halfX + (i / seg) * width).toFixed(6)},${(-halfY + (j / seg) * height).toFixed(6)},${(-halfZ).toFixed(6)}`), // Back
        (i, k) => vertexMap.get(`${(-halfX + (i / seg) * width).toFixed(6)},${halfY.toFixed(6)},${(-halfZ + (k / seg) * depth).toFixed(6)}`), // Top
        (i, k) => vertexMap.get(`${(-halfX + (i / seg) * width).toFixed(6)},${(-halfY).toFixed(6)},${(-halfZ + (k / seg) * depth).toFixed(6)}`), // Bottom
        (j, k) => vertexMap.get(`${halfX.toFixed(6)},${(-halfY + (j / seg) * height).toFixed(6)},${(-halfZ + (k / seg) * depth).toFixed(6)}`), // Right
        (j, k) => vertexMap.get(`${(-halfX).toFixed(6)},${(-halfY + (j / seg) * height).toFixed(6)},${(-halfZ + (k / seg) * depth).toFixed(6)}`) // Left
    ];

    faceConfigs.forEach(config => addFaceIndices(config));

    geo.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geo.setIndex(indices);
    geo.computeVertexNormals();

    return geo;
}

/**
 * Create 3D shape
 */
function createShape(type) {
    if (appState.visuals.mesh) {
        appState.visuals.scene.remove(appState.visuals.mesh);
        if (appState.visuals.mesh.geometry) appState.visuals.mesh.geometry.dispose();
        if (appState.visuals.mesh.material) appState.visuals.mesh.material.dispose();
    }

    let geo;
    type = parseInt(type);

    switch (type) {
        case 0: geo = new THREE.SphereGeometry(1, 128, 128); break;  // Smooth sphere
        case 1: geo = createConnectedCube(1.0, 1.8, 1.0, CONSTANTS.CUBE_SUBDIVISIONS); break;  // Tall rectangular prism (width, height, depth)
        case 2: geo = new THREE.TorusGeometry(0.8, 0.4, 64, 128); break;
        case 3: geo = new THREE.ConeGeometry(1, 2, 64, 64); break;
        default: geo = new THREE.CylinderGeometry(0.8, 0.8, 2, 64, 64); break;
    }

    const mat = new THREE.ShaderMaterial({
        uniforms: appState.visuals.uniforms,
        vertexShader: type === 1 ? cubeVertexShader : vertexShader,
        fragmentShader,
        wireframe: true,
        transparent: true
    });

    appState.visuals.mesh = new THREE.Mesh(geo, mat);
    appState.visuals.scene.add(appState.visuals.mesh);
}

// --- Audio & ML Engine ---

/**
 * Initialize audio context and ML brain
 */
async function initEngine() {
    appState.audio.ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (appState.audio.ctx.state === 'suspended') {
        await appState.audio.ctx.resume();
    }

    appState.audio.analyser = appState.audio.ctx.createAnalyser();
    appState.audio.analyser.fftSize = CONSTANTS.FFT_SIZE;

    appState.ml.brain = ml5.neuralNetwork({
        inputs: ['loudness', 'pitch', 'brightness', 'roughness'],
        outputs: ['y1', 'y2', 'y3', 'y4', 'shape'],
        task: 'regression',
        debug: false
    });

    // Cache DOM elements
    cacheUIElements();

    appState.ui.elements.btnEngine.style.display = 'none';
    appState.ui.elements.btnMain.style.display = 'block';
    appState.ui.elements.btnUpload.style.display = 'block';
    appState.ui.elements.saveLoadZone.style.display = 'block';

    await loadTrainingData();
    initThree();
    updateAllUIText();
    updateStatus(translations[appState.ui.lang].statusReady, 'status-idle');
}

/**
 * Cache all DOM elements for better performance
 */
function cacheUIElements() {
    const elements = appState.ui.elements;
    elements.btnMain = document.getElementById('btn-main');
    elements.btnPlay = document.getElementById('btn-play');
    elements.btnConfirm = document.getElementById('btn-confirm');
    elements.btnUpload = document.getElementById('btn-upload');
    elements.btnEngine = document.getElementById('btn-engine');
    elements.y1 = document.getElementById('y1');
    elements.y2 = document.getElementById('y2');
    elements.y3 = document.getElementById('y3');
    elements.y4 = document.getElementById('y4');
    elements.shapeSelector = document.getElementById('shape-selector');
    elements.valLoud = document.getElementById('val-loud');
    elements.valPitch = document.getElementById('val-pitch');
    elements.valBright = document.getElementById('val-bright');
    elements.valRough = document.getElementById('val-rough');
    elements.dataCount = document.getElementById('data-count');
    elements.labelingZone = document.getElementById('labeling-zone');
    elements.saveLoadZone = document.getElementById('save-load-zone');
    elements.status = document.getElementById('status');
    elements.trainingOverlay = document.getElementById('training-overlay');
    elements.trainingMessage = document.getElementById('training-message');
    elements.trainingProgress = document.getElementById('training-progress');
}

/**
 * Handle record/stop button
 */
async function handleRecord() {
    const t = translations[appState.ui.lang];

    if (appState.audio.ctx && appState.audio.ctx.state === 'suspended') {
        await appState.audio.ctx.resume();
    }

    if (appState.ui.state === 'IDLE' || appState.ui.state === 'REVIEWING') {
        // Start recording
        appState.ui.state = 'RECORDING';
        appState.audio.chunks = [];
        appState.audio.recorded = {
            loudness: 0,
            pitch: 0,
            brightness: 0,
            roughness: 0,
            count: 0,
            // Track peak values for more distinctive shape classification
            peakLoudness: 0,
            peakPitch: 0,
            peakBrightness: 0,
            peakRoughness: 0
        };

        // Reset to Sphere when starting new recording
        appState.visuals.current.shape = 0;
        appState.visuals.target.shape = 0;
        appState.visuals.previousShape = -1;
        createShape(0);
        appState.ui.elements.shapeSelector.value = 0;
        updateShapeNameDisplay();

        if (appState.audio.audioTag) {
            appState.audio.audioTag.pause();
            appState.audio.isPlaying = false;
        }

        try {
            if (!appState.audio.stream) {
                appState.audio.stream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        echoCancellation: false,
                        noiseSuppression: false,
                        autoGainControl: false,
                        sampleRate: CONSTANTS.SAMPLE_RATE
                    }
                });

                appState.audio.microphone = appState.audio.ctx.createMediaStreamSource(appState.audio.stream);

                let mimeType = 'audio/webm';
                if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
                    mimeType = 'audio/webm;codecs=opus';
                }

                appState.audio.recorder = new MediaRecorder(appState.audio.stream, {
                    mimeType: mimeType,
                    audioBitsPerSecond: CONSTANTS.AUDIO_BITS_PER_SECOND
                });

                appState.audio.recorder.ondataavailable = e => {
                    if (e.data.size > 0) appState.audio.chunks.push(e.data);
                };

                appState.audio.recorder.onstop = saveRecording;
            }

            appState.audio.microphone.connect(appState.audio.analyser);
            appState.audio.recorder.start(CONSTANTS.RECORDER_TIMESLICE);
            appState.ui.elements.btnMain.innerText = t.btnStop;
            updateStatus(t.statusRecording, 'status-recording');
            appState.ui.elements.labelingZone.style.display = 'none';
        } catch (err) {
            alert('마이크 권한이 필요합니다.');
            appState.ui.state = 'IDLE';
        }
    } else {
        // Stop recording
        appState.audio.recorder.stop();
        appState.ui.state = 'REVIEWING';

        if (appState.audio.microphone) {
            appState.audio.microphone.disconnect();
        }

        appState.ui.elements.labelingZone.style.display = 'block';
        appState.ui.elements.btnPlay.style.display = 'block';
        appState.ui.elements.btnConfirm.style.display = 'block';

        updateAllUIText();
        updateStatus(t.statusReviewing, 'status-review');
    }
}

/**
 * Save recording and analyze
 */
function saveRecording() {
    let mimeType = 'audio/webm';
    if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
        mimeType = 'audio/webm;codecs=opus';
    }

    const blob = new Blob(appState.audio.chunks, { type: mimeType });
    const audioURL = createTrackedObjectURL(blob);
    appState.audio.audioTag = new Audio(audioURL);
    appState.audio.audioTag.loop = true;

    // Clear chunks after use
    appState.audio.chunks = [];

    if (appState.audio.recorded.count > 0) {
        // Average the recorded values
        appState.audio.recorded.loudness /= appState.audio.recorded.count;
        appState.audio.recorded.pitch /= appState.audio.recorded.count;
        appState.audio.recorded.brightness /= appState.audio.recorded.count;
        appState.audio.recorded.roughness /= appState.audio.recorded.count;

        // Blend average (30%) with peak (70%) for distinctive shape classification
        const blendedFeatures = {
            loudness: appState.audio.recorded.loudness * 0.3 + appState.audio.recorded.peakLoudness * 0.7,
            pitch: appState.audio.recorded.pitch * 0.3 + appState.audio.recorded.peakPitch * 0.7,
            brightness: appState.audio.recorded.brightness * 0.3 + appState.audio.recorded.peakBrightness * 0.7,
            roughness: appState.audio.recorded.roughness * 0.3 + appState.audio.recorded.peakRoughness * 0.7
        };

        // Store blended features for use in REVIEWING state
        appState.audio.recorded = blendedFeatures;

        console.log('📝 Recorded audio features:', appState.audio.recorded);
        console.log('🔢 Blended features (avg 30% + peak 70%):', blendedFeatures, 'normalized:', normalizeAudioFeatures(blendedFeatures));

        // Predict or classify shape
        if (appState.ml.trainingData.length > 0) {
            performAIPrediction(blendedFeatures, (prediction) => {
                if (prediction) {
                    appState.ui.elements.shapeSelector.value = prediction.shape;
                    appState.visuals.target.shape = prediction.shape;
                    appState.visuals.current.shape = prediction.shape;
                    appState.visuals.previousShape = -1;
                    createShape(prediction.shape);
                    console.log('🤖 AI predicted shape:', SHAPE_NAMES[prediction.shape]);
                    updateShapeNameDisplay();
                }
            });
        } else {
            const autoShape = autoClassifyShape(blendedFeatures);
            appState.ui.elements.shapeSelector.value = autoShape;
            appState.visuals.target.shape = autoShape;
            appState.visuals.current.shape = autoShape;
            appState.visuals.previousShape = -1;
            createShape(autoShape);
            console.log('✏️ Rule-based shape:', SHAPE_NAMES[autoShape]);
            updateShapeNameDisplay();
        }
    }
}

/**
 * Handle file upload
 */
async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const t = translations[appState.ui.lang];

    try {
        // Clean up previous audio
        if (appState.audio.audioTag) {
            appState.audio.audioTag.pause();
            appState.audio.audioTag = null;
        }

        if (appState.audio.sourceNode) {
            appState.audio.sourceNode.disconnect();
            appState.audio.sourceNode = null;
        }

        // Load the uploaded file
        const fileURL = createTrackedObjectURL(file);
        appState.audio.audioTag = new Audio(fileURL);
        appState.audio.audioTag.loop = true;

        await new Promise((resolve, reject) => {
            appState.audio.audioTag.onloadedmetadata = resolve;
            appState.audio.audioTag.onerror = reject;
        });

        console.log('Analyzing uploaded file:', file.name);

        appState.audio.recorded = {
            loudness: 0,
            pitch: 0,
            brightness: 0,
            roughness: 0,
            count: 0,
            // Track peak values for more distinctive shape classification
            peakLoudness: 0,
            peakPitch: 0,
            peakBrightness: 0,
            peakRoughness: 0
        };

        // Create source node and analyze
        appState.audio.sourceNode = appState.audio.ctx.createMediaElementSource(appState.audio.audioTag);
        appState.audio.sourceNode.connect(appState.audio.analyser);
        appState.audio.analyser.connect(appState.audio.ctx.destination);

        await appState.audio.audioTag.play();
        appState.audio.isPlaying = true;

        // Analyze for specified duration
        const maxCount = CONSTANTS.ANALYZE_DURATION / CONSTANTS.ANALYZE_INTERVAL;
        let analyzeCount = 0;

        const analyzeTimer = setInterval(() => {
            const frequencyData = new Uint8Array(appState.audio.analyser.frequencyBinCount);
            const timeDomainData = new Uint8Array(appState.audio.analyser.frequencyBinCount);
            appState.audio.analyser.getByteFrequencyData(frequencyData);
            appState.audio.analyser.getByteTimeDomainData(timeDomainData);

            const features = calculateAudioFeatures(frequencyData, timeDomainData);

            appState.audio.recorded.loudness += features.loudness;
            appState.audio.recorded.pitch += features.pitch;
            appState.audio.recorded.brightness += features.brightness;
            appState.audio.recorded.roughness += features.roughness;
            appState.audio.recorded.count++;

            // Track peak values
            appState.audio.recorded.peakLoudness = Math.max(appState.audio.recorded.peakLoudness, features.loudness);
            appState.audio.recorded.peakPitch = Math.max(appState.audio.recorded.peakPitch, features.pitch);
            appState.audio.recorded.peakBrightness = Math.max(appState.audio.recorded.peakBrightness, features.brightness);
            appState.audio.recorded.peakRoughness = Math.max(appState.audio.recorded.peakRoughness, features.roughness);

            analyzeCount++;
            if (analyzeCount >= maxCount) {
                clearInterval(analyzeTimer);

                if (appState.audio.recorded.count > 0) {
                    appState.audio.recorded.loudness /= appState.audio.recorded.count;
                    appState.audio.recorded.pitch /= appState.audio.recorded.count;
                    appState.audio.recorded.brightness /= appState.audio.recorded.count;
                    appState.audio.recorded.roughness /= appState.audio.recorded.count;
                }

                // Blend average (30%) with peak (70%) for distinctive shape classification
                const blendedFeatures = {
                    loudness: appState.audio.recorded.loudness * 0.3 + appState.audio.recorded.peakLoudness * 0.7,
                    pitch: appState.audio.recorded.pitch * 0.3 + appState.audio.recorded.peakPitch * 0.7,
                    brightness: appState.audio.recorded.brightness * 0.3 + appState.audio.recorded.peakBrightness * 0.7,
                    roughness: appState.audio.recorded.roughness * 0.3 + appState.audio.recorded.peakRoughness * 0.7
                };

                // Store blended features for use in REVIEWING state
                appState.audio.recorded = blendedFeatures;

                console.log('File analysis complete:', appState.audio.recorded);
                console.log('🔢 Blended features (avg 30% + peak 70%):', blendedFeatures, 'normalized:', normalizeAudioFeatures(blendedFeatures));

                const autoShape = autoClassifyShape(blendedFeatures);
                appState.ui.elements.shapeSelector.value = autoShape;
                appState.visuals.target.shape = autoShape;
                appState.visuals.current.shape = autoShape;
                appState.visuals.previousShape = -1;
                createShape(autoShape);
                updateShapeNameDisplay();

                console.log('🎯 Auto-classified shape for uploaded file:', SHAPE_NAMES[autoShape]);

                appState.audio.audioTag.pause();
                appState.audio.audioTag.currentTime = 0;
                appState.audio.isPlaying = false;

                appState.ui.state = 'REVIEWING';
                appState.ui.elements.labelingZone.style.display = 'block';
                appState.ui.elements.btnPlay.style.display = 'block';
                appState.ui.elements.btnConfirm.style.display = 'block';
                updateAllUIText();
                updateStatus(t.statusReviewing, 'status-review');
            }
        }, CONSTANTS.ANALYZE_INTERVAL);

        event.target.value = '';

    } catch (err) {
        console.error('File upload error:', err);
        alert(appState.ui.lang === 'KR' ? '파일 로드 실패: ' + err.message : 'File load failed: ' + err.message);
        event.target.value = '';
    }
}

/**
 * Toggle audio playback
 */
function togglePlayback() {
    const t = translations[appState.ui.lang];
    if (!appState.audio.audioTag) return;

    if (appState.audio.audioTag.paused) {
        if (!appState.audio.sourceNode) {
            appState.audio.sourceNode = appState.audio.ctx.createMediaElementSource(appState.audio.audioTag);
            appState.audio.sourceNode.connect(appState.audio.analyser);
            appState.audio.analyser.connect(appState.audio.ctx.destination);
        }
        appState.audio.audioTag.play();
        appState.audio.isPlaying = true;
        appState.ui.elements.btnPlay.innerText = t.btnPause;
    } else {
        appState.audio.audioTag.pause();
        appState.audio.isPlaying = false;
        appState.ui.elements.btnPlay.innerText = t.btnPlay;
    }
}

/**
 * Main animation loop
 */
function animate() {
    requestAnimationFrame(animate);

    if (appState.audio.analyser && appState.audio.ctx && appState.audio.ctx.state === 'running') {
        analyzeAudio();
    }

    appState.visuals.uniforms.uTime.value += CONSTANTS.TIME_INCREMENT;
    // Clamp loudness to prevent graphics from becoming too large
    appState.visuals.uniforms.uLoudness.value = Math.min(appState.audio.features.loudness, 0.8);

    // Update target visuals based on state
    updateTargetVisuals();

    // Smooth interpolation with threshold-based optimization
    const lerpSpeed = appState.ui.state === 'REVIEWING' ?
        CONSTANTS.LERP_SPEED_REVIEWING : CONSTANTS.LERP_SPEED_LIVE;

    const params = ['y1', 'y2', 'y3', 'y4', 'shape'];
    params.forEach(param => {
        const delta = appState.visuals.target[param] - appState.visuals.current[param];
        if (Math.abs(delta) > CONSTANTS.LERP_THRESHOLD) {
            appState.visuals.current[param] += delta * lerpSpeed;
        }
    });

    // Shape change detection (optimized)
    const roundedShape = Math.round(appState.visuals.current.shape);
    if (roundedShape !== appState.visuals.previousShape && roundedShape >= 0 && roundedShape <= 4) {
        appState.visuals.previousShape = roundedShape;
        createShape(roundedShape);
        updateShapeNameDisplay();
    }

    // Update shader uniforms directly (optimized)
    appState.visuals.uniforms.uY1.value = appState.visuals.current.y1;
    appState.visuals.uniforms.uY2.value = appState.visuals.current.y2;
    appState.visuals.uniforms.uY3.value = appState.visuals.current.y3;
    appState.visuals.uniforms.uY4.value = appState.visuals.current.y4;

    if (appState.visuals.mesh) {
        appState.visuals.mesh.rotation.y += CONSTANTS.ROTATION_SPEED;
    }

    if (appState.visuals.renderer && appState.visuals.scene && appState.visuals.camera) {
        appState.visuals.renderer.render(appState.visuals.scene, appState.visuals.camera);
    }
}

/**
 * Analyze audio and update display (throttled)
 */
function analyzeAudio() {
    const frequencyData = new Uint8Array(appState.audio.analyser.frequencyBinCount);
    const timeDomainData = new Uint8Array(appState.audio.analyser.frequencyBinCount);
    appState.audio.analyser.getByteFrequencyData(frequencyData);
    appState.audio.analyser.getByteTimeDomainData(timeDomainData);

    if (appState.ui.state === 'REVIEWING' && !appState.audio.isPlaying) {
        appState.audio.features = { ...appState.audio.recorded };
    } else {
        const features = calculateAudioFeatures(frequencyData, timeDomainData);
        appState.audio.features = features;

        if (appState.ui.state === 'RECORDING') {
            appState.audio.recorded.loudness += features.loudness;
            appState.audio.recorded.pitch += features.pitch;
            appState.audio.recorded.brightness += features.brightness;
            appState.audio.recorded.roughness += features.roughness;
            appState.audio.recorded.count++;

            // Track peak values for distinctive moments
            appState.audio.recorded.peakLoudness = Math.max(appState.audio.recorded.peakLoudness, features.loudness);
            appState.audio.recorded.peakPitch = Math.max(appState.audio.recorded.peakPitch, features.pitch);
            appState.audio.recorded.peakBrightness = Math.max(appState.audio.recorded.peakBrightness, features.brightness);
            appState.audio.recorded.peakRoughness = Math.max(appState.audio.recorded.peakRoughness, features.roughness);
        }
    }

    // Throttle DOM updates for audio display
    const now = Date.now();
    if (now - appState.ui.lastAudioDisplayUpdate > CONSTANTS.DOM_UPDATE_INTERVAL) {
        const shouldUpdate =
            Math.abs(appState.audio.features.loudness - appState.ui.lastAudioValues.loudness) > CONSTANTS.DOM_UPDATE_THRESHOLD ||
            Math.abs(appState.audio.features.pitch - appState.ui.lastAudioValues.pitch) > CONSTANTS.DOM_UPDATE_THRESHOLD ||
            Math.abs(appState.audio.features.brightness - appState.ui.lastAudioValues.brightness) > CONSTANTS.DOM_UPDATE_THRESHOLD ||
            Math.abs(appState.audio.features.roughness - appState.ui.lastAudioValues.roughness) > CONSTANTS.DOM_UPDATE_THRESHOLD;

        if (shouldUpdate) {
            if (appState.ui.elements.valLoud) {
                appState.ui.elements.valLoud.innerText = appState.audio.features.loudness.toFixed(2);
            }
            if (appState.ui.elements.valPitch) {
                appState.ui.elements.valPitch.innerText = appState.audio.features.pitch.toFixed(2);
            }
            if (appState.ui.elements.valBright) {
                appState.ui.elements.valBright.innerText = appState.audio.features.brightness.toFixed(2);
            }
            if (appState.ui.elements.valRough) {
                appState.ui.elements.valRough.innerText = appState.audio.features.roughness.toFixed(2);
            }

            appState.ui.lastAudioValues = { ...appState.audio.features };
            appState.ui.lastAudioDisplayUpdate = now;
        }
    }
}

/**
 * Confirm training data and train model
 */
function confirmTrainingWrapper() {
    if (!appState.ml.brain) {
        console.error('Brain not initialized');
        return;
    }

    // Validate slider inputs
    const y1 = parseFloat(appState.ui.elements.y1.value);
    const y2 = parseFloat(appState.ui.elements.y2.value);
    const y3 = parseFloat(appState.ui.elements.y3.value);
    const y4 = parseFloat(appState.ui.elements.y4.value);
    const shape = parseInt(appState.ui.elements.shapeSelector.value);

    if (isNaN(y1) || isNaN(y2) || isNaN(y3) || isNaN(y4) || isNaN(shape)) {
        console.error('Invalid slider values');
        return;
    }

    if (y1 < 0 || y1 > 1 || y2 < 0 || y2 > 1 || y3 < 0 || y3 > 1 || y4 < 0 || y4 > 1) {
        console.error('Slider values out of range');
        return;
    }

    if (shape < 0 || shape > 5) {
        console.error('Invalid shape value');
        return;
    }

    const labels = [y1, y2, y3, y4, shape / 5.0];

    appState.ml.brain.addData(
        [
            appState.audio.recorded.loudness,
            appState.audio.recorded.pitch,
            appState.audio.recorded.brightness,
            appState.audio.recorded.roughness
        ],
        labels
    );

    appState.ml.trainingData.push({
        x: { ...appState.audio.recorded },
        y: labels
    });

    saveTrainingData();

    if (appState.ui.elements.dataCount) {
        appState.ui.elements.dataCount.innerText = appState.ml.trainingData.length;
    }

    // Hide UI elements before training
    appState.ui.elements.labelingZone.style.display = 'none';
    appState.ui.elements.btnPlay.style.display = 'none';
    appState.ui.elements.btnConfirm.style.display = 'none';
    appState.ui.state = 'IDLE';
    updateStatus(translations[appState.ui.lang].statusReady, 'status-idle');
    updateAllUIText();

    // Normalize and train
    appState.ml.brain.normalizeData();

    // Show training overlay
    if (appState.ui.elements.trainingOverlay) {
        appState.ui.elements.trainingOverlay.style.display = 'flex';
        appState.ui.elements.trainingMessage.textContent =
            appState.ui.lang === 'KR' ? 'AI 모델 학습 중...' : 'Training AI Model...';
        appState.ui.elements.trainingProgress.textContent =
            appState.ui.lang === 'KR' ? '잠시만 기다려주세요 (50 epochs)' : 'Please wait (50 epochs)';
    }

    setTimeout(() => {
        console.log('Starting training in background...');

        try {
            appState.ml.brain.train({ epochs: CONSTANTS.TRAINING_EPOCHS }, () => {
                appState.ml.isTrained = true;
                console.log('Training complete! AI mode enabled.');

                setTimeout(() => {
                    location.reload();
                }, CONSTANTS.RELOAD_DELAY);
            });
        } catch (err) {
            console.error('Training failed:', err);
            if (appState.ui.elements.trainingOverlay) {
                appState.ui.elements.trainingOverlay.style.display = 'none';
            }
            alert(appState.ui.lang === 'KR' ? '학습 실패: ' + err.message : 'Training failed: ' + err.message);
        }
    }, CONSTANTS.TRAINING_DELAY);
}

/**
 * Save training data to localStorage and server
 */
async function saveTrainingData() {
    try {
        const dataToSave = JSON.stringify({ data: appState.ml.trainingData });
        localStorage.setItem('soundTo3D_data', dataToSave);
        console.log('Training data saved to localStorage:', appState.ml.trainingData.length, 'samples');

        if (USE_SERVER) {
            const lastData = appState.ml.trainingData[appState.ml.trainingData.length - 1];

            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000); // 5s timeout

            try {
                const response = await fetch(`${API_URL}/api/data`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(lastData),
                    signal: controller.signal
                });
                clearTimeout(timeoutId);

                const result = await response.json();
                console.log('Training data saved to server:', result.count, 'total samples');
            } catch (fetchErr) {
                if (fetchErr.name === 'AbortError') {
                    console.warn('Server save timed out');
                } else {
                    throw fetchErr;
                }
            }
        }
    } catch (e) {
        console.error('Failed to save training data:', e);
    }
}

/**
 * Load training data from server or localStorage
 */
async function loadTrainingData() {
    try {
        if (USE_SERVER) {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000);

            try {
                const response = await fetch(`${API_URL}/api/data`, {
                    signal: controller.signal
                });
                clearTimeout(timeoutId);

                const result = await response.json();
                if (result.success && result.data.length > 0) {
                    appState.ml.trainingData = result.data;
                    console.log('Training data loaded from server:', appState.ml.trainingData.length, 'samples');

                    const dataToSave = JSON.stringify({ data: appState.ml.trainingData });
                    localStorage.setItem('soundTo3D_data', dataToSave);
                }
            } catch (fetchErr) {
                if (fetchErr.name === 'AbortError') {
                    console.warn('Server load timed out, using localStorage');
                } else {
                    throw fetchErr;
                }
            }
        }

        // Fallback to localStorage
        if (appState.ml.trainingData.length === 0) {
            const saved = localStorage.getItem('soundTo3D_data');
            if (saved) {
                const obj = JSON.parse(saved);
                appState.ml.trainingData = obj.data || [];
                console.log('Training data loaded from localStorage:', appState.ml.trainingData.length, 'samples');
            }
        }

        if (appState.ui.elements.dataCount) {
            appState.ui.elements.dataCount.innerText = appState.ml.trainingData.length;
        }

        if (appState.ml.brain && appState.ml.trainingData.length > 0) {
            console.log('Loading previous training data into brain...');
            appState.ml.trainingData.forEach(i =>
                appState.ml.brain.addData(
                    [i.x.loudness, i.x.pitch, i.x.brightness, i.x.roughness],
                    i.y
                )
            );
            appState.ml.brain.normalizeData();
            console.log('Data loaded. Model will train when you add new data.');
        }
    } catch (e) {
        console.error('Failed to load training data:', e);

        // Final fallback
        const saved = localStorage.getItem('soundTo3D_data');
        if (saved) {
            try {
                const obj = JSON.parse(saved);
                appState.ml.trainingData = obj.data || [];
                console.log('Fallback: Training data loaded from localStorage:', appState.ml.trainingData.length, 'samples');
            } catch (parseErr) {
                console.error('Failed to parse localStorage data:', parseErr);
            }
        }
    }
}

/**
 * Clear all training data
 */
async function clearAllData() {
    const message = appState.ui.lang === 'KR' ?
        "정말로 모든 학습 데이터를 삭제하시겠습니까?" :
        "Are you sure you want to delete all training data?";

    if (confirm(message)) {
        try {
            localStorage.removeItem('soundTo3D_data');
            appState.ml.trainingData = [];
            appState.ml.isTrained = false;

            if (USE_SERVER) {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 5000);

                try {
                    await fetch(`${API_URL}/api/data`, {
                        method: 'DELETE',
                        signal: controller.signal
                    });
                    clearTimeout(timeoutId);
                    console.log('Server data cleared');
                } catch (fetchErr) {
                    console.warn('Server clear failed or timed out');
                }
            }

            location.reload();
        } catch (e) {
            console.error('Failed to clear data:', e);
            location.reload();
        }
    }
}

// --- UI & Translation ---

const translations = {
    KR: {
        title: "IML Research",
        btnEngine: "오디오 엔진 가동",
        btnRecord: "녹음 시작",
        btnStop: "중단",
        btnReRecord: "다시 녹음",
        btnPlay: "소리 재생",
        btnPause: "재생 중지",
        btnConfirm: "데이터 확정 및 학습",
        btnExport: "CSV 추출",
        btnUpload: "파일 업로드",
        btnClear: "모든 데이터 삭제",
        statusReady: "준비 완료",
        statusRecording: "녹음 중...",
        statusReviewing: "검토 및 라벨링",
        labelLoud: "음량",
        labelPitch: "음높이",
        labelBright: "밝기",
        labelRough: "거칠기",
        y1Label: "y1: 각짐",
        y1Left: "둥근",
        y1Right: "각진",
        y2Label: "y2: 뾰족함",
        y2Left: "부드러운",
        y2Right: "뾰족한",
        y3Label: "y3: 거칠기",
        y3Left: "매끈한",
        y3Right: "거친",
        y4Label: "y4: 복잡도",
        y4Left: "단순",
        y4Right: "복잡",
        shapeLabel: "기본 형태",
        dataLabel: "학습 데이터:",
        samplesLabel: "개",
        shapeNames: ['구', '정육면체', '토러스', '원뿔', '원기둥']
    },
    EN: {
        title: "IML Research",
        btnEngine: "Start Engine",
        btnRecord: "Record",
        btnStop: "Stop",
        btnReRecord: "Re-record",
        btnPlay: "Play",
        btnPause: "Pause",
        btnConfirm: "Confirm & Train",
        btnExport: "Export CSV",
        btnUpload: "Upload File",
        btnClear: "Clear All Data",
        statusReady: "Ready",
        statusRecording: "Recording...",
        statusReviewing: "Reviewing...",
        labelLoud: "Loudness",
        labelPitch: "Pitch",
        labelBright: "Brightness",
        labelRough: "Roughness",
        y1Label: "y1: Angularity",
        y1Left: "Round",
        y1Right: "Angular",
        y2Label: "y2: Spikiness",
        y2Left: "Smooth",
        y2Right: "Spiky",
        y3Label: "y3: Texture",
        y3Left: "Sleek",
        y3Right: "Rough",
        y4Label: "y4: Density",
        y4Left: "Simple",
        y4Right: "Complex",
        shapeLabel: "Base Shape",
        dataLabel: "Data:",
        samplesLabel: "samples",
        shapeNames: ['Sphere', 'Cube', 'Torus', 'Cone', 'Cylinder']
    }
};

function toggleLanguage() {
    appState.ui.lang = appState.ui.lang === 'KR' ? 'EN' : 'KR';
    updateAllUIText();
}

function updateAllUIText() {
    const t = translations[appState.ui.lang];
    const langBtn = document.getElementById('lang-toggle');
    if (langBtn) langBtn.innerText = appState.ui.lang === 'KR' ? 'EN' : 'KR';

    const mapping = {
        'title': t.title,
        'btn-engine': t.btnEngine,
        'btn-confirm': t.btnConfirm,
        'btn-play': appState.audio.isPlaying ? t.btnPause : t.btnPlay,
        'btn-upload': t.btnUpload,
        'btn-export': t.btnExport,
        'btn-clear': t.btnClear,
        'label-loud': t.labelLoud,
        'label-pitch': t.labelPitch,
        'label-bright': t.labelBright,
        'label-rough': t.labelRough,
        'y1-label': t.y1Label,
        'y1-left': t.y1Left,
        'y1-right': t.y1Right,
        'y2-label': t.y2Label,
        'y2-left': t.y2Left,
        'y2-right': t.y2Right,
        'y3-label': t.y3Label,
        'y3-left': t.y3Left,
        'y3-right': t.y3Right,
        'y4-label': t.y4Label,
        'y4-left': t.y4Left,
        'y4-right': t.y4Right,
        'shape-label': t.shapeLabel,
        'data-label': t.dataLabel,
        'samples-label': t.samplesLabel
    };

    for (let id in mapping) {
        const el = document.getElementById(id);
        if (el) el.innerText = mapping[id];
    }

    const mainBtn = document.getElementById('btn-main');
    if (mainBtn) {
        if (appState.ui.state === 'IDLE') mainBtn.innerText = t.btnRecord;
        else if (appState.ui.state === 'RECORDING') mainBtn.innerText = t.btnStop;
        else if (appState.ui.state === 'REVIEWING') mainBtn.innerText = t.btnReRecord;
    }

    updateShapeNameDisplay();
}

function updateShapeNameDisplay() {
    const t = translations[appState.ui.lang];
    const shapeSelector = document.getElementById('shape-selector');
    if (!shapeSelector) return;
    const shapeName = document.getElementById('shape-name');
    if (shapeName) {
        shapeName.innerText = t.shapeNames[parseInt(shapeSelector.value)];
    }
}

function updateStatus(msg, cls) {
    const statusEl = document.getElementById('status');
    if (statusEl) {
        statusEl.innerText = msg;
        statusEl.className = 'status-badge ' + cls;
    }
}

function changeShape(val) {
    updateShapeNameDisplay();
    createShape(val);
}

function exportCSV() {
    let csv = "loudness,pitch,brightness,roughness,y1,y2,y3,y4,shape\n";
    appState.ml.trainingData.forEach(d => {
        csv += `${d.x.loudness},${d.x.pitch},${d.x.brightness},${d.x.roughness},${d.y.join(',')}\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = createTrackedObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'IML_Data.csv';
    a.click();
}

window.addEventListener('DOMContentLoaded', () => {
    updateAllUIText();
});
