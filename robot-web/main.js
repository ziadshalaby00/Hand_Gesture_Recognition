import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

// ==============
// Three.js Scene, Camera, and Renderer Initialization
// ==============
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ canvas: document.getElementById("scene"), antialias: true });

const cameraFPS = document.getElementById("pipeline-fps");
const inferenceMS = document.getElementById("inference-ms");
const processingMS = document.getElementById("processing-ms");

renderer.setSize(window.innerWidth, window.innerHeight * 0.55);
camera.position.set(0, 1, 5);

// ==============
// Lighting Setup
// ==============
scene.add(new THREE.AmbientLight(0xffffff, 0.8));
const light = new THREE.DirectionalLight(0xffffff, 1);
light.position.set(2, 5, 5);
scene.add(light);

// ==============
// Global Animation and Robot State Variables
// ==============
let robot;
let mixer;
let animations = {}; 
let currentActionName = '';
const clock = new THREE.Clock();
const loader = new GLTFLoader();

let targetX = 0;
let moveSpeed = 0.05;
const STEP = 0.1;

// ==============
// Animation File Paths Configuration
// ==============
const animationFiles = [
    { name: 'run', url: './source/Fast Run.glb' },
    { name: 'punch', url: './source/Punching Bag.glb' }, 
    { name: 'dance', url: './source/Rumba Dancing.glb' }, 
    { name: 'sit', url: './source/Sitting.glb' }, 
    { name: 'walk-left', url: './source/Walk Strafe Left.glb' }, 
    { name: 'walk-right', url: './source/Walk Strafe right.glb' }, 
    { name: 'Walking', url: './source/Walking.glb' }, 
];

// ==============
// Base Robot Model Loading and Setup
// ==============
loader.load("./source/sin_nombre1.glb", (gltf) => {
    robot = gltf.scene;
    
    const box = new THREE.Box3().setFromObject(robot);
    const center = box.getCenter(new THREE.Vector3());
    robot.scale.set(0.5, 0.5, 0.5);
    robot.position.sub(center);
    robot.position.y = -1; 

    scene.add(robot);

    mixer = new THREE.AnimationMixer(robot);

    // ==============
    // Async Loading of Animation Clips
    // ==============
    const loadPromises = animationFiles.map(file => {
        return new Promise((resolve) => {
            loader.load(file.url, (animGltf) => {
                if (animGltf.animations && animGltf.animations.length > 0) {
                    const clip = animGltf.animations[0];
                    const action = mixer.clipAction(clip);
                    animations[file.name] = action;
                }
                resolve();
            }, undefined, (error) => {
                console.error(`Error loading: ${file.url}`, error);
                resolve();
            });
        });
    });

    Promise.all(loadPromises).then(async () => {

    });
});

// ==============
// Animation Blending and Playback Control
// ==============
function playAction(name) {
    if (currentActionName === name || !animations[name]) return;

    const newAction = animations[name];
    const oldAction = animations[currentActionName];

    newAction.reset();
    newAction.enabled = true;
    newAction.setEffectiveTimeScale(1);
    newAction.setEffectiveWeight(1);

    if (oldAction) {
        newAction.crossFadeFrom(oldAction, 0.75, true);
    }
    
    newAction.play();
    currentActionName = name;
}

// ==============
// Idle State: Fade Out All Animations
// ==============
function idleRobot() {
    Object.values(animations).forEach(action => action.fadeOut(0));
    currentActionName = '';
}

// ==============
// Main Render Loop with Position Interpolation
// ==============
function animate() {
    requestAnimationFrame(animate);

    const delta = clock.getDelta();

    if (mixer) {
        mixer.update(delta);
    }

    if (robot) {
        robot.position.x = THREE.MathUtils.lerp(
            robot.position.x,
            targetX,
            0.08
        );
    }
    renderer.render(scene, camera);
}

animate();

// ==============
// Socket.IO Client Connection Setup
// ==============
const socket = io("http://localhost:8765");

socket.on("connect", () => {
    console.log("Connected to Socket.IO ✔️");
});

// ==============
// DOM Element References for UI Updates
// ==============
const img = document.getElementById("frame");
const frameInfo = document.getElementById("frame-info");

let lastUIUpdate = 0;
// ==============
// Real-time Hand Data Handler: Animation Trigger and UI Rendering
// ==============
socket.on("hand_data", (data) => {
    if (data) {
        if(data["robot_action"] === 'stop' || data["hand_present"] !== 'Yes') idleRobot();
        else playAction(data["robot_action"]);

        if (data["robot_action"] === "walk-right") {
            targetX -= STEP;
        }

        if (data["robot_action"] === "walk-left") {
            targetX += STEP;
        }

        const now = performance.now();
        if (now - lastUIUpdate > 500) {
            cameraFPS.textContent = data.pipeline_fps;
            inferenceMS.textContent = `${data.inference_ms} ms`;
            processingMS.textContent = `${data.processing_ms} ms`;

            lastUIUpdate = now;
        }

        let hand_present_color =
            data["hand_present"] === "Yes"
                ? "rgb(20, 238, 12)"
                : "#c12323";

        let status_text_color =
            data["status_text"] === "Open"
                ? "rgb(20, 238, 12)"
                : "#c12323";

        frameInfo.innerHTML = `
            <span class="sp-1">Hand Present:
                <span class="sp-2" style="color: ${hand_present_color}">
                    ${data["hand_present"]}
                </span>
            </span>

            <span class="sp-1">Status Text:
                <span class="sp-2" style="color: ${status_text_color}">
                    ${data["status_text"]}
                </span>
            </span>

            <span class="sp-1">Fingers Count:
                <span class="sp-2" style="color: #0ec7e7;">
                    ${data["fingers_count"]}
                </span>
            </span>

            <span class="sp-1">Robot Action:
                <span class="sp-2" style="color: #0ec7e7;">
                    ${data["robot_action"]}
                </span>
            </span>
        `;
    }

    console.log(data);
});

// ==============
// Reference: Supported Robot Action Types
// ==============
// Action Type
// ['walk-right', 'Walking', 'dance', 'sit', 'punch', 'walk-left', 'run']