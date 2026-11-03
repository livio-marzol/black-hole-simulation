import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

// ============================================================================
// 1. DETECÇÃO DE DISPOSITIVO E PARÂMETROS GLOBAIS DA SIMULAÇÃO
// ============================================================================
const isMobileDevice = /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent) || window.innerWidth <= 768;

const PARAMS = {
  mass: 1.0,              // Multiplicador de massa do Buraco Negro (1.0 = 100 Massas Solares)
  baseRs: 2.2,            // Raio de Schwarzschild base em unidades de mundo
  lensing: 1.0,           // Intensidade da curvatura geodésica da luz
  diskSpeed: 1.0,         // Velocidade de rotação kepleriana do disco de acreção
  doppler: 1.0,           // Intensidade do efeito Doppler relativístico (beaming)
  diskDensity: 1.0,       // Densidade óptica do plasma no disco
  spawnRate: 1.2,         // Taxa de surgimento automático de astros por segundo
  spaghettification: 1.0, // Multiplicador da força de maré (esticamento radial)
  bloomStrength: 0.65,    // Intensidade do pós-processamento UnrealBloom
  // Começa com qualidade alta (0.85x) e só reduz dinamicamente se o hardware pedir!
  renderScale: 0.85,
  maxSteps: 56,           // Passos geodésicos no shader (ajustado apenas se FPS cair)
  autoQuality: true,      // Governador automático de qualidade baseado no FPS real
  swallowedCount: 0,
  absorbedMassBonus: 0.0
};

function getEffectiveMass() {
  return PARAMS.mass + PARAMS.absorbedMassBonus;
}

function getSchwarzschildRadius() {
  return PARAMS.baseRs * getEffectiveMass();
}

// Ajusta a distância da câmera em telas verticais (retrato, ex: Moto G9 Power 20:9)
// para que o Disco de Acreção inteiro caiba perfeitamente na largura da tela.
function getResponsiveCameraPos(baseVec) {
  const aspect = window.innerWidth / Math.max(window.innerHeight, 1);
  if (aspect < 1.0) {
    const portraitZoomOut = Math.min(1.0 / Math.pow(aspect, 0.58), 1.85);
    return baseVec.clone().multiplyScalar(portraitZoomOut);
  }
  return baseVec.clone();
}

// ============================================================================
// 2. INICIALIZAÇÃO DO RENDERIZADOR, EXTENSÕES MOBILE E CÂMERA
// ============================================================================
const container = document.getElementById('canvas-container');
const renderer = new THREE.WebGLRenderer({
  antialias: false,
  powerPreference: 'high-performance',
  stencil: false,
  depth: false
});

// Ativa extensões de buffer float para GPUs mobile (Adreno 610 / Mali)
renderer.extensions.get('EXT_color_buffer_float');
renderer.extensions.get('EXT_color_buffer_half_float');

renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(PARAMS.renderScale);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
container.appendChild(renderer.domElement);

const orbitalScene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(
  52,
  window.innerWidth / window.innerHeight,
  0.1,
  350.0
);
camera.position.copy(getResponsiveCameraPos(new THREE.Vector3(0.0, 5.2, 34.0)));

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.minDistance = 5.5;
controls.maxDistance = 150.0;
controls.target.set(0, 0, 0);

// ============================================================================
// 3. RENDER TARGET RGBA16F (RGB = Cor HDR, A = Distância Euclidiana Assinada)
// ============================================================================
function createSceneRenderTarget(width, height) {
  return new THREE.WebGLRenderTarget(width, height, {
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
    format: THREE.RGBAFormat,
    type: THREE.HalfFloatType,
    depthBuffer: true,
    stencilBuffer: false
  });
}

let sceneRenderTarget = createSceneRenderTarget(
  Math.max(1, Math.floor(window.innerWidth * PARAMS.renderScale)),
  Math.max(1, Math.floor(window.innerHeight * PARAMS.renderScale))
);

// ============================================================================
// 4. SHADER DE RAYMARCHING GEODÉSICO (COMPATÍVEL COM ADRENO 610 & DESKTOP)
// ============================================================================
const BlackHoleRaymarchShader = {
  uniforms: {
    tSceneData: { value: null },
    uResolution: { value: new THREE.Vector2(window.innerWidth * PARAMS.renderScale, window.innerHeight * PARAMS.renderScale) },
    uCameraPos: { value: new THREE.Vector3() },
    uInvProjMatrix: { value: new THREE.Matrix4() },
    uCameraWorldMatrix: { value: new THREE.Matrix4() },
    uTime: { value: 0.0 },
    uRs: { value: getSchwarzschildRadius() },
    uLensing: { value: PARAMS.lensing },
    uDiskSpeed: { value: PARAMS.diskSpeed },
    uDoppler: { value: PARAMS.doppler },
    uDiskDensity: { value: PARAMS.diskDensity },
    uFlashIntensity: { value: 0.0 },
    uFlashPos: { value: new THREE.Vector3(1, 0, 0) },
    uMaxSteps: { value: PARAMS.maxSteps }
  },

  vertexShader: /* glsl */ `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,

  fragmentShader: /* glsl */ `
    precision highp float;

    varying vec2 vUv;

    uniform sampler2D tSceneData;
    uniform vec2 uResolution;
    uniform vec3 uCameraPos;
    uniform mat4 uInvProjMatrix;
    uniform mat4 uCameraWorldMatrix;
    uniform float uTime;
    uniform float uRs;
    uniform float uLensing;
    uniform float uDiskSpeed;
    uniform float uDoppler;
    uniform float uDiskDensity;
    uniform float uFlashIntensity;
    uniform vec3 uFlashPos;
    uniform int uMaxSteps;

    #define HARD_MAX_STEPS 60

    float hash12(vec2 p) {
      vec3 p3 = fract(vec3(p.xyx) * 0.1031);
      p3 += dot(p3, p3.yzx + 33.33);
      return fract((p3.x + p3.y) * p3.z);
    }

    float hash13(vec3 p3) {
      p3 = fract(p3 * 0.1031);
      p3 += dot(p3, p3.zyx + 31.32);
      return fract((p3.x + p3.y) * p3.z);
    }

    // Ruído 2D C2-contínuo (quintic) — zero aspecto granulado ou pixelado
    float smoothNoise2D(vec2 p) {
      vec2 i = floor(p);
      vec2 f = fract(p);
      vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
      float a = hash12(i);
      float b = hash12(i + vec2(1.0, 0.0));
      float c = hash12(i + vec2(0.0, 1.0));
      float d = hash12(i + vec2(1.0, 1.0));
      return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
    }

    float diskFbm(vec2 p) {
      float v = 0.65 * smoothNoise2D(p);
      mat2 rot = mat2(0.8, 0.6, -0.6, 0.8);
      v += 0.35 * smoothNoise2D(rot * p * 2.1 + vec2(17.3, 9.2));
      return v;
    }

    vec3 blackbodyDopplerColor(float tempFactor, float freqShift) {
      float t = clamp(tempFactor * pow(freqShift, 0.85), 0.05, 2.65);
      vec3 deepRed   = vec3(0.55, 0.06, 0.01);
      vec3 warmAmber = vec3(0.98, 0.38, 0.06);
      vec3 hotGold   = vec3(1.00, 0.80, 0.42);
      vec3 blueWhite = vec3(0.72, 0.88, 1.32);

      vec3 col = mix(deepRed, warmAmber, smoothstep(0.05, 0.52, t));
      col = mix(col, hotGold, smoothstep(0.52, 1.15, t));
      col = mix(col, blueWhite, smoothstep(1.15, 2.25, t));
      return col;
    }

    vec3 sampleStarfieldAndNebula(vec3 dir) {
      vec3 d = normalize(dir);

      vec3 galAxis = normalize(vec3(0.25, 0.88, -0.40));
      float galLat = abs(dot(d, galAxis));
      float galacticBand = exp(-galLat * 5.5);

      vec2 skyUv = vec2(atan(d.z, d.x), d.y * 2.5);
      float n1 = smoothNoise2D(skyUv * 2.5 + vec2(1.2, 3.4));
      float n2 = smoothNoise2D(skyUv * 5.5 - vec2(2.1, 0.8));
      float neb = 0.62 * n1 + 0.38 * n2;

      vec3 bg = vec3(0.004, 0.007, 0.014);
      bg += vec3(0.10, 0.04, 0.09) * smoothstep(0.35, 0.82, neb) * galacticBand;
      bg += vec3(0.02, 0.06, 0.13) * smoothstep(0.40, 0.85, n2) * (0.35 + 0.65 * galacticBand);

      vec3 starGrid = d * 120.0;
      vec3 cell = floor(starGrid);
      float rnd = hash13(cell);
      if (rnd > 0.972) {
        vec3 frac = fract(starGrid) - 0.5;
        vec3 starOffset = (vec3(rnd, fract(rnd * 31.7), fract(rnd * 73.1)) - 0.5) * 0.45;
        float dist2 = dot(frac - starOffset, frac - starOffset);
        float core = exp(-dist2 * 50.0) * smoothstep(0.972, 1.0, rnd) * 2.4;
        vec3 starCol = mix(vec3(1.0, 0.80, 0.60), vec3(0.70, 0.86, 1.0), fract(rnd * 113.0));
        bg += starCol * core;
      }

      return bg;
    }

    void sampleDiskAtCrossing(
      vec3 hitPos,
      vec3 rayDir,
      out vec3 diskColorOut,
      out float diskOpacityOut
    ) {
      diskColorOut = vec3(0.0);
      diskOpacityOut = 0.0;

      float rIn = 2.55 * uRs;
      float rOut = 14.2 * uRs;
      float radialLen = length(hitPos.xz);

      if (radialLen <= rIn * 0.80 || radialLen >= rOut) return;

      float innerFade = smoothstep(rIn * 0.82, rIn * 1.30, radialLen);
      float outerFade = 1.0 - smoothstep(rIn * 1.5, rOut, radialLen);
      float radialProfile = innerFade * outerFade * outerFade;

      if (radialProfile < 0.001) return;

      float phi = atan(hitPos.z, hitPos.x);
      float normR = clamp((radialLen - rIn) / (rOut - rIn), 0.0, 1.0);
      float logR = log(max(radialLen / uRs, 1.001));
      float t = uTime * uDiskSpeed;

      float rings1 = 0.5 + 0.5 * sin(24.0 * logR - t * 1.6);
      float rings2 = 0.5 + 0.5 * cos(44.0 * logR + t * 0.9);
      float spiralArm1 = 0.5 + 0.5 * sin(2.0 * phi - 10.0 * logR + t * 2.4);
      float spiralArm2 = 0.5 + 0.5 * cos(3.0 * phi - 16.0 * logR + t * 3.6);

      float rotAngle = t * 1.2;
      float c = cos(rotAngle), s = sin(rotAngle);
      vec2 rotXZ = vec2(c * hitPos.x - s * hitPos.z, s * hitPos.x + c * hitPos.z) / uRs;

      float shear = diskFbm(rotXZ * 0.75 + vec2(rings1 * 0.5, rings2 * 0.5));
      float structure = mix(rings1 * 0.6 + rings2 * 0.4, spiralArm1 * 0.55 + spiralArm2 * 0.45, 0.5);
      float plasmaPattern = smoothstep(0.10, 0.90, structure * 0.62 + shear * 0.52) * (0.65 + 0.65 * shear);

      vec3 tangentDir = vec3(-hitPos.z, 0.0, hitPos.x) / max(radialLen, 0.001);
      float beta = clamp(sqrt(uRs / max(2.0 * (radialLen - uRs), 0.18)), 0.05, 0.66);
      float gammaInv = sqrt(max(1.0 - beta * beta, 0.1));

      float cosTheta = dot(tangentDir, -rayDir);
      float dopplerFactor = gammaInv / max(1.0 - beta * cosTheta * uDoppler, 0.28);

      float gravRedshift = sqrt(clamp(1.0 - (uRs / radialLen), 0.15, 1.0));
      float freqShift = clamp(dopplerFactor * gravRedshift, 0.40, 2.05);
      float beamingBoost = pow(freqShift, 2.55);

      float baseTemp = mix(1.38, 0.24, pow(normR, 0.6)) * (0.65 + 0.65 * plasmaPattern);
      vec3 col = blackbodyDopplerColor(baseTemp, freqShift);

      float density = clamp(radialProfile * plasmaPattern * uDiskDensity * 1.55, 0.0, 0.95);
      diskOpacityOut = density;
      diskColorOut = col * density * beamingBoost * 2.25;
    }

    void main() {
      vec2 ndc = vUv * 2.0 - 1.0;
      vec4 viewRay = uInvProjMatrix * vec4(ndc, -1.0, 1.0);
      viewRay.xyz /= viewRay.w;
      vec3 worldRayDir = normalize((uCameraWorldMatrix * vec4(viewRay.xyz, 0.0)).xyz);

      vec4 sceneData = texture2D(tSceneData, vUv);
      float signedDist = sceneData.a;
      bool hasSceneObject = (abs(signedDist) > 0.01);
      bool isSolidObject = (signedDist > 0.01);
      float sceneRayDist = abs(signedDist);

      vec3 pos = uCameraPos;
      vec3 dir = worldRayDir;
      float totalDist = 0.0;
      bool sceneComposited = !hasSceneObject;

      vec3 accumulatedLight = vec3(0.0);
      float transmittance = 1.0;
      bool swallowedByHorizon = false;
      float minRadiusReached = 1e6;

      vec3 hVec = cross(pos, dir);
      float h2 = dot(hVec, hVec);

      for (int i = 0; i < HARD_MAX_STEPS; ++i) {
        if (i >= uMaxSteps) break;

        float r2 = dot(pos, pos);
        float r = sqrt(r2);

        float dt = max(0.075 * uRs, (r - 0.92 * uRs) * 0.175);

        // Equação geodésica escrita sem r^5 explícito para evitar overflow em GPUs Mobile (Adreno 610)
        vec3 midPos = pos + dir * (0.5 * dt);
        float midR2 = max(dot(midPos, midPos), 0.01);
        float midR = sqrt(midR2);
        vec3 unitMid = midPos / midR;
        float invR2 = 1.0 / midR2;
        vec3 accel = (-1.5 * uRs * uLensing * (h2 * invR2) * invR2) * unitMid;

        vec3 nextDir = normalize(dir + accel * dt);
        vec3 nextPos = pos + normalize(dir + nextDir) * dt;

        vec3 seg = nextPos - pos;
        float segLen2 = max(dot(seg, seg), 1e-6);
        float sClosest = clamp(-dot(pos, seg) / segLen2, 0.0, 1.0);
        vec3 closestPt = pos + seg * sClosest;
        float segMinR = length(closestPt);
        minRadiusReached = min(minRadiusReached, segMinR);

        if (!sceneComposited && (totalDist + dt >= sceneRayDist)) {
          if (segMinR > uRs * 1.005 || sceneRayDist < totalDist + sClosest * dt) {
            accumulatedLight += transmittance * sceneData.rgb;
            if (isSolidObject) {
              transmittance = 0.0;
              sceneComposited = true;
              break;
            } else {
              transmittance *= 0.35;
              sceneComposited = true;
            }
          }
        }

        if (segMinR <= uRs * 1.005) {
          swallowedByHorizon = true;
          break;
        }

        // Interseção exata com o plano equatorial Y = 0
        if (pos.y * nextPos.y <= 0.0) {
          float denomY = pos.y - nextPos.y;
          float alpha = abs(denomY) > 1e-7 ? clamp(pos.y / denomY, 0.0, 1.0) : 0.5;
          vec3 hitPos = mix(pos, nextPos, alpha);
          vec3 hitDir = normalize(mix(dir, nextDir, alpha));

          if (length(hitPos) > uRs * 1.05) {
            vec3 diskCol;
            float diskOpac;
            sampleDiskAtCrossing(hitPos, hitDir, diskCol, diskOpac);

            if (diskOpac > 0.001) {
              accumulatedLight += transmittance * diskCol;
              transmittance *= (1.0 - diskOpac * 0.80);
            }
          }
        }

        if (uFlashIntensity > 0.005 && segMinR < 3.4 * uRs) {
          float flashDist = length(closestPt - uFlashPos);
          float shellGlow = exp(-abs(segMinR - 1.25 * uRs) * 3.5) * exp(-flashDist * 0.6);
          accumulatedLight += transmittance * vec3(0.72, 0.90, 1.45) * shellGlow * uFlashIntensity * dt * 2.2;
        }

        pos = nextPos;
        dir = nextDir;
        totalDist += dt;

        if (transmittance < 0.01) break;

        if (r > 4.2 * uRs && dot(pos, dir) > 0.0 && (pos.y * dir.y > 0.0) && abs(pos.y) > 0.3 * uRs) {
          break;
        }

        if (r > 180.0 && dot(pos, dir) > 0.0) break;
      }

      if (!sceneComposited && !swallowedByHorizon) {
        accumulatedLight += transmittance * sceneData.rgb;
        if (isSolidObject) transmittance = 0.0;
      }

      if (!swallowedByHorizon) {
        float normClosest = (minRadiusReached - uRs) / uRs;
        float photonRing = exp(-pow((normClosest - 0.48) * 8.5, 2.0)) * 0.90;
        float horizonAntiAlias = smoothstep(0.0, 0.05, normClosest);

        accumulatedLight += transmittance * vec3(1.0, 0.84, 0.55) * photonRing * horizonAntiAlias;

        if (transmittance > 0.008) {
          accumulatedLight += transmittance * sampleStarfieldAndNebula(dir) * horizonAntiAlias;
        }
      }

      vec2 vigUv = vUv * (1.0 - vUv.yx);
      float vig = clamp(pow(vigUv.x * vigUv.y * 16.0, 0.14), 0.0, 1.0);
      accumulatedLight *= vig;

      gl_FragColor = vec4(accumulatedLight, 1.0);
    }
  `
};

// ============================================================================
// 5. COMPOSER DE PÓS-PROCESSAMENTO E ESCALONADOR DINÂMICO DE RESOLUÇÃO
// ============================================================================
const composer = new EffectComposer(renderer);
composer.setPixelRatio(PARAMS.renderScale);

const blackHolePass = new ShaderPass(BlackHoleRaymarchShader);
composer.addPass(blackHolePass);

const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(
    Math.floor(window.innerWidth * PARAMS.renderScale * 0.5),
    Math.floor(window.innerHeight * PARAMS.renderScale * 0.5)
  ),
  PARAMS.bloomStrength,
  0.50,
  0.78
);
composer.addPass(bloomPass);

const outputPass = new OutputPass();
composer.addPass(outputPass);

function applyResolutionScale(scale, updateSlider = true) {
  const clamped = Math.max(0.40, Math.min(1.0, Number(scale.toFixed(2))));
  if (Math.abs(clamped - PARAMS.renderScale) < 0.01 && updateSlider === false) return;

  PARAMS.renderScale = clamped;
  const w = window.innerWidth;
  const h = window.innerHeight;
  const scaledW = Math.max(1, Math.floor(w * clamped));
  const scaledH = Math.max(1, Math.floor(h * clamped));

  renderer.setPixelRatio(clamped);
  renderer.setSize(w, h);
  composer.setPixelRatio(clamped);
  composer.setSize(w, h);

  sceneRenderTarget.dispose();
  sceneRenderTarget = createSceneRenderTarget(scaledW, scaledH);

  blackHolePass.uniforms.uResolution.value.set(scaledW, scaledH);
  bloomPass.resolution.set(Math.floor(scaledW * 0.5), Math.floor(scaledH * 0.5));

  const resSlider = document.getElementById('ctrl-res-scale');
  const resVal = document.getElementById('val-res-scale');
  if (resSlider && updateSlider) resSlider.value = String(clamped);
  if (resVal) resVal.textContent = `${clamped.toFixed(2)}x`;
}

// ============================================================================
// 6. SHADER DE ESPAGUETIFICAÇÃO E AQUECIMENTO DE MARÉ PARA CORPOS CELESTES 3D
// ============================================================================
function createSpaghettifiedMaterial(type) {
  const isStar = type === 'star';
  const isComet = type === 'comet';

  const baseColor = isStar
    ? new THREE.Color(0xffb84d)
    : isComet
      ? new THREE.Color(0x6be8ff)
      : new THREE.Color(0x7c746b);

  const hotColor = isStar
    ? new THREE.Color(0xd8f3ff)
    : isComet
      ? new THREE.Color(0xffffff)
      : new THREE.Color(0xff6a1a);

  return new THREE.ShaderMaterial({
    blending: THREE.NoBlending,
    depthWrite: true,
    depthTest: true,
    uniforms: {
      uBaseColor: { value: baseColor },
      uHotColor: { value: hotColor },
      uTidalHeat: { value: 0.0 },
      uTime: { value: 0.0 },
      uIsStar: { value: isStar ? 1.0 : 0.0 },
      uSeed: { value: Math.random() * 100.0 }
    },
    vertexShader: /* glsl */ `
      varying vec3 vNormal;
      varying vec3 vWorldPos;
      varying vec3 vLocalPos;
      varying float vCamDist;
      uniform float uTime;
      uniform float uSeed;
      uniform float uIsStar;

      float hash(vec3 p) {
        p = fract(p * 0.3183099 + 0.1);
        p *= 17.0;
        return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
      }

      float noise(vec3 x) {
        vec3 i = floor(x);
        vec3 f = fract(x);
        f = f * f * (3.0 - 2.0 * f);
        return mix(
          mix(mix(hash(i + vec3(0,0,0)), hash(i + vec3(1,0,0)), f.x),
              mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
          mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x),
              mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y),
          f.z
        );
      }

      void main() {
        vLocalPos = position;
        float disp = noise(position * 2.6 + uSeed);
        float displacementScale = (uIsStar > 0.5) ? 0.04 * sin(uTime * 4.0 + disp * 6.0) : (disp - 0.5) * 0.28;
        vec3 displaced = position + normal * displacementScale;

        vec4 worldPosition = modelMatrix * vec4(displaced, 1.0);
        vWorldPos = worldPosition.xyz;
        vCamDist = max(length(worldPosition.xyz - cameraPosition), 0.1);
        vNormal = normalize(mat3(modelMatrix) * normal);
        gl_Position = projectionMatrix * viewMatrix * worldPosition;
      }
    `,
    fragmentShader: /* glsl */ `
      varying vec3 vNormal;
      varying vec3 vWorldPos;
      varying vec3 vLocalPos;
      varying float vCamDist;

      uniform vec3 uBaseColor;
      uniform vec3 uHotColor;
      uniform float uTidalHeat;
      uniform float uTime;
      uniform float uIsStar;
      uniform float uSeed;

      float hash(vec3 p) {
        p = fract(p * 0.3183099 + 0.1);
        p *= 17.0;
        return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
      }

      float noise(vec3 x) {
        vec3 i = floor(x);
        vec3 f = fract(x);
        f = f * f * (3.0 - 2.0 * f);
        return mix(
          mix(mix(hash(i + vec3(0,0,0)), hash(i + vec3(1,0,0)), f.x),
              mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
          mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x),
              mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y),
          f.z
        );
      }

      void main() {
        vec3 toCenter = normalize(-vWorldPos);
        float diff = max(dot(normalize(vNormal), toCenter), 0.08);

        float surfPattern = noise(vLocalPos * 4.5 + uSeed + vec3(uTime * 0.4 * uIsStar));
        vec3 surfaceColor = uBaseColor * (0.65 + 0.65 * surfPattern);

        vec3 finalCol;
        if (uIsStar > 0.5) {
          vec3 starCore = mix(uBaseColor, vec3(1.0, 0.95, 0.8), surfPattern);
          finalCol = mix(starCore * 2.2, uHotColor * 4.5, clamp(uTidalHeat, 0.0, 1.0));
        } else {
          float magmaCracks = smoothstep(0.45, 0.82, surfPattern + uTidalHeat * 0.65) * uTidalHeat;
          vec3 litRock = surfaceColor * (0.25 + 1.35 * diff);
          vec3 glowingMagma = mix(vec3(1.0, 0.35, 0.05), uHotColor * 3.8, uTidalHeat) * magmaCracks * 2.8;
          finalCol = mix(litRock + glowingMagma, uHotColor * 4.2, uTidalHeat * uTidalHeat);
        }

        gl_FragColor = vec4(finalCol, vCamDist);
      }
    `
  });
}

const sphereGeometry = new THREE.IcosahedronGeometry(1.0, 3);

// ============================================================================
// 7. SISTEMA DE PARTÍCULAS DE RUPTURA DE MARÉ (DETRITOS E CLARÕES DE ABSORÇÃO)
// ============================================================================
const MAX_DEBRIS_PARTICLES = 500;
const debrisPositions = new Float32Array(MAX_DEBRIS_PARTICLES * 3);
const debrisVelocities = new Float32Array(MAX_DEBRIS_PARTICLES * 3);
const debrisColors = new Float32Array(MAX_DEBRIS_PARTICLES * 3);
const debrisLife = new Float32Array(MAX_DEBRIS_PARTICLES);
let debrisCursor = 0;

const debrisGeometry = new THREE.BufferGeometry();
debrisGeometry.setAttribute('position', new THREE.BufferAttribute(debrisPositions, 3));
debrisGeometry.setAttribute('color', new THREE.BufferAttribute(debrisColors, 3));
debrisGeometry.setAttribute('aLife', new THREE.BufferAttribute(debrisLife, 1));

const debrisMaterial = new THREE.ShaderMaterial({
  blending: THREE.NoBlending,
  depthWrite: true,
  depthTest: true,
  vertexShader: /* glsl */ `
    attribute vec3 color;
    attribute float aLife;
    varying vec3 vColor;
    varying float vLife;
    varying float vCamDist;

    void main() {
      vColor = color;
      vLife = aLife;
      vec4 worldPosition = modelMatrix * vec4(position, 1.0);
      vCamDist = max(length(worldPosition.xyz - cameraPosition), 0.1);
      vec4 mvPosition = viewMatrix * worldPosition;
      gl_PointSize = (aLife > 0.01) ? clamp((30.0 * aLife) / -mvPosition.z, 1.5, 8.0) : 0.0;
      gl_Position = projectionMatrix * mvPosition;
    }
  `,
  fragmentShader: /* glsl */ `
    varying vec3 vColor;
    varying float vLife;
    varying float vCamDist;

    void main() {
      if (vLife <= 0.02) discard;
      vec2 cxy = 2.0 * gl_PointCoord - 1.0;
      float r2 = dot(cxy, cxy);
      if (r2 > 1.0) discard;
      float glow = (1.0 - r2) * (1.0 - r2) * clamp(vLife, 0.0, 1.0);
      if (glow < 0.06) discard;
      gl_FragColor = vec4(vColor * glow * 2.2, -vCamDist);
    }
  `
});

const debrisPoints = new THREE.Points(debrisGeometry, debrisMaterial);
orbitalScene.add(debrisPoints);

function emitTidalDebris(pos, vel, color, count = 2, speedSpread = 1.1) {
  for (let i = 0; i < count; i++) {
    const idx = debrisCursor;
    debrisCursor = (debrisCursor + 1) % MAX_DEBRIS_PARTICLES;

    const i3 = idx * 3;
    debrisPositions[i3]     = pos.x + (Math.random() - 0.5) * 0.3;
    debrisPositions[i3 + 1] = pos.y + (Math.random() - 0.5) * 0.3;
    debrisPositions[i3 + 2] = pos.z + (Math.random() - 0.5) * 0.3;

    debrisVelocities[i3]     = vel.x + (Math.random() - 0.5) * speedSpread;
    debrisVelocities[i3 + 1] = vel.y + (Math.random() - 0.5) * speedSpread;
    debrisVelocities[i3 + 2] = vel.z + (Math.random() - 0.5) * speedSpread;

    debrisColors[i3]     = color.r;
    debrisColors[i3 + 1] = color.g;
    debrisColors[i3 + 2] = color.b;

    debrisLife[idx] = 0.75 + Math.random() * 0.4;
  }
}

function updateDebrisParticles(dt, Rs) {
  const GM = 18.0 * getEffectiveMass();
  for (let i = 0; i < MAX_DEBRIS_PARTICLES; i++) {
    if (debrisLife[i] <= 0.0) continue;

    const i3 = i * 3;
    const x = debrisPositions[i3];
    const y = debrisPositions[i3 + 1];
    const z = debrisPositions[i3 + 2];

    const r = Math.sqrt(x * x + y * y + z * z);
    if (r <= Rs * 1.03) {
      debrisLife[i] = 0.0;
      debrisPositions[i3] = 9999.0;
      continue;
    }

    const denom = Math.max(r - Rs, 0.35);
    const accelMag = -GM / (denom * denom * r);

    debrisVelocities[i3]     += x * accelMag * dt;
    debrisVelocities[i3 + 1] += y * accelMag * dt;
    debrisVelocities[i3 + 2] += z * accelMag * dt;

    debrisPositions[i3]     += debrisVelocities[i3] * dt;
    debrisPositions[i3 + 1] += debrisVelocities[i3 + 1] * dt;
    debrisPositions[i3 + 2] += debrisVelocities[i3 + 2] * dt;

    debrisLife[i] -= dt * 0.48;
  }

  debrisGeometry.attributes.position.needsUpdate = true;
  debrisGeometry.attributes.color.needsUpdate = true;
  debrisGeometry.attributes.aLife.needsUpdate = true;
}

// ============================================================================
// 8. FÍSICA ORBITAL RELATIVÍSTICA & ESPAGUETIFICAÇÃO DE TENSORES DE MARÉ
// ============================================================================
const celestialBodies = [];
const MAX_ACTIVE_BODIES = 26;
const TRAIL_LENGTH = 36;

const tmpRadialDir = new THREE.Vector3();
const tmpTangentDir = new THREE.Vector3();
const tmpBinormalDir = new THREE.Vector3();
const tmpRotMatrix = new THREE.Matrix4();
const tmpScaleMatrix = new THREE.Matrix4();

class CelestialBody {
  constructor(type = 'asteroid', customPos = null, customVel = null) {
    this.type = type;
    this.alive = true;
    this.age = 0.0;

    if (type === 'star') {
      this.baseRadius = 0.62 + Math.random() * 0.25;
      this.trailColor = new THREE.Color(0xffaa44);
      this.label = 'Estrela';
    } else if (type === 'comet') {
      this.baseRadius = 0.28 + Math.random() * 0.14;
      this.trailColor = new THREE.Color(0x5ce1e6);
      this.label = 'Cometa Gelado';
    } else {
      this.baseRadius = 0.32 + Math.random() * 0.22;
      this.trailColor = new THREE.Color(0xff7733);
      this.label = 'Asteroide Rochoso';
    }

    const Rs = getSchwarzschildRadius();
    const GM = 18.0 * getEffectiveMass();

    if (customPos && customVel) {
      this.position = customPos.clone();
      this.velocity = customVel.clone();
    } else {
      const spawnDist = Rs * (6.2 + Math.random() * 7.5);
      const angle = Math.random() * Math.PI * 2;
      const inclination = (Math.random() - 0.5) * (type === 'comet' ? 1.1 : 0.45);

      this.position = new THREE.Vector3(
        Math.cos(angle) * spawnDist,
        Math.sin(inclination) * spawnDist * 0.45,
        Math.sin(angle) * spawnDist
      );

      const vCirc = Math.sqrt(GM / Math.max(spawnDist - Rs, 1.0));
      const eccentricityFactor = type === 'star' ? 0.72 : (0.68 + Math.random() * 0.24);
      const tangent = new THREE.Vector3(-Math.sin(angle), (Math.random() - 0.5) * 0.15, Math.cos(angle)).normalize();
      const inward = this.position.clone().normalize().multiplyScalar(-vCirc * (0.08 + Math.random() * 0.12));

      this.velocity = tangent.multiplyScalar(vCirc * eccentricityFactor).add(inward);
    }

    this.material = createSpaghettifiedMaterial(type);
    this.mesh = new THREE.Mesh(sphereGeometry, this.material);
    this.mesh.matrixAutoUpdate = false;
    orbitalScene.add(this.mesh);

    this.trailPoints = [];
    for (let i = 0; i < TRAIL_LENGTH; i++) {
      this.trailPoints.push(this.position.clone());
    }

    const trailPosArray = new Float32Array(TRAIL_LENGTH * 3);
    const trailAlphaArray = new Float32Array(TRAIL_LENGTH);
    for (let i = 0; i < TRAIL_LENGTH; i++) {
      trailAlphaArray[i] = 1.0 - i / (TRAIL_LENGTH - 1);
    }

    this.trailGeo = new THREE.BufferGeometry();
    this.trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPosArray, 3));
    this.trailGeo.setAttribute('aAlpha', new THREE.BufferAttribute(trailAlphaArray, 1));

    this.trailMat = new THREE.ShaderMaterial({
      blending: THREE.NoBlending,
      depthWrite: true,
      depthTest: true,
      uniforms: {
        uColor: { value: this.trailColor },
        uHeat: { value: 0.0 }
      },
      vertexShader: /* glsl */ `
        attribute float aAlpha;
        varying float vAlpha;
        varying float vCamDist;
        void main() {
          vAlpha = aAlpha;
          vec4 worldPosition = modelMatrix * vec4(position, 1.0);
          vCamDist = max(length(worldPosition.xyz - cameraPosition), 0.1);
          gl_Position = projectionMatrix * viewMatrix * worldPosition;
        }
      `,
      fragmentShader: /* glsl */ `
        uniform vec3 uColor;
        uniform float uHeat;
        varying float vAlpha;
        varying float vCamDist;
        void main() {
          if (vAlpha < 0.06) discard;
          vec3 hotCol = mix(uColor, vec3(0.85, 0.95, 1.4), uHeat);
          gl_FragColor = vec4(hotCol * (1.1 + uHeat * 1.6) * vAlpha, -vCamDist);
        }
      `
    });

    this.trailLine = new THREE.Line(this.trailGeo, this.trailMat);
    orbitalScene.add(this.trailLine);
  }

  update(dt, elapsedTime) {
    if (!this.alive) return;
    this.age += dt;

    const Rs = getSchwarzschildRadius();
    const GM = 18.0 * getEffectiveMass();

    const subSteps = 3;
    const subDt = dt / subSteps;

    for (let s = 0; s < subSteps; s++) {
      const r = this.position.length();

      if (r <= Rs * 1.04) {
        this.onSwallowed(Rs);
        return;
      }

      const effectiveDenom = Math.max(r - Rs, 0.22 * Rs);
      const gravMag = -GM / (effectiveDenom * effectiveDenom);

      tmpRadialDir.copy(this.position).divideScalar(r);
      const dragFactor = 0.045 * Math.pow(Rs / r, 2.2);

      this.velocity.addScaledVector(tmpRadialDir, gravMag * subDt);
      this.velocity.multiplyScalar(1.0 - dragFactor * subDt);
      this.position.addScaledVector(this.velocity, subDt);
    }

    const r = this.position.length();
    if (r > 140.0) {
      this.dispose();
      return;
    }

    const normalizedDist = Math.max((r - Rs) / Rs, 0.04);
    const tidalFactor = PARAMS.spaghettification * (2.6 / Math.pow(normalizedDist + 0.35, 2.15));
    const stretchRadial = 1.0 + Math.min(tidalFactor * (this.type === 'star' ? 2.4 : 1.75), 9.5);
    const compressTransverse = 1.0 / Math.pow(stretchRadial, 0.44);

    const tidalHeat = Math.min(tidalFactor * 0.48, 1.0);
    this.material.uniforms.uTidalHeat.value = tidalHeat;
    this.material.uniforms.uTime.value = elapsedTime;
    this.trailMat.uniforms.uHeat.value = tidalHeat;

    tmpRadialDir.copy(this.position).normalize().negate();
    const velDir = this.velocity.clone().normalize();
    const stretchAxis = tmpRadialDir.lerp(velDir, 0.28).normalize();

    tmpBinormalDir.crossVectors(stretchAxis, new THREE.Vector3(0, 1, 0));
    if (tmpBinormalDir.lengthSq() < 0.001) {
      tmpBinormalDir.crossVectors(stretchAxis, new THREE.Vector3(0, 0, 1));
    }
    tmpBinormalDir.normalize();
    tmpTangentDir.crossVectors(tmpBinormalDir, stretchAxis).normalize();

    tmpRotMatrix.makeBasis(stretchAxis, tmpTangentDir, tmpBinormalDir);
    tmpScaleMatrix.makeScale(
      this.baseRadius * stretchRadial,
      this.baseRadius * compressTransverse,
      this.baseRadius * compressTransverse
    );

    this.mesh.matrix.copy(tmpRotMatrix).multiply(tmpScaleMatrix);
    this.mesh.matrix.setPosition(this.position);

    if (tidalHeat > 0.42) {
      const tipFront = this.position.clone().addScaledVector(stretchAxis, this.baseRadius * stretchRadial * 0.85);
      emitTidalDebris(tipFront, this.velocity, this.trailColor, 1, 0.8);
    }

    for (let i = TRAIL_LENGTH - 1; i > 0; i--) {
      this.trailPoints[i].copy(this.trailPoints[i - 1]);
    }
    this.trailPoints[0].copy(this.position);

    const posAttr = this.trailGeo.attributes.position;
    for (let i = 0; i < TRAIL_LENGTH; i++) {
      const pt = this.trailPoints[i];
      posAttr.setXYZ(i, pt.x, pt.y, pt.z);
    }
    posAttr.needsUpdate = true;
  }

  onSwallowed(Rs) {
    const flashMultiplier = this.type === 'star' ? 2.2 : 1.0;
    blackHolePass.uniforms.uFlashIntensity.value = Math.min(
      blackHolePass.uniforms.uFlashIntensity.value + flashMultiplier,
      3.0
    );
    blackHolePass.uniforms.uFlashPos.value.copy(this.position).normalize().multiplyScalar(Rs * 1.15);

    emitTidalDebris(
      this.position,
      this.velocity.clone().multiplyScalar(0.35),
      new THREE.Color(0xaad8ff),
      this.type === 'star' ? 22 : 10,
      3.2
    );

    PARAMS.swallowedCount++;
    PARAMS.absorbedMassBonus = Math.min(PARAMS.absorbedMassBonus + (this.type === 'star' ? 0.015 : 0.003), 0.6);

    addEventFeedItem(
      `Horizonte absorveu ${this.label} (+energia relativística)`,
      this.type === 'star' ? 'tde' : 'swallow'
    );

    this.dispose();
  }

  dispose() {
    this.alive = false;
    orbitalScene.remove(this.mesh);
    orbitalScene.remove(this.trailLine);
    this.material.dispose();
    this.trailGeo.dispose();
    this.trailMat.dispose();
  }
}

// ============================================================================
// 9. GERENCIAMENTO DE SPAWN E INTERAÇÃO (MOUSE + TOUCH NO CELULAR)
// ============================================================================
function spawnBody(type = 'random', customPos = null, customVel = null, logEvent = false) {
  if (celestialBodies.length >= MAX_ACTIVE_BODIES) {
    const oldest = celestialBodies.shift();
    if (oldest) oldest.dispose();
  }

  let chosenType = type;
  if (type === 'random') {
    const r = Math.random();
    if (r < 0.16) chosenType = 'star';
    else if (r < 0.45) chosenType = 'comet';
    else chosenType = 'asteroid';
  }

  const body = new CelestialBody(chosenType, customPos, customVel);
  celestialBodies.push(body);

  if (logEvent) {
    addEventFeedItem(
      `${body.label} lançado em órbita de captura`,
      chosenType === 'star' ? 'tde' : 'spawn'
    );
  }
  return body;
}

for (let i = 0; i < 7; i++) {
  spawnBody('random');
}

const raycaster = new THREE.Raycaster();
const mouseNDC = new THREE.Vector2();

function launchAtScreenCoords(clientX, clientY) {
  mouseNDC.x = (clientX / window.innerWidth) * 2 - 1;
  mouseNDC.y = -(clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(mouseNDC, camera);

  const Rs = getSchwarzschildRadius();
  const GM = 18.0 * getEffectiveMass();

  const launchDist = Rs * 8.5;
  const targetPoint = new THREE.Vector3()
    .copy(raycaster.ray.origin)
    .addScaledVector(raycaster.ray.direction, camera.position.length() * 0.85);

  if (targetPoint.length() < Rs * 2.5) {
    targetPoint.normalize().multiplyScalar(launchDist);
  }

  const vCirc = Math.sqrt(GM / Math.max(targetPoint.length() - Rs, 1.0));
  const tangent = new THREE.Vector3(-targetPoint.z, (Math.random() - 0.5) * 0.25, targetPoint.x).normalize();
  const inward = targetPoint.clone().normalize().multiplyScalar(-vCirc * 0.18);
  const vel = tangent.multiplyScalar(vCirc * 0.76).add(inward);

  spawnBody('random', targetPoint, vel, true);
}

// Duplo clique no Desktop
renderer.domElement.addEventListener('dblclick', (e) => {
  launchAtScreenCoords(e.clientX, e.clientY);
});

// Duplo toque (Double-Tap) no Celular (Moto G9 Power)
let lastTapTime = 0;
let touchStartPos = { x: 0, y: 0 };

renderer.domElement.addEventListener('touchstart', (e) => {
  if (e.touches.length === 1) {
    touchStartPos.x = e.touches[0].clientX;
    touchStartPos.y = e.touches[0].clientY;
  }
}, { passive: true });

renderer.domElement.addEventListener('touchend', (e) => {
  if (e.changedTouches.length !== 1) return;
  const touch = e.changedTouches[0];
  const moveDist = Math.hypot(touch.clientX - touchStartPos.x, touch.clientY - touchStartPos.y);
  if (moveDist > 18) return; // Ignora se foi um arrasto de câmera

  const now = performance.now();
  if (now - lastTapTime < 300) {
    launchAtScreenCoords(touch.clientX, touch.clientY);
    lastTapTime = 0;
  } else {
    lastTapTime = now;
  }
}, { passive: true });

// ============================================================================
// 10. GOVERNADOR INTELIGENTE DE QUALIDADE ADAPTATIVA (AUTO-FPS)
// ============================================================================
// Mantém a qualidade alta e SÓ reduz se o dispositivo realmente precisar!
const hudMass = document.getElementById('hud-mass');
const hudRs = document.getElementById('hud-rs');
const hudPhoton = document.getElementById('hud-photon');
const hudIsco = document.getElementById('hud-isco');
const hudActive = document.getElementById('hud-active');
const hudSwallowed = document.getElementById('hud-swallowed');
const hudFps = document.getElementById('hud-fps');
const hudQualityMode = document.getElementById('hud-quality-mode');
const eventFeed = document.getElementById('event-feed');

function addEventFeedItem(text, className = '') {
  const item = document.createElement('div');
  item.className = `feed-item ${className}`;
  const now = new Date();
  const timeStr = now.toTimeString().slice(0, 8);
  item.textContent = `[${timeStr}] ${text}`;
  eventFeed.appendChild(item);

  while (eventFeed.children.length > 3) {
    eventFeed.removeChild(eventFeed.firstChild);
  }
}

let fpsFrames = 0;
let fpsLastTime = performance.now();
let lowFpsStreak = 0;
let highFpsStreak = 0;
const startupGraceTime = performance.now() + 1800; // Ignora compilação inicial do shader

function evaluateAdaptiveQuality(currentFps) {
  if (!PARAMS.autoQuality || performance.now() < startupGraceTime) return;

  // Se o FPS cair abaixo de 38 FPS no celular/PC, reduz suavemente a escala para recuperar fluidez
  if (currentFps < 38) {
    lowFpsStreak++;
    highFpsStreak = 0;

    if (lowFpsStreak >= 2) {
      lowFpsStreak = 0;
      if (PARAMS.renderScale > 0.45) {
        const nextScale = Math.max(0.45, PARAMS.renderScale - 0.08);
        applyResolutionScale(nextScale, true);
      }
      if (PARAMS.renderScale <= 0.62 && PARAMS.maxSteps > 40) {
        PARAMS.maxSteps = 40;
        blackHolePass.uniforms.uMaxSteps.value = 40;
      }
      if (hudQualityMode) {
        hudQualityMode.textContent = `Qualidade Adaptada (${Math.round(PARAMS.renderScale * 100)}%)`;
      }
    }
  } else if (currentFps >= 56 && PARAMS.renderScale < 0.85) {
    // Se o dispositivo estiver sobrando (56+ FPS constantes), sobe a qualidade de volta!
    highFpsStreak++;
    lowFpsStreak = 0;

    if (highFpsStreak >= 5) {
      highFpsStreak = 0;
      const nextScale = Math.min(0.85, PARAMS.renderScale + 0.05);
      applyResolutionScale(nextScale, true);
      if (nextScale >= 0.70 && PARAMS.maxSteps < 56) {
        PARAMS.maxSteps = 56;
        blackHolePass.uniforms.uMaxSteps.value = 56;
      }
      if (hudQualityMode) {
        hudQualityMode.textContent = nextScale >= 0.84
          ? 'Qualidade Máxima Ativa'
          : `Qualidade Adaptativa (${Math.round(nextScale * 100)}%)`;
      }
    }
  } else {
    lowFpsStreak = 0;
  }
}

function updateTelemetryHUD() {
  const effMass = getEffectiveMass();
  const Rs = getSchwarzschildRadius();
  hudMass.textContent = `${(effMass * 100.0).toFixed(1)} M☉`;
  hudRs.textContent = `${Rs.toFixed(2)} u`;
  hudPhoton.textContent = `${(Rs * 1.5).toFixed(2)} u`;
  hudIsco.textContent = `${(Rs * 3.0).toFixed(2)} u`;
  hudActive.textContent = `${celestialBodies.length}`;
  hudSwallowed.textContent = `${PARAMS.swallowedCount}`;

  fpsFrames++;
  const now = performance.now();
  if (now - fpsLastTime >= 600) {
    const currentFps = Math.round((fpsFrames * 1000) / (now - fpsLastTime));
    if (hudFps) {
      hudFps.textContent = `${currentFps} FPS`;
      hudFps.style.color = currentFps >= 48 ? '#4ade80' : currentFps >= 30 ? '#fbbf24' : '#f87171';
    }
    evaluateAdaptiveQuality(currentFps);
    fpsFrames = 0;
    fpsLastTime = now;
  }
}

// ============================================================================
// 11. CONTROLES DA INTERFACE & RESPONSIVIDADE MOBILE (MOTO G9 POWER)
// ============================================================================
function bindSlider(id, valId, paramKey, formatFn, onChangeFn = null) {
  const slider = document.getElementById(id);
  const valSpan = document.getElementById(valId);
  if (!slider || !valSpan) return;
  slider.addEventListener('input', () => {
    const v = parseFloat(slider.value);
    PARAMS[paramKey] = v;
    valSpan.textContent = formatFn(v);
    if (onChangeFn) onChangeFn(v);
  });
}

bindSlider('ctrl-mass', 'val-mass', 'mass', (v) => `${v.toFixed(2)}x`);
bindSlider('ctrl-lensing', 'val-lensing', 'lensing', (v) => `${v.toFixed(2)}x`);
bindSlider('ctrl-disk-speed', 'val-disk-speed', 'diskSpeed', (v) => `${v.toFixed(2)}x`);
bindSlider('ctrl-doppler', 'val-doppler', 'doppler', (v) => `${v.toFixed(2)}x`);
bindSlider('ctrl-disk-density', 'val-disk-density', 'diskDensity', (v) => `${v.toFixed(2)}x`);
bindSlider('ctrl-spawn-rate', 'val-spawn-rate', 'spawnRate', (v) => `${v.toFixed(1)} /s`);
bindSlider('ctrl-spaghetti', 'val-spaghetti', 'spaghettification', (v) => `${v.toFixed(2)}x`);
bindSlider('ctrl-bloom', 'val-bloom', 'bloomStrength', (v) => `${v.toFixed(2)}`, (v) => {
  bloomPass.strength = v;
});
bindSlider('ctrl-res-scale', 'val-res-scale', 'renderScale', (v) => `${v.toFixed(2)}x`, (v) => {
  applyResolutionScale(v, false);
});

const autoQualityCheckbox = document.getElementById('ctrl-auto-quality');
if (autoQualityCheckbox) {
  autoQualityCheckbox.addEventListener('change', () => {
    PARAMS.autoQuality = autoQualityCheckbox.checked;
    if (hudQualityMode) {
      hudQualityMode.textContent = PARAMS.autoQuality
        ? 'Qualidade Adaptativa Ativa'
        : `Qualidade Manual (${Math.round(PARAMS.renderScale * 100)}%)`;
    }
  });
}

// Minimizar / Expandir Painel de Parâmetros e Painel de Telemetria
const togglePanelBtn = document.getElementById('toggle-panel-btn');
const panelHeaderBar = document.getElementById('panel-header-bar');
const panelBody = document.getElementById('panel-body');

function toggleControlPanel() {
  panelBody.classList.toggle('collapsed');
  togglePanelBtn.textContent = panelBody.classList.contains('collapsed') ? '+' : '−';
}

panelHeaderBar.addEventListener('click', toggleControlPanel);

const toggleHudBtn = document.getElementById('toggle-hud-btn');
const hudBody = document.getElementById('hud-body');
if (toggleHudBtn && hudBody) {
  toggleHudBtn.addEventListener('click', () => {
    hudBody.classList.toggle('collapsed');
    toggleHudBtn.textContent = hudBody.classList.contains('collapsed') ? '+' : '−';
  });
}

// Em telas pequenas (Moto G9 Power), inicia o painel de controles recolhido
// para deixar o Buraco Negro livre em tela cheia!
if (window.innerWidth <= 768) {
  panelBody.classList.add('collapsed');
  togglePanelBtn.textContent = '+';
}

function triggerStarTDE() {
  const Rs = getSchwarzschildRadius();
  const GM = 18.0 * getEffectiveMass();
  const angle = Math.random() * Math.PI * 2;
  const dist = Rs * 5.2;
  const pos = new THREE.Vector3(Math.cos(angle) * dist, (Math.random() - 0.5) * Rs, Math.sin(angle) * dist);
  const vCirc = Math.sqrt(GM / (dist - Rs));
  const vel = new THREE.Vector3(-Math.sin(angle), 0.05, Math.cos(angle))
    .normalize()
    .multiplyScalar(vCirc * 0.66);
  spawnBody('star', pos, vel, true);
}

function triggerSwarm() {
  for (let i = 0; i < 12; i++) {
    spawnBody('random');
  }
  addEventFeedItem('Chuva de 12 corpos celestes injetada na órbita!', 'spawn');
}

document.getElementById('btn-spawn-asteroid').addEventListener('click', () => {
  spawnBody('asteroid', null, null, true);
});

document.getElementById('btn-spawn-comet').addEventListener('click', () => {
  spawnBody('comet', null, null, true);
});

document.getElementById('btn-spawn-star').addEventListener('click', triggerStarTDE);
document.getElementById('btn-spawn-swarm').addEventListener('click', triggerSwarm);

// Botões rápidos flutuantes para Mobile
const mobSpawnStar = document.getElementById('mob-spawn-star');
const mobSpawnSwarm = document.getElementById('mob-spawn-swarm');
const mobCamCycle = document.getElementById('mob-cam-cycle');

if (mobSpawnStar) mobSpawnStar.addEventListener('click', triggerStarTDE);
if (mobSpawnSwarm) mobSpawnSwarm.addEventListener('click', triggerSwarm);

let cameraTransition = null;
let currentPresetIdx = 0;
const cameraPresets = [
  new THREE.Vector3(0.0, 4.6, 33.0),
  new THREE.Vector3(18.0, 1.15, 22.0),
  new THREE.Vector3(0.1, 38.0, 4.0)
];

function animateCameraTo(baseTargetPos) {
  const responsiveTarget = getResponsiveCameraPos(baseTargetPos);
  cameraTransition = {
    startPos: camera.position.clone(),
    endPos: responsiveTarget,
    progress: 0.0
  };
}

document.getElementById('cam-gargantua').addEventListener('click', () => {
  currentPresetIdx = 0;
  animateCameraTo(cameraPresets[0]);
});

document.getElementById('cam-grazing').addEventListener('click', () => {
  currentPresetIdx = 1;
  animateCameraTo(cameraPresets[1]);
});

document.getElementById('cam-polar').addEventListener('click', () => {
  currentPresetIdx = 2;
  animateCameraTo(cameraPresets[2]);
});

if (mobCamCycle) {
  mobCamCycle.addEventListener('click', () => {
    currentPresetIdx = (currentPresetIdx + 1) % cameraPresets.length;
    animateCameraTo(cameraPresets[currentPresetIdx]);
  });
}

controls.addEventListener('start', () => {
  cameraTransition = null;
});

// ============================================================================
// 12. LOOP PRINCIPAL DE ANIMAÇÃO E RENDERIZAÇÃO HÍBRIDA
// ============================================================================
const clock = new THREE.Clock();
let spawnAccumulator = 0.0;

function animate() {
  requestAnimationFrame(animate);

  const rawDt = clock.getDelta();
  const dt = Math.min(rawDt, 0.05);
  const elapsedTime = clock.getElapsedTime();

  if (cameraTransition) {
    cameraTransition.progress += dt * 1.8;
    const t = Math.min(cameraTransition.progress, 1.0);
    const ease = 1.0 - Math.pow(1.0 - t, 3.0);
    camera.position.lerpVectors(cameraTransition.startPos, cameraTransition.endPos, ease);
    if (t >= 1.0) cameraTransition = null;
  }

  controls.update();

  const maxAutoBodies = PARAMS.renderScale < 0.55 ? 12 : 18;
  if (PARAMS.spawnRate > 0.0 && celestialBodies.length < maxAutoBodies) {
    spawnAccumulator += dt * PARAMS.spawnRate;
    if (spawnAccumulator >= 1.0) {
      spawnAccumulator -= 1.0;
      spawnBody('random');
    }
  }

  for (let i = celestialBodies.length - 1; i >= 0; i--) {
    const body = celestialBodies[i];
    body.update(dt, elapsedTime);
    if (!body.alive) {
      celestialBodies.splice(i, 1);
    }
  }

  const Rs = getSchwarzschildRadius();
  updateDebrisParticles(dt, Rs);

  blackHolePass.uniforms.uFlashIntensity.value *= Math.exp(-dt * 3.2);

  // 1º PASSO: Renderiza os objetos 3D no RenderTarget RGBA16F
  renderer.setRenderTarget(sceneRenderTarget);
  renderer.setClearColor(0x000000, 0.0);
  renderer.clear(true, true, false);
  renderer.render(orbitalScene, camera);

  // 2º PASSO: Atualiza matrizes e uniforms do Raymarcher
  camera.updateMatrixWorld();
  blackHolePass.uniforms.tSceneData.value = sceneRenderTarget.texture;
  blackHolePass.uniforms.uCameraPos.value.copy(camera.position);
  blackHolePass.uniforms.uInvProjMatrix.value.copy(camera.projectionMatrixInverse);
  blackHolePass.uniforms.uCameraWorldMatrix.value.copy(camera.matrixWorld);
  blackHolePass.uniforms.uTime.value = elapsedTime;
  blackHolePass.uniforms.uRs.value = Rs;
  blackHolePass.uniforms.uLensing.value = PARAMS.lensing;
  blackHolePass.uniforms.uDiskSpeed.value = PARAMS.diskSpeed;
  blackHolePass.uniforms.uDoppler.value = PARAMS.doppler;
  blackHolePass.uniforms.uDiskDensity.value = PARAMS.diskDensity;

  // 3º PASSO: Renderiza o Composer
  composer.render();

  updateTelemetryHUD();
}

// ============================================================================
// 13. REDIMENSIONAMENTO DE TELA E ROTAÇÃO DO CELULAR (RETRATO <-> PAISAGEM)
// ============================================================================
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  applyResolutionScale(PARAMS.renderScale, false);
});

animate();
