// WebGL half of ThreadWaveform: the thread/ribbon shaders, the theme palette, the scene setup and
// the PCM -> envelope reduction. Copied from the legacy WaveformDeck
// (frontend/src/components/workbench/WaveformDeck.tsx), which stays untouched for the legacy app;
// nothing here imports it or any legacy module (`@/services/api`, `@/types`).

import * as THREE from 'three';

/** Envelope samples across the call (one texel each). */
export const COUNT = 2048;
/** Fixed drawing-buffer height; the canvas CSS fills the stage and the compositor scales it. */
export const STAGE_BUFFER_H = 128;
/** A window whose peak stays under this is digital silence: Store's PII mute writes exact zeros. */
const SILENT_PEAK = 1e-4;
/** Shortest flat run worth marking, in seconds. */
const MIN_SILENT_SECONDS = 0.25;

/* Theme-aware palette for the shader ribbons. The CSS layer themes the deck chrome; the WebGL
 * shader colors cannot read CSS variables, so they live here and update on theme change.
 *
 * Color-space contract (unchanged from legacy): the fragment shader performs raw vec3 arithmetic
 * and writes gl_FragColor directly — no colorspace_fragment include — so the uniform values ARE
 * the displayed sRGB channels. setRGB stores raw values, unlike Color(hex), which would linearize
 * and darken the ribbons. Agent = teal, Caller = amber on both themes; light uses deeper tints
 * that stay readable on the Frost & Ink canvas. */
function ribbonColor(r: number, g: number, b: number): THREE.Color {
  return new THREE.Color().setRGB(r, g, b);
}

export function resolveRibbonPalette(theme: string) {
  if (theme === 'light') {
    return {
      tint: ribbonColor(0.02, 0.43, 0.56),
      pale: ribbonColor(0.82, 0.94, 0.98),
      callerTint: ribbonColor(0.7, 0.33, 0.04),
      callerPale: ribbonColor(0.98, 0.95, 0.88),
    };
  }
  return {
    tint: ribbonColor(0.12, 0.78, 0.98),
    pale: ribbonColor(0.72, 0.95, 0.99),
    callerTint: ribbonColor(0.89, 0.63, 0.36),
    callerPale: ribbonColor(1.0, 0.94, 0.84),
  };
}

export const VERTEX_SHADER = `
  uniform sampler2D uEnvelope;
  uniform float uTime, uCursor, uHover, uY, uClick, uLayer;
  varying vec2 vUv;
  varying vec3 vPosition;
  void main(){
    vUv=uv;
    vec4 sampleValue=texture2D(uEnvelope,vec2(uv.x,0.5));
    float amp=uLayer<0.5?sampleValue.r:sampleValue.g;
    float edge=pow(sin(uv.x*3.14159265),0.45);
    float phase=uv.x*15.0+uLayer*2.2;
    float twist=phase+0.22*sin(uTime*0.5+uv.x*12.0);
    float width=(0.10+amp*0.64)*edge;
    float across=(uv.y-0.5)*2.0;
    float center=sin(phase)*(.08+amp*.20)*edge;
    float distance=uv.x-uCursor;
    float wake=exp(-distance*distance*95.0)*uHover;
    float age=uTime-uClick;
    float ripple=sin(abs(distance)*48.0-age*9.0)*exp(-abs(distance)*5.0-age*2.5)*step(0.0,age);
    vec3 p=vec3((uv.x-.5)*1.84,center+across*width*cos(twist),across*width*sin(twist));
    p.y+=sin(across*8.0+phase*2.0+uTime*.6)*amp*.028*edge;
    p.y+=wake*(uY*.22+sin(across*3.0+uTime)*.045)+ripple*.10*edge;
    p.z+=wake*.18;
    vPosition=p;
    gl_Position=projectionMatrix*modelViewMatrix*vec4(p,1.0);
  }`;

export const FRAGMENT_SHADER = `
  precision highp float;
  uniform float uLayer,uRole,uProgress,uCursor,uHover;
  uniform vec3 uTint,uPale,uCallerTint,uCallerPale;
  varying vec2 vUv;
  varying vec3 vPosition;
  void main(){
    vec3 normal=normalize(cross(dFdx(vPosition),dFdy(vPosition)));
    float facing=abs(normal.z);
    float silk=pow(abs(dot(normal,normalize(vec3(-.3,.8,1.0)))),9.0);
    float rim=pow(1.0-facing,2.0);
    vec3 tint=uRole<.5?uTint:uCallerTint;
    vec3 pale=uRole<.5?uPale:uCallerPale;
    float thread=.5+.5*sin(vUv.y*900.0);
    float hem=pow(abs(vUv.y-.5)*2.0,20.0);
    float inspected=exp(-pow((vUv.x-uCursor)*12.0,2.0))*uHover;
    float played=1.0-smoothstep(uProgress-.003,uProgress+.003,vUv.x);
    vec3 color=tint*(.42+.30*facing+.18*thread)+pale*(silk*.7+rim*.40+hem*.3+inspected*.13);
    color*=.8+played*.22;
    float fade=smoothstep(0.0,.035,vUv.x)*(1.0-smoothstep(.965,1.0,vUv.x));
    gl_FragColor=vec4(color,fade*(.72+silk*.14));
  }`;

export interface Scene3 {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.OrthographicCamera;
  material: THREE.ShaderMaterial;
  layers: THREE.Mesh[];
  texture: THREE.DataTexture;
}

/** Build the two-layer thread scene on `canvas`. Throws when WebGL is unavailable. */
export function createScene(canvas: HTMLCanvasElement, envelope: Float32Array): Scene3 {
  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, powerPreference: 'low-power' });
  try {
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);
    const scene = new THREE.Scene();
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 20);
    camera.position.z = 5;
    const texture = new THREE.DataTexture(envelope, COUNT, 1, THREE.RGBAFormat, THREE.FloatType);
    texture.minFilter = texture.magFilter = THREE.LinearFilter;
    texture.needsUpdate = true;
    const material = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      uniforms: {
        uEnvelope: { value: texture },
        uTime: { value: 0 },
        uCursor: { value: 0.5 },
        uHover: { value: 0 },
        uY: { value: 0 },
        uClick: { value: -10 },
        uProgress: { value: 0 },
        uLayer: { value: 0 },
        uRole: { value: 0 },
        uTint: { value: new THREE.Color() },
        uPale: { value: new THREE.Color() },
        uCallerTint: { value: new THREE.Color() },
        uCallerPale: { value: new THREE.Color() },
      },
      vertexShader: VERTEX_SHADER,
      fragmentShader: FRAGMENT_SHADER,
    });
    const layers: THREE.Mesh[] = [];
    for (let layer = 0; layer < 2; layer++) {
      const mat = material.clone();
      mat.uniforms.uEnvelope.value = texture;
      mat.uniforms.uLayer.value = layer;
      mat.uniforms.uRole.value = layer; // provisional; corrected from the turn->channel map on load
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 1, 512, 24), mat);
      mesh.frustumCulled = false;
      scene.add(mesh);
      layers.push(mesh);
    }
    return { renderer, scene, camera, material, layers, texture };
  } catch (error) {
    renderer.dispose();
    throw error;
  }
}

export function syncPalette(layers: THREE.Mesh[]) {
  const theme = document.documentElement.getAttribute('data-theme') || 'dark';
  const palette = resolveRibbonPalette(theme);
  for (const mesh of layers) {
    const u = (mesh.material as THREE.ShaderMaterial).uniforms;
    u.uTint.value = palette.tint;
    u.uPale.value = palette.pale;
    u.uCallerTint.value = palette.callerTint;
    u.uCallerPale.value = palette.callerPale;
  }
}

export function disposeScene(s: Scene3) {
  for (const mesh of s.layers) {
    mesh.geometry.dispose();
    (mesh.material as THREE.ShaderMaterial).dispose();
  }
  s.material.dispose();
  s.texture.dispose();
  s.renderer.dispose();
}

/**
 * Reduce decoded PCM to the ribbon envelope, in place in `out` (RGBA texels: R = channel 0,
 * G = channel 1; mono feeds both). Same math as legacy — RMS windows, normalized with a 0.65
 * gamma, smoothed into cloth folds — plus one honest addition: windows that are digital silence
 * (Store mutes PII by zeroing samples) are pinned back to zero after smoothing, so a muted span
 * reads as a flat thread instead of being filled in by its neighbours. Returns those flat runs as
 * [start, end] fractions of the call.
 */
export function computeEnvelope(buffer: AudioBuffer, out: Float32Array): Array<[number, number]> {
  out.fill(0);
  const channels = [buffer.getChannelData(0), buffer.getChannelData(Math.min(1, buffer.numberOfChannels - 1))];
  const silent = new Uint8Array(COUNT);
  let maximum = 0;
  for (let i = 0; i < COUNT; i++) {
    const start = Math.floor((i * buffer.length) / COUNT);
    const end = Math.floor(((i + 1) * buffer.length) / COUNT);
    let peak = 0;
    for (let channel = 0; channel < 2; channel++) {
      const data = channels[channel];
      let sum = 0;
      for (let s = start; s < end; s++) {
        const v = data[s];
        sum += v * v;
        const a = v < 0 ? -v : v;
        if (a > peak) peak = a;
      }
      const value = Math.sqrt(sum / Math.max(1, end - start));
      out[i * 4 + channel] = value;
      maximum = Math.max(maximum, value);
    }
    silent[i] = end > start && peak < SILENT_PEAK ? 1 : 0;
  }
  if (maximum) {
    for (let i = 0; i < COUNT; i++) {
      for (let c = 0; c < 2; c++) out[i * 4 + c] = Math.pow(out[i * 4 + c] / maximum, 0.65);
    }
  }
  const peaks = out.slice();
  for (let i = 0; i < COUNT; i++) {
    for (let c = 0; c < 2; c++) {
      if (silent[i]) {
        out[i * 4 + c] = 0;
        continue;
      }
      let sum = 0;
      let weight = 0;
      for (let j = -18; j <= 18; j++) {
        const w = 19 - Math.abs(j);
        sum += peaks[Math.max(0, Math.min(COUNT - 1, i + j)) * 4 + c] * w;
        weight += w;
      }
      out[i * 4 + c] = sum / weight;
    }
  }

  const minWindows = Math.max(2, Math.ceil((MIN_SILENT_SECONDS / Math.max(buffer.duration, 1e-6)) * COUNT));
  const runs: Array<[number, number]> = [];
  let runStart = -1;
  for (let i = 0; i <= COUNT; i++) {
    const s = i < COUNT && silent[i] === 1;
    if (s && runStart < 0) runStart = i;
    if (!s && runStart >= 0) {
      if (i - runStart >= minWindows) runs.push([runStart / COUNT, i / COUNT]);
      runStart = -1;
    }
  }
  return runs;
}
