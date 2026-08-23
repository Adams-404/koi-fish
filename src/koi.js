import * as THREE from "three/webgpu";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import {
  Fn,
  Loop,
  float,
  vec2,
  vec3,
  uniform,
  positionGeometry,
  positionLocal,
  normalLocal,
  positionWorld,
  normalWorld,
  sin,
  cos,
  abs,
  pow,
  exp,
  step,
  clamp,
  saturate,
  smoothstep,
  max,
  length,
  oneMinus,
  mx_fractal_noise_float,
  normalView,
  mix,
  texture,
  dot,
  normalize,
} from "three/tsl";

/**
 * A koi in water.
 *
 * The GLB (Tripo) is authored nose-toward -X, dorsal +Y. We want a top-down
 * koi, so `base` re-bases the model to nose +X / dorsal +Z / lateral ±Y; after
 * that the whole fish steers with a single rotation about world Z.
 *
 *   swimmer (position)  ->  .rotation.z = heading + head crane
 *     pitch             ->  .rotation.y = nose up / down toward the camera
 *       bank            ->  .rotation.x = roll about its own spine
 *         base (fixed)  ->  the GLB
 *
 * Everything else — the body bend, the caustics, the ripples — is TSL, so it
 * runs on the GPU and the CPU only pushes a handful of scalars per frame.
 */

const FOV = 15;
const CAM_Z = 11;
const TAU = Math.PI * 2;

const clamp01 = (v, a, b) => (v < a ? a : v > b ? b : v);
// frame-rate independent easing: 0 -> 1 over time, k = "how eager"
const approach = (k, dt) => 1 - Math.exp(-k * dt);
const smooth01 = (t) => (t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t));
// a smooth 0 -> 1 -> 0 bump centred on c, half-width w
const bump = (u, c, w) => {
  const k = (u - c) / w;
  return k * k >= 1 ? 0 : (1 - k * k) ** 2;
};

// ---------------------------------------------------------------- the spine
//
// The GLB's four morph targets cannot bend this fish. Rendering the mesh at
// morph amplitudes from 0 to 2.15 shows only a tail flick — and past ~1.2 the
// caudal fin collapses into a spike. So the body is bent here instead.
//
// Curvature kappa(s) along the body gives a tangent angle; stepping along that
// angle gives the deformed centreline. Because every step is the same arc
// length the body cannot stretch, which is what lets it curl right round
// without tearing. Each vertex then rides that centreline at its own lateral
// offset. Morph targets 0/1 stay at zero; 2/3 still drive the pectoral fins,
// and their displacement is carried through the bend.
const SPINE_N = 32;
const WAVE = 4.6; // radians of travelling wave across the body
const HEAD_A = 0.03;
const HEAD_B = 0.3;

// Curvature profile for a head turn: a bump over the front quarter of the body
// with unit integral, so feeding it `-angle` cranes the head by exactly that
// much relative to the rest of the fish, which frame() then pays back.
const headBump = Fn(([s]) => {
  const span = float(HEAD_B - HEAD_A);
  const u = s.sub(HEAD_A).div(span);
  const inside = step(float(HEAD_A), s).mul(step(s, float(HEAD_B)));
  return oneMinus(cos(u.mul(TAU)))
    .div(span)
    .mul(inside);
});

// Integrate the centreline from the nose out to `s`. The partial weight on the
// straddling step keeps it smooth instead of quantised, and branch-free.
// The caudal fin — everything past x > 0.30, some 5,000 vertices and only
// 0.02-0.03 units thick — is a membrane, not a paddle. It trails the tail base
// by about a fifth of a cycle and flexes harder than the peduncle it hangs off,
// which is what gives a swimming fish its whip. FIN_LAG is that delay.
const FIN_LAG = 1.4; // radians the fin trails the tail base

// Integrate the centreline from the nose out to `s`. The partial weight on the
// straddling step keeps it smooth instead of quantised, and branch-free.
const spineAt = Fn(([s, wave, ph, hold, crane]) => {
  const ds = float(1 / SPINE_N);
  const x = float(-0.5).toVar();
  const z = float(0).toVar();
  const th = float(0).toVar();
  Loop(SPINE_N, ({ i }) => {
    const u = float(i).mul(ds);
    const w = clamp(s.sub(u).div(ds), 0, 1);
    const fin = smoothstep(0.76, 0.96, u);
    // Two envelopes, because the travelling wave and a turn are different
    // moves. Carps are subcarangiform: cruising undulation is confined to the
    // back half and grows to the tail tip. A turn is a C through the whole
    // body, so it starts much further forward.
    const envWave = smoothstep(0.25, 0.62, u)
      .mul(float(0.3).add(u.mul(u).mul(0.7)))
      .mul(fin.mul(0.6).add(1));
    const envHold = smoothstep(0.06, 0.45, u).mul(fin.mul(0.3).add(1));
    // ...and the fin lags, so it is still coming through as the body starts back
    const wavePh = ph.add(u.mul(WAVE)).sub(fin.mul(FIN_LAG));
    const kappa = envWave
      .mul(wave)
      .mul(sin(wavePh))
      .add(envHold.mul(hold))
      .sub(crane.mul(headBump(u)));
    th.addAssign(kappa.mul(ds).mul(w));
    x.addAssign(cos(th).mul(ds).mul(w));
    z.addAssign(sin(th).mul(ds).mul(w));
  });
  return vec3(x, z, th);
});

// ---------------------------------------------------------------- the water
const RIPPLES = 10; // click rings plus the koi's own wake
const RIPPLE_SPEED = 1.6; // world units / second
const RIPPLE_LIFE = 1.05; // 1/e decay

// Ridged fractal noise reads as caustics: the ridges are the thin bright
// filaments light makes on a pool floor.
const caustic = Fn(([p, t]) => {
  // Two ridged layers at different scales, drifting against each other. The
  // ridges are the thin bright filaments light makes on a pool floor; crossing
  // two sets is what gives caustics their shifting interference.
  const d1 = vec2(sin(t.mul(0.13)).mul(0.35), cos(t.mul(0.11)).mul(0.35));
  const n1 = mx_fractal_noise_float(
    vec3(p.mul(3.2).add(d1), t.mul(0.085)),
    3,
    2,
    0.5,
    1,
  );
  const r1 = pow(saturate(oneMinus(abs(n1))), 9);

  const d2 = vec2(cos(t.mul(0.09)).mul(0.5), sin(t.mul(0.12)).mul(0.5));
  const n2 = mx_fractal_noise_float(
    vec3(p.mul(5.1).add(d2).add(11.3), t.mul(0.11)),
    2,
    2,
    0.5,
    1,
  );
  const r2 = pow(saturate(oneMinus(abs(n2))), 12);

  // a slow coarse mask, so the surface has calm patches instead of being
  // uniformly busy from edge to edge
  const calm = smoothstep(
    -0.35,
    0.55,
    mx_fractal_noise_float(vec3(p.mul(0.85), t.mul(0.04)), 2, 2, 0.5, 1),
  );
  return r1.add(r2.mul(0.8)).mul(calm.mul(0.75).add(0.25));
});

// ------------------------------------------------------------------ the eyes
//
// The GLB has no eye geometry — the eyes are painted into a 2K atlas whose UV
// layout is auto-generated and unusable for a UV-space trick. So they were
// found in 3D instead: sample the texture at every head vertex, keep the ones
// that come back near-black with a sideways-facing normal, and cluster. That
// lands two tight blobs, which are the pupils.
//
// At this scale a pupil is a handful of pixels, so shifting it would be
// invisible. What is visible on a real fish is the specular highlight — the
// eye is wet and glossy, and the glint sits wherever your reflection falls on
// it. So the glint is what moves: its offset is the direction to the camera
// resolved into the eye's own frame, which means it drifts across the eye as
// the koi turns and centres when it comes round to face you.
const EYE_L = [-0.3896, 0.2447, 0.067];
const EYE_R_ = [-0.4262, 0.2454, -0.0671];
const EYE_R = 0.021; // the wet eyeball, a little wider than the painted pupil
const JAW_HINGE = [-0.415, 0.205]; // where the lower jaw articulates
const BODY_AXIS_Y = 0.21; // roughly the centreline height, for axial twist
const GLINT_R = 0.0045;

export function createKoi({ mount, url, onProgress }) {
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const coarse = matchMedia("(hover: none)").matches;

  const renderer = new THREE.WebGPURenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.12;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  mount.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(FOV, 1, 0.1, 60);
  camera.position.set(0, 0, CAM_Z);

  // ---------------------------------------------------------------- uniforms
  const uWave = uniform(1.6);
  const uPhase = uniform(0);
  const uHold = uniform(0);
  const uCrane = uniform(0);
  const uNow = uniform(0);
  const uScale = uniform(1); // world units per body length, for the caustics
  // x, y = where the glint sits within the eye; z = how bright it is
  const uGlint = uniform(new THREE.Vector3(0, 0, 0.5));
  // buccal pump: the jaw drops, then the gill covers flare to push the water
  // back out. Out of phase with each other, the way a fish actually breathes.
  const uJaw = uniform(0);
  const uGill = uniform(0);
  // 0 at the glass, 1 down at the bottom. Water absorbs, so the koi should
  // wash out toward the water's own colour as it sinks and warm up as it rises.
  const uDeep = uniform(0.5);
  // The dorsal, anal and pelvic fins have no morph targets, so they are the
  // last rigid parts of the koi. uFin flutters them a beat behind the body,
  // uErect stands the dorsal up under way, uTwist lets roll propagate down the
  // spine instead of the whole fish rotating as one piece.
  const uFin = uniform(0.008);
  const uErect = uniform(0);
  const uTwist = uniform(0);
  const uRipple = Array.from({ length: RIPPLES }, () =>
    uniform(new THREE.Vector4(0, 0, -999, 0)),
  );
  let nextRipple = 0;

  // Every ripple is a decaying ring travelling outward. It shows up twice: as
  // a bright crest of its own, and as a displacement of the caustic pattern —
  // which is what actually sells it as a disturbance on a water surface.
  const rippleAt = (p) => {
    let crest = float(0);
    let push = vec2(0, 0);
    for (const r of uRipple) {
      const d = p.sub(r.xy);
      const dist = max(length(d), 0.0001);
      const age = uNow.sub(r.z);
      const x = dist.sub(age.mul(RIPPLE_SPEED));
      const env = exp(x.mul(x).mul(-7))
        .mul(exp(age.mul(-1 / RIPPLE_LIFE)))
        .mul(r.w)
        .mul(step(float(0), age));
      const wv = sin(x.mul(-24)).mul(env);
      crest = crest.add(wv);
      push = push.add(d.div(dist).mul(wv).mul(0.55));
    }
    return { crest, push };
  };

  const litWater = (p) => {
    const { crest, push } = rippleAt(p);
    const c = caustic(p.div(uScale).add(push), uNow);
    return c.add(crest.mul(1.15));
  };

  // ---------------------------------------------------------------- lighting
  scene.add(new THREE.HemisphereLight(0xffe9dd, 0x8e1d10, 1.5));

  // Modelling light — raked, so the scales and fins get some form.
  const key = new THREE.DirectionalLight(0xfff4e8, 3.1);
  key.position.set(-2.4, 3.1, 5.4);
  scene.add(key);

  const rim = new THREE.DirectionalLight(0xffd2b4, 1.15);
  rim.position.set(2.6, -0.8, -1.6);
  scene.add(rim);

  // Shadow light — contributes no colour, exists only to throw the drop shadow.
  // Near head-on, because a raked light smears the shadow half a body away.
  const caster = new THREE.DirectionalLight(0xffffff, 0.0001);
  caster.position.set(-1.3, 1.75, 13);
  caster.castShadow = true;
  caster.shadow.mapSize.set(2048, 2048);
  caster.shadow.radius = 7;
  caster.shadow.bias = -0.0009;
  caster.shadow.camera.near = 1;
  caster.shadow.camera.far = 26;
  scene.add(caster, caster.target);

  // ------------------------------------------------------------ water layer
  // Additive light over the red page: dappled caustics plus any live ripples.
  const waterMat = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
  });
  waterMat.colorNode = vec3(1, 0.94, 0.84);
  waterMat.opacityNode = clamp(litWater(positionWorld.xy).mul(0.19), 0, 0.36);
  const water = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), waterMat);
  water.renderOrder = -2;
  scene.add(water);

  // ------------------------------------------------------------ shadow catcher
  // Invisible except where the koi shadows it, so the shadow lands on the red
  // page itself — including across the big type behind the canvas.
  const catcher = new THREE.Mesh(
    new THREE.PlaneGeometry(1, 1),
    new THREE.ShadowNodeMaterial({
      color: 0x2b0700, // keep the shadow inside the red family, not grey
      opacity: 0.3,
      transparent: true,
    }),
  );
  catcher.receiveShadow = true;
  catcher.renderOrder = -1;
  scene.add(catcher);

  // ------------------------------------------------------------------- rig
  const swimmer = new THREE.Group();
  const pitch = new THREE.Group();
  const bank = new THREE.Group();
  const base = new THREE.Group();
  base.setRotationFromMatrix(
    new THREE.Matrix4().makeBasis(
      new THREE.Vector3(-1, 0, 0), // model +X (tail) -> rig -X
      new THREE.Vector3(0, 0, 1), // model +Y (dorsal) -> rig +Z, faces camera
      new THREE.Vector3(0, 1, 0), // model +Z (flank) -> rig +Y
    ),
  );
  bank.add(base);
  pitch.add(bank);
  swimmer.add(pitch);
  scene.add(swimmer);

  // ----------------------------------------------------------------- state
  //
  // Calibrated against a screen capture of the original. Tracking the koi's
  // orange body frame by frame gave a median travel of 0.054 viewport widths
  // per second (p90 0.13, peak 0.30), and a body-bend ratio — lateral extent
  // over length — swinging between 0.40 (near straight) and 1.73 (curled past
  // a U) on a ~0.3 Hz cycle. So: very slow, and very bendy.
  const SPEED_MAX = 0.42; // viewport widths / second
  const SPEED_MIN = 0.01; // it never fully stops, but it nearly does
  // Koi routine turns run 88-1050 deg/s; even their slowest is above the
  // 54 deg/s this was peaking at. 2.2 rad/s is 126 deg/s — the bottom of
  // their range, which suits an unhurried hero animation.
  const YAW_MAX = 2.2; // rad/s
  // Measured koi turns bend the body 0.17-0.40 body lengths laterally; this
  // gain lands at the top of that.
  const HOLD_MAX = 1.8;
  // Swimming animals hold Strouhal number St = f*A/U between 0.25 and 0.35,
  // and tail amplitude A between 0.15 and 0.25 body lengths. Ours was sweeping
  // 0.43-0.69 L at a frequency far too low for the speed it was making — St
  // near 0.56, well outside the band — which is why the tail looked like it
  // was thrashing without driving anything. Frequency is now DERIVED from the
  // amplitude and speed instead of being tuned independently.
  const FIN_DUTY = 0.62; // fraction of the pectoral cycle that is propulsive
  const BREATH_HZ = 0.95; // ~57 a minute; a resting carp runs about 60
  const STROUHAL = 0.3;
  const AMP_PER_GAIN = 0.169; // A/L per unit of wave gain, from the spine model

  // Coming up to look at you. Let the cursor rest and the koi drifts over,
  // rolls upright and pitches its nose toward the glass — the pose in the
  // reference still — then goes back to swimming the moment you move.
  const STILL_T = 0.85; // s of a motionless pointer before it turns to look
  const GAZE_R = 0.8; // body lengths — it has to be near you first
  const GAZE_PITCH = -1.1; // rad, ~63 degrees: clearly at you, still a 3/4
  const GAZE_RISE = 0.34; // barely closes the distance — it turns, it does not lunge

  const view = {
    w: 4,
    h: 2.8,
    len: 2,
    bx: 1.3,
    by: 0.7,
    roamX: 1,
    roamY: 0.3,
    homeY: 0,
  };
  const pointer = new THREE.Vector2(0.13, 0.02); // -0.5 .. 0.5 of the viewport
  const target = new THREE.Vector3();

  let mesh = null;
  let weights = null;
  let heading = Math.atan2(-0.55, -0.84); // nose down-left, as in the comp
  let yaw = 0; // rad/s
  let speed = SPEED_MIN; // viewport widths / s, always along `heading`
  let curve = 0; // -1..1 steady body curvature; the body IS the rudder
  let curveSlow = 0; // slow follower of `curve`, so roll can be washed out
  let phase = 0; // travelling wave along the body
  let finPhase = 0;
  let bankAngle = 0.14;
  let pitchAngle = 0;
  let depth = 0;
  let look = 0; // head yaw relative to the body, radians
  let gaze = 0; // 0..1 swimming <-> facing you
  let flick = 0; // brief tail-flick when a ripple lands on it
  let lastStroke = 0; // which half-beat we last shed a wake ripple on
  let breathBrace = 0; // pectorals countering the opercular jet
  // Idle tics. A fish holding station is never perfectly still: every 12-32 s
  // it gapes, shivers, or flicks a pectoral. Rare enough to feel involuntary.
  let ticKind = 0;
  let ticT = -1;
  let ticAt = 9;
  let pointerSpeed = 0; // viewport widths / second
  let engaged = false; // has the visitor moved the pointer at all yet?
  let lastMove = -10;
  let running = false;
  let visible = true;
  let raf = 0;

  const TIC_DUR = [2.4, 0.85, 0.7]; // gape, shimmy, fin flick

  const pos = new THREE.Vector3();
  const camLocal = new THREE.Vector3();
  const invRot = new THREE.Quaternion();
  const clock = { elapsedTime: 0, last: 0 };
  const advance = (now) => {
    const dt = clock.last ? Math.min((now - clock.last) / 1000, 0.05) : 0;
    clock.last = now;
    clock.elapsedTime += dt;
    return dt;
  };

  function layout() {
    const w = mount.clientWidth || innerWidth;
    const h = mount.clientHeight || innerHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();

    view.h = 2 * CAM_Z * Math.tan(((FOV / 2) * Math.PI) / 180);
    view.w = view.h * camera.aspect;
    // Nose-to-tail on screen. A phone in portrait has far less width to spend,
    // so the koi takes a bigger share of it there than on a wide desktop.
    const wide = clamp01((camera.aspect - 0.7) / 0.9, 0, 1);
    view.len = Math.min(view.w * (0.74 - 0.26 * wide), view.h * 0.78);
    view.bx = view.w * 0.3;
    view.by = view.h * 0.18;
    // Idle drift is held to whatever keeps the whole fish on screen, so an
    // untouched page never ends up with a koi parked half out of frame.
    view.roamX = Math.max(view.w * 0.06, view.w / 2 - view.len * 0.45);
    view.roamY = Math.max(view.h * 0.06, view.h / 2 - view.len * 0.45);
    // On a phone the type block sits above centre, so the koi idles below it
    // instead of parking on top of the headline.
    view.homeY = -view.h * 0.16 * (1 - wide);

    swimmer.scale.setScalar(view.len);
    uScale.value = view.len;

    water.scale.set(view.w * 1.2, view.h * 1.2, 1);
    water.position.z = -view.len * 0.42;
    catcher.scale.set(view.w * 2.4, view.h * 2.4, 1);
    catcher.position.z = -view.len * 0.34;

    const r = Math.max(view.w, view.h) * 0.7;
    Object.assign(caster.shadow.camera, {
      left: -r,
      right: r,
      top: r,
      bottom: -r,
    });
    caster.shadow.camera.updateProjectionMatrix();
  }

  function setSpine(wave, ph, hold, crane) {
    uWave.value = wave;
    uPhase.value = ph;
    uHold.value = hold;
    uCrane.value = crane;
  }

  // A fixed, composed pose — nose down-left across the type, the way the koi
  // sits in the comp. Used when the visitor has asked for reduced motion.
  function pose() {
    swimmer.position.set(view.w * 0.13, view.homeY, 0);
    heading = Math.atan2(-0.55, -0.84);
    swimmer.rotation.z = heading;
    pitch.rotation.y = 0;
    bank.rotation.x = 0.14;
    uFin.value = 0.005 + eased * 0.016;
    uErect.value = eased;
    setSpine(1.2, 0.9, 0.5, 0);
    if (weights) {
      weights[0] = weights[1] = 0;
      weights[2] = weights[3] = 0.15;
    }
    renderer.render(scene, camera);
  }

  // A last-resort wall. The koi turns back on its own long before this bites;
  // it only matters if the viewport is resized out from under it.
  function bound(p, axis, limit, dt, k = 1.4) {
    const over = Math.abs(p[axis]) - limit;
    if (over <= 0) return;
    p[axis] -= Math.sign(p[axis]) * over * approach(k, dt);
  }

  function spawnRipple(wx, wy, strength) {
    const u = uRipple[nextRipple % RIPPLES];
    nextRipple++;
    u.value.set(wx, wy, clock.elapsedTime, strength);
  }

  // Every half-beat the tail throws water sideways, so it sheds a ring from
  // wherever the tail happens to be. Strung together as the koi moves, those
  // rings are its wake — the surface was previously glassy no matter how hard
  // it was swimming, which is the one thing you would never see in a pond.
  function shedWake(t, drive) {
    const stroke = Math.floor(phase / Math.PI);
    if (stroke === lastStroke) return;
    lastStroke = stroke;
    const s = clamp01(drive.speed, 0, 1);
    if (s < 0.06) return;
    const reach = view.len * 0.5 * Math.cos(pitchAngle);
    spawnRipple(
      pos.x - Math.cos(heading) * reach,
      pos.y - Math.sin(heading) * reach,
      0.1 + s * 0.3,
    );
  }

  // --------------------------------------------------------------- steering
  //
  // The koi goes where the cursor goes. Leave the cursor still and, once it has
  // caught up, it turns and looks at you.
  function steer(t, dt) {
    pointerSpeed *= Math.exp(-6 * dt);
    flick *= Math.exp(-2.4 * dt);

    if (ticT < 0) {
      if (t > ticAt) {
        ticT = 0;
        ticKind = Math.floor(Math.random() * 3);
      }
    } else {
      ticT += dt;
      if (ticT > TIC_DUR[ticKind]) {
        ticT = -1;
        ticAt = t + 12 + Math.random() * 20;
      }
    }
    // a smooth 0 -> 1 -> 0 over the tic, so nothing starts or stops abruptly
    const ticE =
      ticT < 0 ? 0 : Math.sin((ticT / TIC_DUR[ticKind]) * Math.PI) ** 1.5;
    const gape = ticKind === 0 ? ticE : 0;
    const shimmy = ticKind === 1 ? ticE : 0;
    const finFlick = ticKind === 2 ? ticE : 0;

    const curX = pointer.x * view.w;
    const curY = pointer.y * view.h;
    const nx = Math.cos(heading);
    const ny = Math.sin(heading);
    // How far the nose sits ahead of `pos` *on screen*. Once the koi pitches
    // up to face you the body is 70-odd degrees out of the xy plane, so its
    // nose is barely ahead of its centre at all — the flat half-length below
    // is only correct when it is swimming level, hence the cosine.
    const reach = view.len * 0.5 * Math.cos(pitchAngle);
    const headX = pos.x + nx * reach;
    const headY = pos.y + ny * reach;
    const dCur = Math.hypot(curX - headX, curY - headY) / view.len;

    // ------------------------------------------------------------ the gaze
    // Slow to come round, quick to break off — being looked at should feel
    // earned, and losing it should be instant.
    const still = t - lastMove;
    // Hysteresis, so the two cannot chatter: it is much easier to hold a gaze
    // than to start one. Without this the koi flipped between facing you and
    // swimming several times a second.
    const holdR = GAZE_R * (1 + gaze * 1.6);
    const wantGaze = engaged && still > STILL_T && dCur < holdR ? 1 : 0;
    gaze += (wantGaze - gaze) * approach(wantGaze ? 0.42 : 1.1, dt);

    // ---------------------------------------------------------- the course
    target.set(
      clamp01(curX, -view.bx, view.bx),
      clamp01(curY, -view.by, view.by),
      0,
    );

    let err = Math.atan2(target.y - pos.y, target.x - pos.x) - heading;
    err = Math.atan2(Math.sin(err), Math.cos(err));
    const dist = Math.hypot(target.x - pos.x, target.y - pos.y) / view.w;

    // A fish steers by bending; with no water flowing past it, it has almost
    // no authority to turn. That coupling is what stops it pirouetting.
    // Only a light coupling to speed now. It exists so a parked koi cannot
    // pirouette, but leaning on it hard also stopped it pivoting when slow —
    // which is precisely when a real fish turns tightest.
    const authority = 0.75 + 0.25 * clamp01(speed / (SPEED_MAX * 0.55), 0, 1);
    const meander = 0.16 * Math.sin(t * 0.21) + 0.08 * Math.sin(t * 0.53 + 2.1);
    // The koi is nearly as long as the hero is tall, so nose-up it simply does
    // not fit. This settles it back onto a horizontal course.
    const level = -0.26 * Math.sin(2 * heading);
    // It still eases into the turn rather than snapping onto a bearing, but it
    // now commits in about half a second instead of a second and a half. At
    // the old rate a reversal peaked at 31 deg/s and took eight and a half
    // seconds to come round, which is what read as sluggish.
    // High gain into a hard clamp, so anything past about 35 degrees off
    // commands a full-lock turn and HOLDS it until it comes round. At low gain
    // the command bled away as fast as the error did, which is why a reversal
    // crawled round over eight seconds instead of committing to it.
    const wantYaw =
      clamp01(err * 2.0 + meander + level, -YAW_MAX, YAW_MAX) * authority;
    // Target zero while gazing rather than freezing the update: multiplying
    // the whole step by (1 - gaze) left a stale yaw turning the heading for as
    // long as the koi held still.
    yaw += (wantYaw * (1 - gaze) - yaw) * approach(1.6, dt);
    heading += yaw * dt;

    // coming round to face you, it also rights itself
    const g = smooth01(gaze); // soft at both ends of the turn
    if (g > 0.001) {
      let up = -Math.PI / 2 - heading;
      up = Math.atan2(Math.sin(up), Math.cos(up));
      heading += up * approach(0.9 * g, dt);
    }

    // A fish slows into a turn and picks the pace back up on the way out.
    // Without this the arc is wider than the viewport, because a low yaw rate
    // at full speed sweeps a turning circle bigger than the hero.
    const turnCost = 1 - 0.45 * clamp01(Math.abs(err) / Math.PI, 0, 1);
    const wantSpeed =
      dist < 0.1
        ? SPEED_MIN
        : clamp01(dist * 0.55 * turnCost, SPEED_MIN, SPEED_MAX);
    speed += (wantSpeed - speed) * approach(1.1, dt);
    speed *= 1 - 0.55 * Math.abs(yaw / YAW_MAX) * dt;
    speed *= 1 - g * 0.88; // still creeping, never quite parked

    // Thrust comes in pulses, twice a beat, so it surges and coasts inside
    // every stroke instead of gliding along at a constant rate.
    const surge = 1 + 0.16 * Math.abs(Math.sin(phase)) * (1 - gaze);
    const u = speed * view.w * surge * (1 + flick * 0.9);
    pos.x += Math.cos(heading) * u * dt;
    pos.y += Math.sin(heading) * u * dt;

    // once it is actually looking at you, park it on the cursor
    if (g > 0.35) {
      const k = approach(0.8 * ((g - 0.35) / 0.65), dt);
      pos.x += (target.x - pos.x) * k;
      pos.y += (target.y - pos.y) * k;
    }
    bound(pos, "x", view.bx * 1.1, dt);
    bound(pos, "y", view.by * 1.05, dt);

    const wasDepth = depth;
    depth += (view.len * 0.09 * Math.sin(t * 0.31) - depth) * approach(1.1, dt);
    const swimPitch = clamp01(
      (-(depth - wasDepth) / Math.max(dt, 1e-3)) * 0.5,
      -0.2,
      0.2,
    );
    const wantPitch = swimPitch * (1 - g) + GAZE_PITCH * g;
    pitchAngle += (wantPitch - pitchAngle) * approach(1.3, dt);

    // curvature tracks the turn and lags it, so the body is still unwinding
    // after the heading has settled
    curve += (yaw / YAW_MAX - curve) * approach(2.4, dt) * (1 - gaze);

    // Roll, washed out. Banking straight off `curve` meant a sustained turn
    // held a sustained lean — it wound up to 30 degrees over six seconds and
    // stayed there, which reads as capsizing rather than carving. A real fish
    // rolls in as the turn STARTS and comes back to level while still turning,
    // so bank follows only the part of the curvature that is still new: the
    // slow follower below is subtracted off, leaving the transient.
    curveSlow += (curve - curveSlow) * approach(0.85, dt);
    const onset = curve - curveSlow;
    const roll = 0.05 * Math.sin(phase + 0.9) * (1 - gaze);
    const wantBank = (0.12 - onset * 0.55 + roll) * (1 - gaze);
    const wasBank = bankAngle;
    bankAngle += (wantBank - bankAngle) * approach(2.6, dt);
    bank.rotation.x = bankAngle;
    pitch.rotation.y = pitchAngle;

    // ...and the roll travels down the body: the tail is still coming over
    // after the head has levelled. This uniform was declared and read by the
    // shader but never written — the twist has been dead until now.
    const rollRate = (bankAngle - wasBank) / Math.max(dt, 1e-3);
    uTwist.value +=
      (clamp01(-rollRate * 0.6, -0.26, 0.26) - uTwist.value) * approach(4, dt);

    // the head leads a turn, and swings round to you when it settles
    const wantLook = clamp01(yaw * 0.35, -0.3, 0.3) * (1 - gaze);
    look += (wantLook - look) * approach(3, dt);

    return { speed: speed / SPEED_MAX, curve, dCur, gape, shimmy, finFlick };
  }

  // ------------------------------------------------------------ body + fins
  function swim(dt, drive) {
    const lock = clamp01(Math.abs(drive.curve) * 1.7, 0, 1);
    const eased = clamp01(drive.speed, 0, 1);

    // Amplitude first: 0.15 body lengths peak-to-peak idling, 0.25 flat out —
    // the range swimming animals actually hold.
    const waveGain =
      (0.9 + eased * 0.7 + flick * 1.3 + drive.shimmy * 0.5) *
      (1 - gaze * 0.62);
    const ampL = Math.max(AMP_PER_GAIN * waveGain, 0.05);

    // ...then frequency is DERIVED from it, not tuned separately: swimming
    // animals hold Strouhal number St = f*A/U between 0.25 and 0.35, so
    // f = St * U / A with everything in body lengths. Tuning the two apart is
    // what let the tail thrash without driving anything. The floor keeps a
    // hovering koi alive when U is nearly zero.
    const bodyPerSec = (speed * view.w) / view.len;
    const freq =
      Math.max(0.28, (STROUHAL * bodyPerSec) / ampL) +
      flick * 1.6 +
      drive.shimmy * 5;
    phase += freq * TAU * dt * (1 - 0.8 * lock);

    uFin.value = 0.005 + eased * 0.016;
    uErect.value = eased;
    setSpine(waveGain, phase, drive.curve * HOLD_MAX, look);

    if (!weights) return;

    // A pectoral stroke is abduction, then adduction, then a REFRACTORY period
    // with the fin held in against the body — not the continuous oscillation
    // this was. Measured beat frequency runs about 1.2 Hz at a slow cruise up
    // to 2.1 Hz near one body length per second.
    finPhase += (1.2 + eased * 0.9) * (1 - 0.35 * gaze) * TAU * dt;
    const fc = (finPhase / TAU) % 1;
    const beat = (c) => (c < FIN_DUTY ? Math.sin((c / FIN_DUTY) * TAU) : 0);
    const strokeA = beat(fc);
    const strokeB = beat((fc + 0.1) % 1);

    // Damped hard while it holds station: head-on, a wide pectoral arc sweeps
    // across the front of the face and reads as the mouth working.
    const fan = (0.22 + 0.24 * (1 - eased)) * (1 - 0.82 * gaze);
    // nothing to brace against while hovering
    const brace = clamp01(drive.curve * 0.45, -0.45, 0.45) * (1 - gaze);
    // and they counter the jet leaving the opercular slit, once per breath
    const puff = breathBrace * 0.12;

    weights[0] = weights[1] = 0;
    // clamped: past +-1 the fins extrapolate into shapes the mesh was never
    // sculpted for, and head-on that reads as the face doing something odd
    weights[2] = clamp01(
      fan * strokeA + brace + puff + drive.finFlick * 0.55,
      -1,
      1,
    );
    weights[3] = clamp01(fan * strokeB - brace + puff, -1, 1);
  }

  function frame(now) {
    raf = requestAnimationFrame(frame);
    const dt = advance(now);
    const t = clock.elapsedTime;
    uNow.value = t;

    const drive = steer(t, dt);
    swim(dt, drive);
    shedWake(t, drive);

    // Buccal and opercular cavities expand and contract SYNCHRONOUSLY in the
    // carp — I had them alternating, mouth then gills, which is not how the
    // pump works. The two chambers move together and it is the valves that
    // alternate; the floor of the mouth may lead the lateral expansion
    // slightly, hence the small offset between these two centres.
    // Rate is ~60 breaths a minute for a resting carp, not the 0.62 Hz I had.
    const cyc = (t * BREATH_HZ) % 1;
    uJaw.value = bump(cyc, 0.25, 0.24) * 0.85 + drive.gape * 1.7;
    uGill.value = bump(cyc, 0.31, 0.26);
    // C: the pectorals brace against the jet leaving the opercular slit, just
    // after the chambers compress.
    breathBrace = bump(cyc, 0.52, 0.17);

    // The whole body recoils against the tail stroke, and the head yaws a
    // couple of degrees out of phase with it.
    const sway =
      view.len * 0.02 * Math.sin(phase + 1.7) * clamp01(drive.speed, 0, 1.5);
    // Coming up to the glass. The lens is long and far back, so simply turning
    // to face you would barely change its size — it has to actually close the
    // distance. Scaling x/y by the same ratio keeps it pinned under the cursor
    // while it grows.
    const z = depth + smooth01(gaze) * view.len * GAZE_RISE;
    // the water plane sits at -0.42 len; the glass is a little above the gaze
    uDeep.value = clamp01((view.len * 0.4 - z) / (view.len * 0.82), 0, 1);
    const keep = (CAM_Z - z) / CAM_Z;
    swimmer.position.set(
      (pos.x - Math.sin(heading) * sway) * keep,
      (pos.y + Math.cos(heading) * sway) * keep,
      z,
    );
    // `look` is paid back here: the spine bent the head by -look relative to
    // the body, so rotating everything by +look leaves the body on course and
    // the head turned.
    swimmer.rotation.z = heading + look + 0.05 * Math.sin(phase + 2.4);

    // Where does the viewer's reflection land on the eye? Take the direction
    // from the head to the camera into the model's own frame; its x/y are the
    // offset across the eye, since the eyeball faces along local z.
    swimmer.updateMatrixWorld(true);
    base.getWorldQuaternion(invRot).invert();
    camLocal
      .set(
        camera.position.x - swimmer.position.x,
        camera.position.y - swimmer.position.y,
        camera.position.z - swimmer.position.z,
      )
      .normalize()
      .applyQuaternion(invRot);
    uGlint.value.set(
      clamp01(camLocal.x * 1.35, -1, 1),
      clamp01(camLocal.y * 1.35, -1, 1),
      0.5 + 0.5 * gaze,
    );

    renderer.render(scene, camera);
  }

  // ---------------------------------------------------------------- events
  const onPointer = (e) => {
    const nx = e.clientX / innerWidth - 0.5;
    const ny = 0.5 - e.clientY / innerHeight;
    const now = clock.elapsedTime;
    const gap = Math.max(now - lastMove, 1 / 120);
    if (now - lastMove < 0.4) {
      // peak-held, decayed in steer() — a flick reads as fast for a moment
      pointerSpeed = Math.max(
        pointerSpeed,
        Math.hypot(nx - pointer.x, ny - pointer.y) / gap,
      );
    }
    pointer.set(nx, ny);
    lastMove = now;
    engaged = true;
  };

  const onTap = (e) => {
    onPointer(e);
    if (reduced) return;
    const wx = pointer.x * view.w;
    const wy = pointer.y * view.h;
    spawnRipple(wx, wy, 1);
    // a ring landing on top of it makes it flick its tail and put on a spurt
    const d = Math.hypot(wx - pos.x, wy - pos.y) / view.len;
    if (d < 1.2) flick = Math.min(1, flick + 1 - d / 1.2);
  };

  addEventListener("pointermove", onPointer, { passive: true });
  addEventListener("pointerdown", onTap, { passive: true });
  const onResize = () => {
    layout();
    if (running && reduced) pose();
  };
  addEventListener("resize", onResize);
  addEventListener("orientationchange", onResize);
  layout();

  // ------------------------------------------------------------------ load
  const api = {
    // How hard the surface is moving where the headline sits, 0..1. The type is
    // HTML behind a transparent canvas, so WebGL cannot refract it — but it can
    // report the disturbance and let CSS do the bending.
    surfaceEnergy() {
      let sum = 0;
      for (const r of uRipple) {
        const v = r.value;
        const age = clock.elapsedTime - v.z;
        if (age < 0 || age > 6) continue;
        const d = Math.hypot(v.x, v.y);
        const x = d - age * RIPPLE_SPEED;
        sum += Math.exp(-7 * x * x) * Math.exp(-age / RIPPLE_LIFE) * v.w;
      }
      return clamp01(sum, 0, 1);
    },
    setVisible(v) {
      if (v === visible) return;
      visible = v;
      if (running) v ? start() : stop();
    },
    dispose() {
      stop();
      removeEventListener("pointermove", onPointer);
      removeEventListener("pointerdown", onTap);
      removeEventListener("resize", onResize);
      removeEventListener("orientationchange", onResize);
      renderer.dispose();
    },
  };

  function start() {
    if (raf) return;
    if (reduced) return pose(); // hold still; no loop, no power drawn
    clock.last = 0;
    raf = requestAnimationFrame(frame);
  }
  function stop() {
    cancelAnimationFrame(raf);
    raf = 0;
  }

  return (async () => {
    await renderer.init();
    const gltf = await new GLTFLoader().loadAsync(url, (xhr) => {
      if (xhr.total) onProgress?.(xhr.loaded / xhr.total);
    });

    mesh = gltf.scene.getObjectByProperty("type", "Mesh");
    if (!mesh) throw new Error("no mesh in the GLB");
    weights = mesh.morphTargetInfluences;
    if (!weights || weights.length < 4)
      throw new Error("expected 4 morph targets");

    mesh.castShadow = true;
    mesh.frustumCulled = false;

    const src = mesh.material;
    const mat = new THREE.MeshPhysicalNodeMaterial({
      map: src.map,
      // Tripo ships baseColorFactor 0.8, which dims the texture 20%.
      color: 0xffffff,
      roughness: src.roughness ?? 0.5,
      metalness: src.metalness ?? 0,
      side: src.side,
    });
    // Koi scales are pearlescent — the colour shifts with viewing angle. That
    // is a thin-film effect, which MeshStandardNodeMaterial cannot express at
    // all; the physical one has a real iridescence term.
    // Koi iridophores stack anhydrous guanine crystals — refractive index
    // 1.83, not the 1.32 I had guessed — into multilayer reflectors, and over
    // 95% of the plates lie parallel to the scale surface, so the sheen is a
    // surface-aligned mirror rather than a diffuse shimmer. The iridophores
    // concentrate in the pale markings, so it is masked by how light the skin
    // is rather than applied flat over the whole fish.
    mat.iridescence = 1;
    mat.iridescenceIOR = 1.83;
    mat.iridescenceThicknessRange = [120, 420];
    const skin = texture(src.map);
    const lum = skin.r.mul(0.3).add(skin.g.mul(0.6)).add(skin.b.mul(0.1));
    mat.iridescenceNode = smoothstep(0.34, 0.8, lum).mul(0.5).add(0.07);

    // ---- the bend, on the GPU
    const s = positionGeometry.x.add(0.5).clamp(0, 1);
    const sp = spineAt(s, uWave, uPhase, uHold, uCrane);
    const ct = cos(sp.z);
    const st = sin(sp.z);
    mat.positionNode = Fn(() => {
      // Breathing, in two strokes. The jaw ROTATES about its articulation —
      // translating it straight down just slides the whole chin off the face,
      // which is what made the old version look wrong. Then the gill covers
      // flare to push the water back out.
      const jaw = smoothstep(-0.4, -0.46, positionGeometry.x).mul(
        smoothstep(0.235, 0.17, positionGeometry.y),
      );
      const ang = uJaw.mul(0.16).mul(jaw);
      const ca = cos(ang);
      const sa = sin(ang);
      const hx = positionGeometry.x.sub(JAW_HINGE[0]);
      const hy = positionGeometry.y.sub(JAW_HINGE[1]);
      const raw = positionLocal.sub(positionGeometry); // whatever the fins added
      const d = vec3(
        raw.x.add(hx.mul(ca).sub(hy.mul(sa)).sub(hx)),
        raw.y.add(hx.mul(sa).add(hy.mul(ca)).sub(hy)),
        raw.z,
      );

      // The operculum is the upper flank, not the throat — at x = -0.30 the
      // half-width jumps 0.09 -> 0.20, and it is the gill cover doing it. So
      // the flare is masked in y as well as x.
      const gillBand = smoothstep(0.13, 0.19, s).mul(smoothstep(0.3, 0.23, s));
      const gillHi = smoothstep(0.11, 0.2, positionGeometry.y).mul(
        smoothstep(0.32, 0.24, positionGeometry.y),
      );
      const flare = gillBand.mul(gillHi).mul(uGill).mul(0.3).add(1);
      // The three fins that have no morph targets. Masks are straight off the
      // measured bands: dorsal is the tall bit around x -0.10..+0.06, anal sits
      // low at x +0.06..+0.30, pelvic low around the belly.
      const px = positionGeometry.x;
      const py = positionGeometry.y;

      // The two pairs of barbels — sensory whiskers at x -0.50..-0.44, below
      // y 0.20 — are part of the head mesh and so were dead rigid. They are
      // soft filaments; they trail and sweep.
      const barbel = smoothstep(-0.435, -0.478, px).mul(
        smoothstep(0.206, 0.174, py),
      );
      const bSway = sin(uNow.mul(2.1).add(px.mul(9)))
        .mul(barbel)
        .mul(0.011);
      const bLift = sin(uNow.mul(1.5).add(2.2)).mul(barbel).mul(0.006);

      const dorsal = smoothstep(0.31, 0.36, py)
        .mul(smoothstep(-0.17, -0.1, px))
        .mul(smoothstep(0.14, 0.06, px));
      const anal = smoothstep(0.13, 0.07, py)
        .mul(smoothstep(0.06, 0.12, px))
        .mul(smoothstep(0.3, 0.22, px));
      const pelvic = smoothstep(0.09, 0.02, py)
        .mul(smoothstep(-0.13, -0.06, px))
        .mul(smoothstep(0.09, 0.02, px));
      const membrane = dorsal.add(anal).add(pelvic);
      // a beat behind the body, like the caudal
      const flutter = sin(uPhase.add(s.mul(WAVE)).sub(1.1))
        .mul(uFin)
        .mul(membrane);

      const off0 = positionGeometry.z
        .add(d.z)
        .mul(flare)
        .add(flutter)
        .add(bSway);
      // the dorsal stands up under way and folds back down when hovering
      const y0 = positionGeometry.y
        .add(d.y)
        .add(dorsal.mul(uErect).mul(0.035))
        .add(bLift);

      // Roll travels down the body rather than the whole fish turning as one
      // piece, so the tail is still coming round after the head has levelled.
      const tw = uTwist.mul(smoothstep(0.15, 1, s));
      const yc = y0.sub(BODY_AXIS_Y);
      const cw = cos(tw);
      const sw = sin(tw);
      const off = yc.mul(sw).add(off0.mul(cw));
      const yTwisted = yc.mul(cw).sub(off0.mul(sw)).add(BODY_AXIS_Y);
      // the cross-section rotates with the tangent, so the normal must too
      normalLocal.assign(
        vec3(
          normalLocal.x.mul(ct).sub(normalLocal.z.mul(st)),
          normalLocal.y,
          normalLocal.x.mul(st).add(normalLocal.z.mul(ct)),
        ).normalize(),
      );
      return vec3(
        sp.x.add(d.x.mul(ct)).sub(off.mul(st)),
        yTwisted,
        sp.y.add(d.x.mul(st)).add(off.mul(ct)),
      );
    })();

    // ---- caustics crawling over it. Sampled in world space, so the light
    // stays put and the fish swims through it rather than dragging it along.
    const facing = saturate(normalWorld.z).mul(0.75).add(0.25);
    // Clamped, unlike the plane: light can only ever be added. The signed
    // ripple crest used to push this negative in a trough, and a negative
    // tinted emissive subtracts unevenly per channel — which showed up as the
    // fish changing colour as a ring swept over it.
    const water = vec3(1, 0.88, 0.74)
      .mul(clamp(litWater(positionWorld.xy), 0, 1.6))
      .mul(facing)
      .mul(oneMinus(uDeep.mul(0.45)))
      .mul(0.22);

    // ---- the water column. Beer-Lambert in spirit: the deeper it sits, the
    // more of the water's own colour is between it and you, so it desaturates
    // toward that colour and loses contrast. This is what the gaze rise trades
    // on — it brightens as it comes up, instead of lunging at you.
    mat.colorNode = mix(
      texture(src.map),
      vec3(0.42, 0.13, 0.09),
      uDeep.mul(0.42),
    );

    // ---- wetness. A fish is wet: it has a hard grazing-angle sheen, and its
    // body is far glossier than its fins. The caudal membrane past x = 0.26 is
    // matte by comparison.
    mat.roughnessNode = float(0.34).add(
      smoothstep(0.26, 0.42, positionGeometry.x).mul(0.3),
    );
    const graze = oneMinus(saturate(normalView.z));
    const sheen = vec3(1, 0.95, 0.88)
      .mul(pow(graze, 3.5))
      .mul(oneMinus(uDeep.mul(0.5)))
      .mul(0.17);

    // ---- and the fins are 0.02-0.03 thick, so light gets through them
    const thin = max(
      smoothstep(0.26, 0.4, positionGeometry.x),
      smoothstep(0.34, 0.42, positionGeometry.y),
    );
    const through = vec3(1, 0.86, 0.72)
      .mul(thin)
      .mul(graze.mul(0.5).add(0.5))
      .mul(0.16);

    const glintAt = (c) => {
      // Measured in 3D, not in the xy plane. An xy-only distance is really an
      // infinite cylinder along z, and that cylinder cuts the curved eyeball in
      // a long arc — which is why the catchlight came out as a streak.
      const off = uGlint.xy.mul(EYE_R * 0.4);
      const eye = vec3(c[0], c[1], c[2]);
      const spot = vec3(
        float(c[0]).add(off.x),
        float(c[1]).add(off.y),
        float(c[2]),
      );
      // A wet convex eye carries two reflections: the hard specular of the
      // viewer, and a broad soft fill from the sky opposite it.
      const fill = vec3(
        float(c[0]).sub(off.x.mul(0.7)),
        float(c[1]).sub(off.y.mul(0.7)),
        float(c[2]),
      );
      const hard = smoothstep(
        GLINT_R,
        GLINT_R * 0.3,
        length(positionGeometry.sub(spot)),
      );
      const soft = smoothstep(
        EYE_R * 0.55,
        EYE_R * 0.12,
        length(positionGeometry.sub(fill)),
      );
      const wet = smoothstep(
        EYE_R,
        EYE_R * 0.6,
        length(positionGeometry.sub(eye)),
      );
      return hard.mul(0.8).add(soft.mul(0.09)).add(wet.mul(0.03));
    };
    const eyes = glintAt(EYE_L).add(glintAt(EYE_R_)).mul(uGlint.z);

    mat.emissiveNode = water
      .add(vec3(1, 0.97, 0.93).mul(eyes).mul(0.95))
      .add(sheen)
      .add(through);

    mesh.material = mat;

    gltf.scene.position.y = -0.24; // model sits on y = 0; centre it
    base.add(gltf.scene);

    pos.set(view.w * 0.13, view.homeY + view.h * 0.02, 0);
    swimmer.position.copy(pos);
    running = true;
    if (visible) start();
    return api;
  })();
}
