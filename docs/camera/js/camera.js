/* The live page: camera in, MediaPipe landmarks, this project's rules out.
 *
 * Nothing in here decides anything. Every verdict on screen comes from
 * js/guidance.js -- the checked port of pipeline/guidance.py, byte-identical
 * to the copy the slider demo runs and compared case by case against the
 * Python by tools/build_static.py. This file's whole job is:
 *
 *   1. get frames from the visitor's camera into a canvas,
 *   2. hand them to MediaPipe Face Landmarker,
 *   3. translate its output into the measurements guidance.py reads
 *      (js/measure.js, which labels every approximation),
 *   4. draw what came back.
 *
 * The frame is mirrored before anything measures it, which is what
 * pipeline/camera.py does (`cv2.flip(frame, 1)`, before the clean copy is
 * taken). Mirroring after measuring would flip the sign of yaw and make
 * "turn slightly to the left" mean the opposite of what the app means.
 *
 * No frame, and no measurement taken from one, leaves this tab. There is no
 * fetch of anything but the model and the WebAssembly runtime, both named in
 * the banner, and no upload of any kind.
 */
'use strict';

/* Injected by tools/build_static.py so the pinned version lives in exactly
 * one place. */
const VISION_MODULE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';
const WASM_ROOT = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const DETECT_SCORE = LANDMARK_CONSTANTS.DETECT_SCORE;

/* How many faces to ask for. More than one so "One person only" can actually
 * be demonstrated; not many more, because each one costs a mesh fit. */
const MAX_FACES = 3;

const el = (id) => document.getElementById(id);

const ui = {
  start: el('startBtn'),
  stop: el('stopBtn'),
  stage: el('stage'),
  video: el('video'),
  status: el('status'),
  severity: el('severity'),
  message: el('message'),
  detail: el('detail'),
  checklist: el('checklist'),
  measurements: el('measurements'),
  chain: el('chain'),
  fps: el('fps')
};

const ctx = ui.stage.getContext('2d', { willReadFrequently: true });

/* One import, started as soon as the page loads. Started early on purpose:
 * it is the only way to find out whether the runtime is reachable at all
 * before the visitor presses a button and waits for nothing. It fetches the
 * library, not the camera -- the camera is not touched until Start. */
const visionPromise = import(VISION_MODULE);

let landmarker = null;
let drawing = null;
let FaceLandmarkerClass = null;
let stream = null;
let frameHandle = null;
let lastVideoTime = -1;
let running = false;
let frames = 0;
let fpsMark = 0;

// ------------------------------------------------------------------ status

function setStatus(text, kind) {
  ui.status.textContent = text;
  ui.status.className = 'status' + (kind ? ' status--' + kind : '');
}

/* The page must be honest about failing too. Every path that cannot produce
 * a real measurement ends here rather than leaving the last good frame on
 * screen looking live. */
function fail(text) {
  setStatus(text, 'bad');
  ui.start.disabled = false;
  ui.stop.disabled = true;
}

// ----------------------------------------------------------- loading assets

/* A dynamic import rather than a static one so that a CDN that is blocked,
 * offline or slow produces a sentence on the page instead of a module-level
 * exception in the console and a button that does nothing. */
async function loadMediaPipe() {
  if (landmarker) { return landmarker; }

  setStatus('Loading the MediaPipe runtime…', 'busy');
  let vision;
  try {
    vision = await visionPromise;
  } catch (error) {
    throw new Error(
      'The MediaPipe library could not be loaded from ' + VISION_MODULE +
      '. It is served from a pinned CDN, so this usually means no network, ' +
      'an offline cache miss, or a blocker. Nothing on this page works ' +
      'without it. (' + error.message + ')');
  }

  let fileset;
  try {
    fileset = await vision.FilesetResolver.forVisionTasks(WASM_ROOT);
  } catch (error) {
    throw new Error(
      'The MediaPipe WebAssembly runtime could not be loaded from ' +
      WASM_ROOT + '. (' + error.message + ')');
  }

  setStatus('Loading the face landmark model (about 3.7 MB)…', 'busy');
  try {
    landmarker = await vision.FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
      runningMode: 'VIDEO',
      numFaces: MAX_FACES,
      /* Matched to this project's own DETECT_SCORE -- the bar for "there is
       * a face here worth measuring". It is a different detector's score on
       * a different scale, so the number is transcribed, not equivalent. */
      minFaceDetectionConfidence: DETECT_SCORE,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false
    });
  } catch (error) {
    throw new Error(
      'The face landmark model could not be loaded from ' + MODEL_URL +
      '. (' + error.message + ')');
  }

  drawing = new vision.DrawingUtils(ctx);
  FaceLandmarkerClass = vision.FaceLandmarker;
  window.__faceLandmarkerReady = true;
  return landmarker;
}

// ------------------------------------------------------------------ camera

function cameraError(error) {
  const name = error && error.name;
  if (name === 'NotAllowedError' || name === 'SecurityError') {
    return 'Camera permission was refused. Nothing else on this page can ' +
           'run without it. Allow the camera in your browser’s site ' +
           'settings and press Start again — the video still never ' +
           'leaves this tab.';
  }
  if (name === 'NotFoundError' || name === 'OverconstrainedError') {
    return 'No camera was found on this device.';
  }
  if (name === 'NotReadableError') {
    return 'The camera is there but something else is already using it. ' +
           'Close the other application and press Start again.';
  }
  return 'The camera could not be opened: ' +
         ((error && error.message) || String(error));
}

async function openCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      'This browser will not give a page the camera here. getUserMedia ' +
      'needs a secure context, so an https:// or localhost URL — a page ' +
      'opened straight off the filesystem cannot ask.');
  }
  setStatus('Asking for the camera…', 'busy');
  return navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 } },
    audio: false
  });
}

// ----------------------------------------------------------------- drawing

/* The mesh, drawn with MediaPipe's own tesselation. Deliberately the same
 * shape of overlay as pipeline/landmarks.py draw() -- a faint mesh, brighter
 * feature outlines -- but these are MediaPipe's 478 points, not the LBF
 * model's 68, so it is a denser mesh than the app draws. */
function drawMesh(faces, severity) {
  if (!drawing || !FaceLandmarkerClass || !faces || !faces.length) { return; }
  const F = FaceLandmarkerClass;
  const tone = severity === 'ok' ? '#0ca30c'
             : severity === 'warn' ? '#fab219' : '#d03b3b';
  for (const points of faces) {
    drawing.drawConnectors(points, F.FACE_LANDMARKS_TESSELATION,
                           { color: 'rgba(255,255,255,0.22)', lineWidth: 1 });
    for (const set of [F.FACE_LANDMARKS_FACE_OVAL, F.FACE_LANDMARKS_LEFT_EYE,
                       F.FACE_LANDMARKS_RIGHT_EYE, F.FACE_LANDMARKS_LIPS]) {
      drawing.drawConnectors(points, set, { color: tone, lineWidth: 1.6 });
    }
  }
}

function drawBox(traits, severity) {
  if (!traits.box) { return; }
  const tone = severity === 'ok' ? '#0ca30c'
             : severity === 'warn' ? '#fab219' : '#d03b3b';
  ctx.save();
  ctx.strokeStyle = tone;
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.strokeRect(traits.box[0], traits.box[1], traits.box[2], traits.box[3]);
  ctx.restore();
}

// -------------------------------------------------------------- the panels

function severityWord(severity) {
  return severity === 'ok' ? 'ready'
       : severity === 'warn' ? 'warning' : 'blocking';
}

function renderInstruction(verdict) {
  ui.severity.textContent = severityWord(verdict.severity);
  ui.severity.className = 'sev sev--' + verdict.severity;
  ui.message.textContent = verdict.message;
  ui.detail.textContent = verdict.detail || '';
}

/* Which measurement each checklist row is really standing on, so a row that
 * is green because nothing measured it says so instead of reading as a pass.
 * The labels are guidance.py's own. */
const ROW_SOURCE = {
  'Face visible': 'detected',
  'Only one person': 'faces',
  'Close enough': 'facePx',
  'Facing forward': 'yaw',
  'Head upright': 'roll',
  'Eyes unobstructed': 'eyeMismatch',
  'Eyes open': 'flags',
  'Neutral expression': 'flags',
  'Both sides visible': 'flags',
  'Sharp': 'sharpness',
  'Good quality': 'qualityScore'
};

function renderChecklist(rows) {
  const list = document.createDocumentFragment();
  for (const row of rows) {
    const key = ROW_SOURCE[row.label];
    const info = key ? CameraMeasure.PROVENANCE[key] : null;
    const unavailable = info && info.kind === 'unavailable';

    const item = document.createElement('li');
    item.className = 'chk' + (unavailable ? ' chk--none'
                                          : row.ok ? ' chk--ok' : ' chk--bad');

    const mark = document.createElement('span');
    mark.className = 'chk__mark';
    mark.textContent = unavailable ? '—' : row.ok ? '✓' : '×';
    item.appendChild(mark);

    const body = document.createElement('span');
    body.className = 'chk__body';
    const label = document.createElement('b');
    label.textContent = row.label;
    body.appendChild(label);

    if (unavailable) {
      const tag = document.createElement('em');
      tag.className = 'chk__note';
      tag.textContent = ' not measured on this page';
      body.appendChild(tag);
    } else if (info && info.kind === 'approximate') {
      const tag = document.createElement('em');
      tag.className = 'chk__note';
      tag.textContent = ' approximated';
      body.appendChild(tag);
    }

    if (!row.ok && !unavailable) {
      const fix = document.createElement('span');
      fix.className = 'chk__fix';
      fix.textContent = row.fix;
      body.appendChild(fix);
    }
    item.appendChild(body);
    if (info) { item.title = info.note; }
    list.appendChild(item);
  }
  ui.checklist.replaceChildren(list);
}

function renderChain(rows) {
  const list = document.createDocumentFragment();
  let chosen = false;
  for (const row of rows) {
    const item = document.createElement('li');
    item.className = 'rule' + (row.chosen ? ' rule--chosen' : '')
                   + (row.fired && !row.chosen ? ' rule--queued' : '');
    const name = document.createElement('code');
    name.textContent = row.name;
    item.appendChild(name);
    if (row.chosen) {
      const tag = document.createElement('span');
      tag.textContent = 'shown';
      tag.className = 'rule__tag';
      item.appendChild(tag);
      chosen = true;
    } else if (row.fired) {
      const tag = document.createElement('span');
      tag.textContent = 'also firing';
      tag.className = 'rule__tag';
      item.appendChild(tag);
    }
    list.appendChild(item);
  }
  if (!chosen && rows.length) {
    const item = document.createElement('li');
    item.className = 'rule rule--clear';
    item.textContent = 'nothing fired';
    list.appendChild(item);
  }
  ui.chain.replaceChildren(list);
}

const MEASUREMENT_ROWS = [
  ['detected', 'detected', (t) => String(Boolean(t.detected))],
  ['faces', 'faces', (t) => String(t.faces)],
  ['facePx', 'facePx', (t) => t.facePx === null ? '—' : t.facePx + ' px'],
  ['yaw', 'yaw', (t) => t.yaw === null ? '—' : t.yaw + '°'],
  ['roll', 'roll', (t) => t.roll === null ? '—' : t.roll + '°'],
  ['sharpness', 'sharpness',
   (t) => t.sharpness === null ? '—' : String(t.sharpness)],
  ['shadowClip', 'shadowClip',
   (t) => t.shadowClip === null ? '—' : String(t.shadowClip)],
  ['highlightClip', 'highlightClip',
   (t) => t.highlightClip === null ? '—' : String(t.highlightClip)],
  ['qualityScore', 'qualityScore', () => 'null'],
  ['parts.eyeMismatch', 'eyeMismatch',
   (t) => t.parts ? String(t.parts.eyeMismatch) : '—'],
  ['parts.flags', 'flags',
   (t) => t.parts && t.parts.flags.length ? t.parts.flags.join(', ')
                                          : '(none)']
];

const KIND_WORD = {
  exact: 'same measurement',
  approximate: 'approximated',
  unavailable: 'not measured'
};

function renderMeasurements(traits) {
  const body = document.createDocumentFragment();
  for (const [label, key, read] of MEASUREMENT_ROWS) {
    const info = CameraMeasure.PROVENANCE[key];
    const row = document.createElement('tr');
    row.className = 'prov prov--' + info.kind;

    const name = document.createElement('th');
    name.scope = 'row';
    name.textContent = label;
    row.appendChild(name);

    const value = document.createElement('td');
    value.className = 'num';
    value.textContent = read(traits);
    row.appendChild(value);

    const kind = document.createElement('td');
    const tag = document.createElement('span');
    tag.className = 'tag tag--' + info.kind;
    tag.textContent = KIND_WORD[info.kind];
    kind.appendChild(tag);
    row.appendChild(kind);

    const note = document.createElement('td');
    note.className = 'note';
    note.textContent = info.note;
    row.appendChild(note);

    body.appendChild(row);
  }
  ui.measurements.replaceChildren(body);
}

// -------------------------------------------------------------- the loop

function sampler(width, height) {
  /* ImageData for a rectangle of the clean frame. Taken before the mesh is
   * drawn, for the same reason pipeline/camera.py keeps an untouched copy:
   * measuring sharpness through a wireframe drawn over the face measures the
   * wireframe. */
  return (rect) => {
    const w = Math.min(rect[2], width - rect[0]);
    const h = Math.min(rect[3], height - rect[1]);
    if (w <= 0 || h <= 0) { return null; }
    return ctx.getImageData(rect[0], rect[1], w, h);
  };
}

function tick() {
  if (!running) { return; }
  frameHandle = requestAnimationFrame(tick);

  const video = ui.video;
  if (!video.videoWidth || video.readyState < 2) { return; }

  /* Nothing new to look at. Leaving the canvas alone keeps the last frame
   * and its mesh on screen rather than redrawing the frame without one,
   * which reads as a flicker. */
  if (video.currentTime === lastVideoTime) { return; }
  lastVideoTime = video.currentTime;

  const width = video.videoWidth;
  const height = video.videoHeight;
  if (ui.stage.width !== width || ui.stage.height !== height) {
    ui.stage.width = width;
    ui.stage.height = height;
  }

  /* Mirror on the way in, so everything downstream -- the measurements and
   * the overlay alike -- lives in the same flipped frame the app measures. */
  ctx.save();
  ctx.translate(width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, width, height);
  ctx.restore();

  let result;
  try {
    result = landmarker.detectForVideo(ui.stage, performance.now());
  } catch (error) {
    stopCamera();
    fail('MediaPipe stopped: ' + error.message);
    return;
  }

  const faces = (result && result.faceLandmarks) || [];
  const traits = CameraMeasure.traitsFor(faces, width, height,
                                         sampler(width, height));

  /* The whole point of the page: the decision is not made here. */
  const frameShape = [height, width];
  const verdict = Guidance.instruction(traits, frameShape, 'idle');

  drawMesh(faces, verdict.severity);
  drawBox(traits, verdict.severity);

  renderInstruction(verdict);
  renderChecklist(Guidance.checklist(traits, frameShape));
  renderChain(Guidance.chain(traits, frameShape));
  renderMeasurements(traits);

  window.__lastVerdict = verdict;
  window.__lastTraits = traits;
  window.__faceCount = faces.length;

  frames += 1;
  const now = performance.now();
  if (now - fpsMark > 1000) {
    ui.fps.textContent = frames + ' fps · ' + width + '×' + height;
    frames = 0;
    fpsMark = now;
  }
}

// ------------------------------------------------------------- start / stop

async function startCamera() {
  ui.start.disabled = true;
  try {
    await loadMediaPipe();
  } catch (error) {
    fail(error.message);
    return;
  }

  try {
    stream = await openCamera();
  } catch (error) {
    fail(cameraError(error));
    return;
  }

  ui.video.srcObject = stream;
  try {
    await ui.video.play();
  } catch (error) {
    stopCamera();
    fail('The video stream would not start: ' + error.message);
    return;
  }

  running = true;
  lastVideoTime = -1;
  fpsMark = performance.now();
  ui.stop.disabled = false;
  document.body.classList.add('is-live');
  setStatus('Live. The video and every number below stay in this tab.', 'good');
  frameHandle = requestAnimationFrame(tick);
}

function stopCamera() {
  running = false;
  if (frameHandle) { cancelAnimationFrame(frameHandle); frameHandle = null; }
  if (stream) {
    for (const track of stream.getTracks()) { track.stop(); }
    stream = null;
  }
  ui.video.srcObject = null;
  ctx.clearRect(0, 0, ui.stage.width, ui.stage.height);
  ui.start.disabled = false;
  ui.stop.disabled = true;
  ui.fps.textContent = '';
  document.body.classList.remove('is-live');
  setStatus('Camera stopped. Nothing was recorded and nothing was sent.', '');
}

ui.start.addEventListener('click', startCamera);
ui.stop.addEventListener('click', stopCamera);
window.addEventListener('pagehide', stopCamera);

/* Draw the panels once from an empty read, so the page shows what it will
 * show rather than three blank boxes. `{}` is the "no trait read at all"
 * case guidance.py answers with "Starting camera". */
renderInstruction(Guidance.instruction({}, null, 'idle'));
renderChecklist(Guidance.checklist(CameraMeasure.traitsFor([], 0, 0, null),
                                   null));
renderMeasurements(CameraMeasure.traitsFor([], 0, 0, null));
renderChain([]);
setStatus('Camera off. Press Start — the camera is not touched until ' +
          'you do.', '');

/* Find out now whether the library is reachable, rather than letting the
 * visitor press Start and wait on a fetch that was never going to arrive.
 * The camera is untouched either way. */
visionPromise.then(function (vision) {
  FaceLandmarkerClass = vision.FaceLandmarker;
  window.__mediapipeReachable = true;
}, function (error) {
  window.__mediapipeReachable = false;
  ui.start.disabled = true;
  setStatus(
    'The MediaPipe library at ' + VISION_MODULE + ' could not be reached, so ' +
    'the camera demo cannot run here. Everything this page says about what ' +
    'it would do is still accurate. (' + error.message + ')', 'bad');
});
