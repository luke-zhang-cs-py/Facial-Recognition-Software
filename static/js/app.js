/*
 * app.js — dashboard wiring.
 *
 * The browser does no image work at all: the video element is an MJPEG
 * stream produced by OpenCV on the server, and everything else is a poll
 * of /api/status once a second. That keeps this file to plumbing.
 */

const $ = (id) => document.getElementById(id);

const video = $('video');
const placeholder = $('placeholder');
const camBtn = $('camBtn');
const regBtn = $('regBtn');
const trainBtn = $('trainBtn');
const attBtn = $('attBtn');
const nameInput = $('nameInput');
const errBox = $('err');

let state = { running: false, mode: 'idle' };

async function post(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  const data = await res.json().catch(() => ({ ok: false, error: 'Bad response' }));
  if (!res.ok || data.ok === false) throw new Error(data.error || 'Request failed');
  return data;
}

function showError(msg) {
  errBox.textContent = msg;
  errBox.style.display = msg ? 'block' : 'none';
}

/* The MJPEG stream is only attached while the camera is on. Clearing src
 * on stop is what actually closes the connection browser-side. */
function setStream(on) {
  if (on && !video.src) {
    video.src = '/video_feed?t=' + Date.now();
    video.classList.add('live');
    placeholder.style.display = 'none';
  } else if (!on && video.src) {
    video.removeAttribute('src');
    video.classList.remove('live');
    placeholder.style.display = 'block';
  }
}

const MODES = {
  idle: { text: 'Idle — camera on', dot: 'on' },
  register: { text: 'Registering…', dot: 'rec' },
  attendance: { text: 'Attendance running', dot: 'live' },
};

function render(s) {
  state = s;
  setStream(s.running);

  const mode = s.running ? (MODES[s.mode] || MODES.idle) : { text: 'Camera off', dot: '' };
  $('modeflagText').textContent = mode.text;
  $('modedot').className = 'dot ' + mode.dot;

  camBtn.textContent = s.running ? 'Stop camera' : 'Start camera';
  camBtn.classList.toggle('active', s.running);

  const busy = s.mode === 'register';
  regBtn.disabled = busy;
  regBtn.textContent = busy ? 'Capturing…' : 'Capture 30 samples';

  attBtn.textContent = s.mode === 'attendance' ? 'Stop attendance' : 'Start attendance';
  attBtn.classList.toggle('active', s.mode === 'attendance');
  attBtn.disabled = !s.modelExists && s.mode !== 'attendance';

  $('modelState').textContent = s.modelExists ? 'trained' : 'not trained yet';

  // registration progress
  const r = s.register || {};
  const showProgress = s.mode === 'register' || r.finished;
  $('regProgress').style.display = showProgress ? 'block' : 'none';
  if (showProgress) {
    $('regFill').style.width = (100 * (r.captured || 0) / (r.target || 30)) + '%';
    $('regLabel').textContent = r.name || '—';
    $('regCount').textContent = `${r.captured || 0}/${r.target || 30}`;
  }

  fillList($('userList'), s.users, (u) => `[${u.id}] ${u.name}`, 'none yet');
  fillList($('todayList'), s.today,
    (a) => `${a.name}<span class="when">${a.timestamp.slice(11, 19)}</span>`, 'nobody yet');
  fillList($('eventList'), s.events,
    (e) => `${e.message}<span class="when">${e.at}</span>`, '—',
    (e) => e.kind === 'success');

  $('todayCount').textContent = (s.today || []).length;

  if (s.error) showError(s.error);
}

function fillList(el, items, fmt, empty, isSuccess) {
  if (!items || !items.length) {
    el.innerHTML = `<li class="muted">${empty}</li>`;
    return;
  }
  el.innerHTML = items
    .map((it) => `<li class="${isSuccess && isSuccess(it) ? 'success' : ''}">${fmt(it)}</li>`)
    .join('');
}

async function refresh() {
  try {
    render(await (await fetch('/api/status')).json());
  } catch (_) {
    /* server restarting or gone — the next tick will pick it back up */
  }
}

camBtn.onclick = async () => {
  showError('');
  try {
    await post(state.running ? '/api/camera/stop' : '/api/camera/start');
    if (state.running) setStream(false);   // drop the stream immediately
  } catch (e) { showError(e.message); }
  refresh();
};

regBtn.onclick = async () => {
  showError('');
  try {
    await post('/api/register', { name: nameInput.value });
    nameInput.value = '';
  } catch (e) { showError(e.message); }
  refresh();
};

trainBtn.onclick = async () => {
  showError('');
  trainBtn.disabled = true;
  trainBtn.textContent = 'Training…';
  try {
    const r = await post('/api/train');
    trainBtn.textContent = `Trained on ${r.images} images`;
    setTimeout(() => { trainBtn.textContent = 'Retrain model'; }, 2500);
  } catch (e) {
    showError(e.message);
    trainBtn.textContent = 'Retrain model';
  }
  trainBtn.disabled = false;
  refresh();
};

attBtn.onclick = async () => {
  showError('');
  const stopping = state.mode === 'attendance';
  try {
    await post(stopping ? '/api/attendance/stop' : '/api/attendance/start');
  } catch (e) { showError(e.message); }
  refresh();
};

nameInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') regBtn.click(); });

refresh();
setInterval(refresh, 1000);
