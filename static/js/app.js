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

  renderTraits(s);

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

/* ------------------------------------------------------------ live traits */

const DASH = '—';

function renderTraits(s) {
  const t = s.liveTraits;
  const traitsBtn = $('traitsBtn');
  traitsBtn.textContent = 'Trait readout: ' + (s.traitsOn ? 'ON' : 'OFF');
  traitsBtn.classList.toggle('active', !!s.traitsOn);

  if (!t || t.error) {
    ['ltDetected', 'ltSharp', 'ltBright', 'ltQuality', 'ltPose', 'ltAge', 'ltGender']
      .forEach((id) => { $(id).textContent = DASH; });
    $('ltFlags').innerHTML = t && t.error
      ? `<div class="chips"><span class="chip">${t.error}</span></div>` : '';
    return;
  }

  $('ltDetected').textContent = t.detected ? `yes (${t.faces})` : 'no face';
  $('ltSharp').textContent = t.sharpness != null ? t.sharpness.toFixed(0) : DASH;
  $('ltBright').textContent = t.brightness != null ? t.brightness.toFixed(0) : DASH;
  $('ltQuality').textContent = t.qualityScore != null ? t.qualityScore.toFixed(2) : DASH;
  $('ltPose').textContent = (t.yaw != null)
    ? `yaw ${t.yaw.toFixed(0)}° roll ${t.roll.toFixed(0)}°` : DASH;

  /* Both estimates carry their uncertainty in the UI, not just the JSON —
   * a bare label reads as fact in a way the number never does. */
  const noFace = t.demographicsSkipped ? 'no face' : DASH;
  $('ltAge').textContent = t.age
    ? `${t.age.label}  ${(t.age.confidence * 100).toFixed(0)}%${t.age.uncertain ? ' ?' : ''}`
    : noFace;
  $('ltGender').textContent = t.gender
    ? `${t.gender.label}  ${(t.gender.confidence * 100).toFixed(0)}%${t.gender.uncertain ? ' ?' : ''}`
    : noFace;

  $('ltFlags').innerHTML = (t.flags && t.flags.length)
    ? `<div class="chips">${t.flags.map((f) => `<span class="chip">${f}</span>`).join('')}</div>`
    : '';
}

$('traitsBtn').onclick = async () => {
  try {
    await post('/api/traits', { enabled: !state.traitsOn });
  } catch (e) { showError(e.message); }
  refresh();
};

/* --------------------------------------------------------------- analysis */

const fmt = (s, k) => (s && s[k] != null ? s[k] : DASH);

function sweepTable(block) {
  if (!block.available) return `<div class="note">${block.reason}</div>`;
  const rows = block.sweep.map((r) => {
    const cls = [];
    if (r.threshold === block.recommendedThreshold) cls.push('rec');
    if (r.threshold === block.currentThreshold) cls.push('cur');
    return `<tr class="${cls.join(' ')}"><td>${r.threshold}</td>
      <td>${r.accept}</td><td>${r.falseMatch}</td></tr>`;
  }).join('');
  return `
    <div class="metrics">
      <span>protocol</span><span>${block.protocol}</span>
      <span>accuracy</span><span>${block.accuracy}% over ${block.samples}</span>
    </div>
    <table class="sweep">
      <tr><th>thresh</th><th>accept %</th><th>false match %</th></tr>${rows}
    </table>
    <div class="note">Recommended <b>${block.recommendedThreshold}</b> &mdash;
      ${block.recommendedAccept}% accepted, ${block.recommendedFalseMatch}% false matches.
      ${block.currentThreshold != null && block.currentThreshold !== block.recommendedThreshold
        ? `attendance.py currently uses ${block.currentThreshold}.` : ''}</div>`;
}

function userCard(u) {
  const pct = u.samples ? Math.round(100 * u.usable / u.samples) : 0;
  const metric = (label, s, key) =>
    s ? `<span>${label}</span><span>${s[key]}</span>` : '';

  return `
    <div class="card">
      <div class="who">
        <h3>${u.name}</h3>
        <span class="pill ${u.verdict === 'good' ? 'good' : 'warn'}">${u.verdict}</span>
      </div>
      <div class="metrics">
        <span>usable</span><span>${u.usable}/${u.samples} (${pct}%)</span>
        ${metric('sharpness', u.sharpness, 'mean')}
        ${metric('brightness', u.brightness, 'mean')}
        ${metric('quality', u.quality, 'mean')}
        <span>pose spread</span><span>${u.yawSpread != null ? u.yawSpread + '°' : 'n/a'}</span>
        ${u.age ? `<span>age est.</span><span>${u.age.label} (${Math.round(u.age.agreement * 100)}% agree)</span>` : ''}
        ${u.gender ? `<span>gender est.</span><span>${u.gender.label} (${Math.round(u.gender.agreement * 100)}% agree)</span>` : ''}
      </div>
      ${Object.keys(u.flags).length
        ? `<div class="chips">${Object.entries(u.flags)
            .map(([k, v]) => `<span class="chip">${k} ×${v}</span>`).join('')}</div>` : ''}
      ${u.worstSamples.length
        ? `<div class="chips">${u.worstSamples.slice(0, 4)
            .map((w) => `<span class="chip n">${w.file}</span>`).join('')}</div>` : ''}
      ${u.recommendations.length
        ? `<ul class="tips">${u.recommendations.map((r) => `<li>${r}</li>`).join('')}</ul>` : ''}
    </div>`;
}

function renderAnalysis(rep) {
  const body = $('analysisBody');
  if (!rep) { body.innerHTML = ''; return; }

  if (!rep.totalSamples) {
    body.innerHTML = `<div class="note">No samples under dataset/ yet — register
      someone first, then run this.</div>`;
    return;
  }

  body.innerHTML = `
    <div class="agrid">${rep.users.map(userCard).join('')}</div>
    <div class="agrid">
      <div class="card"><div class="who"><h3>LBPH</h3>
        <span class="pill warn">in use today</span></div>${sweepTable(rep.lbph)}</div>
      <div class="card"><div class="who"><h3>SFace embeddings</h3>
        <span class="pill good">128-d</span></div>${sweepTable(rep.sface)}
        ${rep.sface.available ? `<div class="note">Genuine ${rep.sface.genuine.mean}
          vs impostor ${rep.sface.impostor.mean} (margin ${rep.sface.margin}).
          ${rep.sface.weakestPairs.length ? `Most confusable: user
          ${rep.sface.weakestPairs[0].a} vs ${rep.sface.weakestPairs[0].b}
          at ${rep.sface.weakestPairs[0].maxSimilarity}.` : ''}</div>` : ''}
      </div>
    </div>
    <div class="caveat">${rep.notes.map((n) => `&bull; ${n}`).join('<br>')}</div>`;
}

let analysisTimer = null;

async function pollAnalysis() {
  let s;
  try {
    s = await (await fetch('/api/analysis')).json();
  } catch (_) { return; }

  const showing = s.running;
  $('analysisProgress').style.display = showing ? 'flex' : 'none';
  if (showing) {
    const pct = s.total ? 100 * s.done / s.total : 0;
    $('analysisFill').style.width = pct + '%';
    $('analysisProgressText').textContent = `${s.done}/${s.total}`;
  }

  $('analyzeBtn').disabled = s.running;
  $('analyzeFreshBtn').disabled = s.running;
  $('analyzeBtn').textContent = s.running ? 'Analysing…' : 'Run analysis';

  if (s.error) showError(s.error);
  if (s.report) renderAnalysis(s.report);

  if (!s.running && analysisTimer) {
    clearInterval(analysisTimer);
    analysisTimer = null;
  }
}

async function startAnalysis(refreshAll) {
  showError('');
  try {
    await post('/api/analysis/start', { refresh: refreshAll });
  } catch (e) { showError(e.message); return; }
  if (!analysisTimer) analysisTimer = setInterval(pollAnalysis, 500);
  pollAnalysis();
}

$('analyzeBtn').onclick = () => startAnalysis(false);
$('analyzeFreshBtn').onclick = () => startAnalysis(true);

refresh();
pollAnalysis();
setInterval(refresh, 1000);
