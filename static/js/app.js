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
const attBtn = $('attBtn');
const nameInput = $('nameInput');
const errBox = $('err');

let state = { running: false, mode: 'idle' };

async /* Text that came from a person, on its way into innerHTML.
 *
 * Registered names are stored exactly as typed and rendered straight into the
 * dashboard, so a name of `<img src=x onerror=...>` was script that ran in
 * every browser that opened the page. That is stored XSS, and registering a
 * name is the app's main function.
 *
 * Single quotes are escaped too: unlike the email templates, some attributes
 * here are single-quoted. */
const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}[c]));

function post(url, body) {
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

  camBtn.textContent = s.running ? 'Close camera' : 'Open camera';
  camBtn.classList.toggle('active', s.running);

  const busy = s.mode === 'register';
  regBtn.disabled = busy;
  regBtn.textContent = busy ? 'Capturing…' : 'Start capture';

  attBtn.textContent = s.mode === 'attendance' ? 'Stop attendance' : 'Start attendance';
  attBtn.classList.toggle('active', s.mode === 'attendance');
  attBtn.disabled = !s.modelExists && s.mode !== 'attendance';


  // registration progress
  const r = s.register || {};
  const showProgress = s.mode === 'register' || r.finished;
  $('regProgress').style.display = showProgress ? 'block' : 'none';
  if (showProgress) {
    $('regFill').style.width = (100 * (r.captured || 0) / (r.target || 30)) + '%';
    $('regLabel').textContent = r.name || '—';
    $('regCount').textContent = `${r.captured || 0}/${r.target || 30}`;
    /* Which pose is being captured, which are done, which are still to come. */
    const stages = r.stages || [];
    $('stageList').innerHTML = stages.map((st, i) => {
      const done = i < (r.stage || 0);
      const active = i === (r.stage || 0);
      const n = active ? (r.stageCount || 0) : (done ? st.count : 0);
      return `<li class="${done ? 'pass' : (active ? '' : 'fail')}">${esc(st.label)}`
        + `<span class="when">${n}/${st.count}</span></li>`;
    }).join('');
  }

  renderReport(s.report);

  renderGuidance(s);
  renderTraits(s);

  fillList($('userList'), s.users, (u) => `[${esc(u.id)}] ${esc(u.name)}`, 'none yet');
  fillList($('todayList'), s.today,
    (a) => `${esc(a.name)}<span class="when">${esc(a.timestamp.slice(11, 19))}</span>`, 'nobody yet');
  fillList($('eventList'), s.events,
    (e) => `${esc(e.message)}<span class="when">${esc(e.at)}</span>`, '—',
    (e) => e.kind === 'success');

  $('todayCount').textContent = (s.today || []).length;

  if (s.error) showError(s.error);
}

function fillList(el, items, fmt, empty, isSuccess) {
  if (!items || !items.length) {
    el.innerHTML = `<li class="muted">${esc(empty)}</li>`;
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

async function toggleCamera(forceOn) {
  showError('');
  const turningOff = forceOn === undefined ? state.running : !forceOn;
  try {
    await post(turningOff ? '/api/camera/stop' : '/api/camera/start');
    if (turningOff) setStream(false);   // drop the stream immediately
  } catch (e) { showError(e.message); }
  refresh();
}

camBtn.onclick = () => toggleCamera();
$('openCamBtn').onclick = () => toggleCamera(true);

/* Release the camera when the page goes away. Without this the capture thread
 * keeps the device open — and the webcam light stays on — after the tab is
 * closed, which is both alarming and wrong for something that only needs the
 * camera while someone is looking at it. sendBeacon because a normal fetch is
 * cancelled during unload. */
function releaseCamera() {
  if (!state.running) return;
  const body = new Blob(['{}'], { type: 'application/json' });
  navigator.sendBeacon('/api/camera/stop', body);
}
window.addEventListener('pagehide', releaseCamera);
window.addEventListener('beforeunload', releaseCamera);

regBtn.onclick = async () => {
  showError('');
  try {
    await post('/api/register', { name: nameInput.value });
    nameInput.value = '';
  } catch (e) { showError(e.message); }
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

/* ------------------------------------------------------------- report */

let reportDismissed = null;

function stat(label, value) {
  return `<span>${label}</span><span>${value}</span>`;
}

/* A report on the enrollment, not a reading of the person. Everything here
 * is measured from the samples that were just written to disk. */
function renderReport(rep) {
  const panel = $('reportPanel');
  if (!rep || reportDismissed === rep.userId) {
    panel.style.display = 'none';
    return;
  }
  panel.style.display = 'block';
  $('reportSub').textContent =
    `${esc(rep.samples)} samples captured for ${esc(rep.name)}. What the system can now`
    + ` measure, and how well it should recognise them.`;

  const s = rep.sharpness || {}, q = rep.quality || {}, px = rep.facePx || {};
  const poses = rep.poseStages || {};
  const poseRows = Object.keys(poses).map((k) =>
    stat(k, `${poses[k]} samples`)).join('');

  const near = rep.nearestOther
    ? `Closest other enrolled face is <b>${esc(rep.nearestOther.name)}</b> at
       ${esc(rep.nearestOther.similarity)} similarity.
       ${rep.nearestOther.similarity >= rep.threshold
         ? '<b>That is above the recommended threshold — these two could be confused.</b>'
         : 'Comfortably below the recommended threshold.'}`
    : 'Nobody else is enrolled yet, so there is nothing to be confused with.';

  panel.querySelector('#reportBody').innerHTML = `
    <div class="agrid">
      <div class="card">
        <div class="who"><h3>Capture</h3>
          <span class="pill ${esc(rep.verdict === 'good' ? 'good' : 'warn')}">${esc(rep.verdict)}</span></div>
        <div class="metrics">
          ${stat('usable', `${rep.usable}/${rep.samples}`)}
          ${rep.sampleAccuracy != null
            ? stat('expected id rate', (100 * rep.sampleAccuracy).toFixed(0) + '%') : ''}
          ${stat('face size', px.mean != null ? px.mean + ' px' : '—')}
          ${stat('sharpness', s.mean != null ? s.mean : '—')}
          ${stat('quality', q.mean != null ? q.mean : '—')}
          ${stat('yaw spread', rep.yawSpread != null ? rep.yawSpread + '°' : '—')}
          ${stat('yaw range', rep.yawRange ? rep.yawRange[0] + '° to ' + rep.yawRange[1] + '°' : '—')}
        </div>
        ${Object.keys(rep.flags || {}).length
          ? `<div class="chips">${Object.entries(rep.flags)
              .map(([k, v]) => `<span class="chip">${k} ×${v}</span>`).join('')}</div>` : ''}
        ${rep.sampleAdvice ? `<div class="note">${rep.sampleAdvice}</div>` : ''}
        ${(rep.recommendations || []).length
          ? `<ul class="tips">${rep.recommendations.map((r) => `<li>${esc(r)}</li>`).join('')}</ul>` : ''}
      </div>

      <div class="card">
        <div class="who"><h3>Pose coverage</h3>
          <span class="pill good">${Object.keys(poses).length} angles</span></div>
        <div class="metrics">${poseRows}</div>
        <div class="note">Samples spread across angles generalise; thirty frames
          of one angle only recognise that angle.</div>
      </div>

      <div class="card">
        <div class="who"><h3>Recognition</h3>
          <span class="pill ${rep.thresholdReachable ? 'good' : 'warn'}">${rep.gallerySize} enrolled</span></div>
        <div class="metrics">
          ${stat('threshold', rep.threshold)}
          ${stat('false-match risk', (100 * rep.thresholdRisk).toFixed(2) + '%')}
        </div>
        <div class="note">${near}</div>
        ${!rep.thresholdReachable
          ? `<div class="note"><b>At this gallery size no measured threshold holds
             the risk under 1%.</b> Add a second factor.</div>` : ''}
      </div>
    </div>
    ${rep.fairness ? `
    <div class="agrid">
      <div class="card" style="grid-column:1/-1;">
        <div class="who"><h3>How evenly this system performs</h3>
          <span class="pill warn">audit</span></div>
        <div class="note" style="margin-top:0;">Measured across demographic
          groups on ${rep.fairness.corpus}. This measures the software, not
          you &mdash; the system does not infer anyone&rsquo;s ethnicity, and
          nothing about it is stored.</div>
        <table class="sweep">
          <tr><th>check</th><th>result</th><th>disparity</th></tr>
          ${rep.fairness.checks.map((c) => `<tr class="${c.disparity > 1.25 ? 'cur' : 'rec'}">
            <td>${esc(c.name)}</td><td>${c.value}</td><td>${c.disparity.toFixed(2)}x</td></tr>
            <tr><td colspan="3" style="color:var(--muted);font-size:10.5px;padding-top:0;">
            ${c.detail}</td></tr>`).join('')}
        </table>
      </div>
    </div>` : ''}
    <div class="caveat">Measured from the ${rep.samples} images just captured.
      These describe image quality and separability &mdash; how well this
      enrollment will work &mdash; not attributes of the person.</div>`;

  $('reportClose').onclick = () => {
    reportDismissed = rep.userId;
    panel.style.display = 'none';
  };
}

/* ---------------------------------------------------------- guidance */

/* One instruction at a time. The checklist below it explains *why*, but the
 * headline is a single thing to do -- people fix one problem at a time, and
 * a wall of simultaneous corrections gets ignored. */
function renderGuidance(s) {
  const bar = $('instrBar');
  const msg = $('guideMsg');
  const count = $('guideCount');
  const list = $('checkList');
  const idle = '<li class="muted">—</li>';

  const detail = $('guideDetail');

  if (!s.running) {
    bar.className = 'instrbar';
    msg.textContent = 'Camera off';
    detail.textContent = '';
    count.textContent = '';
    list.innerHTML = idle;
    return;
  }

  const t = s.liveTraits;
  const g = t && t.guidance;
  if (!g) {
    bar.className = 'instrbar';
    msg.textContent = 'Reading…';
    detail.textContent = '';
    count.textContent = '';
    list.innerHTML = idle;
    return;
  }

  bar.className = 'instrbar ' + g.severity;
  msg.textContent = g.message;
  /* The bar spans the width of the video, so the sentence explaining the
   * instruction fits beside it — no need to make people look elsewhere for
   * why they are being asked to move. */
  detail.textContent = g.detail || '';

  /* During capture the bar doubles as the progress readout, so the person
   * never has to look away from the lens to know how far along they are. */
  const r = s.register || {};
  const stage = (r.stages || [])[r.stage];
  count.textContent = (s.mode === 'register' && stage)
    ? `${r.stageCount || 0}/${stage.count}  ·  ${r.captured || 0}/${r.target}`
    : '';

  /* Only what is failing, as .steps rows — same shape as Activity and Today. */
  const checks = (t && t.checklist) || [];
  const bad = checks.filter((c) => !c.ok);
  list.innerHTML = bad.length
    ? bad.map((c) => `<li class="fail">${c.fix}</li>`).join('')
    : '<li class="muted">Nothing — you are framed correctly.</li>';
}

/* ------------------------------------------------------------ live traits */

const DASH = '—';

function renderTraits(s) {
  const t = s.liveTraits;
  const traitsBtn = $('traitsBtn');
  traitsBtn.textContent = 'Trait readout: ' + (s.traitsOn ? 'ON' : 'OFF');
  traitsBtn.classList.toggle('active', !!s.traitsOn);

  if (!t || t.error) {
    ['ltDetected', 'ltSharp', 'ltBright', 'ltQuality', 'ltPose', 'ltAge', 'ltGender', 'ltParts', 'ltSym', 'ltCheek', 'ltLive']
      .forEach((id) => { $(id).textContent = DASH; });
    $('ltFlags').innerHTML = t && t.error
      ? `<div class="chips"><span class="chip">${esc(t.error)}</span></div>` : '';
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
  /* A single number with its real hit-rate, rather than a 7-year bucket that
   * looks authoritative. Measured MAE is 12.4 years, so the coverage figure
   * is the honest part of this readout. */
  if (t.age && t.age.estimate) {
    const e = t.age.estimate;
    $('ltAge').textContent =
      `${Math.round(e.years)}y (${e.range[0]}–${e.range[1]}, ${(e.coverage * 100).toFixed(0)}%)`;
  } else {
    $('ltAge').textContent = t.age ? t.age.label : noFace;
  }
  $('ltGender').textContent = t.gender
    ? `${esc(t.gender.label)}  ${(t.gender.confidence * 100).toFixed(0)}%${t.gender.uncertain ? ' ?' : ''}`
    : noFace;

  const pm = t.parts;
  $('ltParts').textContent = pm
    ? `eyes ${pm.eyeOpenRight.toFixed(2)}/${pm.eyeOpenLeft.toFixed(2)}  mouth ${pm.mouthOpen.toFixed(2)}`
    : DASH;
  $('ltSym').textContent = pm
    ? `offset ${pm.centreOffset.toFixed(2)}  eyes ${pm.eyeMismatch.toFixed(2)}`
    : DASH;
  $('ltCheek').textContent = pm && pm.cheekWidthRatio != null
    ? `width ${pm.cheekWidthRatio.toFixed(2)}  prom ${pm.cheekProminence.toFixed(2)}`
    : DASH;

  const lv = s.liveness;
  $('ltLive').textContent = (lv && lv.available)
    ? (lv.score != null ? `${esc(lv.verdict)} ${lv.score.toFixed(2)}` : esc(lv.verdict))
    : 'model missing';

  $('ltFlags').innerHTML = (t.flags && t.flags.length)
    ? `<div class="chips">${t.flags.map((f) => `<span class="chip">${esc(f)}</span>`).join('')}</div>`
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

/* A threshold swept on a handful of enrolled people cannot see false matches,
 * so the report also carries a recommendation calibrated on ~98k identities.
 * Show the warning prominently when the local sweep is not trustworthy. */
function calibrationBlock(sface) {
  if (!sface.available || !sface.calibration) return '';
  const c = sface.calibration;
  const warn = sface.warning
    ? `<div class="err" style="margin-top:10px;">${sface.warning}</div>` : '';
  const unreachable = !c.reachable
    ? `<div class="err" style="margin-top:8px;">At ${c.gallerySize} enrolled,
       no measured threshold keeps the risk under 1%. Embeddings alone are not
       enough at this scale — add a second factor.</div>` : '';
  const contrast = (!sface.localSweepTrusted && sface.localSweepThreshold != null)
    ? `<div class="note">This dataset's own sweep would have said
       <b>${sface.localSweepThreshold}</b> — a
       ${(100 * c.riskAtLocalChoice).toFixed(2)}% gallery-wide false-match risk
       at ${c.gallerySize} enrolled.</div>` : '';
  const d = c.disparity;

  return `
    ${warn}
    <div class="note" style="margin-top:10px;"><b>Gallery-size calibration.</b>
      ${c.summary}</div>
    ${unreachable}
    ${contrast}
    <div class="note">Risk is not evenly shared: at threshold ${d.threshold},
      ${d.worst[0]} faces falsely match someone
      ${(100 * d.worst[1]).toFixed(1)}% of the time vs ${d.best[0]} at
      ${(100 * d.best[1]).toFixed(1)}% (${d.ratio}x).</div>`;
}

function userCard(u) {
  const pct = u.samples ? Math.round(100 * u.usable / u.samples) : 0;
  const metric = (label, s, key) =>
    s ? `<span>${label}</span><span>${s[key]}</span>` : '';

  return `
    <div class="card">
      <div class="who">
        <h3>${esc(u.name)}</h3>
        <span class="pill ${esc(u.verdict === 'good' ? 'good' : 'warn')}">${esc(u.verdict)}</span>
      </div>
      <div class="metrics">
        <span>usable</span><span>${u.usable}/${u.samples} (${pct}%)</span>
        ${metric('sharpness', u.sharpness, 'mean')}
        ${metric('brightness', u.brightness, 'mean')}
        ${metric('quality', u.quality, 'mean')}
        <span>pose spread</span><span>${u.yawSpread != null ? u.yawSpread + '°' : 'n/a'}</span>
        ${u.age ? `<span>age est.</span><span>${esc(u.age.label)} (${Math.round(u.age.agreement * 100)}% agree)</span>` : ''}
        ${u.gender ? `<span>gender est.</span><span>${esc(u.gender.label)} (${Math.round(u.gender.agreement * 100)}% agree)</span>` : ''}
      </div>
      ${Object.keys(u.flags).length
        ? `<div class="chips">${Object.entries(u.flags)
            .map(([k, v]) => `<span class="chip">${k} ×${v}</span>`).join('')}</div>` : ''}
      ${u.worstSamples.length
        ? `<div class="chips">${u.worstSamples.slice(0, 4)
            .map((w) => `<span class="chip n">${esc(w.file)}</span>`).join('')}</div>` : ''}
      ${u.recommendations.length
        ? `<ul class="tips">${u.recommendations.map((r) => `<li>${esc(r)}</li>`).join('')}</ul>` : ''}
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
        ${calibrationBlock(rep.sface)}
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


/* ------------------------------------------------------------- identify */

/* Recognition against a still image. Kept away from the camera path on
 * purpose: the webcam refuses photographs as spoofs, which is right, and
 * would otherwise make still images untestable. */
$('idFile').onchange = async (ev) => {
  const file = ev.target.files && ev.target.files[0];
  if (!file) return;
  const body = new FormData();
  body.append('image', file);
  const out = $('identifyBody');
  out.innerHTML = '<div class="note">Identifying…</div>';
  try {
    const res = await fetch('/api/identify', { method: 'POST', body });
    const d = await res.json();
    if (!res.ok || d.ok === false) throw new Error(d.error || 'Failed');
    renderIdentify(d);
  } catch (e) {
    out.innerHTML = `<div class="err">${esc(e.message)}</div>`;
  }
  ev.target.value = '';
};

function renderIdentify(d) {
  const m = d.match;
  const rows = d.candidates.map((c, i) => `
    <div class="idrow ${i === 0 && m ? 'hit' : (i === 0 ? 'miss' : '')}">
      <span>${i === 0 ? 'best match' : 'also considered'}</span>
      <b>${esc(c.name)} &nbsp; ${c.similarity.toFixed(3)}</b>
    </div>`).join('');

  $('identifyBody').innerHTML = `
    <div class="agrid">
      <div class="card">
        <div class="who"><h3>${esc(m ? m.name : 'No confident match')}</h3>
          <span class="pill ${m ? 'good' : 'warn'}">${m ? 'identified' : 'below threshold'}</span></div>
        <div class="metrics">
          ${stat('similarity', d.best.similarity.toFixed(3))}
          ${stat('threshold', d.threshold)}
          ${stat('margin over 2nd', d.margin != null ? d.margin.toFixed(3) : '—')}
          ${stat('gallery', d.gallerySize + ' enrolled')}
        </div>
        <div class="note">${m
          ? `Above the threshold for a gallery of ${d.gallerySize}, which carries a
             ${(100 * d.galleryRisk).toFixed(2)}% chance of a false match.`
          : `Best similarity ${d.best.similarity.toFixed(3)} is under the
             ${d.threshold} threshold, so this is reported as unknown rather than
             guessed at.`}</div>
      </div>
      <div class="card">
        <div class="who"><h3>Ranking</h3><span class="pill warn">top ${d.candidates.length}</span></div>
        ${rows}
        <div class="note">The gap to the runner-up matters as much as the top
          score: 0.62 means very different things when the next best is 0.20
          versus 0.61.</div>
      </div>
    </div>`;
}
