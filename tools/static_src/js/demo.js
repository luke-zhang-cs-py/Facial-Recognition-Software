/* The page: read the controls, call the two ported modules, draw the answer.
 *
 * Nothing in here is ported from Python and nothing in here decides anything.
 * Every number shown comes from Guidance or Calibration; this file is the
 * wiring and the drawing, which is why the build compares those two and not
 * this one.
 */
(function () {
  'use strict';

  function el(id) { return document.getElementById(id); }

  function text(node, value) { node.textContent = value; }

  function clear(node) {
    while (node.firstChild) { node.removeChild(node.firstChild); }
  }

  function make(tag, className, content) {
    var node = document.createElement(tag);
    if (className) { node.className = className; }
    if (content !== undefined && content !== null) {
      node.textContent = content;
    }
    return node;
  }

  var SVG_NS = 'http://www.w3.org/2000/svg';

  function svg(tag, attrs) {
    var node = document.createElementNS(SVG_NS, tag);
    Object.keys(attrs || {}).forEach(function (key) {
      node.setAttribute(key, attrs[key]);
    });
    return node;
  }

  // =====================================================================
  // Guidance panel
  // =====================================================================
  var FLAGS = ['eyes closed', 'one eye closed', 'mouth open',
               'face partly obscured'];

  /* Which controls the engine stops reading, and when. The instruction is
   * decided before any measurement is looked at in two cases, and a panel
   * that left the sliders live would imply otherwise. */
  function stateOfControls() {
    return {
      noRead: el('g-noread').checked,
      error: el('g-error').checked
    };
  }

  function readTraits() {
    var state = stateOfControls();
    if (state.noRead) { return {}; }
    if (state.error) {
      return { error: el('g-errortext').value };
    }
    var traits = {
      detected: el('g-detected').checked,
      faces: Number(el('g-faces').value),
      facePx: Number(el('g-facepx').value),
      shadowClip: Number(el('g-shadow').value),
      highlightClip: Number(el('g-highlight').value),
      parts: { flags: [], eyeMismatch: null }
    };
    traits.yaw = el('g-yaw-known').checked ? Number(el('g-yaw').value) : null;
    traits.roll = el('g-roll-known').checked ? Number(el('g-roll').value) : null;
    traits.sharpness = el('g-sharp-known').checked
      ? Number(el('g-sharp').value) : null;
    traits.qualityScore = el('g-quality-known').checked
      ? Number(el('g-quality').value) : null;
    traits.parts.eyeMismatch = el('g-mismatch-known').checked
      ? Number(el('g-mismatch').value) : null;
    FLAGS.forEach(function (name, index) {
      if (el('g-flag-' + index).checked) { traits.parts.flags.push(name); }
    });
    return traits;
  }

  function frameShape() {
    if (!el('g-frame-known').checked) { return null; }
    return [Number(el('g-frameh').value), Number(el('g-framew').value)];
  }

  function mode() {
    return el('g-register').checked ? 'register' : 'idle';
  }

  var SEVERITY_WORD = { block: 'Blocking', warn: 'Warning', ok: 'Ready' };

  function renderInstruction(verdict) {
    var box = el('verdict');
    box.className = 'verdict sev-' + verdict.severity;
    text(el('verdictMessage'), verdict.message);
    text(el('verdictDetail'), verdict.detail === null ? '—'
                                                      : verdict.detail);
    text(el('verdictSeverity'), SEVERITY_WORD[verdict.severity]);
    text(el('verdictReady'), verdict.ready ? 'ready' : 'not ready');
  }

  function renderChain(rows, verdict) {
    var list = el('chain');
    clear(list);
    if (!rows.length) {
      var note = make('li', 'chainNote',
        verdict.message === 'Camera error'
          ? 'No rule runs: the read carries an error, which is answered '
            + 'before any measurement is looked at.'
          : 'No rule runs: there is no trait read yet, which is answered '
            + 'before any measurement is looked at.');
      list.appendChild(note);
      return;
    }
    rows.forEach(function (row, index) {
      var item = make('li', 'chainRow'
        + (row.chosen ? ' chosen' : '')
        + (row.fired ? ' fired' : ' quiet'));
      item.appendChild(make('span', 'chainRank', index + 1));
      item.appendChild(make('span', 'chainName', row.name));
      item.appendChild(make('span', 'chainState',
        row.fired ? (row.chosen ? 'shown' : 'also firing') : 'satisfied'));
      item.appendChild(make('span', 'chainMessage',
        row.fired ? row.message : '—'));
      list.appendChild(item);
    });
  }

  function renderChecklist(items) {
    var list = el('checklist');
    clear(list);
    if (!items.length) {
      list.appendChild(make('li', 'chainNote',
        'No checklist: there is no trait read to check.'));
      return;
    }
    items.forEach(function (item) {
      var row = make('li', 'checkRow ' + (item.ok ? 'pass' : 'fail'));
      row.appendChild(make('span', 'checkMark', item.ok ? '✓' : '✗'));
      row.appendChild(make('span', 'checkLabel', item.label));
      row.appendChild(make('span', 'checkFix', item.ok ? '' : item.fix));
      list.appendChild(row);
    });
    var failed = items.filter(function (i) { return !i.ok; }).length;
    text(el('checklistCount'),
      failed === 0 ? 'all ' + items.length + ' satisfied'
                   : failed + ' of ' + items.length + ' failing');
  }

  function syncGuidanceControls() {
    var state = stateOfControls();
    var frozen = state.noRead || state.error;
    el('g-error').disabled = state.noRead;
    el('measurements').classList.toggle('frozen', frozen);
    Array.prototype.forEach.call(
      el('measurements').querySelectorAll('input'),
      function (input) { input.disabled = frozen; });
    if (!frozen) {
      [['g-yaw', 'g-yaw-known'], ['g-roll', 'g-roll-known'],
       ['g-sharp', 'g-sharp-known'], ['g-quality', 'g-quality-known'],
       ['g-mismatch', 'g-mismatch-known'],
       ['g-framew', 'g-frame-known'], ['g-frameh', 'g-frame-known']
      ].forEach(function (pair) {
        el(pair[0]).disabled = !el(pair[1]).checked;
      });
    }
    el('g-errortext').disabled = state.noRead || !state.error;
  }

  function showValue(id, target, digits) {
    var node = el(target);
    var input = el(id);
    text(node, digits === undefined ? input.value
                                    : Number(input.value).toFixed(digits));
  }

  function renderGuidance() {
    syncGuidanceControls();
    showValue('g-faces', 'g-faces-v');
    showValue('g-facepx', 'g-facepx-v');
    showValue('g-framew', 'g-framew-v');
    showValue('g-frameh', 'g-frameh-v');
    showValue('g-yaw', 'g-yaw-v', 1);
    showValue('g-roll', 'g-roll-v', 1);
    showValue('g-sharp', 'g-sharp-v', 1);
    showValue('g-quality', 'g-quality-v', 3);
    showValue('g-mismatch', 'g-mismatch-v', 3);
    showValue('g-shadow', 'g-shadow-v', 3);
    showValue('g-highlight', 'g-highlight-v', 3);

    var traits = readTraits();
    var shape = frameShape();
    var verdict = Guidance.instruction(traits, shape, mode());
    renderInstruction(verdict);
    renderChain(Guidance.chain(traits, shape), verdict);
    renderChecklist(Guidance.checklist(traits, shape));

    var fraction = (shape && traits.facePx)
      ? (traits.facePx * traits.facePx) / (shape[0] * shape[1]) : 0;
    text(el('g-fraction'), shape
      ? fraction.toFixed(3) + ' of the frame'
      : 'no frame size, so this rule cannot fire');
  }

  // =====================================================================
  // Calibration panel
  // =====================================================================
  var RISK_CHOICES = [0.001, 0.005, 0.01, 0.05, 0.1];

  function gallerySize() { return Number(el('c-size').value); }

  function maxRisk() { return Number(el('c-risk').value); }

  /* The gallery size at which no measured threshold is strict enough any
   * more, found by asking rather than by writing the number down. It moves
   * with max_risk, and at the 1% default it is not the round number the
   * README used to imply. */
  function crossover(risk) {
    if (!Calibration.recommendThreshold(2, risk).reachable) { return 2; }
    var lo = 2;
    var hi = 4;
    while (Calibration.recommendThreshold(hi, risk).reachable) {
      lo = hi;
      hi *= 2;
      if (hi > 1e9) { return null; }
    }
    while (lo + 1 < hi) {
      var mid = Math.floor((lo + hi) / 2);
      if (Calibration.recommendThreshold(mid, risk).reachable) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    return hi;
  }

  function pct(value, digits) {
    return (value * 100).toFixed(digits) + '%';
  }

  function renderCalibration() {
    var size = gallerySize();
    var risk = maxRisk();
    var found = Calibration.recommendThreshold(size, risk);
    var edge = crossover(risk);

    text(el('c-size-v'), size.toLocaleString('en-GB'));
    text(el('c-threshold'), String(found.threshold));
    text(el('c-risk-achieved'), pct(found.risk, 3));
    text(el('c-fmr'), Calibration.fmrAt(found.threshold).toExponential(3));
    text(el('c-pairs'), (size < 2 ? 0 : size - 1).toLocaleString('en-GB'));

    var badge = el('c-reachable');
    badge.className = 'badge ' + (found.reachable ? 'ok' : 'critical');
    text(badge, found.reachable ? 'sufficient' : 'not sufficient');

    text(el('c-describe'), Calibration.describe(size, risk));
    text(el('c-edge'), edge === null ? 'never' : edge.toLocaleString('en-GB'));
    text(el('c-size-v2'), size.toLocaleString('en-GB'));

    renderTable(size, found.threshold);
    drawCurves(size, risk, edge);
  }

  function renderTable(size, chosen) {
    var body = el('riskTableBody');
    clear(body);
    Calibration.riskTable(size).forEach(function (row) {
      var tr = make('tr', row.threshold === chosen ? 'chosen' : '');
      tr.appendChild(make('td', 'num', String(row.threshold)));
      tr.appendChild(make('td', 'num', row.fmrPerPair.toExponential(3)));
      tr.appendChild(make('td', 'num', pct(row.galleryRisk, 3)));
      body.appendChild(tr);
    });
  }

  // ------------------------------------------------------------------
  // The two charts. One measure each on its own y axis, never both on one.
  // ------------------------------------------------------------------
  var CHART = { w: 620, h: 210, l: 58, r: 18, t: 16, b: 34 };
  var MIN_N = 1;
  var MAX_N = 100000;

  function xOf(n) {
    var span = Math.log10(MAX_N) - Math.log10(MIN_N);
    var frac = (Math.log10(Math.max(n, MIN_N)) - Math.log10(MIN_N)) / span;
    return CHART.l + frac * (CHART.w - CHART.l - CHART.r);
  }

  function samples() {
    var out = [];
    var steps = 260;
    for (var i = 0; i <= steps; i += 1) {
      var lg = (i / steps) * (Math.log10(MAX_N) - Math.log10(MIN_N));
      out.push(Math.max(1, Math.round(Math.pow(10, lg))));
    }
    return out.filter(function (v, i, a) { return i === 0 || a[i - 1] !== v; });
  }

  function frame(root, title, yLabel) {
    clear(root);
    root.setAttribute('viewBox', '0 0 ' + CHART.w + ' ' + CHART.h);
    root.setAttribute('role', 'img');
    root.setAttribute('aria-label', title + '. ' + yLabel
      + '. The same numbers are in the table below.');
    var ticks = [1, 10, 100, 1000, 10000, 100000];
    ticks.forEach(function (n) {
      root.appendChild(svg('line', {
        x1: xOf(n), x2: xOf(n), y1: CHART.t, y2: CHART.h - CHART.b,
        class: 'grid'
      }));
      var label = svg('text', {
        x: xOf(n), y: CHART.h - CHART.b + 15, class: 'tick', 'text-anchor':
          n === MIN_N ? 'start' : (n === MAX_N ? 'end' : 'middle')
      });
      label.textContent = n.toLocaleString('en-GB');
      root.appendChild(label);
    });
    var axis = svg('text', {
      x: (CHART.l + CHART.w - CHART.r) / 2, y: CHART.h - 3, class: 'axisName',
      'text-anchor': 'middle'
    });
    axis.textContent = 'people enrolled';
    root.appendChild(axis);
    root.appendChild(svg('line', {
      x1: CHART.l, x2: CHART.w - CHART.r,
      y1: CHART.h - CHART.b, y2: CHART.h - CHART.b, class: 'baseline'
    }));
  }

  function shadeUnreachable(root, edge) {
    if (edge === null || edge > MAX_N) { return; }
    root.appendChild(svg('rect', {
      x: xOf(edge), y: CHART.t,
      width: Math.max(0, (CHART.w - CHART.r) - xOf(edge)),
      height: CHART.h - CHART.b - CHART.t, class: 'unreachable'
    }));
    root.appendChild(svg('line', {
      x1: xOf(edge), x2: xOf(edge), y1: CHART.t, y2: CHART.h - CHART.b,
      class: 'edgeLine'
    }));
  }

  function marker(root, n, cls) {
    root.appendChild(svg('line', {
      x1: xOf(n), x2: xOf(n), y1: CHART.t, y2: CHART.h - CHART.b, class: cls
    }));
  }

  function drawThresholdCurve(size, edge) {
    var root = el('chartThreshold');
    frame(root, 'Recommended threshold against gallery size',
      'Recommended SFace cosine threshold');
    shadeUnreachable(root, edge);

    var lo = Calibration.thresholds[0];
    var hi = Calibration.thresholds[Calibration.thresholds.length - 1];
    function yOf(t) {
      var frac = (t - lo) / (hi - lo);
      return (CHART.h - CHART.b) - frac * (CHART.h - CHART.b - CHART.t);
    }
    [lo, (lo + hi) / 2, hi].forEach(function (t) {
      var label = svg('text', {
        x: CHART.l - 8, y: yOf(t) + 4, class: 'tick', 'text-anchor': 'end'
      });
      label.textContent = t.toFixed(3);
      root.appendChild(label);
    });

    /* A step line, not a smooth one: the recommendation can only ever be one
     * of the nineteen thresholds the corpus was counted at, so drawing it as
     * a slope would invent operating points that were never measured. */
    var d = '';
    var previous = null;
    samples().forEach(function (n) {
      var t = Calibration.recommendThreshold(n, maxRisk()).threshold;
      var x = xOf(n);
      if (previous === null) {
        d += 'M' + x + ' ' + yOf(t);
      } else if (t !== previous) {
        d += 'L' + x + ' ' + yOf(previous) + 'L' + x + ' ' + yOf(t);
      } else {
        d += 'L' + x + ' ' + yOf(t);
      }
      previous = t;
    });
    root.appendChild(svg('path', { d: d, class: 'series' }));

    marker(root, size, 'cursor');
    var here = Calibration.recommendThreshold(size, maxRisk());
    root.appendChild(svg('circle', {
      cx: xOf(size), cy: yOf(here.threshold), r: 5, class: 'dot'
    }));
    var tag = svg('text', {
      x: Math.min(xOf(size) + 9, CHART.w - CHART.r - 4),
      y: yOf(here.threshold) - 9, class: 'pointLabel',
      'text-anchor': xOf(size) > CHART.w - 120 ? 'end' : 'start'
    });
    tag.textContent = String(here.threshold);
    root.appendChild(tag);
  }

  function drawRiskCurve(size, risk, edge) {
    var root = el('chartRisk');
    frame(root, 'Chance of at least one false match, at the recommended '
      + 'threshold', 'Probability that someone in the gallery is mistaken '
      + 'for someone else');
    shadeUnreachable(root, edge);

    /* The axis is scaled to the largest risk actually reached across the
     * plotted range, not to a multiple of the target. An axis that stopped
     * near the target would draw the whole runaway as a flat line along the
     * ceiling -- a plateau that is not in the data, and the one part of this
     * chart a reader is here to see. */
    var top = 0;
    samples().forEach(function (n) {
      top = Math.max(top, Calibration.recommendThreshold(n, risk).risk);
    });
    top = Math.max(top, risk * 1.6, 1e-4);
    function yOf(p) {
      var frac = p / top;
      return (CHART.h - CHART.b) - frac * (CHART.h - CHART.b - CHART.t);
    }
    [0, top / 2, top].forEach(function (p) {
      var label = svg('text', {
        x: CHART.l - 8, y: yOf(p) + 4, class: 'tick', 'text-anchor': 'end'
      });
      label.textContent = pct(p, 1);
      root.appendChild(label);
    });

    root.appendChild(svg('line', {
      x1: CHART.l, x2: CHART.w - CHART.r, y1: yOf(risk), y2: yOf(risk),
      class: 'target'
    }));
    var targetTag = svg('text', {
      x: CHART.l + 4, y: yOf(risk) - 5, class: 'targetLabel'
    });
    targetTag.textContent = 'target ' + pct(risk, 1);
    root.appendChild(targetTag);

    var d = '';
    samples().forEach(function (n, index) {
      var found = Calibration.recommendThreshold(n, risk);
      d += (index === 0 ? 'M' : 'L') + xOf(n) + ' ' + yOf(found.risk);
    });
    root.appendChild(svg('path', { d: d, class: 'series' }));

    marker(root, size, 'cursor');
    var here = Calibration.recommendThreshold(size, risk);
    root.appendChild(svg('circle', {
      cx: xOf(size), cy: yOf(here.risk), r: 5,
      class: 'dot' + (here.reachable ? '' : ' over')
    }));
  }

  /* Clicking or dragging on either chart moves the gallery size, because the
   * charts are the thing a reader is looking at and reaching back to the
   * slider to ask "what about here?" is a worse question to have to ask. The
   * dashed cursor and the tiles are the readout, so a hover tooltip would be
   * a second, smaller copy of numbers already on screen. */
  function sizeAtPixel(root, clientX) {
    var box = root.getBoundingClientRect();
    var x = ((clientX - box.left) / box.width) * CHART.w;
    var usable = CHART.w - CHART.l - CHART.r;
    var frac = Math.max(0, Math.min(1, (x - CHART.l) / usable));
    var lg = frac * (Math.log10(MAX_N) - Math.log10(MIN_N));
    return Math.max(MIN_N, Math.min(MAX_N, Math.round(Math.pow(10, lg))));
  }

  function makeScrubbable(root) {
    function scrub(event) {
      el('c-size').value = sizeAtPixel(root, event.clientX);
      renderCalibration();
    }
    root.style.cursor = 'crosshair';
    root.addEventListener('pointerdown', function (event) {
      root.setPointerCapture(event.pointerId);
      scrub(event);
    });
    root.addEventListener('pointermove', function (event) {
      if (event.buttons) { scrub(event); }
    });
  }

  function drawCurves(size, risk, edge) {
    drawThresholdCurve(size, edge);
    drawRiskCurve(size, risk, edge);
    var note = el('edgeNote');
    if (edge === null) {
      text(note, 'At this target every gallery size on the chart is reachable.');
    } else {
      text(note, 'Past ' + edge.toLocaleString('en-GB') + ' people the shaded '
        + 'region begins: no threshold in the measured table is strict enough '
        + 'to hold the risk under ' + pct(risk, 1) + ', and the honest answer '
        + 'is a second factor rather than a tighter number.');
    }
  }

  // =====================================================================
  function buildRiskChoices() {
    var select = el('c-risk');
    RISK_CHOICES.forEach(function (value) {
      var option = document.createElement('option');
      option.value = String(value);
      option.textContent = pct(value, 1);
      if (value === 0.01) { option.selected = true; }
      select.appendChild(option);
    });
  }

  function preset(name) {
    var values = {
      good: { detected: true, faces: 1, facePx: 190, yaw: 4, roll: 2,
              sharp: 90, quality: 0.62, shadow: 0.05, highlight: 0.04,
              mismatch: 0.05, flags: [] },
      far: { detected: true, faces: 1, facePx: 90, yaw: 6, roll: 3,
             sharp: 70, quality: 0.5, shadow: 0.1, highlight: 0.05,
             mismatch: 0.08, flags: [] },
      everything: { detected: true, faces: 3, facePx: 90, yaw: 35, roll: 25,
                    sharp: 8, quality: 0.05, shadow: 0.8, highlight: 0.7,
                    mismatch: 0.6, flags: [0, 1, 2, 3] }
    }[name];
    el('g-noread').checked = false;
    el('g-error').checked = false;
    el('g-detected').checked = values.detected;
    el('g-faces').value = values.faces;
    el('g-facepx').value = values.facePx;
    el('g-yaw').value = values.yaw;
    el('g-roll').value = values.roll;
    el('g-sharp').value = values.sharp;
    el('g-quality').value = values.quality;
    el('g-shadow').value = values.shadow;
    el('g-highlight').value = values.highlight;
    el('g-mismatch').value = values.mismatch;
    ['g-yaw-known', 'g-roll-known', 'g-sharp-known', 'g-quality-known',
     'g-mismatch-known', 'g-frame-known'].forEach(function (id) {
      el(id).checked = true;
    });
    FLAGS.forEach(function (_, index) {
      el('g-flag-' + index).checked = values.flags.indexOf(index) !== -1;
    });
    renderGuidance();
  }

  function start() {
    buildRiskChoices();
    Array.prototype.forEach.call(
      document.querySelectorAll('#guidancePanel input'),
      function (node) {
        node.addEventListener('input', renderGuidance);
        node.addEventListener('change', renderGuidance);
      });
    Array.prototype.forEach.call(
      document.querySelectorAll('[data-preset]'),
      function (node) {
        node.addEventListener('click', function () {
          preset(node.getAttribute('data-preset'));
        });
      });
    el('c-size').addEventListener('input', renderCalibration);
    el('c-risk').addEventListener('change', renderCalibration);
    makeScrubbable(el('chartThreshold'));
    makeScrubbable(el('chartRisk'));
    Array.prototype.forEach.call(
      document.querySelectorAll('[data-size]'),
      function (node) {
        node.addEventListener('click', function () {
          el('c-size').value = node.getAttribute('data-size');
          renderCalibration();
        });
      });

    text(el('corpusName'), Calibration.corpus);
    text(el('referencePoint'), String(Calibration.reference));
    text(el('disparityRatio'), String(Calibration.disparity.ratio));
    text(el('disparityWorst'), Calibration.disparity.worst[0]);
    text(el('disparityBest'), Calibration.disparity.best[0]);
    text(el('disparityWorstPct'), pct(Calibration.disparity.worst[1], 1));
    text(el('disparityBestPct'), pct(Calibration.disparity.best[1], 1));
    text(el('disparityThreshold'), String(Calibration.disparity.threshold));

    renderGuidance();
    renderCalibration();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
}());
