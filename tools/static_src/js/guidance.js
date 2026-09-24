/* A port of pipeline/guidance.py -- the rules only, not the numbers.
 *
 * Every threshold this file compares against is read from GUIDANCE_CONSTANTS
 * in js/constants.js, which tools/build_static.py generates by importing the
 * Python module. So a number can only be wrong here if it is also wrong
 * there: a transcription of MIN_FACE_PX is a second place for it to live, and
 * the whole point of the build is that there is not one.
 *
 * What *is* transcribed is the control flow -- the fourteen rules and, more
 * importantly, the order they are tried in. That is the part with no
 * mechanical link to the Python, so it is the part the build compares
 * case by case against it before publishing anything.
 */
var Guidance = (function () {
  'use strict';

  var K = GUIDANCE_CONSTANTS;

  /* Python's `if not traits:` is false for an empty dict, and `{}` is truthy
   * in JavaScript. A port that wrote `if (!traits)` would answer "Starting
   * camera" for a missing read and something else entirely for an empty one,
   * which is the same bug in the one case that matters. */
  function empty(traits) {
    if (traits === null || traits === undefined) { return true; }
    for (var key in traits) {
      if (Object.prototype.hasOwnProperty.call(traits, key)) { return false; }
    }
    return true;
  }

  function get(traits, key) {
    var value = traits[key];
    return value === undefined ? null : value;
  }

  /* `traits.get("facePx") or 0` -- None and 0 both become 0. */
  function orZero(value) {
    return (value === null || value === undefined || value === 0 ||
            value === false || value === '') ? 0 : value;
  }

  function out(severity, message, detail, ready) {
    return {
      severity: severity,
      message: message,
      detail: detail === undefined ? null : detail,
      ready: ready === true
    };
  }

  /* Python's `if frame_shape` is false for the empty tuple and true for
   * `(0, 0)`; JavaScript's `if (frameShape)` is true for `[]`, which would
   * make the area NaN rather than None. Length, not truthiness. */
  function areaOf(frameShape) {
    if (!frameShape || !frameShape.length) { return null; }
    return frameShape[0] * frameShape[1];
  }

  function Reading(traits, frameShape) {
    this.traits = traits;
    this.frameArea = areaOf(frameShape);
    this.faces = traits.faces === undefined ? 0 : traits.faces;
    this.detected = traits.detected === undefined ? false : traits.detected;
    this.facePx = orZero(get(traits, 'facePx'));
    this.yaw = get(traits, 'yaw');
    this.roll = get(traits, 'roll');
    var parts = traits.parts;
    this.partFlags = (parts && parts.flags) ? parts.flags : [];
  }

  Reading.prototype.get = function (key) {
    return get(this.traits, key);
  };

  Reading.prototype.faceFraction = function () {
    var px = orZero(get(this.traits, 'facePx'));
    if (!this.frameArea || !px) { return 0.0; }
    return (px * px) / this.frameArea;
  };

  function has(flags, name) {
    return flags.indexOf(name) !== -1;
  }

  // -------------------------------------------------------------------
  // The rules. One function each, same names as the Python.
  // -------------------------------------------------------------------
  function ruleNoFace(r) {
    if (!r.detected || r.faces === 0) {
      return out('block', 'Look at the centre of the camera',
        'No face detected — take off sunglasses, or a cap or hood ' +
        'with a brim shading your eyes.');
    }
    return null;
  }

  function ruleCrowd(r) {
    if (r.faces > 1) {
      return out('block', 'One person only',
        r.faces + ' faces in view — others should step out of frame.');
    }
    return null;
  }

  function ruleTooFar(r) {
    if (r.facePx && r.facePx < K.MIN_FACE_PX) {
      return out('block', 'Move closer',
        'Your face is too small in the frame to capture detail.');
    }
    return null;
  }

  function ruleTooClose(r) {
    if (r.frameArea && r.faceFraction() > K.MAX_FACE_FRACTION) {
      return out('warn', 'Move back', 'Your face is filling the frame.');
    }
    return null;
  }

  function ruleYaw(r) {
    if (r.yaw !== null && Math.abs(r.yaw) > K.MAX_YAW) {
      var side = r.yaw > 0 ? 'left' : 'right';
      return out('block', 'Look at the centre of the camera',
        'Turn slightly to the ' + side + '. If it still will not lock on, ' +
        'take off sunglasses or a brim shading your eyes.');
    }
    return null;
  }

  function ruleRoll(r) {
    if (r.roll !== null && Math.abs(r.roll) > K.MAX_ROLL) {
      return out('warn', 'Head upright', 'Your head is tilted.');
    }
    return null;
  }

  function ruleEyesClosed(r) {
    if (has(r.partFlags, 'eyes closed')) {
      return out('block', 'Open your eyes',
        'Both eyes read as closed — the sample would be unusable.');
    }
    return null;
  }

  function ruleOneEyeClosed(r) {
    if (has(r.partFlags, 'one eye closed')) {
      return out('warn', 'Open both eyes', 'One eye reads as closed.');
    }
    return null;
  }

  function ruleObscured(r) {
    if (has(r.partFlags, 'face partly obscured')) {
      return out('block', 'Uncover your face',
        'One side is measuring very differently from the other — ' +
        'something may be covering it, or the light is only hitting ' +
        'one side.');
    }
    return null;
  }

  function ruleMouthOpen(r) {
    if (has(r.partFlags, 'mouth open')) {
      return out('warn', 'Neutral expression',
        'An open mouth changes the shape of the lower face.');
    }
    return null;
  }

  function ruleShadow(r) {
    if (orZero(r.get('shadowClip')) > K.MAX_SHADOW_CLIP) {
      return out('warn', 'More light in front',
        'Detail is being lost in shadow.');
    }
    return null;
  }

  function ruleHighlight(r) {
    if (orZero(r.get('highlightClip')) > K.MAX_HIGHLIGHT_CLIP) {
      return out('warn', 'Less light behind',
        'Move away from the window, or turn to face the light.');
    }
    return null;
  }

  function ruleBlur(r) {
    var sharp = r.get('sharpness');
    if (sharp !== null && sharp < K.MIN_SHARPNESS) {
      return out('warn', 'Hold still', 'The image is blurred.');
    }
    return null;
  }

  function ruleQuality(r) {
    var q = r.get('qualityScore');
    if (q !== null && q < K.MIN_QUALITY) {
      return out('warn', 'Improve lighting',
        'Image quality is low. Try more even light, or clean the lens.');
    }
    return null;
  }

  /* The order, and the order is the design. Read top to bottom: a face at
   * all, then how many, then framing, then pose, then the face itself, then
   * exposure, then general quality.
   *
   * The names ride along because the page draws this list, and because the
   * build compares it position by position -- a reordering that changed no
   * single verdict would still be a different design, and it is caught here
   * rather than by accident. */
  var RULES = [
    { name: 'no_face', fn: ruleNoFace },
    { name: 'crowd', fn: ruleCrowd },
    { name: 'too_far', fn: ruleTooFar },
    { name: 'too_close', fn: ruleTooClose },
    { name: 'yaw', fn: ruleYaw },
    { name: 'roll', fn: ruleRoll },
    { name: 'eyes_closed', fn: ruleEyesClosed },
    { name: 'one_eye_closed', fn: ruleOneEyeClosed },
    { name: 'obscured', fn: ruleObscured },
    { name: 'mouth_open', fn: ruleMouthOpen },
    { name: 'shadow', fn: ruleShadow },
    { name: 'highlight', fn: ruleHighlight },
    { name: 'blur', fn: ruleBlur },
    { name: 'quality', fn: ruleQuality }
  ];

  function instruction(traits, frameShape, mode) {
    if (empty(traits)) { return out('block', 'Starting camera'); }
    if (traits.error) {
      return out('block', 'Camera error', String(traits.error));
    }
    var reading = new Reading(traits, frameShape);
    for (var i = 0; i < RULES.length; i += 1) {
      var verdict = RULES[i].fn(reading);
      if (verdict !== null) { return verdict; }
    }
    if (mode === 'register') {
      return out('ok', 'Hold still', 'Capturing samples.', true);
    }
    return out('ok', K.READY, null, true);
  }

  /* Which rules would fire, in priority order -- what the page draws as the
   * chain. `instruction` returns only the first; this is the same evaluation
   * with nothing discarded, so the reader can see the one that won and the
   * ones queued behind it. */
  function chain(traits, frameShape) {
    if (empty(traits) || traits.error) { return []; }
    var reading = new Reading(traits, frameShape);
    var rows = [];
    var chosen = false;
    for (var i = 0; i < RULES.length; i += 1) {
      var verdict = RULES[i].fn(reading);
      rows.push({
        name: RULES[i].name,
        fired: verdict !== null,
        chosen: verdict !== null && !chosen,
        severity: verdict === null ? null : verdict.severity,
        message: verdict === null ? null : verdict.message
      });
      if (verdict !== null) { chosen = true; }
    }
    return rows;
  }

  function checklist(traits, frameShape) {
    if (empty(traits)) { return []; }
    var detected = traits.detected === undefined ? false : traits.detected;
    var faces = traits.faces === undefined ? 0 : traits.faces;
    var px = orZero(get(traits, 'facePx'));
    var yaw = get(traits, 'yaw');
    var roll = get(traits, 'roll');
    var q = get(traits, 'qualityScore');
    var sharp = get(traits, 'sharpness');
    var parts = traits.parts ? traits.parts : {};
    var pflags = parts.flags ? parts.flags : [];
    var eyeMismatch = parts.eyeMismatch === undefined ? null : parts.eyeMismatch;

    var items = [
      ['Face visible', Boolean(detected) && faces >= 1,
       'Look straight into the camera'],
      ['Only one person', faces <= 1, 'Others should step out of frame'],
      ['Close enough', Boolean(px) && px >= K.MIN_FACE_PX, 'Move closer'],
      ['Facing forward', yaw === null || Math.abs(yaw) <= K.MAX_YAW,
       'Turn to face the camera'],
      ['Head upright', roll === null || Math.abs(roll) <= K.MAX_ROLL,
       'Straighten your head'],
      ['Eyes unobstructed',
       eyeMismatch === null || eyeMismatch <= K.EYE_MISMATCH_LIMIT,
       'Remove sunglasses or a shading brim'],
      ['Eyes open',
       !has(pflags, 'eyes closed') && !has(pflags, 'one eye closed'),
       'Open both eyes'],
      ['Neutral expression', !has(pflags, 'mouth open'), 'Close your mouth'],
      ['Both sides visible', !has(pflags, 'face partly obscured'),
       'Uncover your face, or even out the light'],
      ['Sharp', sharp === null || sharp >= K.MIN_SHARPNESS, 'Hold still'],
      ['Good quality', q === null || q >= K.MIN_QUALITY, 'Improve lighting']
    ];
    return items.map(function (row) {
      return { label: row[0], ok: Boolean(row[1]), fix: row[2] };
    });
  }

  return {
    instruction: instruction,
    checklist: checklist,
    chain: chain,
    RULES: RULES,
    K: K
  };
}());
