/* Turning MediaPipe's face landmarks into the measurements
 * pipeline/guidance.py reads -- and being explicit about every place the
 * translation is approximate.
 *
 * ---------------------------------------------------------------------
 * The point of this file, and the thing it must not do
 * ---------------------------------------------------------------------
 * js/guidance.js is a checked port of pipeline/guidance.py: every rule and
 * every threshold in it is compared against the Python, case by case, by
 * tools/build_static.py before anything is published. It is a real rule
 * engine. Feeding a real rule engine a number that merely *looks* like the
 * one it expects is worse than feeding it nothing, because the verdict comes
 * out looking equally real either way and the reader has no way to tell.
 *
 * So every field below is one of three things, and says which:
 *
 *   EXACT        the same definition, recomputed from this page's pixels.
 *   APPROXIMATE  the same formula, but over landmarks from a different
 *                model than the one the thresholds were tuned against.
 *   UNAVAILABLE  left null. guidance.py skips a rule whose input is None,
 *                so a missing measurement silently disables its rule rather
 *                than answering with a guess. That is the correct
 *                degradation and it is why nothing here is invented.
 *
 * PROVENANCE below carries that label for every field, and the page draws it
 * next to the number. Nothing is fed in without one.
 *
 * ---------------------------------------------------------------------
 * Whose eyes these are
 * ---------------------------------------------------------------------
 * The real pipeline detects with YuNet (falling back to a Haar cascade),
 * fits 68 points with OpenCV's LBF facemark model, and scores image quality
 * with the eDifFIQA network. None of that is here. Here it is MediaPipe Face
 * Landmarker, a different engine with a different mesh, and the numbers it
 * produces are therefore not the numbers those models would produce for the
 * same face. The formulas are this project's; the landmarks are not.
 */
var CameraMeasure = (function () {
  'use strict';

  var LK = LANDMARK_CONSTANTS;

  /* ------------------------------------------------------------------
   * Landmark indices in MediaPipe's 478-point canonical face mesh.
   *
   * Chosen to stand in for the points the Python measures from. The 68-point
   * LBF layout and MediaPipe's 478-point mesh are different topologies, so
   * these are the nearest equivalents, not the same points.
   * ---------------------------------------------------------------------- */

  /* The six-point ring each eye aspect ratio is computed over. Ordered to
   * match pipeline/landmarks.py's 68-point slices, so _aspect() below can be
   * a straight transcription: [outer, upper, upper, inner, lower, lower],
   * which makes pairs (1,5) and (2,4) the two vertical spans and (0,3) the
   * horizontal one. */
  var EYE_RIGHT = [33, 160, 158, 133, 153, 144];   // the subject's right eye
  var EYE_LEFT = [362, 385, 387, 263, 373, 380];   // the subject's left eye

  /* Outer lip. The Python measures the mouth aspect ratio between the outer
   * lip's top and bottom centres (68-point indices 51 and 57) over the
   * corner-to-corner width (48 to 54). */
  var LIP_TOP = 0;
  var LIP_BOTTOM = 17;
  var LIP_CORNER_RIGHT = 61;
  var LIP_CORNER_LEFT = 291;

  /* Top of the nose bridge -- 68-point index 27, the nasion. Used for the
   * centre-offset half of the "face partly obscured" test. */
  var NOSE_BRIDGE_TOP = 168;

  /* The five points pipeline/traits.py's geometry() reads off a YuNet row:
   * two eyes, nose tip, two mouth corners. MediaPipe's iris centres (468 and
   * 473, present only in the 478-point output) are the closest thing to
   * YuNet's eye points; if a model returns only 468 landmarks we fall back
   * to the mean of the eye ring, which sits in much the same place. */
  var IRIS_RIGHT = 468;
  var IRIS_LEFT = 473;
  var NOSE_TIP = 1;

  /* ------------------------------------------------------------------
   * Provenance. One entry per field handed to the rule engine.
   * ---------------------------------------------------------------------- */
  var PROVENANCE = {
    detected: {
      kind: 'approximate',
      note: 'MediaPipe’s face detector, not this project’s YuNet ' +
            'or Haar cascade. A face one finds and the other misses is a ' +
            'real difference, not a rounding one.'
    },
    faces: {
      kind: 'approximate',
      note: 'Every face MediaPipe returns above its detection confidence of ' +
            LK.DETECT_SCORE + ' is counted. The real pipeline only counts a ' +
            'second person at a much stricter ' + LK.PERSON_SCORE +
            ' — MediaPipe does not expose a per-face score, so that ' +
            'filter cannot be applied here. “One person only” can ' +
            'therefore fire on this page where the app would stay quiet.'
    },
    facePx: {
      kind: 'approximate',
      note: 'The longer side of the bounding box of all 478 landmarks, in ' +
            'source-video pixels. The app measures the detector’s own ' +
            'box. MediaPipe’s mesh reaches up over the forehead and out ' +
            'to the temples, so its box is larger than a YuNet box on the ' +
            'same face, and MIN_FACE_PX bites at a slightly greater distance.'
    },
    yaw: {
      kind: 'approximate',
      note: 'This project’s own formula, transcribed from ' +
            'pipeline/traits.py geometry(): how far the nose tip sits off ' +
            'the midpoint of the eyes, over eye separation, times 75, ' +
            'clamped to ±90°. It is a proxy, not a pose solver — ' +
            'the app says so too. The five points come from MediaPipe ' +
            'instead of YuNet, so they sit a little differently.'
    },
    roll: {
      kind: 'approximate',
      note: 'The angle of the line between the two eye centres, as ' +
            'pipeline/traits.py computes it. MediaPipe’s iris centres ' +
            'stand in for YuNet’s eye points.'
    },
    sharpness: {
      kind: 'exact',
      note: 'The same definition the app uses — the variance of a 3×3 ' +
            'Laplacian over the greyscale face crop — recomputed here on ' +
            'canvas pixels, with the same 12% crop padding, the same ' +
            'BGR→grey weights and the same reflect-101 border. Not ' +
            'bit-identical to OpenCV (the browser’s pixels came through a ' +
            'different decode path) but the same measurement, not a stand-in.'
    },
    shadowClip: {
      kind: 'exact',
      note: 'Fraction of greyscale pixels at or below 8 in the face crop — ' +
            'the definition in pipeline/traits.py exposure_metrics(), ' +
            'recomputed here.'
    },
    highlightClip: {
      kind: 'exact',
      note: 'Fraction of greyscale pixels at or above 247 in the face crop, ' +
            'same source.'
    },
    qualityScore: {
      kind: 'unavailable',
      note: 'Not measured here. The app’s quality score is eDifFIQA, a ' +
            'neural network run over the face crop; there is no honest way ' +
            'to approximate its output from landmarks. It is passed as null, ' +
            'which makes guidance.py skip its rule entirely rather than ' +
            'judge on a made-up number — so “Improve lighting” ' +
            'can never fire on this page, and the “Good quality” ' +
            'checklist row is not a pass, it is an absence.'
    },
    eyeMismatch: {
      kind: 'approximate',
      note: 'pipeline/landmarks.py’s formula — the difference between ' +
            'the two eye aspect ratios over the larger of them — but over ' +
            'MediaPipe’s eyelid points rather than the LBF model’s. ' +
            'The limit of ' + LK.ASYMMETRY_LIMIT + ' was tuned against LBF, ' +
            'so it does not mean quite the same thing here.'
    },
    flags: {
      kind: 'approximate',
      note: 'Eyes closed below an aspect ratio of ' + LK.EAR_CLOSED +
            ', mouth open above ' + LK.MAR_OPEN + ', face partly obscured ' +
            'past ' + LK.ASYMMETRY_LIMIT + ' — this project’s ' +
            'thresholds, applied to a different model’s landmarks. ' +
            'Expect them to trip at slightly different moments than the app.'
    },
    frame: {
      kind: 'exact',
      note: 'The video’s own pixel dimensions, which is what the app ' +
            'passes as frame_shape.'
    }
  };

  // ------------------------------------------------------------------ maths

  function dist(a, b) {
    var dx = a[0] - b[0];
    var dy = a[1] - b[1];
    return Math.sqrt(dx * dx + dy * dy);
  }

  function round(value, places) {
    var scale = Math.pow(10, places);
    return Math.round(value * scale) / scale;
  }

  /* A landmark in source-video pixels. MediaPipe reports normalised
   * coordinates; every threshold in guidance.py that involves a length is in
   * pixels, so this conversion is load-bearing. */
  function at(points, index, width, height) {
    var p = points[index];
    if (!p) { return null; }
    return [p.x * width, p.y * height];
  }

  function centre(points, ring, width, height) {
    var sx = 0;
    var sy = 0;
    for (var i = 0; i < ring.length; i += 1) {
      var p = points[ring[i]];
      sx += p.x * width;
      sy += p.y * height;
    }
    return [(sx / ring.length) * 1, (sy / ring.length) * 1];
  }

  /* pipeline/landmarks.py _aspect(): vertical opening over horizontal width,
   * for an eye or a mouth. Transcribed, including the `or 1.0` guard. */
  function aspect(pts) {
    var width = dist(pts[0], pts[3]) || 1.0;
    var a = dist(pts[1], pts[5]);
    var b = dist(pts[2], pts[4]);
    return (a + b) / (2.0 * width);
  }

  function ring(points, indices, width, height) {
    return indices.map(function (index) {
      return at(points, index, width, height);
    });
  }

  // ------------------------------------------------- geometry: box and pose

  /* The bounding box of every landmark, in pixels.
   *
   * APPROXIMATE, and this is the substitution most worth knowing about: the
   * app measures facePx off the *detector's* box. This is the mesh's extent,
   * which is not the same rectangle.
   */
  function boxOf(points, width, height) {
    var minX = Infinity;
    var minY = Infinity;
    var maxX = -Infinity;
    var maxY = -Infinity;
    for (var i = 0; i < points.length; i += 1) {
      var x = points[i].x * width;
      var y = points[i].y * height;
      if (x < minX) { minX = x; }
      if (y < minY) { minY = y; }
      if (x > maxX) { maxX = x; }
      if (y > maxY) { maxY = y; }
    }
    return [minX, minY, maxX - minX, maxY - minY];
  }

  /* A transcription of pipeline/traits.py geometry(), including both guards
   * on the yaw denominator. Those guards exist because the raw ratio runs
   * away as the head turns towards profile -- benchmarking there produced
   * yaw values from -689 to +470 degrees -- so dropping them here would
   * reproduce a bug the project already fixed.
   */
  function pose(points, box, width, height) {
    var w = box[2];
    var h = box[3];

    var rightEye = at(points, IRIS_RIGHT, width, height)
                   || centre(points, EYE_RIGHT, width, height);
    var leftEye = at(points, IRIS_LEFT, width, height)
                  || centre(points, EYE_LEFT, width, height);
    var nose = at(points, NOSE_TIP, width, height);

    var eyeMid = [(rightEye[0] + leftEye[0]) / 2.0,
                  (rightEye[1] + leftEye[1]) / 2.0];
    var eyeDist = dist(leftEye, rightEye);

    var roll = Math.atan2(leftEye[1] - rightEye[1],
                          leftEye[0] - rightEye[0]) * 180.0 / Math.PI;

    var minSep = Math.max(1.0, 0.10 * Math.max(w, h));
    var yawRatio = (nose[0] - eyeMid[0]) / Math.max(eyeDist, minSep);
    var yaw = Math.max(-90.0, Math.min(90.0, yawRatio * 75.0));

    return { yaw: round(yaw, 1), roll: round(roll, 1) };
  }

  // -------------------------------------------------- the part measurements

  /* pipeline/landmarks.py metrics(), reduced to the three things guidance.py
   * actually reads: the flags, and eyeMismatch for the checklist row. The
   * brow raise, cheek prominence and the rest of that dict are measured for
   * the app's traits panel and no rule looks at them, so they are not
   * reproduced here.
   */
  function parts(points, width, height) {
    var earR = aspect(ring(points, EYE_RIGHT, width, height));
    var earL = aspect(ring(points, EYE_LEFT, width, height));

    var top = at(points, LIP_TOP, width, height);
    var bottom = at(points, LIP_BOTTOM, width, height);
    var cornerR = at(points, LIP_CORNER_RIGHT, width, height);
    var cornerL = at(points, LIP_CORNER_LEFT, width, height);
    var mar = dist(top, bottom) / (dist(cornerR, cornerL) || 1.0);

    var eyeRC = centre(points, EYE_RIGHT, width, height);
    var eyeLC = centre(points, EYE_LEFT, width, height);
    var interocular = dist(eyeLC, eyeRC) || 1.0;

    var noseTop = at(points, NOSE_BRIDGE_TOP, width, height);
    var eyeMidX = (eyeRC[0] + eyeLC[0]) / 2.0;
    var centreOffset = Math.abs(noseTop[0] - eyeMidX) / interocular;
    var eyeMismatch = Math.abs(earR - earL) / (Math.max(earR, earL) || 1.0);

    /* Same order and same elif structure as the Python: both eyes below the
     * bar is "eyes closed", exactly one is "one eye closed", never both. */
    var flags = [];
    if (earR < LK.EAR_CLOSED && earL < LK.EAR_CLOSED) {
      flags.push('eyes closed');
    } else if (earR < LK.EAR_CLOSED || earL < LK.EAR_CLOSED) {
      flags.push('one eye closed');
    }
    if (mar > LK.MAR_OPEN) { flags.push('mouth open'); }
    if (eyeMismatch > LK.ASYMMETRY_LIMIT ||
        centreOffset > LK.ASYMMETRY_LIMIT) {
      flags.push('face partly obscured');
    }

    return {
      eyeOpenRight: round(earR, 3),
      eyeOpenLeft: round(earL, 3),
      mouthOpen: round(mar, 3),
      interocularPx: round(interocular, 1),
      centreOffset: round(centreOffset, 3),
      eyeMismatch: round(eyeMismatch, 3),
      flags: flags
    };
  }

  // ------------------------------------------------------ the pixel metrics

  /* The crop pipeline/traits.py analyze() measures quality over: the
   * detection box grown by 12% of its longer side, clamped to the frame.
   * Transcribed, including the truncation in `int(0.12 * max(w, h))`.
   */
  function cropRect(box, width, height) {
    var x = Math.round(box[0]);
    var y = Math.round(box[1]);
    var w = Math.round(box[2]);
    var h = Math.round(box[3]);
    var pad = Math.trunc(0.12 * Math.max(w, h));
    var x0 = Math.max(0, x - pad);
    var y0 = Math.max(0, y - pad);
    var x1 = Math.min(width, x + w + pad);
    var y1 = Math.min(height, y + h + pad);
    return [x0, y0, Math.max(0, x1 - x0), Math.max(0, y1 - y0)];
  }

  /* OpenCV's BGR->GRAY weights, rounded to an integer the way cvtColor's
   * 8-bit path does, so the <= 8 and >= 247 clipping counts below are
   * counting the same thing the Python counts.
   */
  function greyscale(data, w, h) {
    var grey = new Float64Array(w * h);
    for (var i = 0, p = 0; i < grey.length; i += 1, p += 4) {
      grey[i] = Math.round(0.299 * data[p] + 0.587 * data[p + 1] +
                           0.114 * data[p + 2]);
    }
    return grey;
  }

  /* cv2.Laplacian(gray, CV_64F).var().
   *
   * The default kernel at ksize=1 is [[0,1,0],[1,-4,1],[0,1,0]] and the
   * default border is BORDER_REFLECT_101, which mirrors without repeating
   * the edge pixel -- index -1 reads index 1, index n reads index n-2. Both
   * are reproduced rather than approximated with an interior-only pass,
   * because on a small crop the border is a real fraction of the pixels and
   * numpy's .var() is a population variance over all of them.
   */
  function reflect101(index, n) {
    if (n <= 1) { return 0; }
    while (index < 0 || index >= n) {
      if (index < 0) { index = -index; }
      if (index >= n) { index = 2 * n - 2 - index; }
    }
    return index;
  }

  function laplacianVariance(grey, w, h) {
    if (w < 1 || h < 1) { return 0.0; }
    var total = w * h;
    var sum = 0.0;
    var sumSq = 0.0;
    for (var y = 0; y < h; y += 1) {
      var up = reflect101(y - 1, h) * w;
      var down = reflect101(y + 1, h) * w;
      var row = y * w;
      for (var x = 0; x < w; x += 1) {
        var left = reflect101(x - 1, w);
        var right = reflect101(x + 1, w);
        var value = grey[up + x] + grey[down + x] + grey[row + left] +
                    grey[row + right] - 4 * grey[row + x];
        sum += value;
        sumSq += value * value;
      }
    }
    var mean = sum / total;
    return sumSq / total - mean * mean;
  }

  /* sharpness, shadowClip and highlightClip for one crop of canvas pixels.
   * Same definitions as pipeline/traits.py; same rounding, so the numbers on
   * screen have the precision the app's do.
   */
  function pixelMetrics(imageData) {
    var w = imageData.width;
    var h = imageData.height;
    if (!w || !h) { return null; }
    var grey = greyscale(imageData.data, w, h);
    var total = grey.length || 1;
    var dark = 0;
    var bright = 0;
    for (var i = 0; i < grey.length; i += 1) {
      if (grey[i] <= 8) { dark += 1; }
      if (grey[i] >= 247) { bright += 1; }
    }
    return {
      sharpness: round(laplacianVariance(grey, w, h), 1),
      shadowClip: round(dark / total, 4),
      highlightClip: round(bright / total, 4)
    };
  }

  // ------------------------------------------------------------- assembling

  /* The traits dictionary pipeline/readout.py hands to guidance.instruction()
   * -- the same key names, because js/guidance.js is the real port and reads
   * exactly those.
   *
   * `landmarks` is MediaPipe's result for one face, `width`/`height` the
   * source frame's pixel size, and `sample` a function that returns
   * ImageData for a rectangle of the clean frame (or null when the page has
   * chosen not to pay for it).
   */
  function traitsFor(faces, width, height, sample) {
    var count = faces ? faces.length : 0;
    if (!count) {
      /* No face. Everything downstream of detection is genuinely unknown, so
       * it is left null rather than zeroed -- guidance.py's first rule fires
       * on this and nothing after it is consulted. */
      return {
        detected: false,
        faces: 0,
        facePx: null,
        yaw: null,
        roll: null,
        sharpness: null,
        qualityScore: null,
        shadowClip: null,
        highlightClip: null,
        parts: null
      };
    }

    /* MediaPipe returns faces in no documented order. The app measures the
     * highest-scoring detection; with no scores exposed, this takes the
     * largest face, which is the nearest available meaning of "the person in
     * front of the camera". */
    var chosen = faces[0];
    var chosenBox = boxOf(chosen, width, height);
    for (var i = 1; i < count; i += 1) {
      var box = boxOf(faces[i], width, height);
      if (Math.max(box[2], box[3]) > Math.max(chosenBox[2], chosenBox[3])) {
        chosen = faces[i];
        chosenBox = box;
      }
    }

    var geom = pose(chosen, chosenBox, width, height);
    var part = parts(chosen, width, height);

    var pixels = null;
    if (sample) {
      var rect = cropRect(chosenBox, width, height);
      var data = rect[2] && rect[3] ? sample(rect) : null;
      pixels = data ? pixelMetrics(data) : null;
    }

    return {
      detected: true,
      faces: count,
      /* int(round(max(w, h))), as geometry() reports it. */
      facePx: Math.round(Math.max(chosenBox[2], chosenBox[3])),
      yaw: geom.yaw,
      roll: geom.roll,
      /* UNAVAILABLE. Deliberately null -- see PROVENANCE.qualityScore. */
      qualityScore: null,
      sharpness: pixels ? pixels.sharpness : null,
      shadowClip: pixels ? pixels.shadowClip : null,
      highlightClip: pixels ? pixels.highlightClip : null,
      parts: part,
      box: chosenBox
    };
  }

  return {
    traitsFor: traitsFor,
    boxOf: boxOf,
    cropRect: cropRect,
    pixelMetrics: pixelMetrics,
    PROVENANCE: PROVENANCE,
    EYE_RIGHT: EYE_RIGHT,
    EYE_LEFT: EYE_LEFT
  };
}());
