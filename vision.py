"""
vision.py
----------
The numbers the LBPH capture path shares, in one place.

The same problem `paths.py` solves for files. Five modules each wrote out the
face crop size, three each wrote out the detector's tuning triple, and two
declared their own `CONFIDENCE_THRESHOLD = 70` -- while `analytics.py`, which
exists to *recommend* a threshold, reported the current one as a typed
literal.

Why that was a bug and not only a smell
=======================================
`analytics.py` returns `currentThreshold` in its report, and four places
consume it: the sweep table marks that row as "current", `analyze_faces.py`
prints "attendance.py has CONFIDENCE_THRESHOLD = N; your data supports M",
and the web UI highlights the row and raises a recommendation banner. All
four read a literal `70`. Change the threshold the recogniser actually uses
and every one of them keeps saying 70 -- so the tool whose job is to tell you
what your configuration is would confidently misreport it, and its
recommendation would be computed against a baseline that no longer existed.

The crop size is the same kind of trap with a quieter failure. LBPH compares
texture histograms over a fixed grid, so a gallery captured at one size and a
query resized to another are not comparable: recognition would simply get
worse, with nothing to point at. Five call sites agreed on (200, 200) by
coincidence of everyone copying the same line.

What is deliberately *not* here
===============================
`traits.py` keeps `SFACE_INPUT_SIZE`, and `recognition.py` keeps `MIN_MARGIN`
and its calibrated cosine thresholds. Those belong to the SFace path, which
is a different algorithm with a different metric -- a threshold in a cosine
space and an LBPH distance have nothing to do with each other, and putting
them in one module would invite exactly the confusion this file is trying to
remove. `analytics.py` keeps its sweep ranges and quality fractions: it is
the only thing that uses them.
"""

# The LBPH distance below which a match is accepted. Lower is a closer match,
# which is the opposite of what "confidence" suggests; the name is OpenCV's.
#
# 70 is a starting guess, not a measurement, and it does not transfer between
# cameras, lighting or gallery sizes -- which is why `analytics.py` sweeps it
# against local data and recommends a value. Run `python analyze_faces.py` and
# change it here, once, rather than in each module that matches a face.
CONFIDENCE_THRESHOLD = 70

# Every face crop is normalised to this before it is trained on or matched
# against. LBPH histograms are computed over a fixed grid, so the gallery and
# the query have to agree; changing this means retraining, and changing it in
# one place only means silently worse recognition.
LBPH_INPUT_SIZE = (200, 200)

# The Haar cascade's tuning triple. Kept together because they are read
# together and only mean anything as a set: a smaller scale factor with the
# same neighbour count is a different detector, not a slightly slower one.
#
# 1.1 rescales the image in 10% steps -- fine enough not to miss faces between
# scales, and the cost is linear in how many steps it takes. 5 is how many
# overlapping detections a region needs before it counts, which is the main
# dial against false positives. (80, 80) is the smallest face worth trying to
# recognise: an LBPH crop scaled up from less than that is mostly
# interpolation, and it was already the floor every caller used.
DETECT_SCALE_FACTOR = 1.1
DETECT_MIN_NEIGHBOURS = 5
MIN_FACE_SIZE = (80, 80)

# JPEG quality for the MJPEG preview stream. A preview, not evidence: the
# frames that get stored go through the capture path, not this one.
STREAM_JPEG_QUALITY = 80
