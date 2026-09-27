"""
cli/fetch_models.py
--------------------
Download the pretrained weights the trait analysis needs into models/.

    python -m cli.fetch_models           # everything missing
    python -m cli.fetch_models --force   # re-download even if present
    python -m cli.fetch_models --list    # just show what is and isn't there

Roughly 196 MB total, mostly the two Caffe demographic nets and the LBF
landmark model. The CLI's LBPH pipeline needs none of it. The web app's
attendance loop does need the liveness net: it will not mark anybody present
until it can tell a face from a photograph of one.

The OpenCV Zoo files are stored in Git LFS, so they come from the
media.githubusercontent.com endpoint; the normal raw URL returns a ~130 byte
pointer file rather than the model, which then fails to load with a confusing
protobuf error.
"""

import argparse
import hashlib
import os
import sys
import urllib.request

from core.facemodels import models_dir, SPECS, have, path_for

ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
LEARNOPENCV = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender"
AGEGENDER_WEIGHTS = "https://github.com/eveningglow/age-and-gender-classification/raw/master/model"
# Same weights as the Silent-Face-Anti-Spoofing release, exported to ONNX.
# Checked against the copy this project was developed with: identical logits.
MINIFASNET = "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx"
LBF = "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml"

URLS = {
    "face_detection_yunet_2023mar.onnx": f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "ediffiqa_tiny_jun2024.onnx": f"{ZOO}/face_image_quality_assessment_ediffiqa/ediffiqa_tiny_jun2024.onnx",
    "age_deploy.prototxt": f"{LEARNOPENCV}/age_deploy.prototxt",
    "gender_deploy.prototxt": f"{LEARNOPENCV}/gender_deploy.prototxt",
    "age_net.caffemodel": f"{AGEGENDER_WEIGHTS}/age_net.caffemodel",
    "gender_net.caffemodel": f"{AGEGENDER_WEIGHTS}/gender_net.caffemodel",
    "lbfmodel.yaml": LBF,
    "minifasnet_v2.onnx": MINIFASNET,
}

# What each file must hash to. The downloads come from six third-party repos,
# any of which could change or be replaced; a file that doesn't match is
# refused rather than loaded. Recorded from the files at these URLs on
# 2026-09-27 (all but MiniFASNet are byte-identical to the copies this
# project was developed with). A deliberate model update changes its line here.
SHA256 = {
    "face_detection_yunet_2023mar.onnx": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    "face_recognition_sface_2021dec.onnx": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    "ediffiqa_tiny_jun2024.onnx": "9426c899cc0f01665240cb7d9e7f98e18e24e456c178326c771a43da289bfc6a",
    "age_deploy.prototxt": "f58b73e2e20766f54c583cb1a9404f45dab8901773da6864d94b63212ed37ca0",
    "gender_deploy.prototxt": "7379953e048e5bffad9dfc6b3a8807f7fc826e2f27432df56d4c2670784b8e78",
    "age_net.caffemodel": "6dde5d07df5ca1d66ff39e525693f05ccfb9d2c437e188fdd1a10d42e57fabd6",
    "gender_net.caffemodel": "ac7571b281ae078817764b645a20541bd6aa1babeac20a45e6d8de7d61ba0e50",
    "lbfmodel.yaml": "70dd8b1657c42d1595d6bd13d97d932877b3bed54a95d3c4733a0f740d1fd66b",
    "minifasnet_v2.onnx": "b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907",
}

# A Git LFS pointer is a few hundred bytes of text; a real model is not.
MIN_PLAUSIBLE_BYTES = 2048


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(filename, url, force=False):
    dest = path_for(filename)
    if os.path.exists(dest) and not force:
        print(f"  have    {filename}")
        return True

    print(f"  fetch   {filename} ... ", end="", flush=True)
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                fh.write(chunk)
    except Exception as exc:
        print(f"FAILED ({exc})")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False

    size = os.path.getsize(tmp)
    if size < MIN_PLAUSIBLE_BYTES and not filename.endswith(".prototxt"):
        os.remove(tmp)
        print(f"FAILED (got {size} bytes — looks like an LFS pointer, not the model)")
        return False

    got = sha256_of(tmp)
    if got != SHA256[filename]:
        os.remove(tmp)
        print(f"FAILED (checksum {got[:12]}... is not the expected {SHA256[filename][:12]}...; refusing it)")
        return False
    os.replace(tmp, dest)
    print(f"ok ({size / 1024:.0f} KB, checksum verified)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download files already present")
    ap.add_argument("--list", action="store_true", help="show status and exit")
    args = ap.parse_args()

    os.makedirs(models_dir(), exist_ok=True)

    if args.list:
        for name, (files, desc) in SPECS.items():
            mark = "ready" if have(name) else "MISSING"
            print(f"  {mark:<8}{name:<10}{desc}")
            for f in files:
                where = "ok" if os.path.exists(path_for(f)) else "not downloaded"
                print(f"              {f:<44}{where}")
        return 0

    print(f"Downloading models into {models_dir()}")
    failed = [f for f, url in URLS.items() if not download(f, url, args.force)]

    print()
    for name, (_, desc) in SPECS.items():
        print(f"  {'ready' if have(name) else 'MISSING':<8}{name:<10}{desc}")

    if failed:
        print(f"\n{len(failed)} file(s) failed: {', '.join(failed)}")
        print("The app still runs; the features backed by those models stay off.")
        return 1

    print("\nAll models present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
