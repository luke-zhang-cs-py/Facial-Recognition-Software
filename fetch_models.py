"""
fetch_models.py
----------------
Download the pretrained weights the trait analysis needs into models/.

    python fetch_models.py           # everything missing
    python fetch_models.py --force   # re-download even if present
    python fetch_models.py --list    # just show what is and isn't there

Roughly 134 MB total, mostly the two Caffe demographic nets. Nothing here is
needed for the original LBPH attendance pipeline — skip it and register/
train/attendance still work, you just lose the analysis tab.

The OpenCV Zoo files are stored in Git LFS, so they come from the
media.githubusercontent.com endpoint; the normal raw URL returns a ~130 byte
pointer file rather than the model, which then fails to load with a confusing
protobuf error.
"""

import argparse
import os
import sys
import urllib.request

from facemodels import MODELS_DIR, SPECS, have, path_for

ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
LEARNOPENCV = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender"
AGEGENDER_WEIGHTS = "https://github.com/eveningglow/age-and-gender-classification/raw/master/model"

URLS = {
    "face_detection_yunet_2023mar.onnx": f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "ediffiqa_tiny_jun2024.onnx": f"{ZOO}/face_image_quality_assessment_ediffiqa/ediffiqa_tiny_jun2024.onnx",
    "age_deploy.prototxt": f"{LEARNOPENCV}/age_deploy.prototxt",
    "gender_deploy.prototxt": f"{LEARNOPENCV}/gender_deploy.prototxt",
    "age_net.caffemodel": f"{AGEGENDER_WEIGHTS}/age_net.caffemodel",
    "gender_net.caffemodel": f"{AGEGENDER_WEIGHTS}/gender_net.caffemodel",
}

# A Git LFS pointer is a few hundred bytes of text; a real model is not.
MIN_PLAUSIBLE_BYTES = 2048


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

    os.replace(tmp, dest)
    print(f"ok ({size / 1024:.0f} KB)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download files already present")
    ap.add_argument("--list", action="store_true", help="show status and exit")
    args = ap.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    if args.list:
        for name, (files, desc) in SPECS.items():
            mark = "ready" if have(name) else "MISSING"
            print(f"  {mark:<8}{name:<10}{desc}")
            for f in files:
                where = "ok" if os.path.exists(path_for(f)) else "not downloaded"
                print(f"              {f:<44}{where}")
        return 0

    print(f"Downloading models into {MODELS_DIR}")
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
