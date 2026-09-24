"""Fill the sample-frame directory from a video or a folder of photographs.

    python tools/sample_frames.py clip.mp4
    python tools/sample_frames.py ~/photos --keep 3 --stride 10
    python tools/sample_frames.py %TEMP%/lfw/lfw.parquet
    python tools/sample_frames.py clip.mp4 --dry-run

Three tests skip without a real photograph of a face, and nothing in this
project wrote the files they look for. This writes them.

Bring your own footage. This deliberately downloads nothing: scraping faces
off the internet to build a face-recognition database is prohibited under
the EU AI Act, and a face template is special-category data under UK GDPR,
so a public URL is not a licence. A clip of yourself works. So do the still
corpora the benchmark tools already fetch, which come with licences.
Those arrive as parquet, and this reads them directly -- seed_demo.py
prints the one-line curl that fetches the LFW shard.

The filtering is sampleframes.py, which is the project's own quality gate:
one face in shot, sharp enough, exposed, frontal, and not three copies of
the same instant.
"""
import argparse
import os
import sys

# The project root -- one level up, because this file lives in tools/.
# Needed before the two imports below: run as `python tools/sample_frames.py`,
# sys.path[0] is tools/, and neither `core` nor `pipeline` is under it.
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

from core import corpus_paths       # noqa: E402
from pipeline import sampleframes   # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source",
                        help="a video file, an image, or a folder of images")
    parser.add_argument("--keep", type=int, default=len(sampleframes.NAMES),
                        help="how many frames to write (default %d)"
                             % len(sampleframes.NAMES))
    parser.add_argument("--stride", type=int, default=sampleframes.STRIDE,
                        help="consider every Nth video frame (default %d)"
                             % sampleframes.STRIDE)
    parser.add_argument("--into", default=None,
                        help="write somewhere other than the sample-frame "
                             "directory")
    parser.add_argument("--limit", type=int, default=sampleframes.LIMIT,
                        help="stop after examining N frames; 0 for all "
                             "(default %d, because a corpus shard holds "
                             "thousands and only %d are kept)"
                             % (sampleframes.LIMIT, len(sampleframes.NAMES)))
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be kept, write nothing")
    args = parser.parse_args(argv)

    where = args.into or corpus_paths.sample_frame_dir()
    print("source : %s" % args.source)
    print("target : %s%s" % (where, "  (dry run)" if args.dry_run else ""))
    print()

    try:
        if args.dry_run:
            candidates, rejected = sampleframes.scan(
                args.source, args.stride, args.limit)
            chosen = sampleframes.choose(candidates, args.keep)
            print("%d frame(s) usable, %d would be kept"
                  % (len(candidates), len(chosen)))
            for line in sampleframes.summarise([], rejected):
                print(line)
            return 0

        written, rejected = sampleframes.build(
            args.source, keep=args.keep, stride=args.stride,
            directory=args.into, limit=args.limit)
    except (FileNotFoundError, ValueError) as exc:
        print("could not read the source: %s" % exc, file=sys.stderr)
        return 1

    for line in sampleframes.summarise(written, rejected):
        print(line)
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
