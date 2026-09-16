"""Fill the sample-frame directory from a video or a folder of photographs.

    python tools_sample_frames.py clip.mp4
    python tools_sample_frames.py ~/photos --keep 3 --stride 10
    python tools_sample_frames.py clip.mp4 --dry-run

Three tests skip without a real photograph of a face, and nothing in this
project wrote the files they look for. This writes them.

Bring your own footage. This deliberately downloads nothing: scraping faces
off the internet to build a face-recognition database is prohibited under
the EU AI Act, and a face template is special-category data under UK GDPR,
so a public URL is not a licence. A clip of yourself works. So do the still
corpora the benchmark tools already fetch, which come with licences --
point this at the folder once one is unpacked.

The filtering is sampleframes.py, which is the project's own quality gate:
one face in shot, sharp enough, exposed, frontal, and not three copies of
the same instant.
"""
import argparse
import sys

import corpus_paths
import sampleframes


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
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be kept, write nothing")
    args = parser.parse_args(argv)

    where = args.into or corpus_paths.sample_frame_dir()
    print("source : %s" % args.source)
    print("target : %s%s" % (where, "  (dry run)" if args.dry_run else ""))
    print()

    try:
        if args.dry_run:
            candidates, rejected = sampleframes.scan(args.source, args.stride)
            chosen = sampleframes.choose(candidates, args.keep)
            print("%d frame(s) usable, %d would be kept"
                  % (len(candidates), len(chosen)))
            for line in sampleframes.summarise([], rejected):
                print(line)
            return 0

        written, rejected = sampleframes.build(
            args.source, keep=args.keep, stride=args.stride,
            directory=args.into)
    except (FileNotFoundError, ValueError) as exc:
        print("could not read the source: %s" % exc, file=sys.stderr)
        return 1

    for line in sampleframes.summarise(written, rejected):
        print(line)
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
