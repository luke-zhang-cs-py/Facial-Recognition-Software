"""
face_attendance.py
-------------------
Single-command front end for the whole pipeline.

    python face_attendance.py register "Jane Doe"
    python face_attendance.py train
    python face_attendance.py attendance
    python face_attendance.py report

This used to be a genuinely self-contained script: 462 lines that
re-implemented twelve functions already living in the modules, including the
entire contents of db.py. Two copies of the schema and the attendance rules,
and only one of them ever received a fix. By the time it was audited the
copies had diverged on three separate bugs -- the connection leak that locked
the database, the detection threshold that discarded real faces, and the yaw
calculation that ran to +/-689 degrees. None of those repairs reached this
file, because nobody remembers to fix a bug twice.

So the convenience is kept and the duplication is not: this is now one
argument parser over the real modules. Every fix lands in one place.
"""

import argparse
import sys

import db
import register_user
import train_model
import attendance
import view_report


def main():
    parser = argparse.ArgumentParser(
        description="Face recognition attendance -- all steps in one command")
    sub = parser.add_subparsers(dest="command", required=True)

    p_reg = sub.add_parser("register", help="capture face samples for a person")
    p_reg.add_argument("name", help="the person's full name")

    sub.add_parser("train", help="rebuild the recogniser from dataset/")
    sub.add_parser("attendance", help="run live recognition and log attendance")
    sub.add_parser("report", help="print users and attendance records")

    args = parser.parse_args()
    db.init_db()

    if args.command == "register":
        register_user.register_user(args.name)
        # Registering without training leaves somebody enrolled who cannot be
        # recognised, which looks identical to the system being broken.
        train_model.train()
    elif args.command == "train":
        train_model.train()
    elif args.command == "attendance":
        attendance.run_attendance()
    elif args.command == "report":
        view_report.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
