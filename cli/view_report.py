"""
cli/view_report.py
------------------
Quick CLI to inspect what's in the SQL database — all users, and every
attendance record (not just today's). Handy for demos and debugging.

Usage:
    python -m cli.view_report
"""

from core import db


def main():
    db.init_db()

    print("=== Registered Users ===")
    users = db.get_all_users()
    if not users:
        print("  (none yet — run register_user.py)")
    for user_id, name in users:
        print(f"  [{user_id}] {name}")

    print("\n=== All Attendance Records ===")
    rows = db.get_all_attendance_with_method()

    if not rows:
        print("  (no attendance logged yet)")
    for name, timestamp, confidence, method in rows:
        # The two recognisers' numbers run opposite ways; say which this is.
        reading = (f"similarity={confidence:.3f} (SFace)" if method == "sface"
                   else f"distance={confidence:.1f} (LBPH)")
        print(f"  {timestamp}  |  {name}  |  {reading}")


if __name__ == "__main__":
    main()
