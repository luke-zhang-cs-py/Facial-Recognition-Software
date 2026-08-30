"""
view_report.py
---------------
Quick CLI to inspect what's in the SQL database — all users, and every
attendance record (not just today's). Handy for demos and debugging.

Usage:
    python view_report.py
"""

import db


def main():
    db.init_db()

    print("=== Registered Users ===")
    users = db.get_all_users()
    if not users:
        print("  (none yet — run register_user.py)")
    for user_id, name in users:
        print(f"  [{user_id}] {name}")

    print("\n=== All Attendance Records ===")
    conn = db.get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT u.name, a.timestamp, a.confidence
        FROM attendance a
        JOIN users u ON u.id = a.user_id
        ORDER BY a.timestamp DESC
    """)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("  (no attendance logged yet)")
    for name, timestamp, confidence in rows:
        print(f"  {timestamp}  |  {name}  |  confidence={confidence:.1f}")


if __name__ == "__main__":
    main()
