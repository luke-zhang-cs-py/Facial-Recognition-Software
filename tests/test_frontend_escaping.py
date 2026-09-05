"""Stored XSS: a registered name is script.

Names are stored exactly as typed -- correctly, since a name is not the
system's business to rewrite -- and the dashboard rendered them straight into
`innerHTML`. Registering somebody as `<img src=x onerror=...>` put running
script in every browser that opened the page, and registering people is what
this application is for.

The fix is escaping at the point of display, which is where it belongs: the
database keeps the name, the browser is told it is text. These tests check
the frontend never lost that, since there is no JavaScript test runner here
to check it the direct way.
"""

import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_JS = os.path.join(ROOT, "static", "js", "app.js")


@pytest.fixture(scope="module")
def source():
    return open(APP_JS, encoding="utf-8").read()


def interpolations(source):
    """Every ${...} in the file."""
    return re.findall(r"\$\{[^{}]*\}", source)


def test_the_escaping_helper_exists(source):
    assert "const esc =" in source, "no escaping helper in app.js"


@pytest.mark.parametrize("char", ["&", "<", ">", '"', "'"])
def test_the_helper_covers_every_dangerous_character(source, char):
    """Single quotes included: unlike the email templates, some attributes in
    this file are single-quoted, so &#39; is load-bearing here."""
    helper = source[source.index("const esc ="):source.index("const esc =") + 400]
    assert repr(char)[1:-1] in helper or char in helper, char


DATA_FIELDS = re.compile(r"\b\w+\.(name|message|error|file|label|verdict)\b")


def test_no_person_supplied_field_reaches_the_dom_unescaped(source):
    """The specific failure: `${u.name}` inside a template literal that is
    assigned to innerHTML."""
    offenders = [chunk for chunk in interpolations(source)
                 if DATA_FIELDS.search(chunk) and "esc(" not in chunk]
    assert not offenders, f"wrap these in esc(): {offenders}"


def test_the_user_list_escapes_its_names(source):
    """The list that shows everyone registered -- the first place a hostile
    name would land."""
    assert "esc(u.name)" in source


def test_the_attendance_list_escapes_its_names(source):
    assert "esc(a.name)" in source


def test_the_capture_report_escapes_its_names(source):
    """Both the subject and the nearest other enrolled face."""
    assert "esc(rep.name)" in source
    assert "esc(rep.nearestOther.name)" in source


def test_the_template_literals_are_still_balanced(source):
    """A blunt check that the escaping edit did not break the file: an odd
    number of backticks means a template literal was left open."""
    assert source.count("`") % 2 == 0
    assert source.count("{") == source.count("}")
    assert source.count("(") == source.count(")")


# --------------------------------------------------------------- the server

def test_a_name_is_stored_exactly_as_given(isolated_db):
    """Deliberate. Rewriting somebody's name to make it safe to display is
    the wrong layer -- names contain apostrophes and ampersands and those are
    not attacks. The display escapes; the store remembers.
    """
    payload = "<img src=x onerror=alert(1)>"
    uid = isolated_db.add_user(payload)
    assert isolated_db.get_user_name(uid) == payload


def test_the_api_hands_the_name_back_unchanged(isolated_db):
    """So the escaping really is the only thing standing between a stored
    name and the DOM."""
    payload = "Ampersand & <script>"
    uid = isolated_db.add_user(payload)
    assert (uid, payload) in isolated_db.get_all_users()
