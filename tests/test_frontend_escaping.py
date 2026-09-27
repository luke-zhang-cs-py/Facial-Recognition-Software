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


TEXT_CONTENT = re.compile(r"\.textContent\s*=[^;]*;")


def without_text_content(source):
    """The file with every `x.textContent = ...;` statement blanked out.

    textContent is the one sink that must *not* be escaped, so it is checked
    separately below rather than being held to the innerHTML rule."""
    return TEXT_CONTENT.sub(";", source)


def test_no_person_supplied_field_reaches_the_dom_unescaped(source):
    """The specific failure: `${u.name}` inside a template literal that is
    assigned to innerHTML."""
    offenders = [chunk for chunk in interpolations(without_text_content(source))
                 if DATA_FIELDS.search(chunk) and "esc(" not in chunk]
    assert not offenders, f"wrap these in esc(): {offenders}"


def test_text_content_is_never_escaped(source):
    """The opposite mistake. textContent is text already, so esc() there puts
    the entities on screen: the capture report for O'Brien was headed
    "O&#39;Brien", and "Smith & Jones" came out as "Smith &amp; Jones"."""
    statements = TEXT_CONTENT.findall(source)
    assert len(statements) > 20, "the pattern stopped matching the file"
    offenders = [s for s in statements if "esc(" in s]
    assert not offenders, f"drop esc() from these: {offenders}"


def test_an_ambiguous_identification_is_not_called_below_threshold(source):
    """recognition.identify refuses a match that clears the threshold when the
    runner-up is too close behind, and says so in `ambiguous`. The panel read
    only `match`, so it reported those as "below threshold" and printed a
    sentence saying the similarity was under a threshold it was over."""
    start = source.index("function renderIdentify")
    body = source[start:source.index("\n}\n", start)]
    assert "d.ambiguous" in body
    assert "too close to call" in body


def test_the_user_list_escapes_its_names(source):
    """The list that shows everyone registered -- the first place a hostile
    name would land."""
    assert "esc(u.name)" in source


def test_the_attendance_list_escapes_its_names(source):
    assert "esc(a.name)" in source


def test_the_capture_report_keeps_both_names_text(source):
    """The nearest other enrolled face goes into innerHTML, so it is escaped.
    The subject goes into a textContent heading, which is text already --
    this test used to demand esc() there too, which is what put "&#39;" on
    screen."""
    assert "esc(rep.nearestOther.name)" in source
    heading = source[source.index("$('reportSub').textContent"):]
    heading = heading[:heading.index(";")]
    assert "rep.name" in heading and "esc(" not in heading


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
