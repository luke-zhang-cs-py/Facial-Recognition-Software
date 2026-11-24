"""The theme tokens: complete in every theme, WCAG AA in every theme, dark with
no JavaScript, and a toggle that switches, remembers, and falls back.

Two palettes are checked. static/css/tokens.css is the app's: the Flask app
loads it and tools/build_static.py copies it into both demos. docs/index.html
carries the shared write-up template's own block, inline.

The pairings are checked from the tokens rather than only from a rendered
page, because a selected row, an error, a badge or a hover state is a pair a
scan of the page at rest never sees. Each matrix below is the pairs the
stylesheets actually use: a new `color: var(--x)` on `var(--y)` belongs here.

The first half needs nothing but Python, so it runs everywhere. The second
half drives a real browser and skips where Playwright or its Chromium is
missing, which includes the CI runner (see tests/test_cov_build_static.py).
"""
import io
import mimetypes
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(*parts):
    with io.open(os.path.join(ROOT, *parts), encoding="utf-8") as handle:
        return handle.read()


TOKENS = read("static", "css", "tokens.css")
THEME_JS = read("static", "js", "theme.js")
WRITEUP = read("docs", "index.html")
WRITEUP_CSS = re.search(r"<style>(.*?)</style>", WRITEUP, re.S).group(1)

# Every page that carries the theme, and where its colours come from.
PAGES = {
    "docs/app/index.html": TOKENS,
    "docs/camera/index.html": TOKENS,
    "templates/index.html": TOKENS,
    "docs/index.html": WRITEUP_CSS,
}

# The colour contract: every theme block sets at least these.
CONTRACT = {"bg", "surface", "surface-subtle", "border", "border-strong",
            "text-primary", "text-muted", "text-subtle", "accent",
            "accent-hover", "accent-subtle", "text-on-accent", "danger"}


def strip_comments(css):
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


def theme_blocks(css):
    """{theme name: {token: value}} for every `.theme-<name>` block."""
    blocks = {}
    for selector, body in re.findall(r"([^{}]*)\{([^}]*)\}", strip_comments(css)):
        names = re.findall(r"\.theme-([a-z]+)", selector)
        tokens = dict(re.findall(r"--color-([\w-]+)\s*:\s*([^;]+);", body))
        for name in names:
            blocks[name] = {k: v.strip() for k, v in tokens.items()}
    return blocks


def themes_in(source):
    """The THEMES list a page's script declares."""
    found = re.search(r"var THEMES = \[([^\]]*)\];", source)
    assert found, "no THEMES list"
    return re.findall(r"'([a-z]+)'", found.group(1))


def luminance(hex_colour):
    h = hex_colour.lstrip("#")
    assert re.fullmatch(r"[0-9a-fA-F]{6}", h), "not a #rrggbb colour: %s" % hex_colour

    def channel(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (channel(int(h[i:i + 2], 16)) for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


# ---------------------------------------------------------- pair matrices
# (foreground token, background token). Text at 4.5:1, UI boundaries at 3:1.
_PLAIN = ("bg", "surface", "surface-subtle")
APP_TEXT = (
    [(fg, bg) for fg in ("text-primary", "text-muted", "text-subtle", "accent")
     for bg in _PLAIN]
    # the chosen rule in the camera chain, and the app's soft buttons
    + [(fg, "accent-subtle") for fg in ("text-primary", "text-muted",
                                        "text-subtle", "accent")]
    # the go button, at rest and hovered
    + [("text-on-accent", "accent"), ("text-on-accent", "accent-hover")]
    # status text on panels and on its own tint (tags, pills, chips, errors)
    + [(s, bg) for s in ("success", "warning", "danger")
       for bg in _PLAIN + (s + "-subtle",)]
    # severity badges: text on a solid status fill
    + [("text-on-status", s) for s in ("success", "warning", "danger",
                                       "text-subtle")]
    # the app's neutral chip
    + [("text-muted", "grid")]
)
APP_UI = (
    # form fields and buttons; the chart series and focus rings; the edge line
    [("border-strong", bg) for bg in _PLAIN]
    + [("accent", bg) for bg in _PLAIN]
    + [("danger", "surface")]
)
WRITEUP_TEXT = (
    [(fg, bg) for fg in ("text-primary", "text-muted", "text-subtle", "accent")
     for bg in _PLAIN]
    # the status flags (glyph + word) in the tables
    + [(s, bg) for s in ("success", "danger", "warning-text")
       for bg in ("bg", "surface")]
    + [("text-on-accent", "accent")]
)
WRITEUP_UI = ([("border-strong", bg) for bg in ("bg", "surface")]
              + [("accent", "surface"), ("accent", "track")])

PALETTES = {
    "static/css/tokens.css": (TOKENS, THEME_JS, APP_TEXT, APP_UI),
    "docs/index.html": (WRITEUP_CSS, WRITEUP, WRITEUP_TEXT, WRITEUP_UI),
}


# ------------------------------------------------- 1. complete theme blocks

@pytest.mark.parametrize("where", sorted(PALETTES))
def test_every_theme_sets_the_same_tokens_as_the_default(where):
    css, script, _, _ = PALETTES[where]
    blocks, themes = theme_blocks(css), themes_in(script)
    assert themes[0] == "dark", "the markup opens dark, so THEMES must start there"
    default = blocks.get(themes[0])
    assert default, "no .theme-%s block" % themes[0]
    assert CONTRACT <= set(default), "missing %s" % sorted(CONTRACT - set(default))
    for name in themes:
        assert name in blocks, "THEMES names %r but there is no .theme-%s block" % (name, name)
        assert set(blocks[name]) == set(default), (
            "%s: .theme-%s differs from .theme-%s by %s" % (
                where, name, themes[0], sorted(set(blocks[name]) ^ set(default))))


def test_the_default_block_is_also_the_no_class_fallback():
    for where, (css, _, _, _) in PALETTES.items():
        assert re.search(r":where\(:root\)\s*,\s*\.theme-dark\s*\{", strip_comments(css)), where


# ---------------------------------------------------- 2. AA in every theme

@pytest.mark.parametrize("where", sorted(PALETTES))
def test_every_pairing_meets_aa_in_every_theme(where):
    css, script, text, ui = PALETTES[where]
    blocks = theme_blocks(css)
    short = []
    for name in themes_in(script):
        tokens = blocks[name]
        for pairs, need in ((text, 4.5), (ui, 3.0)):
            for fg, bg in pairs:
                ratio = contrast(tokens[fg], tokens[bg])
                if ratio < need:
                    short.append("%s: %s on %s is %.2f:1, needs %s" % (name, fg, bg, ratio, need))
    assert not short, "\n".join(short)


def test_the_overlay_colours_read_on_the_video_in_any_theme():
    """The mode flag and instruction bar sit on the camera picture, so their
    colours are theme-independent: checked against the stage and the glass."""
    style = read("static", "css", "style.css")
    overlay = dict(re.findall(r"--overlay-([\w-]+):(#[0-9a-fA-F]{6})", style))
    glass = re.search(r"--overlay-glass:rgba\((\d+),(\d+),(\d+),", style).groups()
    overlay["glass"] = "#%02x%02x%02x" % tuple(int(c) for c in glass)
    for fg in ("ink", "muted", "good", "bad", "warn"):
        for bg in ("stage", "glass"):
            assert contrast(overlay[fg], overlay[bg]) >= 4.5, (fg, bg)


# ------------------------------------------- 3. dark from the markup alone

@pytest.mark.parametrize("page", sorted(PAGES))
def test_the_markup_opens_dark_and_nothing_follows_the_system(page):
    text = read(*page.split("/"))
    assert re.search(r'<html lang="en" class="theme-dark">', text), page
    assert "prefers-color-scheme" not in text
    assert 'id="themeBtn"' in text and 'aria-label="Switch to light theme"' in text


def test_no_stylesheet_switches_on_the_system_theme():
    for path in ("static/css/tokens.css", "static/css/style.css",
                 "tools/static_src/css/demo.css", "tools/static_src/camera/css/camera.css"):
        assert "prefers-color-scheme" not in read(*path.split("/")), path


@pytest.mark.parametrize("path", ["static/css/style.css", "tools/static_src/css/demo.css",
                                  "tools/static_src/camera/css/camera.css"])
def test_component_rules_read_tokens_not_raw_colours(path):
    """Raw colours are allowed only in token definitions (:root and the theme
    blocks), which is where the overlay and stage colours live."""
    css = strip_comments(read(*path.split("/")))
    rules = re.sub(r":root\s*\{[^}]*\}", "", css)
    raw = re.findall(r"#[0-9a-fA-F]{3,8}\b|rgba?\(|hsla?\(", rules)
    assert not raw, "%s: raw colours outside the tokens: %s" % (path, raw)


# ------------------------------------------------------------ the browser
# 4 and 5: the toggle and the rendered contrast, in a real Chromium.

SCAN = r"""() => {
  const parse = c => { const n = c.match(/[\d.]+/g).map(Number); const unit = c.startsWith('color(') ? 1 : 255;
                       return { rgb: n.slice(0, 3).map(v => v / unit), a: n.length > 3 ? n[3] : 1 }; };
  const lum = c => { const [r, g, b] = parse(c).rgb.map(v => v <= .03928 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4);
                     return .2126 * r + .7152 * g + .0722 * b; };
  const behind = el => { for (let e = el; e; e = e.parentElement) { const bg = getComputedStyle(e).backgroundColor;
                     if (/\d/.test(bg) && parse(bg).a > .5) return bg; } return getComputedStyle(document.body).backgroundColor; };
  const out = [];
  for (const el of document.querySelectorAll('body *')) {
    if (![...el.childNodes].some(n => n.nodeType === 3 && n.textContent.trim())) continue;
    const s = getComputedStyle(el); if (s.visibility === 'hidden' || s.display === 'none' || !el.getClientRects().length) continue;
    if (el.closest('[hidden], noscript, option')) continue;
    if (parseFloat(s.opacity) < 1 || el.closest('fieldset.frozen, :disabled')) continue;   // disabled: exempt from AA
    const big = parseFloat(s.fontSize) >= 24 || (parseFloat(s.fontSize) >= 18.66 && +s.fontWeight >= 700);
    const [L1, L2] = [lum(s.color), lum(behind(el))].sort((a, b) => b - a);
    const r = (L1 + .05) / (L2 + .05);
    if (r < (big ? 3 : 4.5)) out.push(el.tagName + '.' + el.className + ' ' + r.toFixed(2) + ' "' + el.textContent.trim().slice(0, 30) + '"');
  }
  return [...new Set(out)];
}"""
THEME = r"(document.documentElement.className.match(/(?:^|\s)theme-(\S+)/) || [])[1]"
BG = "getComputedStyle(document.body).backgroundColor"


def rgb(hex_colour):
    h = hex_colour.lstrip("#")
    return "rgb(%d, %d, %d)" % tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


@pytest.fixture(scope="module")
def chromium():
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as pw:
        try:
            browser = pw.chromium.launch()
        except Exception as error:      # no browser downloaded on this machine
            pytest.skip("Playwright's Chromium is not installed: %s" % str(error).splitlines()[0])
        yield browser
        browser.close()


@pytest.fixture
def flask_html(isolated_root):
    """The app's page as Flask renders it, with nothing behind it running."""
    import app as flask_app
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_request_context():
        return flask_app.render_template("index.html")


ORIGIN = "http://pages.test/"
# page -> (path served for it, the palette it reads)
SERVED = {
    "docs/app/index.html": ("docs/app/", TOKENS),
    "docs/camera/index.html": ("docs/camera/", TOKENS),
    "docs/index.html": ("docs/", WRITEUP_CSS),
    "templates/index.html": ("", TOKENS),
}


def serve(page, flask_html):
    """Answer every request from the repository, so no server is started and
    nothing leaves the machine (fonts and MediaPipe are refused)."""
    def handler(route):
        url = route.request.url
        if not url.startswith(ORIGIN):
            return route.abort()
        path = url[len(ORIGIN):].split("?")[0]
        if page == "templates/index.html":
            if path == "":
                return route.fulfill(body=flask_html, content_type="text/html")
            if path.startswith("api/"):
                return route.fulfill(status=503, body="{}", content_type="application/json")
        if path.endswith("/") or path == "":
            path += "index.html"
        full = os.path.join(ROOT, *path.split("/"))
        if not os.path.isfile(full):
            return route.fulfill(status=404, body="")
        with open(full, "rb") as handle:
            body = handle.read()
        return route.fulfill(body=body, content_type=mimetypes.guess_type(full)[0] or "text/plain")
    return handler


def open_page(browser, page, flask_html, seed=None, js=True):
    context = browser.new_context(java_script_enabled=js, viewport={"width": 1280, "height": 900})
    context.route("**/*", serve(page, flask_html))
    if seed is not None:
        context.add_init_script("localStorage.setItem('theme', %r)" % seed)
    tab = context.new_page()
    tab.goto(ORIGIN + SERVED[page][0])
    tab.wait_for_timeout(300)
    return context, tab


@pytest.mark.parametrize("page", sorted(SERVED))
def test_without_javascript_the_page_is_dark(chromium, flask_html, page):
    context, tab = open_page(chromium, page, flask_html, js=False)
    dark = theme_blocks(SERVED[page][1])["dark"]
    assert tab.evaluate("document.documentElement.className") == "theme-dark"
    assert tab.evaluate(BG) == rgb(dark["bg"])
    assert tab.evaluate("getComputedStyle(document.documentElement).colorScheme") == "dark"
    context.close()


@pytest.mark.parametrize("page", sorted(SERVED))
def test_the_toggle_switches_and_remembers(chromium, flask_html, page):
    context, tab = open_page(chromium, page, flask_html)
    light = theme_blocks(SERVED[page][1])["light"]
    assert tab.evaluate(THEME) == "dark"
    assert tab.get_attribute("#themeBtn", "aria-label") == "Switch to light theme"
    tab.click("#themeBtn")
    assert tab.evaluate(THEME) == "light"
    assert tab.evaluate(BG) == rgb(light["bg"])
    assert tab.get_attribute("#themeBtn", "aria-label") == "Switch to dark theme"
    assert tab.evaluate("localStorage.getItem('theme')") == "light"
    tab.reload()
    tab.wait_for_timeout(300)
    assert tab.evaluate(THEME) == "light", "the choice was not remembered"
    tab.click("#themeBtn")
    assert tab.evaluate(THEME) == "dark"
    context.close()


@pytest.mark.parametrize("page", sorted(SERVED))
@pytest.mark.parametrize("seed, want", [("light", "light"), ("sepia", "dark")])
def test_a_saved_theme_is_restored_and_an_unknown_one_falls_back(chromium, flask_html, page, seed, want):
    context, tab = open_page(chromium, page, flask_html, seed=seed)
    classes = tab.evaluate(r"document.documentElement.className.match(/theme-\S+/g)")
    assert classes == ["theme-" + want]
    assert tab.evaluate(BG) == rgb(theme_blocks(SERVED[page][1])[want]["bg"])
    context.close()


@pytest.mark.parametrize("page", sorted(SERVED))
def test_rendered_text_meets_aa_in_every_theme(chromium, flask_html, page):
    context, tab = open_page(chromium, page, flask_html)
    tab.wait_for_timeout(500)
    if page == "docs/app/index.html":
        # every rule firing at once: the chosen row, the fired rows, the failing checklist
        tab.click("text=everything wrong")
    found = {}
    for name in themes_in(THEME_JS if SERVED[page][1] is TOKENS else WRITEUP):
        tab.evaluate("name => { const r = document.documentElement;"
                     " r.className = r.className.replace(/(^|\\s)theme-\\S+/g, ' ').trim() + ' theme-' + name; }",
                     name)
        bad = tab.evaluate(SCAN)
        if bad:
            found[name] = bad[:6]
    context.close()
    assert not found, found
