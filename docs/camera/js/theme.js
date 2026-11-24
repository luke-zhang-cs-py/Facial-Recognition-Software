// theme.js -- loaded from <head>, before the stylesheets apply. Not needed to
// theme the page: class="theme-dark" on <html> already does that, with or
// without JavaScript. This restores a theme the visitor picked before, ahead
// of first paint so it never flashes, then wires the #themeBtn toggle.
//
// Shared by the Flask app (static/js/) and both demos, which
// tools/build_static.py copies it into byte for byte.
//
// THEMES in order; each name needs a matching .theme-<name> block in
// css/tokens.css, and the first is the one written into the markup. The key
// is 'theme' on purpose: every page on luke-zhang-cs-py.github.io shares it,
// so one choice carries across the whole site.
var THEMES = ['dark', 'light'];

(function () {
  var root = document.documentElement;
  var saved = null;
  try { saved = localStorage.getItem('theme'); } catch (e) { /* storage blocked */ }
  if (/^[a-z]+$/.test(saved)) { root.classList.replace('theme-dark', 'theme-' + saved); }

  function current() {
    return THEMES.filter(function (t) { return root.classList.contains('theme-' + t); })[0] || THEMES[0];
  }
  function next(name) { return THEMES[(THEMES.indexOf(name) + 1) % THEMES.length]; }

  function apply(name, button) {
    root.className = (root.className.replace(/(^|\s)theme-\S+/g, ' ').trim() + ' theme-' + name).trim();
    if (button) {
      var to = next(name);
      button.setAttribute('aria-label', 'Switch to ' + to + ' theme');
      button.textContent = to.charAt(0).toUpperCase() + to.slice(1);
    }
    // the browser's own toolbar colour follows the page, where a page sets one
    var bg = getComputedStyle(root).getPropertyValue('--color-bg').trim();
    var metas = document.querySelectorAll('meta[name="theme-color"]');
    for (var i = 0; i < metas.length; i++) { metas[i].content = bg; }
  }

  document.addEventListener('DOMContentLoaded', function () {
    var button = document.getElementById('themeBtn');
    apply(current(), button);   // also drops a saved theme that no longer exists
    if (!button) { return; }
    button.addEventListener('click', function () {
      var to = next(current());
      try { localStorage.setItem('theme', to); } catch (e) { /* not remembered */ }
      apply(to, button);
    });
  });
})();
