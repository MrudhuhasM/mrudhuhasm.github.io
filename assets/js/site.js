// Theme toggle and the article contents strip. Both are optional extras:
// the site reads fine with JavaScript off.
(function () {
  var root = document.documentElement;

  // --- Light / dark toggle ---------------------------------------------
  var btn = document.querySelector('.theme-toggle');
  if (btn) {
    btn.addEventListener('click', function () {
      var current = root.getAttribute('data-theme');
      if (!current) {
        current = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
      }
      var next = current === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('theme', next); } catch (e) {}
    });
  }

  // --- Contents strip, built from the article's h2 headings ------------
  var toc = document.querySelector('.toc');
  var prose = document.querySelector('.prose');
  if (!toc || !prose) return;
  var heads = Array.prototype.filter.call(prose.querySelectorAll('h2'), function (h) { return h.id; });
  if (heads.length < 3) return;
  heads.forEach(function (h) {
    var a = document.createElement('a');
    a.href = '#' + h.id;
    a.textContent = h.textContent;
    toc.appendChild(a);
  });
  toc.hidden = false;
})();
