// Builds the "Contents" list in the article sidebar from the post's h2 headings.
(function () {
  var toc = document.querySelector('.doc-toc');
  var prose = document.querySelector('.prose');
  if (!toc || !prose) return;

  var heads = Array.prototype.filter.call(prose.querySelectorAll('h2'), function (h) { return h.id; });
  if (heads.length < 2) return;

  var list = toc.querySelector('ol');
  var links = heads.map(function (h) {
    var li = document.createElement('li');
    var a = document.createElement('a');
    a.href = '#' + h.id;
    a.textContent = h.textContent;
    li.appendChild(a);
    list.appendChild(li);
    return a;
  });
  toc.hidden = false;

  if (!('IntersectionObserver' in window)) return;
  var current = null;
  var observer = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting) {
        var i = heads.indexOf(e.target);
        if (current) current.classList.remove('is-current');
        current = links[i];
        current.classList.add('is-current');
      }
    });
  }, { rootMargin: '0px 0px -70% 0px' });
  heads.forEach(function (h) { observer.observe(h); });
})();
