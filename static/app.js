(function () {
  var box = document.getElementById("caption");
  var stats = document.getElementById("caption-stats");
  if (!box || !stats) { return; }

  function update() {
    var words = box.value.split(/\s+/).filter(Boolean);
    var tags = words.filter(function (w) { return w.charAt(0) === "#"; });
    var text = words.filter(function (w) { return w.charAt(0) !== "#"; }).join(" ");
    stats.textContent = text.length + " characters, " + tags.length +
      (tags.length === 1 ? " hashtag" : " hashtags");
  }

  box.addEventListener("input", update);
  if (box.value) { update(); }
})();
