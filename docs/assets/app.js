// Statlas — search, family filter, copy button, lightbox. No dependencies.
(function () {
  const q = document.getElementById("q");
  const chips = Array.from(document.querySelectorAll(".chip"));
  const sections = Array.from(document.querySelectorAll(".cat"));
  const empty = document.querySelector(".empty");
  let family = "all";

  const norm = (s) => s.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "");

  function apply() {
    const terms = q ? norm(q.value.trim()).split(/\s+/).filter(Boolean) : [];
    let shown = 0;
    sections.forEach((sec) => {
      let n = 0;
      const famOk = family === "all" || sec.dataset.cat === family;
      sec.querySelectorAll(".card").forEach((card) => {
        const k = norm(card.dataset.k);
        const ok = famOk && terms.every((t) => k.includes(t));
        card.hidden = !ok;
        if (ok) n++;
      });
      sec.hidden = n === 0;
      shown += n;
    });
    if (empty) empty.hidden = shown !== 0;
    try {
      const url = new URL(location);
      q && q.value ? url.searchParams.set("q", q.value) : url.searchParams.delete("q");
      family !== "all" ? url.searchParams.set("family", family) : url.searchParams.delete("family");
      history.replaceState(null, "", url);
    } catch (e) {}
  }

  function setFamily(f) {
    family = f;
    chips.forEach((c) => c.classList.toggle("on", c.dataset.filter === f));
    apply();
  }

  if (q) {
    const params = new URLSearchParams(location.search);
    if (params.get("q")) q.value = params.get("q");
    if (params.get("family")) family = params.get("family");
    q.addEventListener("input", () => {
      apply();
      if (q.value) document.getElementById("gallery").scrollIntoView({ block: "start" });
    });
    chips.forEach((c) => c.addEventListener("click", () => setFamily(c.dataset.filter)));
    document.querySelectorAll("[data-jump]").forEach((t) =>
      t.addEventListener("click", (e) => {
        e.preventDefault();
        setFamily(t.dataset.jump);
        document.getElementById("gallery").scrollIntoView({ behavior: "smooth" });
      })
    );
    setFamily(family);
  }

  // "/" or ⌘K focuses search
  document.addEventListener("keydown", (e) => {
    const typing = /input|textarea/i.test(document.activeElement.tagName);
    if ((e.key === "/" && !typing) || (e.key === "k" && (e.metaKey || e.ctrlKey))) {
      if (q) { e.preventDefault(); q.focus(); q.select(); }
      else if (e.key === "/" ) { location.href = document.querySelector(".brand").getAttribute("href") + "#gallery"; }
    }
    if (e.key === "Escape") {
      const lb = document.querySelector(".lightbox");
      if (lb) lb.remove();
      else if (q && document.activeElement === q) { q.value = ""; apply(); q.blur(); }
    }
  });

  // Copy code
  document.querySelectorAll("[data-copy]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const text = btn.closest(".code-card").querySelector("pre").innerText;
      try { await navigator.clipboard.writeText(text); } catch (e) {
        const ta = document.createElement("textarea"); ta.value = text; document.body.appendChild(ta);
        ta.select(); document.execCommand("copy"); ta.remove();
      }
      const old = btn.textContent;
      btn.textContent = document.documentElement.dataset.copied || "Copied ✓"; btn.classList.add("done");
      setTimeout(() => { btn.textContent = old; btn.classList.remove("done"); }, 1600);
    });
  });

  // Copy a citation
  document.querySelectorAll("[data-copy-text]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(btn.dataset.copyText); } catch (e) {}
      const old = btn.textContent;
      btn.textContent = document.documentElement.dataset.copied || "Copied ✓"; btn.classList.add("done");
      setTimeout(() => { btn.textContent = old; btn.classList.remove("done"); }, 1600);
    });
  });

  // Close the language menu when clicking elsewhere
  document.addEventListener("click", (e) => {
    document.querySelectorAll("details.lang[open]").forEach((d) => { if (!d.contains(e.target)) d.removeAttribute("open"); });
  });

  // Lightbox for the main figure
  const fig = document.querySelector(".figure a");
  if (fig) {
    fig.addEventListener("click", (e) => {
      e.preventDefault();
      const lb = document.createElement("div");
      lb.className = "lightbox";
      lb.innerHTML = `<img src="${fig.getAttribute("href")}" alt="">`;
      lb.addEventListener("click", () => lb.remove());
      document.body.appendChild(lb);
    });
  }
})();
