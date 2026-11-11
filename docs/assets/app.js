// PyStatlas — search, family filter, copy button, lightbox. No dependencies.
(function () {
  const q = document.getElementById("q");
  const chips = Array.from(document.querySelectorAll(".chip"));
  const sections = Array.from(document.querySelectorAll(".cat"));
  const empty = document.querySelector(".empty");
  const box = document.getElementById("suggest");
  let family = "all";

  const norm = (s) => s.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "");
  const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  // Search index: one entry per chart, with its text split by field —
  // chart (title, summary), key (tags, R function), data (data source and its
  // reference), topic (what the data are about), ref (packages and authors it mirrors), text (its explanation).
  const WEIGHT = { chart: 6, key: 4, data: 3, topic: 3, ref: 3, text: 1 };
  // Which field best explains a match, when several hold as many terms.
  const TELLING = { data: 5, topic: 4, ref: 4, key: 2, text: 1 };

  // Where a term starts a word in the text (so "election" does not match
  // "selection"); scripts written without spaces match anywhere.
  const CJK = /[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]/;
  function find(hay, t, from = 0) {
    for (let j = hay.indexOf(t, from); j !== -1; j = hay.indexOf(t, j + 1)) {
      if (j === 0 || CJK.test(t) || !/[a-z0-9]/.test(hay[j - 1])) return j;
    }
    return -1;
  }
  const has = (hay, t) => find(hay, t) !== -1;
  let docs = [];
  try {
    docs = JSON.parse(document.getElementById("search-index").textContent).map((d) => ({
      ...d, nf: d.f.map(([kind, text]) => [kind, text, norm(text)]),
    }));
  } catch (e) {}
  const byId = new Map(docs.map((d) => [d.u, d]));

  // Score a chart against the search terms: every term must appear somewhere;
  // a term found in the title counts more than one found in the explanation.
  function score(d, terms) {
    let total = 0;
    for (const t of terms) {
      let best = 0;
      for (const f of d.nf) if (has(f[2], t) && WEIGHT[f[0]] > best) best = WEIGHT[f[0]];
      if (!best) return null;
      total += best;
    }
    if (find(norm(d.t), terms[0]) === 0) total += 2;
    // Why it matched: the field (other than the title) that holds the most
    // search terms, the more telling field first when two hold as many.
    let why = null, most = 0;
    if (!terms.every((t) => has(norm(d.t), t))) {
      for (const f of d.nf) {
        if (f[0] === "chart") continue;
        const n = terms.filter((t) => has(f[2], t)).length;
        if (n > most || (n === most && n && TELLING[f[0]] > TELLING[why[0]])) { most = n; why = f; }
      }
    }
    return { total, why };
  }

  // A short excerpt of the matched field, with the search terms highlighted.
  function excerpt(text, terms) {
    const n = norm(text);
    const first = terms.find((t) => has(n, t)) || terms[0];
    const at = Math.max(0, find(n, first));
    const a = at > 70 ? text.lastIndexOf(" ", at - 50) + 1 : 0;
    const b = Math.min(text.length, Math.max(at + 110, a + 150));
    const piece = text.slice(a, b);
    const np = norm(piece);
    const on = new Array(piece.length).fill(false);
    if (np.length === piece.length) {        // highlight only when positions line up
      for (const t of terms) {
        for (let j = find(np, t); j !== -1; j = find(np, t, j + t.length)) on.fill(true, j, j + t.length);
      }
    }
    let html = "", i = 0;
    while (i < piece.length) {
      let j = i; while (j < piece.length && on[j] === on[i]) j++;
      html += on[i] ? `<mark>${esc(piece.slice(i, j))}</mark>` : esc(piece.slice(i, j));
      i = j;
    }
    return (a > 0 ? "… " : "") + html + (b < text.length ? " …" : "");
  }

  let results = [], active = 0;

  function renderSuggest(terms) {
    if (!box) return;
    if (!terms.length || document.activeElement !== q) { box.hidden = true; return; }
    const root = box.dataset.root || "";
    const label = { key: box.dataset.mKey, data: box.dataset.mData, topic: box.dataset.mTopic, ref: box.dataset.mRef, text: box.dataset.mText };
    if (!results.length) {
      box.innerHTML = `<p class="sg-none">${esc(box.dataset.none)}</p>`;
    } else {
      box.innerHTML = results.slice(0, 8).map((r, i) => {
        const why = r.why ? `<span class="sg-why"><b>${esc(label[r.why[0]])}</b> ${excerpt(r.why[1], terms)}</span>` : "";
        return `<a class="sg-item${i === active ? " on" : ""}" role="option" href="${r.d.u}" data-i="${i}">
          <img src="${root}${r.d.u}thumb.webp" alt="" loading="lazy">
          <span class="sg-text"><span class="sg-fam">${esc(r.d.c)}</span><span class="sg-title">${esc(r.d.t)}</span>${why}</span></a>`;
      }).join("") + `<button class="sg-all" type="button">${esc(box.dataset.seeAll.replace("{n}", results.length))} ↓</button>`;
      box.querySelector(".sg-all").addEventListener("click", () => {
        box.hidden = true; q.blur();
        document.getElementById("gallery").scrollIntoView({ behavior: "smooth" });
      });
    }
    box.hidden = false;
  }

  function apply() {
    const terms = q ? norm(q.value.trim()).split(/\s+/).filter(Boolean) : [];
    results = [];
    if (terms.length) {
      for (const d of docs) { const s = score(d, terms); if (s) results.push({ d, ...s }); }
      results.sort((a, b) => b.total - a.total || a.d.t.localeCompare(b.d.t));
    }
    const hits = new Set(results.map((r) => r.d.u));
    active = 0;
    let shown = 0;
    sections.forEach((sec) => {
      let n = 0;
      const famOk = family === "all" || sec.dataset.cat === family;
      sec.querySelectorAll(".card").forEach((card) => {
        const ok = famOk && (!terms.length || (byId.has(card.dataset.id)
          ? hits.has(card.dataset.id)
          : terms.every((t) => norm(card.dataset.k).includes(t))));
        card.hidden = !ok;
        if (ok) n++;
      });
      sec.querySelectorAll(".sub").forEach((sub) => {
        sub.hidden = !sub.querySelector(".card:not([hidden])");
      });
      sec.hidden = n === 0;
      shown += n;
    });
    if (empty) empty.hidden = shown !== 0;
    renderSuggest(terms);
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

  function highlight() {
    box.querySelectorAll(".sg-item").forEach((el, i) => el.classList.toggle("on", i === active));
    const el = box.querySelector(".sg-item.on");
    if (el) el.scrollIntoView({ block: "nearest" });
  }

  if (q) {
    const params = new URLSearchParams(location.search);
    if (params.get("q")) q.value = params.get("q");
    if (params.get("family")) family = params.get("family");
    q.addEventListener("input", apply);
    q.addEventListener("focus", apply);
    q.addEventListener("blur", () => { if (box) box.hidden = true; });
    // keep focus in the search box while clicking a suggestion, so the click lands
    if (box) box.addEventListener("mousedown", (e) => e.preventDefault());
    q.addEventListener("keydown", (e) => {
      const n = Math.min(results.length, 8);
      if (e.key === "ArrowDown" && n) { e.preventDefault(); active = (active + 1) % n; highlight(); }
      else if (e.key === "ArrowUp" && n) { e.preventDefault(); active = (active - 1 + n) % n; highlight(); }
      else if (e.key === "Enter" && n) { e.preventDefault(); location.href = results[active].d.u; }
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
      else if (q && document.activeElement === q) { if (box && !box.hidden) box.hidden = true; else { q.value = ""; apply(); q.blur(); } }
    }
  });

  // Copy code
  document.querySelectorAll("[data-copy]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const text = btn.closest(".code-card").querySelector("pre:not([hidden])").innerText;
      try { await navigator.clipboard.writeText(text); } catch (e) {
        const ta = document.createElement("textarea"); ta.value = text; document.body.appendChild(ta);
        ta.select(); document.execCommand("copy"); ta.remove();
      }
      const old = btn.textContent;
      btn.textContent = document.documentElement.dataset.copied || "Copied ✓"; btn.classList.add("done");
      setTimeout(() => { btn.textContent = old; btn.classList.remove("done"); }, 1600);
    });
  });

  // Switch the code card between the Python and the R version
  document.querySelectorAll("[data-code-toggle]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const card = btn.closest(".code-card");
      const toR = card.dataset.lang !== "r";
      card.dataset.lang = toR ? "r" : "py";
      card.querySelectorAll("pre.code").forEach((pre) => { pre.hidden = (pre.dataset.lang === "r") !== toR; });
      card.querySelectorAll("[data-py][data-r]").forEach((el) => {
        const v = toR ? el.dataset.r : el.dataset.py;
        if (el.hasAttribute("href")) el.setAttribute("href", v); else el.textContent = v;
      });
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
