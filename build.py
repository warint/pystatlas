#!/usr/bin/env python3
"""
Statlas build script.

The repository folders *are* the gallery:

    gallery/
      01-distributions/            <- a category (order prefix + slug)
        _category.yaml             <- title, summary
        01-histogram-density/      <- an example (order prefix + slug)
          plot.py                  <- metadata docstring + runnable code
          plot.png                 <- rendered by this script
          README.md                <- generated, so GitHub itself is browsable

Running this script renders every example, writes the README files and
builds the static website into ``docs/`` (served by GitHub Pages).

    python build.py              # render changed examples + build site
    python build.py --force      # re-render everything
    python build.py --only coef  # render examples whose path contains "coef"
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from PIL import Image
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

ROOT = Path(__file__).resolve().parent
GALLERY = ROOT / "gallery"
DOCS = ROOT / "docs"
SITE_SRC = ROOT / "site_src"
STYLE = ROOT / "statlas.mplstyle"
CONFIG = yaml.safe_load((ROOT / "statlas.yaml").read_text())

LIB_NAMES = {
    "numpy": "NumPy", "pandas": "pandas", "matplotlib": "Matplotlib",
    "seaborn": "seaborn", "scipy": "SciPy", "statsmodels": "statsmodels",
    "sklearn": "scikit-learn", "lifelines": "lifelines",
}

RUNNER = r"""
import sys, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use(sys.argv[2])
out = sys.argv[3]
_saved = []
def _show(*a, **k):
    fig = plt.gcf()
    fig.savefig(out)
    _saved.append(out)
    plt.close("all")
plt.show = _show
src = open(sys.argv[1], encoding="utf-8").read()
exec(compile(src, sys.argv[1], "exec"), {"__name__": "__main__"})
if not _saved:
    _show()
"""


# --------------------------------------------------------------------------- model
@dataclass
class Example:
    category: "Category"
    folder: Path
    slug: str
    order: int
    title: str
    summary: str
    description: str
    r: str
    tags: list[str]
    code: str
    libs: list[str]

    @property
    def url(self) -> str:  # relative to docs root
        return f"{self.category.slug}/{self.slug}/"

    @property
    def png(self) -> Path:
        return self.folder / "plot.png"

    @property
    def repo_path(self) -> str:
        return self.folder.relative_to(ROOT).as_posix()


@dataclass
class Category:
    folder: Path
    slug: str
    order: int
    title: str
    summary: str
    examples: list[Example] = field(default_factory=list)


def split_prefix(name: str) -> tuple[int, str]:
    m = re.match(r"^(\d+)-(.+)$", name)
    return (int(m.group(1)), m.group(2)) if m else (999, name)


def parse_example(cat: Category, folder: Path) -> Example:
    src = (folder / "plot.py").read_text(encoding="utf-8")
    m = re.match(r'^\s*"""(.*?)"""\s*\n', src, re.S)
    if not m:
        raise SystemExit(f"{folder}: plot.py must start with a metadata docstring")
    meta = yaml.safe_load(m.group(1)) or {}
    code = src[m.end():].strip("\n") + "\n"
    libs = []
    for mod in re.findall(r"^\s*(?:from|import)\s+([a-zA-Z_]+)", code, re.M):
        name = LIB_NAMES.get(mod)
        if name and name not in libs:
            libs.append(name)
    order, slug = split_prefix(folder.name)
    return Example(
        category=cat, folder=folder, slug=slug, order=order,
        title=meta["title"], summary=meta.get("summary", ""),
        description=(meta.get("description") or "").strip(),
        r=meta.get("r", ""), tags=list(meta.get("tags", [])),
        code=code, libs=libs,
    )


def discover() -> list[Category]:
    cats = []
    for cdir in sorted(p for p in GALLERY.iterdir() if p.is_dir()):
        meta = yaml.safe_load((cdir / "_category.yaml").read_text())
        order, slug = split_prefix(cdir.name)
        cat = Category(cdir, slug, order, meta["title"], meta.get("summary", ""))
        for edir in sorted(p for p in cdir.iterdir() if (p / "plot.py").exists()):
            cat.examples.append(parse_example(cat, edir))
        cats.append(cat)
    return sorted(cats, key=lambda c: c.order)


# --------------------------------------------------------------------------- render
def render(ex: Example, force: bool) -> tuple[Example, str | None]:
    src = ex.folder / "plot.py"
    newest_input = max(src.stat().st_mtime, STYLE.stat().st_mtime)
    if not force and ex.png.exists() and ex.png.stat().st_mtime > newest_input:
        return ex, None
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONWARNINGS="ignore")
    proc = subprocess.run(
        [sys.executable, "-c", RUNNER, str(src), str(STYLE), str(ex.png)],
        cwd=ex.folder, env=env, capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        return ex, proc.stderr[-3000:]
    return ex, None


# --------------------------------------------------------------------------- helpers
def esc(s: str) -> str:
    return html.escape(s, quote=True)


def inline_md(s: str) -> str:
    """Tiny markdown: `code`, **bold**, *italic*, paragraphs."""
    s = esc(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", s)
    return "".join(f"<p>{p.strip()}</p>" for p in re.split(r"\n\s*\n", s) if p.strip())


FORMATTER = HtmlFormatter(nowrap=True)


def highlight_py(code: str) -> str:
    return highlight(code, PythonLexer(), FORMATTER)


def page(title: str, body: str, depth: int, description: str = "", active: str = "",
         cite_title: str = "", url: str = "") -> str:
    base = "../" * depth
    cite_title = cite_title or "Statlas: The Python Gallery of Statistical Graphics"
    scholar = f"""<meta name="author" content="{esc(CONFIG['author'])}">
<link rel="author" href="{CONFIG['author_url']}">
<meta name="citation_title" content="{esc(cite_title)}">
<meta name="citation_author" content="Warin, Thierry">
<meta name="citation_publication_date" content="{CONFIG['year']}">
<meta name="citation_online_date" content="{CONFIG['year']}/11/06">
<meta name="citation_publisher" content="{esc(CONFIG['author'])}">
<meta name="citation_public_url" content="{CONFIG['site_url']}{url}">
<meta name="DC.creator" content="Warin, Thierry">
<meta name="DC.title" content="{esc(cite_title)}">
<meta name="DC.date" content="{CONFIG['year']}">
<link rel="canonical" href="{CONFIG['site_url']}{url}">
<script type="application/ld+json">{json.dumps({"@context": "https://schema.org", "@type": "CreativeWork", "name": cite_title,
  "url": CONFIG['site_url'] + url, "datePublished": str(CONFIG['year']), "license": "https://opensource.org/licenses/MIT",
  "author": {"@type": "Person", "name": CONFIG['author'], "url": CONFIG['author_url']},
  "isPartOf": {"@type": "WebSite", "name": "Statlas", "url": CONFIG['site_url']}})}</script>"""
    desc = esc(description or CONFIG["tagline"])
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="{desc}">
{scholar}
<meta property="og:title" content="{esc(title)}">
<meta property="og:description" content="{desc}">
<meta name="theme-color" content="#fbfbfd" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#000000" media="(prefers-color-scheme: dark)">
<link rel="icon" href="{base}assets/favicon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="{base}assets/style.css">
</head>
<body>
<header class="nav">
  <div class="nav-inner">
    <a class="brand" href="{base}index.html">{LOGO}<span>{esc(CONFIG['name'])}</span></a>
    <nav class="nav-links">
      <a href="{base}index.html#gallery" class="{'on' if active == 'gallery' else ''}">Gallery</a>
      <a href="{base}style.html" class="{'on' if active == 'style' else ''}">Style</a>
      <a href="{base}about.html" class="{'on' if active == 'about' else ''}">About</a>
      <a href="{base}about.html#cite">Cite</a>
      <a href="{CONFIG['repo_url']}" class="gh" aria-label="GitHub">{GITHUB_ICON}</a>
    </nav>
  </div>
</header>
<main>
{body}
</main>
<footer class="foot">
  <div class="wrap foot-inner">
    <div><strong>{esc(CONFIG['name'])}</strong> · {esc(CONFIG['tagline'])}<br>Created by <a href="{CONFIG['author_url']}">{esc(CONFIG['author'])}</a> · <a href="{CONFIG['author_url']}">www.warin.ca</a> · <a href="{base}about.html#cite">How to cite</a></div>
    <div class="foot-links"><a href="{CONFIG['repo_url']}">GitHub</a><a href="{base}style.html">Plot style</a><a href="{CONFIG['repo_url']}/blob/main/CONTRIBUTING.md">Contribute</a><span>MIT License</span></div>
  </div>
</footer>
<script src="{base}assets/app.js"></script>
</body>
</html>
"""


LOGO = """<svg class="logo" viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="8" fill="url(#lg)"/><defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0A84FF"/><stop offset="1" stop-color="#5E5CE6"/></linearGradient></defs><rect x="7" y="17" width="4" height="8" rx="2" fill="#fff" opacity=".75"/><rect x="14" y="11" width="4" height="14" rx="2" fill="#fff"/><rect x="21" y="7" width="4" height="18" rx="2" fill="#fff" opacity=".9"/></svg>"""
GITHUB_ICON = """<svg viewBox="0 0 16 16" width="18" height="18" aria-hidden="true"><path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>"""


def card(ex: Example, base: str) -> str:
    keywords = " ".join([ex.title, ex.summary, ex.r, ex.category.title, *ex.tags, *ex.libs]).lower()
    r = f'<span class="card-r">{esc(ex.r)}</span>' if ex.r else ""
    return f"""<a class="card" href="{base}{ex.url}" data-cat="{ex.category.slug}" data-k="{esc(keywords)}">
  <div class="card-img"><img src="{base}{ex.url}thumb.webp" alt="{esc(ex.title)}" loading="lazy" width="720" height="450"></div>
  <div class="card-body"><h3>{esc(ex.title)}</h3><p>{esc(ex.summary)}</p>{r}</div>
</a>"""


# --------------------------------------------------------------------------- citation
def cite_text(ex: "Example | None" = None) -> str:
    a, y, site = CONFIG["author"], CONFIG["year"], CONFIG["site_url"]
    base = f"Warin, T. ({y}). Statlas: The Python Gallery of Statistical Graphics."
    if ex:
        return f"Warin, T. ({y}). {ex.title}. In Statlas: The Python Gallery of Statistical Graphics. {site}{ex.url}"
    return f"{base} {site}"


def bibtex() -> str:
    return f"""@misc{{warin{CONFIG['year']}statlas,
  author       = {{Warin, Thierry}},
  title        = {{Statlas: The Python Gallery of Statistical Graphics}},
  year         = {{{CONFIG['year']}}},
  howpublished = {{\\url{{{CONFIG['site_url']}}}}},
  note         = {{Source code: {CONFIG['repo_url']}}}
}}"""


def cite_block(ex: "Example | None" = None) -> str:
    return f"""<section class="cite" id="cite">
  <div class="cite-head"><h2>How to cite</h2>
  <p>{esc(CONFIG['name'])} is created and maintained by <a href="{CONFIG['author_url']}">{esc(CONFIG['author'])}</a>. If it helped your work, please cite it.</p></div>
  <div class="cite-box"><p class="cite-ref">{esc(cite_text(ex))}</p><button class="btn ghost" data-copy-text="{esc(cite_text(ex))}">Copy</button></div>
  <div class="cite-box"><pre class="cite-bib">{esc(bibtex())}</pre><button class="btn ghost" data-copy-text="{esc(bibtex())}">Copy BibTeX</button></div>
</section>"""


# --------------------------------------------------------------------------- site
def build_site(cats: list[Category]) -> None:
    if DOCS.exists():
        shutil.rmtree(DOCS)
    (DOCS / "assets").mkdir(parents=True)
    for f in SITE_SRC.iterdir():
        shutil.copy(f, DOCS / "assets" / f.name)
    css = (DOCS / "assets" / "style.css").read_text()
    css += "\n/* --- pygments (generated) --- */\n" + pygments_css()
    (DOCS / "assets" / "style.css").write_text(css)
    shutil.copy(STYLE, DOCS / "statlas.mplstyle")
    (DOCS / ".nojekyll").write_text("")

    all_ex = [e for c in cats for e in c.examples]
    n = len(all_ex)

    # ---- per-example pages
    for c in cats:
        for i, ex in enumerate(c.examples):
            out = DOCS / ex.url
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy(ex.png, out / "plot.png")
            shutil.copy(ex.folder / "plot.py", out / f"{ex.slug}.py")
            make_thumb(ex.png, out / "thumb.webp")
            prev_ex = c.examples[i - 1] if i > 0 else None
            next_ex = c.examples[i + 1] if i < len(c.examples) - 1 else None
            (out / "index.html").write_text(example_page(ex, prev_ex, next_ex))

    # ---- category pages
    for c in cats:
        out = DOCS / c.slug
        out.mkdir(parents=True, exist_ok=True)
        cards = "\n".join(card(e, "../") for e in c.examples)
        body = f"""<section class="hero hero-sm"><div class="wrap">
<div class="crumbs"><a href="../index.html">Gallery</a><span>/</span>{esc(c.title)}</div>
<h1>{esc(c.title)}</h1><p class="lede">{esc(c.summary)}</p></div></section>
<section class="wrap"><div class="grid">{cards}</div></section>"""
        (out / "index.html").write_text(page(f"{c.title} · {CONFIG['name']}", body, 1, c.summary, "gallery"))

    # ---- home
    chips = '<button class="chip on" data-filter="all">All</button>' + "".join(
        f'<button class="chip" data-filter="{c.slug}">{esc(c.title)}</button>' for c in cats)
    sections = []
    for c in cats:
        cards = "\n".join(card(e, "") for e in c.examples)
        sections.append(f"""<section class="cat" data-cat="{c.slug}" id="{c.slug}">
  <div class="cat-head"><div><h2>{esc(c.title)}</h2><p>{esc(c.summary)}</p></div>
  <a class="more" href="{c.slug}/index.html">{len(c.examples)} charts →</a></div>
  <div class="grid">{cards}</div>
</section>""")
    tiles = "".join(
        f'<a class="tile" href="#{c.slug}" data-jump="{c.slug}"><img src="{c.examples[0].url}thumb.webp" alt="" loading="lazy">'
        f'<span>{esc(c.title)}</span><em>{len(c.examples)}</em></a>' for c in cats if c.examples)
    body = f"""<section class="hero"><div class="wrap">
  <p class="eyebrow">{n} statistical charts · {len(cats)} families · 100% Python</p>
  <h1>Statistical graphics.<br><span class="grad">Beautifully Pythonic.</span></h1>
  <p class="lede">{esc(CONFIG['lede'])}</p>
  <p class="byline">By <a href="{CONFIG['author_url']}">{esc(CONFIG['author'])}</a></p>
  <div class="search"><svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>
    <input id="q" type="search" placeholder="Search charts, methods or R functions — try “likert”" autocomplete="off" spellcheck="false">
    <kbd>/</kbd></div>
</div></section>
<section class="wrap tiles">{tiles}</section>
<section class="wrap" id="gallery">
  <div class="chips" role="tablist">{chips}</div>
  <p class="empty" hidden>No chart matches your search. Try “regression”, “proportion” or an R function name.</p>
  {''.join(sections)}
</section>
<div class="wrap">{cite_block()}</div>"""
    (DOCS / "index.html").write_text(page(f"{CONFIG['name']} — {CONFIG['tagline']}", body, 0, active="gallery"))

    # ---- style page & about page
    (DOCS / "style.html").write_text(page(f"Plot style · {CONFIG['name']}", style_page(), 0, active="style"))
    (DOCS / "about.html").write_text(page(f"About · {CONFIG['name']}", about_page(cats), 0, active="about"))

    # ---- sitemap + robots (search-engine indexing)
    urls = ["", "about.html", "style.html"] + [f"{c.slug}/" for c in cats] + [e.url for e in all_ex]
    (DOCS / "sitemap.xml").write_text('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "".join(f"  <url><loc>{CONFIG['site_url']}{u}</loc></url>\n" for u in urls) + "</urlset>\n")
    (DOCS / "robots.txt").write_text(f"User-agent: *\nAllow: /\nSitemap: {CONFIG['site_url']}sitemap.xml\n")

    # ---- search index (handy for tooling)
    index = [{"title": e.title, "summary": e.summary, "r": e.r, "category": e.category.title,
              "tags": e.tags, "url": e.url} for e in all_ex]
    (DOCS / "index.json").write_text(json.dumps(index, indent=1))


def pygments_css() -> str:
    # Apple-ish light / dark code palettes, mapped onto pygments token classes.
    return """
.code .k,.code .kn,.code .kd,.code .kc,.code .ow{color:var(--c-kw);font-weight:500}
.code .s,.code .s1,.code .s2,.code .sa,.code .sd,.code .se,.code .si,.code .sb{color:var(--c-str)}
.code .c,.code .c1,.code .cm,.code .ch{color:var(--c-com);font-style:italic}
.code .mi,.code .mf,.code .mh,.code .mo{color:var(--c-num)}
.code .nf,.code .fm{color:var(--c-fn)}
.code .nb,.code .bp{color:var(--c-bi)}
.code .nn{color:var(--c-ns)}
.code .o{color:var(--c-op)}
.code .nd{color:var(--c-kw)}
"""


def make_thumb(src: Path, dst: Path) -> None:
    im = Image.open(src).convert("RGB")
    W, H = 1080, 675  # 16:10 canvas, image contained & centred on white
    im.thumbnail((W - 40, H - 40), Image.LANCZOS)
    canvas = Image.new("RGB", (W, H), "white")
    canvas.paste(im, ((W - im.width) // 2, (H - im.height) // 2))
    canvas.save(dst, "WEBP", quality=86, method=6)


def example_page(ex: Example, prev_ex: Example | None, next_ex: Example | None) -> str:
    c = ex.category
    base = "../../"
    gh = f"{CONFIG['repo_url']}/tree/main/{ex.repo_path}"
    meta = []
    if ex.r:
        meta.append(f'<div class="meta-row"><span>R equivalent</span><code>{esc(ex.r)}</code></div>')
    meta.append(f'<div class="meta-row"><span>Libraries</span><div class="pills">{"".join(f"<i>{esc(l)}</i>" for l in ex.libs)}</div></div>')
    if ex.tags:
        meta.append(f'<div class="meta-row"><span>Topics</span><div class="pills">{"".join(f"<i>{esc(t)}</i>" for t in ex.tags)}</div></div>')
    related = [e for e in c.examples if e is not ex][:3]
    rel = "\n".join(card(e, base) for e in related)
    nav = '<div class="pager">'
    nav += f'<a href="{base}{prev_ex.url}"><small>Previous</small>{esc(prev_ex.title)}</a>' if prev_ex else "<span></span>"
    nav += f'<a class="next" href="{base}{next_ex.url}"><small>Next</small>{esc(next_ex.title)}</a>' if next_ex else "<span></span>"
    nav += "</div>"
    lines = ex.code.count("\n")
    body = f"""<article class="wrap ex">
  <div class="crumbs"><a href="{base}index.html">Gallery</a><span>/</span><a href="{base}{c.slug}/index.html">{esc(c.title)}</a></div>
  <header class="ex-head"><h1>{esc(ex.title)}</h1><p class="lede">{esc(ex.summary)}</p>
  <p class="byline">By <a href="{CONFIG['author_url']}">{esc(CONFIG['author'])}</a> · part of <a href="{base}index.html">Statlas</a></p></header>
  <figure class="figure"><a href="plot.png" target="_blank" rel="noopener"><img src="plot.png" alt="{esc(ex.title)}"></a></figure>
  <div class="ex-cols">
    <div class="prose">{inline_md(ex.description)}</div>
    <aside class="meta">{''.join(meta)}</aside>
  </div>
  <section class="code-card" id="code">
    <div class="code-bar">
      <div class="dots"><i></i><i></i><i></i></div><span class="fname">{ex.slug}.py · {lines} lines</span>
      <div class="code-actions">
        <a class="btn ghost" href="{ex.slug}.py" download>Download</a>
        <a class="btn ghost" href="{gh}">GitHub</a>
        <button class="btn copy" data-copy>Copy code</button>
      </div>
    </div>
    <pre class="code"><code>{highlight_py(ex.code)}</code></pre>
  </section>
  <p class="hint">Figures use the optional <a href="{base}style.html">Statlas style</a>: add <code>plt.style.use("statlas.mplstyle")</code> to match the look exactly.</p>
  {cite_block(ex)}
  {nav}
  {f'<h2 class="rel-h">More in {esc(c.title)}</h2><div class="grid">{rel}</div>' if rel else ''}
</article>"""
    return page(f"{ex.title} · {CONFIG['name']}", body, 2, ex.summary, "gallery",
                cite_title=f"{ex.title} — Statlas: The Python Gallery of Statistical Graphics", url=ex.url)


def style_page() -> str:
    swatches = "".join(f'<div class="sw"><i style="background:#{c}"></i><code>#{c}</code></div>' for c in
                       ["007AFF", "FF9500", "34C759", "FF3B30", "AF52DE", "5AC8FA", "FF2D55", "5856D6", "A2845E", "8E8E93"])
    style_src = highlight(STYLE.read_text(), PythonLexer(), FORMATTER)
    return f"""<section class="hero hero-sm"><div class="wrap">
<h1>The Statlas style</h1><p class="lede">One small Matplotlib style sheet gives every chart the same calm look: system-inspired colours, no chart junk, generous whitespace. It works with Matplotlib, seaborn and pandas plotting.</p></div></section>
<section class="wrap ex">
<div class="steps">
  <div><b>1</b><h3>Download</h3><p><a href="statlas.mplstyle" download>statlas.mplstyle</a> into your project.</p></div>
  <div><b>2</b><h3>Activate</h3><p><code>plt.style.use("statlas.mplstyle")</code></p></div>
  <div><b>3</b><h3>Plot</h3><p>Every example in the gallery now looks identical on your machine.</p></div>
</div>
<h2 class="rel-h">Palette</h2><div class="swatches">{swatches}</div>
<section class="code-card"><div class="code-bar"><div class="dots"><i></i><i></i><i></i></div><span class="fname">statlas.mplstyle</span>
<div class="code-actions"><a class="btn ghost" href="statlas.mplstyle" download>Download</a><button class="btn copy" data-copy>Copy</button></div></div>
<pre class="code"><code>{style_src}</code></pre></section>
</section>"""


def about_page(cats: list[Category]) -> str:
    rows = "".join(f'<tr><td><a href="{c.slug}/index.html">{esc(c.title)}</a></td><td>{len(c.examples)}</td><td>{esc(c.summary)}</td></tr>' for c in cats)
    return f"""<section class="hero hero-sm"><div class="wrap">
<h1>About {esc(CONFIG['name'])}</h1><p class="lede">{esc(CONFIG['lede'])}</p></div></section>
<section class="wrap ex prose wide">
<p>{esc(CONFIG['name'])} is to Python what the R Graph Gallery, <code>ggstats</code> and <code>ggstatsplot</code> are to R: a catalogue of <strong>statistical</strong> charts — model coefficients, Likert scales, survival curves, diagnostics, effect sizes — each with clean, copy‑pasteable code built on the scientific Python stack.</p>
<h2>Principles</h2>
<ul>
<li><strong>Self‑contained.</strong> Every script runs as is. Data comes from bundled datasets (statsmodels, scikit‑learn, lifelines) or a seeded simulation — no downloads.</li>
<li><strong>Statistics first.</strong> Each chart carries the numbers that matter: estimates, intervals, test statistics, effect sizes.</li>
<li><strong>Folder = page.</strong> The repository tree mirrors the site. Every example lives in <code>gallery/&lt;family&gt;/&lt;chart&gt;/plot.py</code> with a rendered <code>plot.png</code> and README.</li>
</ul>
<h2>Families</h2>
<table class="tbl"><thead><tr><th>Family</th><th>Charts</th><th>What you'll find</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Author</h2>
<p>{esc(CONFIG['name'])} is created and maintained by <strong><a href="{CONFIG['author_url']}">{esc(CONFIG['author'])}</a></strong>. More of his work on data science, statistics and AI: <a href="{CONFIG['author_url']}">www.warin.ca</a>.</p>
{cite_block()}
<h2>Contribute</h2>
<p>Add a folder with a <code>plot.py</code>, run <code>python build.py</code>, open a pull request. See <a href="{CONFIG['repo_url']}/blob/main/CONTRIBUTING.md">CONTRIBUTING.md</a>.</p>
</section>"""


# --------------------------------------------------------------------------- READMEs
def write_readmes(cats: list[Category]) -> None:
    site = CONFIG["site_url"]
    for c in cats:
        for ex in c.examples:
            r = f"\n**R equivalent:** `{ex.r}`  " if ex.r else ""
            (ex.folder / "README.md").write_text(
f"""# {ex.title}

> {ex.summary}

![{ex.title}](plot.png)

{ex.description}
{r}
**Libraries:** {', '.join(ex.libs)}

```python
{ex.code.rstrip()}
```

---

**Statlas** by [Thierry Warin](https://www.warin.ca) · [View on the website →]({site}{ex.url}) · [← {c.title}](../README.md)

**Cite:** {cite_text(ex)}
""")
        table = "\n".join(
            f"| [<img src=\"{e.folder.name}/plot.png\" width=\"260\">]({e.folder.name}/) | **[{e.title}]({e.folder.name}/)**<br>{e.summary}" + (f"<br>`{e.r}`" if e.r else "") + " |"
            for e in c.examples)
        (c.folder / "README.md").write_text(
f"""# {c.title}

{c.summary}

| Chart | Description |
|:--|:--|
{table}

[← All families](../../README.md) · [Browse on the website]({site}{c.slug}/)
""")

    fam = "\n".join(
        f"| [<img src=\"gallery/{c.folder.name}/{c.examples[0].folder.name}/plot.png\" width=\"220\">](gallery/{c.folder.name}/) | **[{c.title}](gallery/{c.folder.name}/)** — {c.summary}<br><sub>"
        + " · ".join(f"[{e.title}](gallery/{c.folder.name}/{e.folder.name}/)" for e in c.examples) + "</sub> |"
        for c in cats if c.examples)
    n = sum(len(c.examples) for c in cats)
    tpl = (ROOT / "README.template.md").read_text()
    (ROOT / "README.md").write_text(tpl.replace("{{FAMILIES}}", fam).replace("{{N}}", str(n))
                                    .replace("{{SITE}}", site).replace("{{K}}", str(len(cats))))


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-render every figure")
    ap.add_argument("--only", default="", help="only render examples whose path contains this text")
    ap.add_argument("--no-site", action="store_true", help="render figures only")
    args = ap.parse_args()

    cats = discover()
    todo = [e for c in cats for e in c.examples if args.only in e.repo_path]
    failures = []
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        for ex, err in pool.map(lambda e: render(e, args.force), todo):
            status = "✗" if err else "✓"
            print(f"  {status} {ex.repo_path}")
            if err:
                failures.append((ex, err))
    for ex, err in failures:
        print(f"\n--- {ex.repo_path} ---\n{err}", file=sys.stderr)
    if failures:
        raise SystemExit(f"{len(failures)} example(s) failed")
    if args.no_site:
        return
    missing = [e.repo_path for c in cats for e in c.examples if not e.png.exists()]
    if missing:
        raise SystemExit(f"missing figures: {missing}")
    write_readmes(cats)
    build_site(cats)
    print(f"Built {sum(len(c.examples) for c in cats)} examples in {len(cats)} families → docs/")


if __name__ == "__main__":
    main()
