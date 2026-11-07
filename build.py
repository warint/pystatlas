#!/usr/bin/env python3
"""
PyStatlas build script.

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
STYLE = ROOT / "pystatlas.mplstyle"
CONFIG = yaml.safe_load((ROOT / "statlas.yaml").read_text())
import hashlib
ASSET_V = hashlib.md5(b"".join(f.read_bytes() for f in sorted((ROOT / "site_src").iterdir()))).hexdigest()[:8]

LIB_NAMES = {
    "numpy": "NumPy", "pandas": "pandas", "matplotlib": "Matplotlib",
    "seaborn": "seaborn", "scipy": "SciPy", "statsmodels": "statsmodels",
    "sklearn": "scikit-learn", "lifelines": "lifelines", "shap": "SHAP",
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
    sub: str = ""          # optional subcategory key (see _category.yaml → subcategories)

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
    subs: dict = field(default_factory=dict)   # key → {title, summary}


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
        code=code, libs=libs, sub=meta.get("subcategory", ""),
    )


def discover() -> list[Category]:
    cats = []
    for cdir in sorted(p for p in GALLERY.iterdir() if p.is_dir()):
        meta = yaml.safe_load((cdir / "_category.yaml").read_text())
        order, slug = split_prefix(cdir.name)
        cat = Category(cdir, slug, order, meta["title"], meta.get("summary", ""),
                       subs=meta.get("subcategories") or {})
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


# --------------------------------------------------------------------------- i18n
I18N = ROOT / "i18n"
LANGS = ["en", "fr", "ja", "zh"]
HTML_LANG = {"en": "en", "fr": "fr", "ja": "ja", "zh": "zh-Hans"}
CJK_FONTS = {
    "ja": "family=Noto+Sans+JP:wght@400;500;600;700&",
    "zh": "family=Noto+Sans+SC:wght@400;500;600;700&",
}
EN = yaml.safe_load((I18N / "en.yaml").read_text())
LANG = "en"
TR: dict = {}


def load_lang(lang: str) -> None:
    """Activate a language: UI strings + content translations (fallback: English)."""
    global LANG, TR
    LANG = lang
    data = {}
    if lang != "en" and (I18N / f"{lang}.yaml").exists():
        data = yaml.safe_load((I18N / f"{lang}.yaml").read_text()) or {}
    TR = {"ui": {**EN["ui"], **(data.get("ui") or {})},
          "categories": data.get("categories") or {},
          "examples": data.get("examples") or {}}


def t(key: str, **kw) -> str:
    s = TR["ui"][key]
    for k, v in kw.items():
        s = s.replace("{" + k + "}", str(v))
    return s


def ctitle(c: "Category") -> str:
    return (TR["categories"].get(c.slug) or {}).get("title") or c.title


def csummary(c: "Category") -> str:
    return (TR["categories"].get(c.slug) or {}).get("summary") or c.summary


def subtitle(c: "Category", key: str) -> str:
    tr = ((TR["categories"].get(c.slug) or {}).get("subcategories") or {}).get(key) or {}
    return tr.get("title") or c.subs[key]["title"]


def subsummary(c: "Category", key: str) -> str:
    tr = ((TR["categories"].get(c.slug) or {}).get("subcategories") or {}).get(key) or {}
    return tr.get("summary") or c.subs[key].get("summary", "")


def grouped_grid(c: "Category", home: str, root: str) -> str:
    """Main examples first, then one titled block per subcategory."""
    out = [f'<div class="grid">{"".join(card(e, home, root) for e in c.examples if not e.sub)}</div>']
    for key in c.subs:
        exs = [e for e in c.examples if e.sub == key]
        if exs:
            out.append(f'<div class="sub" id="{c.slug}-{key}"><div class="sub-head"><h3>{esc(subtitle(c, key))}</h3>'
                       f'<p>{esc(subsummary(c, key))}</p></div>'
                       f'<div class="grid">{"".join(card(e, home, root) for e in exs)}</div></div>')
    return "".join(out)


def _ex(ex: "Example") -> dict:
    return TR["examples"].get(f"{ex.category.slug}/{ex.slug}") or {}


def etitle(ex: "Example") -> str:
    return _ex(ex).get("title") or ex.title


def esummary(ex: "Example") -> str:
    return _ex(ex).get("summary") or ex.summary


def edesc(ex: "Example") -> str:
    return (_ex(ex).get("description") or ex.description).strip()


def prefix(lang: str) -> str:
    return "" if lang == "en" else f"{lang}/"


def export_i18n(cats: list["Category"]) -> None:
    """Write i18n/_source.yaml: every English string, ready to translate."""
    src = {"ui": EN["ui"],
           "categories": {c.slug: {"title": c.title, "summary": c.summary,
                                   **({"subcategories": c.subs} if c.subs else {})} for c in cats},
           "examples": {f"{c.slug}/{e.slug}": {"title": e.title, "summary": e.summary, "description": e.description}
                        for c in cats for e in c.examples}}
    (I18N / "_source.yaml").write_text(
        "# Generated by `python build.py --export-i18n`. Translate into i18n/<lang>.yaml (same keys).\n"
        + yaml.safe_dump(src, allow_unicode=True, sort_keys=False, width=100))


# --------------------------------------------------------------------------- page shell
def page(title: str, body: str, depth: int, path: str, description: str = "", active: str = "",
         cite_title: str = "") -> str:
    """depth = folder depth of the page inside its language; path = page path inside a language."""
    home = "../" * depth
    root = "../" * (depth + (LANG != "en"))
    cite_title = cite_title or "PyStatlas: The Python Gallery of Statistical Graphics"
    url = CONFIG["site_url"] + prefix(LANG) + path
    alternates = "\n".join(
        f'<link rel="alternate" hreflang="{HTML_LANG[l]}" href="{CONFIG["site_url"]}{prefix(l)}{path}">' for l in LANGS)
    alternates += f'\n<link rel="alternate" hreflang="x-default" href="{CONFIG["site_url"]}{path}">'
    scholar = f"""<meta name="author" content="{esc(CONFIG['author'])}">
<link rel="author" href="{CONFIG['author_url']}">
<meta name="citation_title" content="{esc(cite_title)}">
<meta name="citation_author" content="Warin, Thierry">
<meta name="citation_publication_date" content="{CONFIG['year']}">
<meta name="citation_online_date" content="{CONFIG['year']}/11/06">
<meta name="citation_publisher" content="{esc(CONFIG['author'])}">
<meta name="citation_language" content="{LANG}">
<meta name="citation_public_url" content="{url}">
<meta name="DC.creator" content="Warin, Thierry">
<meta name="DC.title" content="{esc(cite_title)}">
<meta name="DC.date" content="{CONFIG['year']}">
<link rel="canonical" href="{url}">
{alternates}
<script type="application/ld+json">{json.dumps({"@context": "https://schema.org", "@type": "CreativeWork", "name": cite_title,
  "url": url, "inLanguage": HTML_LANG[LANG], "datePublished": str(CONFIG['year']), "license": "https://opensource.org/licenses/MIT",
  "author": {"@type": "Person", "name": CONFIG['author'], "url": CONFIG['author_url']},
  "isPartOf": {"@type": "WebSite", "name": "PyStatlas", "url": CONFIG['site_url']}}, ensure_ascii=False)}</script>"""
    desc = esc(description or t("tagline"))
    langs = "".join(
        f'<a href="{root}{prefix(l)}{path}" hreflang="{HTML_LANG[l]}" lang="{HTML_LANG[l]}"'
        f'{" class=on aria-current=true" if l == LANG else ""}>{esc(EN_NAMES[l])}</a>' for l in LANGS)
    fonts = ("https://fonts.googleapis.com/css2?" + CJK_FONTS.get(LANG, "")
             + "family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap")
    return f"""<!doctype html>
<html lang="{HTML_LANG[LANG]}" data-copied="{esc(t('copied'))}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="{desc}">
{scholar}
<meta property="og:title" content="{esc(title)}">
<meta property="og:description" content="{desc}">
<meta property="og:locale" content="{HTML_LANG[LANG]}">
<meta name="theme-color" content="#ffffff">
<link rel="icon" href="{root}assets/favicon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="{fonts}" rel="stylesheet">
<link rel="stylesheet" href="{root}assets/style.css?v={ASSET_V}">
</head>
<body>
<header class="nav">
  <div class="nav-inner">
    <a class="brand" href="{home}index.html">{LOGO}<span>{esc(CONFIG['name'])}</span></a>
    <nav class="nav-links">
      <a href="{home}index.html#gallery" class="{'on' if active == 'gallery' else ''}">{esc(t('nav_gallery'))}</a>
      <a href="{home}style.html" class="{'on' if active == 'style' else ''}">{esc(t('nav_style'))}</a>
      <a href="{home}about.html" class="{'on' if active == 'about' else ''}">{esc(t('nav_about'))}</a>
      <a href="{home}about.html#cite" class="hide-sm">{esc(t('nav_cite'))}</a>
      <details class="lang"><summary aria-label="Language">{GLOBE_ICON}<span>{esc(EN_NAMES[LANG])}</span></summary><div class="lang-menu">{langs}</div></details>
      <a href="{CONFIG['repo_url']}" class="gh" aria-label="GitHub">{GITHUB_ICON}</a>
    </nav>
  </div>
</header>
<main>
{body}
</main>
<footer class="foot">
  <div class="wrap foot-inner">
    <div><strong>{esc(CONFIG['name'])}</strong> · {esc(t('tagline'))}<br>{esc(t('created_by'))} <a href="{CONFIG['author_url']}">{esc(CONFIG['author'])}</a> · <a href="{CONFIG['author_url']}">www.warin.ca</a> · <a href="{home}about.html#cite">{esc(t('cite_h'))}</a></div>
    <div class="foot-links"><a href="{CONFIG['repo_url']}">GitHub</a><a href="{home}style.html">{esc(t('plot_style'))}</a><a href="{CONFIG['repo_url']}/blob/main/CONTRIBUTING.md">{esc(t('contribute'))}</a><span>{esc(t('license'))}</span></div>
  </div>
  <div class="wrap foot-langs">{langs}</div>
</footer>
<script src="{root}assets/app.js?v={ASSET_V}"></script>
</body>
</html>
"""


EN_NAMES = {"en": "English", "fr": "Français", "ja": "日本語", "zh": "中文"}
LOGO = """<svg class="logo" viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="7.5" fill="#1d1d1f"/><rect x="7.5" y="17" width="4" height="8.5" rx="2" fill="#fff"/><rect x="14" y="11" width="4" height="14.5" rx="2" fill="#fff"/><rect x="20.5" y="6.5" width="4" height="19" rx="2" fill="#0071e3"/></svg>"""
GLOBE_ICON = """<svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.5 2.7 3.8 5.7 3.8 9s-1.3 6.3-3.8 9c-2.5-2.7-3.8-5.7-3.8-9S9.5 5.7 12 3z"/></svg>"""
GITHUB_ICON = """<svg viewBox="0 0 16 16" width="18" height="18" aria-hidden="true"><path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>"""


def card(ex: "Example", home: str, root: str) -> str:
    keywords = " ".join([etitle(ex), esummary(ex), ex.title, ex.summary, ex.r, ctitle(ex.category),
                         ex.category.title, *ex.tags, *ex.libs]).lower()
    r = f'<span class="card-r">{esc(ex.r)}</span>' if ex.r else ""
    kicker = ctitle(ex.category) + (f" · {subtitle(ex.category, ex.sub)}" if ex.sub else "")
    return f"""<a class="card" href="{home}{ex.url}" data-cat="{ex.category.slug}" data-k="{esc(keywords)}">
  <div class="card-img"><img src="{root}{ex.url}thumb.webp" alt="{esc(etitle(ex))}" loading="lazy" width="720" height="450"></div>
  <div class="card-body"><span class="kicker">{esc(kicker)}</span><h3>{esc(etitle(ex))}</h3><p>{esc(esummary(ex))}</p>{r}</div>
</a>"""


# --------------------------------------------------------------------------- citation
def cite_text(ex: "Example | None" = None) -> str:
    y, site = CONFIG["year"], CONFIG["site_url"]
    if ex:
        return f"Warin, T. ({y}). {ex.title}. In PyStatlas: The Python Gallery of Statistical Graphics. {site}{ex.url}"
    return f"Warin, T. ({y}). PyStatlas: The Python Gallery of Statistical Graphics. {site}"


def bibtex() -> str:
    return f"""@misc{{warin{CONFIG['year']}statlas,
  author       = {{Warin, Thierry}},
  title        = {{PyStatlas: The Python Gallery of Statistical Graphics}},
  year         = {{{CONFIG['year']}}},
  howpublished = {{\\url{{{CONFIG['site_url']}}}}},
  note         = {{Source code: {CONFIG['repo_url']}}}
}}"""


def cite_block(ex: "Example | None" = None) -> str:
    author = f'<a href="{CONFIG["author_url"]}">{esc(CONFIG["author"])}</a>'
    return f"""<section class="cite" id="cite">
  <div class="cite-head"><h2>{esc(t('cite_h'))}</h2>
  <p>{t('cite_p', name=esc(CONFIG['name']), author=author)}</p></div>
  <div class="cite-box"><p class="cite-ref">{esc(cite_text(ex))}</p><button class="btn ghost" data-copy-text="{esc(cite_text(ex))}">{esc(t('copy'))}</button></div>
  <div class="cite-box"><pre class="cite-bib">{esc(bibtex())}</pre><button class="btn ghost" data-copy-text="{esc(bibtex())}">{esc(t('copy_bibtex'))}</button></div>
</section>"""


# --------------------------------------------------------------------------- site
def build_site(cats: list["Category"]) -> None:
    if DOCS.exists():
        shutil.rmtree(DOCS)
    (DOCS / "assets").mkdir(parents=True)
    for f in SITE_SRC.iterdir():
        shutil.copy(f, DOCS / "assets" / f.name)
    css = (DOCS / "assets" / "style.css").read_text()
    css += "\n/* --- pygments (generated) --- */\n" + pygments_css()
    (DOCS / "assets" / "style.css").write_text(css)
    shutil.copy(STYLE, DOCS / "pystatlas.mplstyle")
    shutil.copy(STYLE, DOCS / "statlas.mplstyle")   # old name, kept so earlier download links still work
    (DOCS / ".nojekyll").write_text("")

    # shared, language-independent files: figures, thumbnails, scripts
    for c in cats:
        for ex in c.examples:
            out = DOCS / ex.url
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy(ex.png, out / "plot.png")
            shutil.copy(ex.folder / "plot.py", out / f"{ex.slug}.py")
            make_thumb(ex.png, out / "thumb.webp")

    for lang in LANGS:
        load_lang(lang)
        build_language(cats)
    load_lang("en")

    all_ex = [e for c in cats for e in c.examples]
    paths = ["", "about.html", "style.html"] + [f"{c.slug}/" for c in cats] + [e.url for e in all_ex]
    (DOCS / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "".join(f"  <url><loc>{CONFIG['site_url']}{prefix(l)}{p}</loc></url>\n" for l in LANGS for p in paths)
        + "</urlset>\n")
    (DOCS / "robots.txt").write_text(f"User-agent: *\nAllow: /\nSitemap: {CONFIG['site_url']}sitemap.xml\n")
    index = [{"title": e.title, "summary": e.summary, "r": e.r, "category": e.category.title,
              "tags": e.tags, "url": e.url} for e in all_ex]
    (DOCS / "index.json").write_text(json.dumps(index, indent=1))


def build_language(cats: list["Category"]) -> None:
    out_root = DOCS / prefix(LANG)
    rootp = "../" if LANG != "en" else ""
    all_ex = [e for c in cats for e in c.examples]

    for c in cats:
        for i, ex in enumerate(c.examples):
            out = out_root / ex.url
            out.mkdir(parents=True, exist_ok=True)
            prev_ex = c.examples[i - 1] if i > 0 else None
            next_ex = c.examples[i + 1] if i < len(c.examples) - 1 else None
            (out / "index.html").write_text(example_page(ex, prev_ex, next_ex))

    for c in cats:
        out = out_root / c.slug
        out.mkdir(parents=True, exist_ok=True)
        body = f"""<section class="hero hero-sm"><div class="wrap">
<div class="crumbs"><a href="../index.html">{esc(t('gallery'))}</a><span>/</span>{esc(ctitle(c))}</div>
<h1>{esc(ctitle(c))}</h1><p class="lede">{esc(csummary(c))}</p></div></section>
<section class="wrap section">{grouped_grid(c, "../", "../" + rootp)}</section>"""
        (out / "index.html").write_text(page(f"{ctitle(c)} · {CONFIG['name']}", body, 1, f"{c.slug}/", csummary(c), "gallery"))

    chips = f'<button class="chip on" data-filter="all">{esc(t("all"))}</button>' + "".join(
        f'<button class="chip" data-filter="{c.slug}">{esc(ctitle(c))}</button>' for c in cats)
    sections = []
    for c in cats:
        sections.append(f"""<section class="cat" data-cat="{c.slug}" id="{c.slug}">
  <div class="cat-head"><div><h2>{esc(ctitle(c))}</h2><p>{esc(csummary(c))}</p></div>
  <a class="more" href="{c.slug}/index.html">{esc(t('n_charts', n=len(c.examples)))} ›</a></div>
  {grouped_grid(c, "", rootp)}
</section>""")
    tiles = "".join(
        f'<a class="tile" href="#{c.slug}" data-jump="{c.slug}"><img src="{rootp}{c.examples[0].url}thumb.webp" alt="" loading="lazy">'
        f'<span>{esc(ctitle(c))}</span><em>{len(c.examples)}</em></a>' for c in cats if c.examples)
    body = f"""<section class="hero"><div class="wrap">
  <p class="eyebrow">{esc(t('eyebrow', n=len(all_ex), k=len(cats)))}</p>
  <h1>{esc(t('hero_1'))}<br><span class="h-accent">{esc(t('hero_2'))}</span></h1>
  <p class="lede">{esc(t('lede'))}</p>
  <p class="approach">{esc(t('approach'))}</p>
  <p class="byline">{esc(t('by'))} <a href="{CONFIG['author_url']}">{esc(CONFIG['author'])}</a></p>
  <div class="search"><svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>
    <input id="q" type="search" placeholder="{esc(t('search_placeholder'))}" autocomplete="off" spellcheck="false">
    <kbd>/</kbd></div>
</div></section>
<section class="band"><div class="wrap tiles">{tiles}</div></section>
<section class="wrap" id="gallery">
  <div class="chips" role="tablist">{chips}</div>
  <p class="empty" hidden>{esc(t('empty'))}</p>
  {''.join(sections)}
</section>
<div class="wrap">{cite_block()}</div>"""
    (out_root / "index.html").write_text(page(f"{CONFIG['name']} — {t('tagline')}", body, 0, "", active="gallery"))
    (out_root / "style.html").write_text(page(f"{t('style_title')} · {CONFIG['name']}", style_page(rootp), 0, "style.html", active="style"))
    (out_root / "about.html").write_text(page(f"{t('about_title')} · {CONFIG['name']}", about_page(cats), 0, "about.html", active="about"))


def pygments_css() -> str:
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


def example_page(ex: "Example", prev_ex: "Example | None", next_ex: "Example | None") -> str:
    c = ex.category
    home = "../../"
    root = home + ("../" if LANG != "en" else "")
    asset = root + ex.url  # shared figure / script folder
    gh = f"{CONFIG['repo_url']}/tree/main/{ex.repo_path}"
    meta = []
    if ex.r:
        meta.append(f'<div class="meta-row"><span>{esc(t("r_equivalent"))}</span><code>{esc(ex.r)}</code></div>')
    meta.append(f'<div class="meta-row"><span>{esc(t("libraries"))}</span><div class="pills">{"".join(f"<i>{esc(l)}</i>" for l in ex.libs)}</div></div>')
    if ex.tags:
        meta.append(f'<div class="meta-row"><span>{esc(t("topics"))}</span><div class="pills">{"".join(f"<i>{esc(x)}</i>" for x in ex.tags)}</div></div>')
    related = [e for e in c.examples if e is not ex][:3]
    rel = "\n".join(card(e, home, root) for e in related)
    nav = '<div class="pager">'
    nav += f'<a href="{home}{prev_ex.url}"><small>{esc(t("previous"))}</small>{esc(etitle(prev_ex))}</a>' if prev_ex else "<span></span>"
    nav += f'<a class="next" href="{home}{next_ex.url}"><small>{esc(t("next"))}</small>{esc(etitle(next_ex))}</a>' if next_ex else "<span></span>"
    nav += "</div>"
    lines = ex.code.count("\n")
    note = f'<p class="note">{esc(t("code_note"))}</p>' if LANG != "en" else ""
    hint = t("hint", style_link=f'<a href="{home}style.html">{esc(t("style_link"))}</a>',
             code='<code>plt.style.use("pystatlas.mplstyle")</code>')
    body = f"""<article class="wrap ex">
  <div class="crumbs"><a href="{home}index.html">{esc(t('gallery'))}</a><span>/</span><a href="{home}{c.slug}/index.html">{esc(ctitle(c))}</a></div>
  <header class="ex-head"><h1>{esc(etitle(ex))}</h1><p class="lede">{esc(esummary(ex))}</p>
  <p class="byline">{esc(t('by'))} <a href="{CONFIG['author_url']}">{esc(CONFIG['author'])}</a> · {esc(t('part_of'))} <a href="{home}index.html">PyStatlas</a></p></header>
  <figure class="figure"><a href="{asset}plot.png" target="_blank" rel="noopener"><img src="{asset}plot.png" alt="{esc(etitle(ex))}"></a></figure>
  <div class="ex-cols">
    <div class="prose">{inline_md(edesc(ex))}{note}</div>
    <aside class="meta">{''.join(meta)}</aside>
  </div>
  <section class="code-card" id="code">
    <div class="code-bar">
      <span class="fname">{ex.slug}.py · {esc(t('n_lines', n=lines))}</span>
      <div class="code-actions">
        <a class="btn ghost" href="{asset}{ex.slug}.py" download>{esc(t('download'))}</a>
        <a class="btn ghost" href="{gh}">GitHub</a>
        <button class="btn copy" data-copy>{esc(t('copy_code'))}</button>
      </div>
    </div>
    <pre class="code"><code>{highlight_py(ex.code)}</code></pre>
  </section>
  <p class="hint">{hint}</p>
  {cite_block(ex)}
  {nav}
  {f'<h2 class="rel-h">{esc(t("more_in", cat=ctitle(c)))}</h2><div class="grid">{rel}</div>' if rel else ''}
</article>"""
    return page(f"{etitle(ex)} · {CONFIG['name']}", body, 2, ex.url, esummary(ex), "gallery",
                cite_title=f"{ex.title} — PyStatlas: The Python Gallery of Statistical Graphics")


def style_page(rootp: str) -> str:
    swatches = "".join(f'<div class="sw"><i style="background:#{c}"></i><code>#{c}</code></div>' for c in
                       ["007AFF", "FF9500", "34C759", "FF3B30", "AF52DE", "5AC8FA", "FF2D55", "5856D6", "A2845E", "8E8E93"])
    style_src = highlight(STYLE.read_text(), PythonLexer(), FORMATTER)
    f = f'<a href="{rootp}pystatlas.mplstyle" download>pystatlas.mplstyle</a>'
    return f"""<section class="hero hero-sm"><div class="wrap">
<h1>{esc(t('style_h1'))}</h1><p class="lede">{esc(t('style_lede'))}</p></div></section>
<section class="wrap ex">
<div class="steps">
  <div><b>1</b><h3>{esc(t('step1_h'))}</h3><p>{t('step1_p', file=f)}</p></div>
  <div><b>2</b><h3>{esc(t('step2_h'))}</h3><p><code>plt.style.use("pystatlas.mplstyle")</code></p></div>
  <div><b>3</b><h3>{esc(t('step3_h'))}</h3><p>{esc(t('step3_p'))}</p></div>
</div>
<h2 class="rel-h">{esc(t('palette'))}</h2><div class="swatches">{swatches}</div>
<section class="code-card"><div class="code-bar"><span class="fname">pystatlas.mplstyle</span>
<div class="code-actions"><a class="btn ghost" href="{rootp}pystatlas.mplstyle" download>{esc(t('download'))}</a><button class="btn copy" data-copy>{esc(t('copy'))}</button></div></div>
<pre class="code"><code>{style_src}</code></pre></section>
</section>"""


def thanks_block(cats: list["Category"]) -> str:
    th = CONFIG["thanks"]
    exs = [e for c in cats for e in c.examples]
    link = lambda name, url: f'<a href="{url}">{esc(name)}</a>'
    used = {l for e in exs for l in e.libs}
    py = [l for l in th["python_urls"] if l in used] + sorted(used - set(th["python_urls"]))
    py_html = " · ".join(link(l, th["python_urls"][l]) if l in th["python_urls"] else esc(l) for l in py)
    rpk = sorted({m for e in exs if e.r for m in re.findall(r"([A-Za-z][A-Za-z0-9.]*)::", e.r)}, key=str.lower)
    bioc = {"ComplexHeatmap"}  # Bioconductor, not CRAN
    r_url = lambda m: (f"https://bioconductor.org/packages/{m}/" if m in bioc
                       else f"https://cran.r-project.org/package={m}")
    r_html = " · ".join(link(m, r_url(m)) for m in rpk)
    funcs = {}
    for e in exs:
        for m, f in re.findall(r"([A-Za-z][A-Za-z0-9.]*)::([A-Za-z_][A-Za-z0-9_.]*)", e.r or ""):
            funcs.setdefault(m, [])
            if f not in funcs[m]:
                funcs[m].append(f)
    def item(d):
        fs = [f"<code>{esc(f)}()</code>" for m in d.get("pkgs", []) for f in funcs.get(m, [])]
        return f"<li>{link(d['name'], d['url'])}{' — ' + ', '.join(fs) if fs else ''}</li>"
    items = lambda key: "".join(item(d) for d in th[key])
    return f"""<section class="thanks" id="thanks">
<h2>{esc(t('thanks_h'))}</h2>
<p>{esc(t('thanks_p'))}</p>
<h3>{esc(t('thanks_py'))}</h3><p class="pkgs">{py_html}</p>
<h3>{esc(t('thanks_data'))}</h3><ul>{items('datasets')}</ul>
<h3>{esc(t('thanks_inspiration'))}</h3><ul>{items('inspiration')}</ul>
<h3>{esc(t('thanks_r'))}</h3><p class="pkgs">{r_html}</p>
<h3>{esc(t('thanks_site'))}</h3><p class="pkgs">{" · ".join(link(d['name'], d['url']) for d in th['site'])}</p>
</section>"""


def about_page(cats: list["Category"]) -> str:
    name = esc(CONFIG["name"])
    author = f'<strong><a href="{CONFIG["author_url"]}">{esc(CONFIG["author"])}</a></strong>'
    rows = "".join(f'<tr><td><a href="{c.slug}/index.html">{esc(ctitle(c))}</a></td><td>{len(c.examples)}</td><td>{esc(csummary(c))}</td></tr>' for c in cats)
    return f"""<section class="hero hero-sm"><div class="wrap">
<h1>{esc(t('about_h1', name=CONFIG['name']))}</h1><p class="lede">{esc(t('lede'))}</p></div></section>
<section class="wrap ex prose wide">
<p>{t('about_p1', name=name)}</p>
<h2>{esc(t('principles_h'))}</h2>
<ul><li>{t('principle_1')}</li><li>{t('principle_2')}</li><li>{t('principle_3')}</li></ul>
<h2>{esc(t('families_h'))}</h2>
<table class="tbl"><thead><tr><th>{esc(t('th_family'))}</th><th>{esc(t('th_charts'))}</th><th>{esc(t('th_find'))}</th></tr></thead><tbody>{rows}</tbody></table>
<h2>{esc(t('author_h'))}</h2>
<p>{t('author_p', name=name, author=author, site=f'<a href="{CONFIG["author_url"]}">www.warin.ca</a>')}</p>
{cite_block()}
<h2>{esc(t('contribute_h'))}</h2>
<p>{t('contribute_p', contrib=f'<a href="{CONFIG["repo_url"]}/blob/main/CONTRIBUTING.md">CONTRIBUTING.md</a>')}</p>
{thanks_block(cats)}
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

**PyStatlas** by [Thierry Warin](https://www.warin.ca) · [View on the website →]({site}{ex.url}) · [← {c.title}](../README.md)

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
    ap.add_argument("--export-i18n", action="store_true", help="write i18n/_source.yaml for translators")
    args = ap.parse_args()
    if args.export_i18n:
        export_i18n(discover())
        print("wrote i18n/_source.yaml")
        return

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
