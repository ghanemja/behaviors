"""Assemble the static GitHub Pages site from the demo build in out/site.

The demo page is written as a fragment (styles + body); this wraps it in a full HTML document with link-preview tags,
a favicon and a link back to the code, and copies the data, shards, textured models and media next to it.
Usage: python src/build_pages.py out/site pages_build https://user.github.io/repo/ https://github.com/user/repo
"""
import os
import shutil
import sys

SITE, OUT, BASE, REPO = sys.argv[1:5]
BASE = BASE.rstrip("/") + "/"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TITLE = "Behavior Search for NVIDIA SimReady"
DESC = ("Search 2,167 NVIDIA SimReady 3D assets by how they move in physics simulation, not how they look. "
        "Tested on held-out probes in MuJoCo and PhysX.")
FAVICON = ("data:image/svg+xml," "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
           "%3Ccircle cx='16' cy='16' r='15' fill='%23FFD86B' stroke='%231E1B2E' stroke-width='2'/%3E"
           "%3Ccircle cx='10' cy='19' r='5' fill='%232a78d6' stroke='%231E1B2E' stroke-width='1.5'/%3E"
           "%3Ccircle cx='20' cy='11' r='5' fill='%23eb6834' stroke='%231E1B2E' stroke-width='1.5'/%3E"
           "%3Ccircle cx='22' cy='22' r='4' fill='%231baf7a' stroke='%231E1B2E' stroke-width='1.5'/%3E%3C/svg%3E")

src = open(f"{SITE}/index.html").read()
cut = src.index("</style>") + len("</style>")
head, body = src[:cut], src[cut:]
head = "\n".join(l for l in head.splitlines() if not l.startswith(("<meta charset", "<title>", '<meta name="claude:')))
body = body.replace(
    "<p>Physics: MuJoCo 3.17 and PhysX 5 (SAPIEN 3), both on CPU.",
    f'<p>Code, data and every number: <a href="{REPO}">{REPO.replace("https://", "")}</a>.</p>\n      <p>Physics: MuJoCo 3.17 and PhysX 5 (SAPIEN 3), both on CPU.', 1)
page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{TITLE}</title>
<meta name="description" content="{DESC}">
<meta property="og:type" content="website">
<meta property="og:title" content="{TITLE}">
<meta property="og:description" content="{DESC}">
<meta property="og:url" content="{BASE}">
<meta property="og:image" content="{BASE}cover.jpg">
<meta property="og:image:width" content="1080">
<meta property="og:image:height" content="1350">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="{BASE}cover.jpg">
<link rel="icon" href="{FAVICON}">
{head}
</head>
<body>
{body}
</body>
</html>
"""
if os.path.exists(OUT):
    shutil.rmtree(OUT)
os.makedirs(OUT)
open(f"{OUT}/index.html", "w").write(page)
shutil.copy(f"{SITE}/data.json", OUT)
shutil.copytree(f"{SITE}/d", f"{OUT}/d")
os.makedirs(f"{OUT}/mt")
for f in os.listdir(f"{SITE}/mt"):
    shutil.copy(f"{SITE}/mt/{f}", f"{OUT}/mt/")
shutil.copy(f"{ROOT}/media/cover.jpg", f"{OUT}/cover.jpg")
shutil.copy(f"{ROOT}/media/behavior_search.mp4", f"{OUT}/behavior_search.mp4")
open(f"{OUT}/.nojekyll", "w").close()
size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(OUT) for f in fs)
print(f"pages site: {sum(len(fs) for _, _, fs in os.walk(OUT))} files, {size / 1e6:.0f} MB")
