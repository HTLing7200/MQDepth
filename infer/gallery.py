from html import escape
from pathlib import Path
from urllib.parse import quote


def write_gallery(records: list[tuple[Path, dict]], path: str | Path) -> None:
    path = Path(path)
    cards = []
    for output, metadata in records:
        artifacts = metadata["artifacts"]
        if "preview" not in artifacts:
            continue
        parent = output.parent.relative_to(path.parent)
        links = []
        for name, filename in artifacts.items():
            link = quote((parent / filename).as_posix())
            links.append(f'<a href="{link}">{escape(name)}</a>')
        preview = quote((parent / artifacts["preview"]).as_posix())
        title = escape(metadata["image"])
        camera = metadata["camera"]["fx_original_px"]
        seconds = metadata["timing_seconds"]["total"]
        cards.append(
            f'<article><a href="{preview}"><img src="{preview}" loading="lazy" alt="{title}"></a>'
            f"<h2>{title}</h2><p>fx {camera:g} px · {seconds:.3f} s</p>"
            f'<nav>{" ".join(links)}</nav></article>'
        )
    style = (
        "body{font-family:system-ui,sans-serif;margin:24px;background:rgb(245,246,248);color:rgb(24,28,32)}"
        "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:20px}"
        "article{padding:14px;background:white;border-radius:10px;overflow:hidden}"
        "img{width:100%;height:auto}h1{font-size:24px}h2{font-size:15px;overflow-wrap:anywhere}"
        "p,nav{font-size:13px}nav{display:flex;gap:12px;flex-wrap:wrap}a{color:rgb(25,89,170)}"
    )
    document = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Depth predictions</title><style>{style}</style>"
        f"<h1>Depth predictions</h1><p>{len(cards)} images · RGB and metric depth</p>"
        f'<main>{"".join(cards)}</main></html>'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
