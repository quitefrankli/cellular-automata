"""A read-only gallery for locally generated run artifacts."""
from functools import partial
from html import escape
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
from urllib.parse import quote, unquote, urlsplit


STYLE = """
body{margin:0;background:#111827;color:#e5e7eb;font:16px system-ui,sans-serif}
main{max-width:1100px;margin:auto;padding:32px}h1{margin-bottom:8px}
a{color:#93c5fd}p{color:#9ca3af}nav{display:flex;gap:18px;margin:24px 0;flex-wrap:wrap}
section{margin:28px 0;padding:24px;background:#1f2937;border-radius:12px}
.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:20px}
figure{margin:0;min-width:0}figure img{display:block;max-width:100%;max-height:420px;
object-fit:contain;background:white;border-radius:6px}img.cells{image-rendering:pixelated;width:100%}
figcaption{margin-top:8px;overflow-wrap:anywhere}button{background:#374151;color:white;
border:1px solid #6b7280;border-radius:6px;padding:8px 14px;cursor:pointer}
"""


def gallery_page(root: Path) -> bytes:
    sections = []
    groups: dict[Path, list[Path]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in (".gif", ".png") and path.resolve().is_relative_to(root):
            relative = path.relative_to(root)
            # Keep intermediate snapshots alongside their run's main outputs.
            run = relative.parent.parent if relative.parent.name == "snapshots" else relative.parent
            groups.setdefault(run, []).append(relative)
    for run, files in groups.items():
        name = escape(str(run) if str(run) != "." else root.name)
        figures = []
        for relative in files:
            url = "/artifacts/" + quote(relative.as_posix(), safe="/")
            label = escape(str(relative.relative_to(run)))
            css = "" if relative.name == "loss.png" else "cells"
            figures.append(f'<figure><a href="{url}" target="_blank" rel="noopener">'
                           f'<img class="{css}" src="{url}" alt="{label}" loading="lazy"></a>'
                           f'<figcaption>{label}</figcaption></figure>')
        links = []
        for filename in ("config.json", "metrics.csv"):
            file = root / run / filename
            if file.is_file() and file.resolve().is_relative_to(root):
                url = "/artifacts/" + quote((run / filename).as_posix(), safe="/")
                links.append(f'<a href="{url}">{filename}</a>')
        sections.append(f'<section><h2>{name}</h2><nav>{"".join(links)}</nav>'
                        f'<div class="gallery">{"".join(figures)}</div></section>')
    content = "".join(sections) or '<section>No images yet. Train a model or export a GIF into this folder, then refresh.</section>'
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>NCA local viewer</title><style>{STYLE}</style></head><body><main>'
            f'<h1>Growing Neural Cellular Automata</h1><p>Local run gallery · GIFs play automatically. '
            f'Click an image to view it at full size.</p><button onclick="location.reload()">Refresh outputs</button>'
            f'{content}</main></body></html>').encode()


class GalleryHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, root: Path, **kwargs) -> None:
        self.root = root.resolve()
        super().__init__(*args, directory=str(self.root), **kwargs)

    def do_GET(self) -> None:
        self._respond(head=False)

    def do_HEAD(self) -> None:
        self._respond(head=True)

    def _respond(self, *, head: bool) -> None:
        path = unquote(urlsplit(self.path).path)
        if path == "/":
            body = gallery_page(self.root)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if not head:
                self.wfile.write(body)
            return
        if not path.startswith("/artifacts/"):
            self.send_error(404)
            return
        file = (self.root / path.removeprefix("/artifacts/")).resolve()
        if not file.is_relative_to(self.root) or not file.is_file() or file.suffix.lower() not in (".png", ".gif", ".json", ".csv"):
            self.send_error(404)
            return
        # Delegate streaming and MIME handling, with a validated relative path.
        self.path = "/" + quote(file.relative_to(self.root).as_posix(), safe="/")
        if head:
            super().do_HEAD()
        else:
            super().do_GET()

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:
        # Keep the terminal quiet while browsing animations.
        pass


def local_network_ip() -> str | None:
    """Find the IPv4 address used by the default route without sending data."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 80))
            address = probe.getsockname()[0]
            if address != "0.0.0.0" and not address.startswith("127."):
                return address
    except OSError:
        pass
    try:
        for address in socket.gethostbyname_ex(socket.gethostname())[2]:
            if address != "0.0.0.0" and not address.startswith("127."):
                return address
    except OSError:
        pass
    return None


def make_server(root: Path, port: int = 8000) -> ThreadingHTTPServer:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Run directory does not exist: {root}. Train a model first or choose --runs-dir.")
    return ThreadingHTTPServer(("0.0.0.0", port), partial(GalleryHandler, root=root))
