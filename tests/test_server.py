from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from click.testing import CliRunner
from PIL import Image
from nca.cli import main
from nca.server import make_server


def test_local_gallery_and_artifact_boundaries(tmp_path: Path):
    root = tmp_path / "runs"
    run = root / "circle & green"
    run.mkdir(parents=True)
    Image.new("RGB", (8, 8), "green").save(run / "growth.gif")
    (run / "config.json").write_text('{"mode":"grow"}')
    (run / "checkpoint.pt").write_bytes(b"private checkpoint")
    outside = tmp_path / "outside.png"
    Image.new("RGB", (2, 2)).save(outside)
    (run / "linked.png").symlink_to(outside)
    try:
        server = make_server(root, 0)
    except PermissionError:
        pytest.skip("Environment prohibits local sockets")
    with server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base) as response:
                page = response.read().decode()
                assert "circle &amp; green" in page
                assert "growth.gif" in page
                assert "linked.png" not in page
            with urlopen(base + "/artifacts/circle%20%26%20green/growth.gif") as response:
                assert response.headers["Content-Type"] == "image/gif"
                assert response.read().startswith(b"GIF")
            with urlopen(Request(base, method="HEAD")) as response:
                assert response.read() == b""
            for path in ["/artifacts/../outside.png", "/artifacts/%2e%2e/outside.png",
                         "/artifacts/circle%20%26%20green/linked.png", "/artifacts/circle%20%26%20green/checkpoint.pt"]:
                with pytest.raises(HTTPError) as error:
                    urlopen(base + path)
                assert error.value.code == 404
            # New outputs appear on refresh, without restarting the server.
            Image.new("RGB", (8, 8)).save(run / "loss.png")
            with urlopen(base) as response:
                assert b"loss.png" in response.read()
        finally:
            server.shutdown()
            thread.join(timeout=5)


def test_serve_help_and_missing_folder(tmp_path):
    runner = CliRunner()
    assert runner.invoke(main, ["serve", "--help"]).exit_code == 0
    result = runner.invoke(main, ["serve", "--runs-dir", str(tmp_path / "missing")])
    assert result.exit_code != 0
    assert "Train a model first" in result.output


def test_http_handler_without_sockets(tmp_path):
    """Exercise actual HTTP parsing, headers, and streaming in restricted CI."""
    from io import BytesIO
    from types import SimpleNamespace
    from nca.server import GalleryHandler

    class Connection:
        def __init__(self, request):
            self.input = BytesIO(request)
            self.output = bytearray()

        def makefile(self, *args):
            return self.input

        def sendall(self, data):
            self.output.extend(data)

    root = tmp_path / "runs"
    run = root / "circle & green"
    run.mkdir(parents=True)
    Image.new("RGB", (8, 8), "green").save(run / "growth.gif")
    (run / "config.json").write_text('{"mode":"grow"}')
    (tmp_path / "secret.png").write_bytes(b"secret")
    (run / "linked.png").symlink_to(tmp_path / "secret.png")

    def request(path, method="GET"):
        connection = Connection(f"{method} {path} HTTP/1.0\r\nHost: localhost\r\n\r\n".encode())
        GalleryHandler(connection, ("127.0.0.1", 1234), SimpleNamespace(), root=root)
        headers, body = bytes(connection.output).split(b"\r\n\r\n", 1)
        return headers, body

    headers, body = request("/")
    assert b"200 OK" in headers
    assert b"circle &amp; green" in body
    assert b"growth.gif" in body and b"config.json" in body
    assert b"linked.png" not in body
    headers, body = request("/artifacts/circle%20%26%20green/growth.gif")
    assert b"Content-type: image/gif" in headers
    assert body.startswith(b"GIF")
    headers, body = request("/", "HEAD")
    assert b"200 OK" in headers and body == b""
    for path in ["/artifacts/%2e%2e/secret.png", "/artifacts/circle%20%26%20green/linked.png", "/artifacts/", "/missing"]:
        assert b"404" in request(path)[0]
    Image.new("RGB", (8, 8)).save(run / "loss.png")
    assert b"loss.png" in request("/")[1]
