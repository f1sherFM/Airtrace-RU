#!/usr/bin/env python3
"""Simple static web server for the legacy AirTrace UI."""

from __future__ import annotations

import http.server
import os
import socketserver


def _resolve_api_base_url() -> str:
    return (os.getenv("API_BASE_URL", "").strip() or os.getenv("WEB_API_BASE_URL", "").strip()).rstrip("/")


class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Static file handler with permissive CORS and runtime API config injection."""

    def do_GET(self) -> None:  # noqa: N802 - stdlib hook
        if self.path in {"/", "/index.html"}:
            self._serve_index_with_runtime_config()
            return
        super().do_GET()

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib hook
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args) -> None:
        print(f"[{self.address_string()}] {format % args}")

    def _serve_index_with_runtime_config(self) -> None:
        index_path = self.translate_path("/index.html")
        if not os.path.exists(index_path):
            self.send_error(404, "File not found")
            return

        with open(index_path, "r", encoding="utf-8") as handle:
            content = handle.read()

        api_base_url = _resolve_api_base_url().replace("\\", "\\\\").replace('"', '\\"')
        content = content.replace(
            'window.AIRTRACE_API_BASE_URL = window.AIRTRACE_API_BASE_URL || "";',
            f'window.AIRTRACE_API_BASE_URL = "{api_base_url}";',
            1,
        )

        body = content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_server(port: int = 3000, directory: str | None = None) -> None:
    if directory:
        os.chdir(directory)

    with socketserver.TCPServer(("", port), CORSHTTPRequestHandler) as httpd:
        print("AirTrace RU Web Interface")
        print(f"Server started at http://localhost:{port}")
        print(f"Serving files from: {os.getcwd()}")
        print(f"API Backend: {_resolve_api_base_url() or '[not configured]'}")
        print("Press Ctrl+C to stop")
        print("-" * 50)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped")


if __name__ == "__main__":
    import sys

    port = 3000
    directory = "."

    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("Error: port must be a number")
            raise SystemExit(1)

    if len(sys.argv) > 2:
        directory = sys.argv[2]
        if not os.path.exists(directory):
            print(f"Error: directory {directory} does not exist")
            raise SystemExit(1)

    start_server(port, directory)
