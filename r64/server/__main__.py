"""Run the web server: uv run python -m r64.server"""

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="r64 web server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8064)
    parser.add_argument("--reload", action="store_true", help="restart on code changes")
    args = parser.parse_args()
    uvicorn.run(
        "r64.server.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
