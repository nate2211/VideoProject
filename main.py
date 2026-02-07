from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional

import blocks  # noqa: F401  (ensures block modules register themselves)
from stream import run_stream


def _coerce_scalar(s: str) -> Any:
    """
    Convert "123" -> int, "1.2" -> float, "true" -> bool, "null" -> None,
    JSON objects/arrays -> dict/list, else keep as string.
    """
    ss = s.strip()

    # bool / none
    low = ss.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None

    # int / float
    try:
        if ss.startswith("0") and ss != "0" and not ss.startswith("0."):
            # keep things like "0123" as string
            raise ValueError
        return int(ss)
    except ValueError:
        pass
    try:
        return float(ss)
    except ValueError:
        pass

    # json object/array
    if (ss.startswith("{") and ss.endswith("}")) or (ss.startswith("[") and ss.endswith("]")):
        try:
            return json.loads(ss)
        except Exception:
            return ss

    return ss


def parse_extras(extras: list[str]) -> Dict[str, Dict[str, Any]]:
    """
    extras entries are 'key=value' where key is:
      - 'all.someparam'
      - 'mosaic.tile'
      - 'cartoon.levels'
    Returns: stage_params dict: { "all": {...}, "mosaic": {...}, ... }
    """
    out: Dict[str, Dict[str, Any]] = {"all": {}}

    for item in extras:
        if "=" not in item:
            raise SystemExit(f"--extra expects key=value, got: {item!r}")

        k, v = item.split("=", 1)
        k = k.strip()
        v = _coerce_scalar(v)

        if "." not in k:
            raise SystemExit(f"--extra key must be like 'stage.param' (e.g. mosaic.tile), got: {k!r}")

        stage, param = k.split(".", 1)
        stage = stage.strip().lower()
        param = param.strip()

        if stage not in out:
            out[stage] = {}
        out[stage][param] = v

    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="main.py", description="Window stream + cv2 pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stream", help="Capture a window and apply a cv2 pipeline")
    s.add_argument("--window", required=True, help="Substring of window title (e.g. 'Firefox')")
    s.add_argument("--fps", type=int, default=30, help="Target FPS (default 30)")
    s.add_argument("--max-size", type=int, default=0, help="Max dimension (0 disables scaling)")
    s.add_argument("--pipeline", required=True, help="Pipeline stages, e.g. 'mosaic' or 'cartoon|mosaic'")
    s.add_argument("--preview", action="store_true", help="Show cv2 preview window")
    s.add_argument("--window-native", action="store_true", help="Use Win32 PrintWindow/BitBlt capture (Windows only)")
    s.add_argument("--ffmpeg-bin", default=None, help="Path to ffmpeg.exe (optional)")
    s.add_argument("--ffmpeg-out", default=None, help="Output file if ffmpeg enabled (default capture.mp4)")

    # NEW:
    s.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Stage params like mosaic.tile=64, cartoon.levels=8, all.softness=0.2 (repeatable)",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "stream":
        stage_params = parse_extras(args.extra)
        return run_stream(
            window_title=args.window,
            fps=args.fps,
            max_size=args.max_size,
            pipeline=args.pipeline,
            preview=args.preview,
            window_native=args.window_native,
            ffmpeg_bin=args.ffmpeg_bin,
            ffmpeg_out=args.ffmpeg_out,
            stage_params=stage_params,  # NEW
        )

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
