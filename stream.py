from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from registry import BLOCKS


# =============================================================================
# Window helpers (Windows only)
# =============================================================================

def _is_windows() -> bool:
    return os.name == "nt"


def _find_window_hwnd_by_title(title_substr: str) -> int:
    if not _is_windows():
        raise RuntimeError("Windows only.")

    import win32gui

    title_substr_l = title_substr.lower()
    matches: List[int] = []

    def enum_cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        t = win32gui.GetWindowText(hwnd) or ""
        if title_substr_l in t.lower():
            l, t2, r, b = win32gui.GetWindowRect(hwnd)
            if (r - l) > 200 and (b - t2) > 200:
                matches.append(hwnd)

    win32gui.EnumWindows(enum_cb, None)

    if not matches:
        raise RuntimeError(f"No visible window contains title '{title_substr}'")

    def area(hwnd: int) -> int:
        l, t2, r, b = win32gui.GetWindowRect(hwnd)
        return max(0, r - l) * max(0, b - t2)

    matches.sort(key=area, reverse=True)
    return matches[0]


# =============================================================================
# Capturers
# =============================================================================

class WindowCapturer:
    def grab(self) -> np.ndarray:
        raise NotImplementedError


class Win32ClientPrintCapturer(WindowCapturer):
    """
    Fast/stable Windows client-area capture using PrintWindow -> mem DC,
    then gdi32.GetDIBits into a *preallocated* numpy buffer (BGRA).

    This avoids:
      - Creating DC/bitmap every frame
      - Allocating big Python bytes blobs every frame (GetBitmapBits)
      - Slow growth / fragmentation / MemoryError over time
    """
    def __init__(self, hwnd: int):
        if not _is_windows():
            raise RuntimeError("Windows only.")

        import ctypes
        import win32con
        import win32gui
        import win32ui

        self.ctypes = ctypes
        self.win32con = win32con
        self.win32gui = win32gui
        self.win32ui = win32ui
        self.hwnd = hwnd

        self.user32 = ctypes.windll.user32
        self.gdi32 = ctypes.windll.gdi32

        self.PrintWindow = self.user32.PrintWindow
        self.GetDIBits = self.gdi32.GetDIBits

        # persistent GDI objects
        self._w = 0
        self._h = 0
        self._hwnd_dc = None
        self._srcdc = None
        self._memdc = None
        self._bmp = None

        # persistent pixel buffer + BITMAPINFO
        self._buf = None          # numpy (h,w,4) uint8
        self._bmi = None          # BITMAPINFO

        self._ensure_buffers()

    def _client_size(self) -> tuple[int, int]:
        l, t, r, b = self.win32gui.GetClientRect(self.hwnd)
        return max(1, r - l), max(1, b - t)

    def _free_buffers(self) -> None:
        try:
            if self._hwnd_dc is not None:
                self.win32gui.ReleaseDC(self.hwnd, self._hwnd_dc)
        except Exception:
            pass
        try:
            if self._memdc is not None:
                self._memdc.DeleteDC()
            if self._srcdc is not None:
                self._srcdc.DeleteDC()
        except Exception:
            pass
        try:
            if self._bmp is not None:
                self.win32gui.DeleteObject(self._bmp.GetHandle())
        except Exception:
            pass

        self._hwnd_dc = None
        self._srcdc = None
        self._memdc = None
        self._bmp = None
        self._buf = None
        self._bmi = None
        self._w = 0
        self._h = 0

    def _ensure_buffers(self) -> None:
        w, h = self._client_size()
        if w == self._w and h == self._h and self._memdc is not None and self._buf is not None:
            return

        self._free_buffers()

        self._w, self._h = w, h

        self._hwnd_dc = self.win32gui.GetWindowDC(self.hwnd)
        self._srcdc = self.win32ui.CreateDCFromHandle(self._hwnd_dc)
        self._memdc = self._srcdc.CreateCompatibleDC()

        self._bmp = self.win32ui.CreateBitmap()
        self._bmp.CreateCompatibleBitmap(self._srcdc, w, h)
        self._memdc.SelectObject(self._bmp)

        # preallocate BGRA output buffer
        self._buf = np.empty((h, w, 4), dtype=np.uint8)

        # build BITMAPINFO (top-down, 32-bit)
        ctypes = self.ctypes

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", ctypes.c_uint32),
                ("biWidth", ctypes.c_int32),
                ("biHeight", ctypes.c_int32),
                ("biPlanes", ctypes.c_uint16),
                ("biBitCount", ctypes.c_uint16),
                ("biCompression", ctypes.c_uint32),
                ("biSizeImage", ctypes.c_uint32),
                ("biXPelsPerMeter", ctypes.c_int32),
                ("biYPelsPerMeter", ctypes.c_int32),
                ("biClrUsed", ctypes.c_uint32),
                ("biClrImportant", ctypes.c_uint32),
            ]

        class BITMAPINFO(ctypes.Structure):
            _fields_ = [
                ("bmiHeader", BITMAPINFOHEADER),
                ("bmiColors", ctypes.c_uint32 * 3),  # unused for BI_RGB
            ]

        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = w
        bmi.bmiHeader.biHeight = -h  # negative => top-down
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = self.win32con.BI_RGB
        bmi.bmiHeader.biSizeImage = w * h * 4
        self._bmi = bmi

    def grab(self) -> np.ndarray:
        self._ensure_buffers()

        ok = 0
        try:
            ok = self.PrintWindow(self.hwnd, self._memdc.GetSafeHdc(), 3)
        except Exception:
            ok = 0

        if not ok:
            self._memdc.BitBlt((0, 0), (self._w, self._h), self._srcdc, (0, 0), self.win32con.SRCCOPY)

        # GetDIBits into preallocated numpy buffer (BGRA)
        hdc = self._memdc.GetSafeHdc()
        hbmp = self._bmp.GetHandle()

        dst_ptr = self._buf.ctypes.data_as(self.ctypes.c_void_p)
        lines = self.GetDIBits(
            hdc,
            hbmp,
            0,
            self._h,
            dst_ptr,
            self.ctypes.byref(self._bmi),
            self.win32con.DIB_RGB_COLORS,
        )

        if lines != self._h:
            # fallback: if GetDIBits fails, return a safe black frame
            return np.zeros((self._h, self._w, 3), dtype=np.uint8)

        # convert BGRA -> BGR (and copy to decouple from the persistent buffer)
        return self._buf[:, :, :3].copy()

    def __del__(self):
        self._free_buffers()


class MSSCapturer(WindowCapturer):
    def __init__(self, rect: Tuple[int, int, int, int]):
        from mss import mss
        self._sct = mss()
        l, t, r, b = rect
        self._mon = {"left": l, "top": t, "width": r - l, "height": b - t}

    def grab(self) -> np.ndarray:
        shot = self._sct.grab(self._mon)  # BGRA
        frame = np.array(shot, dtype=np.uint8)
        return frame[:, :, :3]  # BGR


# =============================================================================
# Pipeline
# =============================================================================

def build_pipeline(pipeline_str: str) -> List[str]:
    stages = [s.strip().lower() for s in (pipeline_str or "").split("|") if s.strip()]
    if not stages:
        raise ValueError("Empty --pipeline. Example: --pipeline 'mosaic'")
    return stages


def apply_pipeline(frame_bgr: np.ndarray, stages: List[str], stage_params: Dict[str, Dict[str, object]]) -> np.ndarray:
    cur = frame_bgr
    for s in stages:
        blk = BLOCKS.create(s)
        params = dict(stage_params.get("all", {}))
        params.update(stage_params.get(s, {}))
        cur, _meta = blk.execute(cur, params=params)
    return cur


def _has_cuda() -> bool:
    return hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0


def compute_scaled_size(w: int, h: int, max_size: int) -> tuple[int, int]:
    if max_size <= 0:
        return w, h
    mx = max(w, h)
    if mx <= max_size:
        return w, h
    scale = max_size / float(mx)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return nw, nh


def scale_max_size_cpu(frame: np.ndarray, max_size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    nw, nh = compute_scaled_size(w, h, max_size)
    if (nw, nh) == (w, h):
        return frame
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)


def scale_max_size_gpu(g: "cv2.cuda_GpuMat", w: int, h: int, max_size: int) -> "cv2.cuda_GpuMat":
    nw, nh = compute_scaled_size(w, h, max_size)
    if (nw, nh) == (w, h):
        return g
    return cv2.cuda.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)

def _post_click(hwnd: int, cx: int, cy: int, button: str = "left"):
    import win32con, win32gui

    # client coords -> lParam
    lparam = (cy << 16) | (cx & 0xFFFF)

    # IMPORTANT: do NOT call SetForegroundWindow here.
    # That’s what was stealing focus / flipping tabs.

    # Optional: tell some apps we're not activating (helps certain UI frameworks)
    win32gui.PostMessage(hwnd, win32con.WM_MOUSEACTIVATE, 0, 0)

    if button == "left":
        win32gui.PostMessage(hwnd, win32con.WM_MOUSEMOVE, 0, lparam)
        win32gui.PostMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lparam)
        win32gui.PostMessage(hwnd, win32con.WM_LBUTTONUP, 0, lparam)

    elif button == "right":
        win32gui.PostMessage(hwnd, win32con.WM_MOUSEMOVE, 0, lparam)
        win32gui.PostMessage(hwnd, win32con.WM_RBUTTONDOWN, win32con.MK_RBUTTON, lparam)
        win32gui.PostMessage(hwnd, win32con.WM_RBUTTONUP, 0, lparam)

def _post_mouse_move(hwnd: int, cx: int, cy: int):
    import win32con, win32gui
    lparam = (cy << 16) | (cx & 0xFFFF)

    # No SetForegroundWindow (keeps your stream on top)
    # Some UI stacks respond better if WM_MOUSEACTIVATE is seen
    win32gui.PostMessage(hwnd, win32con.WM_MOUSEACTIVATE, 0, 0)

    # WM_MOUSEMOVE triggers hover UI (YouTube controls, tooltips, etc.)
    win32gui.PostMessage(hwnd, win32con.WM_MOUSEMOVE, 0, lparam)

def _post_mouse_leave(hwnd: int):
    import win32con, win32gui
    # Many apps ignore this unless they called TrackMouseEvent internally,
    # but it's harmless and sometimes helps.
    win32gui.PostMessage(hwnd, win32con.WM_MOUSELEAVE, 0, 0)

def _post_mouse_wheel(hwnd: int, cx: int, cy: int, delta_y: int = 0, delta_x: int = 0, keys: int = 0):
    import win32con, win32gui

    # WM_MOUSEWHEEL expects SCREEN coords in lParam
    sx, sy = win32gui.ClientToScreen(hwnd, (cx, cy))

    def pack_lparam(x: int, y: int) -> int:
        return ((y & 0xFFFF) << 16) | (x & 0xFFFF)

    def pack_wparam(keys_low: int, delta: int) -> int:
        # high word = signed 16-bit delta, low word = key state
        return (keys_low & 0xFFFF) | ((delta & 0xFFFF) << 16)

    # IMPORTANT: send to the window *under the cursor* (often a child)
    target = win32gui.WindowFromPoint((sx, sy))
    if not target:
        target = hwnd

    lparam = pack_lparam(sx, sy)

    if delta_y:
        win32gui.PostMessage(target, win32con.WM_MOUSEWHEEL, pack_wparam(keys, delta_y), lparam)

    if delta_x:
        # WM_MOUSEHWHEEL for horizontal wheel
        WM_MOUSEHWHEEL = getattr(win32con, "WM_MOUSEHWHEEL", 0x020E)
        win32gui.PostMessage(target, WM_MOUSEHWHEEL, pack_wparam(keys, delta_x), lparam)

def _post_key(hwnd: int, vk_code: int, down: bool):
    import win32con, win32gui

    msg = win32con.WM_KEYDOWN if down else win32con.WM_KEYUP

    # Simple lParam construction (repeat count 1, scan code 0)
    # For high-end games, you might need accurate scan codes,
    # but for general apps, VK code in wParam is usually enough.
    lparam = 1  # Repeat count 1
    if not down:
        # Set transition state bit (31) and previous key state bit (30) for KeyUp
        lparam |= 0xC0000000

    win32gui.PostMessage(hwnd, msg, vk_code, lparam)


def _post_mouse_down(hwnd: int, cx: int, cy: int, button: str = "left"):
    import win32con, win32gui
    lparam = (cy << 16) | (cx & 0xFFFF)

    # Ensure window receives input attention
    win32gui.PostMessage(hwnd, win32con.WM_MOUSEACTIVATE, 0, 0)

    # Send Move first to ensure cursor is in position before clicking
    win32gui.PostMessage(hwnd, win32con.WM_MOUSEMOVE, 0, lparam)

    if button == "left":
        win32gui.PostMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lparam)
    elif button == "right":
        win32gui.PostMessage(hwnd, win32con.WM_RBUTTONDOWN, win32con.MK_RBUTTON, lparam)


def _post_mouse_up(hwnd: int, cx: int, cy: int, button: str = "left"):
    import win32con, win32gui
    lparam = (cy << 16) | (cx & 0xFFFF)

    if button == "left":
        win32gui.PostMessage(hwnd, win32con.WM_LBUTTONUP, 0, lparam)
    elif button == "right":
        win32gui.PostMessage(hwnd, win32con.WM_RBUTTONUP, 0, lparam)
# =============================================================================
# Optional recording (only if you pass ffmpeg_bin)
# =============================================================================

@dataclass
class FFmpegSink:
    proc: subprocess.Popen
    w: int
    h: int

    def write(self, frame_bgr: np.ndarray) -> None:
        self.proc.stdin.write(frame_bgr.tobytes())

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass


def start_ffmpeg(ffmpeg_bin: str, w: int, h: int, fps: int, out_path: str, *, hwenc: str = "nvenc") -> FFmpegSink:
    if hwenc == "nvenc":
        vcodec = ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "19"]
    elif hwenc == "qsv":
        vcodec = ["-c:v", "h264_qsv", "-global_quality", "19"]
    elif hwenc == "amf":
        vcodec = ["-c:v", "h264_amf", "-quality", "quality", "-qp_i", "20", "-qp_p", "22"]
    else:
        vcodec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]

    cmd = [
        ffmpeg_bin,
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-r", str(int(fps)),
        "-i", "-",
        "-an",
        *vcodec,
        out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    if proc.stdin is None:
        raise RuntimeError("Failed to open ffmpeg stdin pipe.")
    return FFmpegSink(proc=proc, w=w, h=h)


# =============================================================================
# Main loop: LIVE processed preview stream (no “falling behind”)
# =============================================================================

def run_stream(
    *,
    window_title: str,
    fps: int,
    max_size: int,
    pipeline: str,
    preview: bool,
    window_native: bool,
    ffmpeg_bin: Optional[str],
    ffmpeg_out: Optional[str],
    stage_params: Optional[Dict[str, Dict[str, object]]] = None,  # NEW
) -> int:
    """
    Live preview design:
      - No encoder “frame count/size” concerns unless you enable ffmpeg recording.
      - We DO NOT try to “catch up” if processing is slow.
      - We simply: grab -> process -> show, and optionally cap at fps if we’re faster.

    If your pipeline can only do ~13 fps, you will see ~13 fps (but it will always be the latest frame).
    """
    if not _is_windows():
        raise RuntimeError("This stream implementation is Windows-only for now.")

    hwnd = _find_window_hwnd_by_title(window_title)

    # Capturer
    if window_native:
        capturer: WindowCapturer = Win32ClientPrintCapturer(hwnd)
    else:
        # If you want MSS with client rect, you'd need a client-rect-to-screen helper.
        # Keeping it simple: use window rect via win32gui.GetWindowRect.
        import win32gui
        l, t, r, b = win32gui.GetWindowRect(hwnd)
        capturer = MSSCapturer((l, t, r, b))

    stages = build_pipeline(pipeline)


    use_gpu = _has_cuda()
    g = cv2.cuda_GpuMat() if use_gpu else None

    # Prime 1 frame to lock output size for ffmpeg and for scaling cache
    frame = capturer.grab()
    h0, w0 = frame.shape[:2]
    nw0, nh0 = compute_scaled_size(w0, h0, max_size)

    if use_gpu and g is not None:
        g.upload(frame)
        g = scale_max_size_gpu(g, w0, h0, max_size)
        frame = g.download()
    else:
        frame = scale_max_size_cpu(frame, max_size)

    frame = apply_pipeline(frame, stages, stage_params)

    # Optional recording (OFF unless ffmpeg_bin is provided)
    ff: Optional[FFmpegSink] = None
    if ffmpeg_bin:
        out_path = ffmpeg_out or "capture.mp4"
        ff = start_ffmpeg(ffmpeg_bin, frame.shape[1], frame.shape[0], fps, out_path, hwenc="nvenc")
        print(f"[stream] ffmpeg recording -> {out_path}")

    if preview:
        cv2.namedWindow("stream", cv2.WINDOW_NORMAL)
    clicked = {
        "x": None,
        "y": None,
        "button": None,  # "left" or "right"
    }

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["x"] = x
            clicked["y"] = y
            clicked["button"] = "left"

        elif event == cv2.EVENT_RBUTTONDOWN:
            clicked["x"] = x
            clicked["y"] = y
            clicked["button"] = "right"

    cv2.setMouseCallback("stream", on_mouse)
    # FPS cap only (not “schedule frames”): if we are faster than fps, sleep a bit.
    fps_cap = max(1, int(fps)) if fps and fps > 0 else 0
    min_dt = (1.0 / fps_cap) if fps_cap else 0.0

    # Simple on-screen stats
    t_last_stat = time.perf_counter()
    frames_stat = 0
    shown_fps = 0.0

    try:
        while True:
            t0 = time.perf_counter()

            # grab newest
            frame = capturer.grab()

            # If window size changes, recompute scaling target
            h, w = frame.shape[:2]
            if (w, h) != (w0, h0):
                w0, h0 = w, h
                nw0, nh0 = compute_scaled_size(w0, h0, max_size)
            if clicked["x"] is not None:
                px = max(0, min(clicked["x"], nw0 - 1))
                py = max(0, min(clicked["y"], nh0 - 1))

                # map preview coords → window client coords
                cx = int(px * (w0 / float(nw0)))
                cy = int(py * (h0 / float(nh0)))

                _post_click(hwnd, cx, cy, clicked["button"])

                clicked["x"] = clicked["y"] = clicked["button"] = None
            # scale (+ GPU resize only; blocks are still CPU unless you add execute_gpu to them)
            if use_gpu and g is not None:
                g.upload(frame)
                g = scale_max_size_gpu(g, w0, h0, max_size)
                frame = g.download()
            else:
                frame = scale_max_size_cpu(frame, max_size)

            # pipeline (CPU)
            frame = apply_pipeline(frame, stages, stage_params)

            # optional write
            if ff is not None:
                ff.write(frame)

            # preview
            if preview:
                frames_stat += 1
                now = time.perf_counter()
                if now - t_last_stat >= 1.0:
                    shown_fps = frames_stat / (now - t_last_stat)
                    frames_stat = 0
                    t_last_stat = now

                cv2.putText(
                    frame,
                    f"{shown_fps:.1f} fps",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("stream", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            # FPS cap (only if faster than target)
            if min_dt > 0:
                dt = time.perf_counter() - t0
                if dt < min_dt:
                    time.sleep(min_dt - dt)

    finally:
        if ff is not None:
            ff.close()
        if preview:
            cv2.destroyAllWindows()

    return 0
