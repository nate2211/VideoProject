from __future__ import annotations

import subprocess
import time
from typing import Optional

import cv2
import numpy as np

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSlider, QFileDialog, QSizePolicy, QLineEdit, QMessageBox, QGroupBox
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput, QVideoSink


def _fmt_ms(ms: int) -> str:
    if ms < 0:
        ms = 0
    s = ms // 1000
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:d}:{m:02d}:{sec:02d}" if h > 0 else f"{m:02d}:{sec:02d}"


def _try_resolve_youtube(url: str) -> Optional[str]:
    """
    Resolve a YouTube URL to a direct media URL suitable for QMediaPlayer.
    Uses the native yt_dlp Python API to ensure PyInstaller compatibility.
    """
    try:
        import yt_dlp
    except ImportError:
        print("Error: yt_dlp module not found.")
        return None

    ydl_opts = {
        # Prefer progressive MP4 with BOTH audio+video (best for Qt)
        'format': 'best[ext=mp4][vcodec!=none][acodec!=none]/best[ext=mp4]/best',
        'noplaylist': True,
        'no_warnings': True,
        'ignoreerrors': True,
        'quiet': True,  # Suppress console output
    }

    try:
        # download=False is the API equivalent of the "-g" (get URL) command line flag
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            if not info:
                return None

            # If a playlist URL accidentally slipped through, grab the first entry
            if 'entries' in info and len(info['entries']) > 0:
                return info['entries'][0].get('url')

            # Return the direct media URL
            return info.get('url')

    except Exception as e:
        print(f"Failed to resolve YouTube URL: {e}")
        return None

class VideoPlayerLabel(QLabel):
    def __init__(self):
        super().__init__("No media loaded")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #000; border: 2px solid #111; color: white;")
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)


class EditPane(QWidget):
    """
    Video Editor tab that:
      - loads file / URL / YouTube URL
      - plays audio (QMediaPlayer)
      - scrubs by time
      - processes frames in realtime through shared pipeline callbacks
      - force_refresh() re-runs pipeline on the last frame when paused
    """
    def __init__(self, apply_pipeline_callback, get_active_pipeline_callback, get_stage_params_callback):
        super().__init__()

        # Callbacks to StudioApp's pipeline state
        self.apply_pipeline = apply_pipeline_callback
        self.get_active_pipeline = get_active_pipeline_callback
        self.get_stage_params = get_stage_params_callback

        # Multimedia
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)

        self.sink = QVideoSink(self)
        self.player.setVideoOutput(self.sink)

        # State
        self._duration_ms = 0
        self._dragging = False
        self._last_bgr: Optional[np.ndarray] = None
        self._last_process_t = 0.0
        self._max_preview_fps = 30.0  # throttle processing

        self._init_ui()
        self._wire_signals()

        # default volume
        self.audio.setVolume(0.8)

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # -------------------- Source box --------------------
        src_box = QGroupBox("Source")
        src_layout = QVBoxLayout(src_box)

        row = QHBoxLayout()
        self.url_line = QLineEdit()
        self.url_line.setPlaceholderText("Paste a video URL or YouTube URL here…")
        row.addWidget(self.url_line, 1)

        self.btn_load_url = QPushButton("Load URL")
        self.btn_load_url.clicked.connect(self.load_url)
        row.addWidget(self.btn_load_url)

        self.btn_open_file = QPushButton("Open File")
        self.btn_open_file.clicked.connect(self.open_file)
        row.addWidget(self.btn_open_file)

        src_layout.addLayout(row)

        hint = QLabel("YouTube URLs require yt-dlp installed. Direct .mp4 URLs work without it.")
        hint.setStyleSheet("color: #bbb;")
        src_layout.addWidget(hint)

        layout.addWidget(src_box)

        # -------------------- Video display --------------------
        self.display = VideoPlayerLabel()
        layout.addWidget(self.display, stretch=1)

        # -------------------- Playback controls --------------------
        ctrl_box = QGroupBox("Playback")
        ctrl_layout = QVBoxLayout(ctrl_box)

        top = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self.toggle_playback)
        top.addWidget(self.btn_play)

        self.lbl_time = QLabel("00:00 / 00:00")
        top.addWidget(self.lbl_time, 1)

        top.addWidget(QLabel("Vol"))
        self.vol = QSlider(Qt.Orientation.Horizontal)
        self.vol.setRange(0, 100)
        self.vol.setValue(80)
        self.vol.valueChanged.connect(self._on_volume)
        top.addWidget(self.vol)

        ctrl_layout.addLayout(top)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.setRange(0, 0)
        self.slider.sliderPressed.connect(self._on_scrub_press)
        self.slider.sliderReleased.connect(self._on_scrub_release)
        self.slider.sliderMoved.connect(self._on_scrub_move)
        ctrl_layout.addWidget(self.slider)

        layout.addWidget(ctrl_box)

    def _wire_signals(self):
        self.sink.videoFrameChanged.connect(self._on_frame)
        self.player.durationChanged.connect(self._on_duration)
        self.player.positionChanged.connect(self._on_position)
        self.player.playbackStateChanged.connect(self._on_state)

    # ---------------------------------------------------------------------
    # Loading
    # ---------------------------------------------------------------------

    def open_file(self):
        f, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            "",
            "Video Files (*.mp4 *.mov *.mkv *.webm *.avi);;All Files (*.*)"
        )
        if not f:
            return
        self._set_source(QUrl.fromLocalFile(f))

    def load_url(self):
        url = (self.url_line.text() or "").strip()
        if not url:
            return

        low = url.lower()
        if "youtube.com" in low or "youtu.be" in low:
            direct = _try_resolve_youtube(url)
            if not direct:
                QMessageBox.warning(
                    self,
                    "YouTube URL",
                    "Could not resolve YouTube URL.\n\n"
                    "Install yt-dlp:\n  pip install yt-dlp\n\n"
                    "Then try again."
                )
                return
            self._set_source(QUrl(direct))
            return

        self._set_source(QUrl(url))

    def _set_source(self, qurl: QUrl):
        self.player.stop()
        self._last_bgr = None
        self.display.setText("Loading…")

        self.player.setSource(qurl)

        # enable controls (duration may arrive later)
        self.btn_play.setEnabled(True)
        self.slider.setEnabled(True)

        # start playing immediately
        self.player.play()

    # ---------------------------------------------------------------------
    # Playback UI
    # ---------------------------------------------------------------------

    def toggle_playback(self):
        st = self.player.playbackState()
        if st == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _on_state(self, _):
        st = self.player.playbackState()
        self.btn_play.setText("Pause" if st == QMediaPlayer.PlaybackState.PlayingState else "Play")

    def _on_volume(self, v: int):
        self.audio.setVolume(float(v) / 100.0)

    def _on_duration(self, ms: int):
        self._duration_ms = int(ms or 0)
        self.slider.setRange(0, self._duration_ms)
        self._update_time_label(self.player.position())

    def _on_position(self, ms: int):
        ms = int(ms or 0)
        if not self._dragging:
            self.slider.blockSignals(True)
            self.slider.setValue(ms)
            self.slider.blockSignals(False)
        self._update_time_label(ms)

    def _on_scrub_press(self):
        self._dragging = True

    def _on_scrub_move(self, ms: int):
        self._update_time_label(int(ms))

    def _on_scrub_release(self):
        self._dragging = False
        self.player.setPosition(int(self.slider.value()))

    def _update_time_label(self, pos_ms: int):
        self.lbl_time.setText(f"{_fmt_ms(pos_ms)} / {_fmt_ms(self._duration_ms)}")

    # ---------------------------------------------------------------------
    # Frame processing
    # ---------------------------------------------------------------------

    def _on_frame(self, frame):
        # throttle CPU load
        now = time.monotonic()
        if self._max_preview_fps > 0:
            min_dt = 1.0 / self._max_preview_fps
            if (now - self._last_process_t) < min_dt:
                return
        self._last_process_t = now

        try:
            qimg = frame.toImage()
        except Exception:
            return
        if qimg.isNull():
            return

        qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)
        w = qimg.width()
        h = qimg.height()
        bpl = qimg.bytesPerLine()

        ptr = qimg.bits()
        ptr.setsize(h * bpl)
        arr = np.frombuffer(ptr, np.uint8).reshape((h, bpl // 3, 3))[:, :w, :]

        # convert RGB->BGR for your pipeline
        bgr = arr[:, :, ::-1].copy()
        self._last_bgr = bgr

        self._render_processed(bgr)

    def _render_processed(self, bgr: np.ndarray):
        pipeline = self.get_active_pipeline()
        params = self.get_stage_params()

        try:
            out_bgr = self.apply_pipeline(bgr, pipeline, params) if pipeline else bgr
        except Exception as e:
            self.display.setText(f"Pipeline error:\n{e}")
            return

        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        hh, ww, ch = out_rgb.shape
        q = QImage(out_rgb.data, ww, hh, ch * ww, QImage.Format.Format_RGB888)
        pm = QPixmap.fromImage(q)

        # scale to label
        label_w = max(1, self.display.width())
        label_h = max(1, self.display.height())
        scaled = pm.scaled(
            label_w, label_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.display.setPixmap(scaled)

    def force_refresh(self):
        """
        Called by StudioApp whenever pipeline or params change.
        If we're paused (or even playing), we can re-render the last frame immediately.
        """
        if self._last_bgr is None:
            return
        # If you only want instant refresh when paused, uncomment:
        # if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
        #     return
        self._render_processed(self._last_bgr)
