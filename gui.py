# gui.py
import sys
import cv2
import numpy as np
import win32con
import win32gui

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QGroupBox, QSlider, QLabel,
    QScrollArea, QSplitter, QDialog, QListWidgetItem, QSizePolicy, QTabWidget
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap

from registry import BLOCKS
import blocks
from stream import (
    Win32ClientPrintCapturer, apply_pipeline,
    _post_mouse_move, _post_mouse_leave, _post_mouse_wheel,
    _post_key, _post_mouse_down, _post_mouse_up
)

# Import our new editing pane
from gui_elements import EditPane


# =============================================================================
# Clickable preview label (Handles Mouse & Keyboard for Stream)
# =============================================================================
class ClickableLabel(QLabel):
    # Signals for discrete input states
    pressed = pyqtSignal(int, int, str)
    released = pyqtSignal(int, int, str)
    moved = pyqtSignal(int, int)
    left_sig = pyqtSignal()
    wheeled = pyqtSignal(int, int, int, int)

    def __init__(self, parent_gui, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent_gui = parent_gui
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def enterEvent(self, ev):
        self.setFocus()
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        self.clearFocus()
        self.left_sig.emit()
        super().leaveEvent(ev)

    def mousePressEvent(self, ev):
        self.setFocus()
        btn = "left" if ev.button() == Qt.MouseButton.LeftButton else "right"
        if ev.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self.pressed.emit(int(ev.position().x()), int(ev.position().y()), btn)
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):
        btn = "left" if ev.button() == Qt.MouseButton.LeftButton else "right"
        if ev.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self.released.emit(int(ev.position().x()), int(ev.position().y()), btn)
        super().mouseReleaseEvent(ev)

    def mouseMoveEvent(self, ev):
        self.moved.emit(int(ev.position().x()), int(ev.position().y()))
        super().mouseMoveEvent(ev)

    def wheelEvent(self, ev):
        d = ev.angleDelta()
        self.wheeled.emit(int(ev.position().x()), int(ev.position().y()), int(d.y()), int(d.x()))
        ev.accept()

    def keyPressEvent(self, ev):
        if not ev.isAutoRepeat():
            self._send_key(ev, down=True)

    def keyReleaseEvent(self, ev):
        if not ev.isAutoRepeat():
            self._send_key(ev, down=False)

    def _send_key(self, ev, down):
        # We need to access the capturer via the parent's stream logic
        if not self.parent_gui or not self.parent_gui.capturer:
            return
        vk = ev.nativeVirtualKey()
        if vk > 0:
            try:
                _post_key(self.parent_gui.capturer.hwnd, vk, down)
            except Exception as e:
                print(f"Key error: {e}")


# =============================================================================
# Window selector
# =============================================================================
class WindowSelector(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Window to Stream")
        self.setFixedSize(450, 500)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Select an active window:</b>"))

        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self.accept)
        layout.addWidget(self.list_widget)

        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("Connect")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self._refresh_windows()

    def _refresh_windows(self):
        main_win_title = "Gemini Studio"

        def enum_cb(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title and title != main_win_title and title != "Select Window to Stream":
                    item = QListWidgetItem(title)
                    item.setData(Qt.ItemDataRole.UserRole, hwnd)
                    self.list_widget.addItem(item)

        self.list_widget.clear()
        win32gui.EnumWindows(enum_cb, None)

    def get_selection(self):
        item = self.list_widget.currentItem()
        if item:
            return item.data(Qt.ItemDataRole.UserRole), item.text()
        return None, None


class StreamWorker(QThread):
    # Signals to pass the processed frame back to the main GUI safely
    frame_ready = pyqtSignal(QImage, int, int)  # image, original_w, original_h
    error_occurred = pyqtSignal(str)

    def __init__(self, main_app):
        super().__init__()
        self.app = main_app
        self.is_running = False

    def run(self):
        self.is_running = True
        while self.is_running:
            # If not connected or on the wrong tab, just sleep
            if not self.app.capturer or self.app.tabs.currentIndex() != 0:
                self.msleep(33)
                continue

            try:
                frame = self.app.capturer.grab()
                if frame is None:
                    self.msleep(10)
                    continue

                # Take a snapshot of the pipeline/params to avoid thread crashes
                pipeline = list(self.app.active_pipeline)
                params = dict(self.app.stage_params)

                # 1. Process Pipeline
                processed = apply_pipeline(frame, pipeline, params)

                # 2. Convert to RGB
                rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                bytes_per_line = ch * w

                # 3. Create QImage (MUST call .copy() so memory survives the thread transfer)
                qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()

                self.frame_ready.emit(qt_img, w, h)

                # Small yield to prevent CPU pegging if processing is instantly fast
                self.msleep(1)

            except Exception as e:
                self.error_occurred.emit(str(e))
                self.msleep(100)  # Wait a bit before retrying on error

    def stop(self):
        self.is_running = False
        self.wait()

# =============================================================================
# Main GUI App
# =============================================================================
class StudioApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gemini Studio")
        self.resize(1400, 800)

        # Shared State
        self.active_pipeline = []
        self.stage_params = {"all": {}}

        # Stream specific state
        self.capturer: Win32ClientPrintCapturer | None = None
        self.is_connected = False
        self._last_src_w: int | None = None
        self._last_src_h: int | None = None
        self._last_draw_rect: tuple[int, int, int, int] | None = None

        self._init_ui()



    def _init_ui(self):
        self.setStyleSheet("""
                    QWidget {
                        background-color: #1e1e1e;
                        color: #ffffff;
                        font-family: 'Segoe UI', sans-serif;
                    }
                    QLabel {
                        color: #ffffff;
                        background-color: transparent;
                    }
                    QGroupBox {
                        color: #ffffff;
                        border: 1px solid #555555;
                        margin-top: 1.5ex;
                        padding-top: 15px;
                        font-weight: bold;
                        border-radius: 4px;
                    }
                    QGroupBox::title {
                        subcontrol-origin: margin;
                        subcontrol-position: top left;
                        padding: 0 5px;
                        left: 10px;
                        color: #ffffff;
                    }
                    QPushButton {
                        background-color: #333333;
                        color: #ffffff;
                        border: 1px solid #555555;
                        padding: 8px 12px;
                        font-weight: bold;
                        border-radius: 4px;
                    }
                    QPushButton:hover {
                        background-color: #444444;
                        border: 1px solid #666666;
                    }
                    QPushButton:pressed {
                        background-color: #222222;
                    }
                    QPushButton:disabled {
                        background-color: #2a2a2a;
                        color: #777777;
                        border: 1px solid #333333;
                    }
                    QPushButton#connectBtn {
                        background-color: #2d5a27;
                        border: 1px solid #3d7a35;
                    }
                    QPushButton#connectBtn:hover {
                        background-color: #387030;
                    }
                    QPushButton#disconnectBtn {
                        background-color: #8b0000;
                        border: 1px solid #b30000;
                    }
                    QPushButton#disconnectBtn:hover {
                        background-color: #a60000;
                    }
                    QListWidget {
                        background-color: #252526;
                        color: #ffffff;
                        border: 1px solid #444444;
                        border-radius: 4px;
                        outline: none;
                    }
                    QListWidget::item {
                        padding: 4px;
                    }
                    QListWidget::item:selected {
                        background-color: #007acc;
                        color: #ffffff;
                    }
                    QListWidget::item:hover:!selected {
                        background-color: #3a3d41;
                    }
                    QTabWidget::pane {
                        border: 1px solid #555555;
                        background-color: #1e1e1e;
                        border-radius: 4px;
                    }
                    QTabBar::tab {
                        background-color: #2d2d2d;
                        color: #cccccc;
                        padding: 8px 16px;
                        border: 1px solid #444444;
                        border-bottom: none;
                        border-top-left-radius: 4px;
                        border-top-right-radius: 4px;
                        margin-right: 2px;
                    }
                    QTabBar::tab:selected {
                        background-color: #444444;
                        color: #ffffff;
                        font-weight: bold;
                    }
                    QTabBar::tab:hover:!selected {
                        background-color: #383838;
                        color: #ffffff;
                    }
                    QSlider::groove:horizontal {
                        border: 1px solid #333333;
                        background: #2b2b2b;
                        height: 6px;
                        border-radius: 3px;
                    }
                    QSlider::sub-page:horizontal {
                        background: #007acc;
                        border-radius: 3px;
                    }
                    QSlider::handle:horizontal {
                        background: #ffffff;
                        border: 1px solid #000000;
                        width: 14px;
                        margin-top: -4px;
                        margin-bottom: -4px;
                        border-radius: 7px;
                    }
                    QSlider::handle:horizontal:hover {
                        background: #e0e0e0;
                    }
                    QSlider::handle:horizontal:disabled {
                        background: #555555;
                        border: 1px solid #333333;
                    }
                    QDialog {
                        background-color: #1e1e1e;
                    }
                    QScrollArea, QScrollArea > QWidget > QWidget {
                        background-color: transparent;
                        border: none;
                    }
                """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- SHARED SIDEBAR ---
        sidebar = QWidget()
        side_layout = QVBoxLayout(sidebar)

        side_layout.addWidget(QLabel("<b>Available Blocks</b>"))
        self.list_available = QListWidget()
        self.list_available.addItems(BLOCKS.names())
        self.list_available.itemDoubleClicked.connect(self.add_to_pipeline)
        side_layout.addWidget(self.list_available)

        side_layout.addWidget(QLabel("<b>Active Pipeline</b>"))
        self.list_pipeline = QListWidget()
        self.list_pipeline.itemDoubleClicked.connect(self.remove_from_pipeline)
        side_layout.addWidget(self.list_pipeline)

        self.scroll_params = QScrollArea()
        self.scroll_params.setWidgetResizable(True)
        self.params_container = QWidget()
        self.params_layout = QVBoxLayout(self.params_container)
        self.scroll_params.setWidget(self.params_container)
        side_layout.addWidget(QLabel("<b>Block Parameters</b>"))
        side_layout.addWidget(self.scroll_params)

        splitter.addWidget(sidebar)

        # --- TABBED CONTENT AREA ---
        self.tabs = QTabWidget()

        # 1. Stream Tab
        self.stream_widget = QWidget()
        stream_layout = QVBoxLayout(self.stream_widget)

        self.btn_toggle_connect = QPushButton("Connect to Window")
        self.btn_toggle_connect.setObjectName("connectBtn")
        self.btn_toggle_connect.clicked.connect(self.handle_connection)
        stream_layout.addWidget(self.btn_toggle_connect)

        self.stream_display = ClickableLabel(self, "No active stream")
        self.stream_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stream_display.setStyleSheet("background-color: #000; border: 2px solid #111;")
        self.stream_display.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

        self.stream_display.pressed.connect(self.on_stream_press)
        self.stream_display.released.connect(self.on_stream_release)
        self.stream_display.moved.connect(self.on_stream_hover)
        self.stream_display.left_sig.connect(self.on_stream_leave)
        self.stream_display.wheeled.connect(self.on_stream_wheel)

        self._hover_pending = None
        self._hover_timer = QTimer()
        self._hover_timer.setInterval(16)
        self._hover_timer.timeout.connect(self._flush_hover)
        self._hover_timer.start()

        stream_layout.addWidget(self.stream_display, stretch=1)
        self.tabs.addTab(self.stream_widget, "Live Stream")

        # 2. Edit Tab
        self.edit_pane = EditPane(
            apply_pipeline_callback=apply_pipeline,
            get_active_pipeline_callback=lambda: self.active_pipeline,
            get_stage_params_callback=lambda: self.stage_params
        )
        self.tabs.addTab(self.edit_pane, "Video Editor")

        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 4)
        main_layout.addWidget(splitter)
        # ADD the Background Worker Thread instead
        self.stream_worker = StreamWorker(self)
        self.stream_worker.frame_ready.connect(self._on_frame_ready)
        self.stream_worker.error_occurred.connect(self._on_stream_error)
        self.stream_worker.start()
    # -------------------------------------------------------------------------
    # Pipeline Management
    # -------------------------------------------------------------------------
    def add_to_pipeline(self, item):
        self.active_pipeline.append(item.text())
        self.list_pipeline.addItem(item.text())
        self.rebuild_params_ui()
        self.edit_pane.force_refresh()

    def remove_from_pipeline(self, item):
        row = self.list_pipeline.row(item)
        if row >= 0:
            self.list_pipeline.takeItem(row)
            self.active_pipeline.pop(row)
            self.rebuild_params_ui()
            self.edit_pane.force_refresh()

    def rebuild_params_ui(self):
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        unique_stages = list(dict.fromkeys(self.active_pipeline))

        for stage in unique_stages:
            block_cls = BLOCKS._by_name.get(stage)
            if not block_cls or not hasattr(block_cls, "_gui_params"):
                continue

            group = QGroupBox(stage.upper())
            group.setStyleSheet("QGroupBox { color: white; font-weight: bold; border: 1px solid #555; }")
            vbox = QVBoxLayout()

            for info in block_cls._gui_params:
                if stage not in self.stage_params: self.stage_params[stage] = {}
                if info.name not in self.stage_params[stage]: self.stage_params[stage][info.name] = info.default

                cur_val = self.stage_params[stage][info.name]
                val_str = f"{cur_val:.2f}" if info.vtype == float else f"{int(cur_val)}"
                lbl = QLabel(f"{info.name.title()}: {val_str}")
                vbox.addWidget(lbl)

                sld = QSlider(Qt.Orientation.Horizontal)
                mult = 100 if info.vtype == float else 1
                sld.setRange(int(info.min * mult), int(info.max * mult))
                sld.setValue(int(cur_val * mult))

                def update_fn(val, s=stage, n=info.name, m=mult, l=lbl, is_float=(info.vtype == float)):
                    actual_val = val / m
                    self.stage_params[s][n] = actual_val
                    v_str = f"{actual_val:.2f}" if is_float else f"{int(actual_val)}"
                    l.setText(f"{n.title()}: {v_str}")
                    self.edit_pane.force_refresh()  # Update paused video frame instantly

                sld.valueChanged.connect(update_fn)
                vbox.addWidget(sld)

            group.setLayout(vbox)
            self.params_layout.addWidget(group)
        self.params_layout.addStretch()

    # -------------------------------------------------------------------------
    # Stream Logic
    # -------------------------------------------------------------------------
    def handle_connection(self):
        if not self.is_connected:
            selector = WindowSelector(self)
            if selector.exec() == QDialog.DialogCode.Accepted:
                hwnd, title = selector.get_selection()
                if hwnd:
                    try:
                        self.capturer = Win32ClientPrintCapturer(hwnd)
                        self.is_connected = True
                        self.btn_toggle_connect.setText(f"Disconnect: {title[:15]}...")
                        self.btn_toggle_connect.setObjectName("disconnectBtn")
                        self.btn_toggle_connect.setStyle(self.btn_toggle_connect.style())
                    except Exception as e:
                        self.stream_display.setText(f"Capture Error: {e}")
        else:
            self.disconnect_stream()

    def disconnect_stream(self):
        self.capturer = None
        self.is_connected = False
        self._last_src_w = None
        self._last_src_h = None
        self._last_draw_rect = None
        self.btn_toggle_connect.setText("Connect to Window")
        self.btn_toggle_connect.setObjectName("connectBtn")
        self.btn_toggle_connect.setStyle(self.btn_toggle_connect.style())
        self.stream_display.clear()
        self.stream_display.setText("Disconnected")

    # This replaces update_stream_frame
    def _on_frame_ready(self, qt_img: QImage, orig_w: int, original_h: int):
        # This function only handles UI drawing, making it blazing fast
        self._last_src_w = orig_w
        self._last_src_h = original_h

        label_w = max(1, self.stream_display.width())
        label_h = max(1, self.stream_display.height())

        pixmap = QPixmap.fromImage(qt_img)

        # Use FastTransformation instead of SmoothTransformation.
        # It's infinitely faster and perfectly fine for live previews.
        scaled = pixmap.scaled(
            label_w, label_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation
        )

        self.stream_display.setPixmap(scaled)

        x0 = (label_w - scaled.width()) // 2
        y0 = (label_h - scaled.height()) // 2
        self._last_draw_rect = (x0, y0, scaled.width(), scaled.height())

    def _on_stream_error(self, err_msg):
        print(f"Stream Error: {err_msg}")
        self.disconnect_stream()

    def closeEvent(self, event):
        # Ensure the thread shuts down cleanly when you close the app
        self.stream_worker.stop()
        super().closeEvent(event)

    # -------------------------------------------------------------------------
    # Stream Input Handlers
    # -------------------------------------------------------------------------
    def _map_coords(self, lx: int, ly: int) -> tuple[int, int] | None:
        if not self.capturer or not self._last_draw_rect: return None
        x0, y0, pw, ph = self._last_draw_rect
        if lx < x0 or ly < y0 or lx >= x0 + pw or ly >= y0 + ph: return None
        nx, ny = (lx - x0) / float(pw), (ly - y0) / float(ph)
        cx = max(0, min(int(nx * (self._last_src_w - 1)), self._last_src_w - 1))
        cy = max(0, min(int(ny * (self._last_src_h - 1)), self._last_src_h - 1))
        return cx, cy

    def on_stream_press(self, lx: int, ly: int, button: str):
        coords = self._map_coords(lx, ly)
        if coords: _post_mouse_down(self.capturer.hwnd, coords[0], coords[1], button)

    def on_stream_release(self, lx: int, ly: int, button: str):
        coords = self._map_coords(lx, ly)
        if coords: _post_mouse_up(self.capturer.hwnd, coords[0], coords[1], button)

    def on_stream_hover(self, lx: int, ly: int):
        self._hover_pending = (lx, ly)

    def on_stream_leave(self):
        self._hover_pending = None
        if self.capturer:
            try:
                _post_mouse_leave(self.capturer.hwnd)
            except:
                pass

    def _flush_hover(self):
        if not self.capturer or self._hover_pending is None: return
        coords = self._map_coords(self._hover_pending[0], self._hover_pending[1])
        if coords:
            try:
                _post_mouse_move(self.capturer.hwnd, coords[0], coords[1])
            except:
                pass

    def on_stream_wheel(self, lx: int, ly: int, delta_y: int, delta_x: int):
        if not self.capturer: return
        coords = self._map_coords(lx, ly)
        if not coords: return
        cx, cy = coords
        hwnd = self.capturer.hwnd
        try:
            win32gui.SendMessage(hwnd, win32con.WM_MOUSEACTIVATE, 0, 0)
            sx, sy = win32gui.ClientToScreen(hwnd, (cx, cy))
            lparam = ((sy & 0xFFFF) << 16) | (sx & 0xFFFF)
            if delta_y != 0: win32gui.SendMessage(hwnd, win32con.WM_MOUSEWHEEL, ((delta_y & 0xFFFF) << 16), lparam)
            if delta_x != 0: win32gui.SendMessage(hwnd, 0x020E, ((delta_x & 0xFFFF) << 16), lparam)
        except Exception:
            pass


def launch_app():
    app = QApplication(sys.argv)
    window = StudioApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch_app()