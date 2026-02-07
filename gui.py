import sys
import cv2
import numpy as np
import win32con
import win32gui

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QGroupBox, QSlider, QLabel,
    QScrollArea, QSplitter, QDialog, QListWidgetItem, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

# Logic imports
from registry import BLOCKS
import blocks
# Make sure stream.py has these functions defined (from previous steps)
from stream import (
    Win32ClientPrintCapturer, apply_pipeline,
    _post_mouse_move, _post_mouse_leave, _post_mouse_wheel,
    _post_key, _post_mouse_down, _post_mouse_up
)


# =============================================================================
# Clickable preview label (Handles Mouse & Keyboard)
# =============================================================================

class ClickableLabel(QLabel):
    # Signals for discrete input states
    pressed = pyqtSignal(int, int, str)  # x, y, button
    released = pyqtSignal(int, int, str)  # x, y, button
    moved = pyqtSignal(int, int)  # x, y
    left = pyqtSignal()
    wheeled = pyqtSignal(int, int, int, int)

    def __init__(self, parent_gui, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent_gui = parent_gui
        self.setMouseTracking(True)
        # StrongFocus allows this widget to capture Keyboard events when hovered/clicked
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def enterEvent(self, ev):
        self.setFocus()
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        self.clearFocus()
        self.left.emit()
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
        main_win_title = "Gemini Stream Processor"

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


# =============================================================================
# Main GUI
# =============================================================================

class StreamGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gemini Stream Processor")

        self.active_pipeline = []
        self.stage_params = {"all": {}}
        self.capturer: Win32ClientPrintCapturer | None = None
        self.is_connected = False

        # State for coordinate mapping
        self._last_src_w: int | None = None
        self._last_src_h: int | None = None
        self._last_draw_rect: tuple[int, int, int, int] | None = None  # x0,y0,pw,ph

        self._wheel_accum_y = 0
        self._wheel_accum_x = 0

        self._init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def _init_ui(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QLabel { color: #ffffff; font-family: 'Segoe UI'; }
            QGroupBox { color: #aaaaaa; border: 1px solid #333; margin-top: 10px; padding-top: 10px; font-weight: bold; }
            QPushButton { background-color: #333; color: white; border: none; padding: 10px; font-weight: bold; border-radius: 2px; }
            QPushButton:hover { background-color: #444; }
            QPushButton#connectBtn { background-color: #2d5a27; }
            QPushButton#disconnectBtn { background-color: #8b0000; }
            QListWidget { background-color: #252526; color: white; border: 1px solid #333; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- SIDEBAR ---
        sidebar = QWidget()
        side_layout = QVBoxLayout(sidebar)

        self.btn_toggle_connect = QPushButton("Connect to Window")
        self.btn_toggle_connect.setObjectName("connectBtn")
        self.btn_toggle_connect.clicked.connect(self.handle_connection)
        side_layout.addWidget(self.btn_toggle_connect)

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

        # --- VIEWPORT ---
        self.display = ClickableLabel(self, "No active stream")
        self.display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.display.setStyleSheet("background-color: #000; border: 2px solid #111;")
        self.display.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

        # Connect Input Signals (Press/Release for Dragging)
        self.display.pressed.connect(self.on_stream_press)
        self.display.released.connect(self.on_stream_release)
        self.display.moved.connect(self.on_stream_hover)
        self.display.left.connect(self.on_stream_leave)
        self.display.wheeled.connect(self.on_stream_wheel)

        self._hover_pending = None
        self._hover_timer = QTimer()
        self._hover_timer.setInterval(16)  # ~60hz
        self._hover_timer.timeout.connect(self._flush_hover)
        self._hover_timer.start()

        splitter.addWidget(self.display)
        splitter.setStretchFactor(1, 4)
        main_layout.addWidget(splitter)

    # -------------------------------------------------------------------------
    # Connection / Pipeline
    # -------------------------------------------------------------------------

    def handle_connection(self):
        if not self.is_connected:
            selector = WindowSelector(self)
            if selector.exec() == QDialog.DialogCode.Accepted:
                hwnd, title = selector.get_selection()
                if hwnd:
                    try:
                        self.capturer = Win32ClientPrintCapturer(hwnd)
                        self.timer.start(33)
                        self.is_connected = True
                        self.btn_toggle_connect.setText(f"Disconnect: {title[:15]}...")
                        self.btn_toggle_connect.setObjectName("disconnectBtn")
                        self.btn_toggle_connect.setStyle(self.btn_toggle_connect.style())
                    except Exception as e:
                        self.display.setText(f"Capture Error: {e}")
        else:
            self.disconnect_stream()

    def disconnect_stream(self):
        self.timer.stop()
        self.capturer = None
        self.is_connected = False
        self._last_src_w = None
        self._last_src_h = None
        self._last_draw_rect = None
        self.btn_toggle_connect.setText("Connect to Window")
        self.btn_toggle_connect.setObjectName("connectBtn")
        self.btn_toggle_connect.setStyle(self.btn_toggle_connect.style())
        self.display.clear()
        self.display.setText("Disconnected")

    def add_to_pipeline(self, item):
        self.active_pipeline.append(item.text())
        self.list_pipeline.addItem(item.text())
        self.rebuild_params_ui()

    def remove_from_pipeline(self, item):
        row = self.list_pipeline.row(item)
        if row >= 0:
            self.list_pipeline.takeItem(row)
            self.active_pipeline.pop(row)
            self.rebuild_params_ui()

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
            group.setStyleSheet(
                "QGroupBox { color: black; font-weight: bold; border: 1px solid #555; margin-top: 10px; padding-top: 10px; }")
            vbox = QVBoxLayout()
            vbox.setSpacing(2)

            for info in block_cls._gui_params:
                if stage not in self.stage_params: self.stage_params[stage] = {}
                if info.name not in self.stage_params[stage]: self.stage_params[stage][info.name] = info.default

                cur_val = self.stage_params[stage][info.name]
                val_str = f"{cur_val:.2f}" if info.vtype == float else f"{int(cur_val)}"
                lbl = QLabel(f"{info.name.title()}: {val_str}")
                lbl.setStyleSheet("color: black; font-weight: bold; font-size: 11pt; margin-bottom: 0px;")
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

                sld.valueChanged.connect(update_fn)
                vbox.addWidget(sld)

            group.setLayout(vbox)
            self.params_layout.addWidget(group)
        self.params_layout.addStretch()

    # -------------------------------------------------------------------------
    # Coordinate Mapping Helper (Safe Letterboxing)
    # -------------------------------------------------------------------------
    def _map_coords(self, lx: int, ly: int) -> tuple[int, int] | None:
        """
        Maps local label coordinates (lx, ly) to the target window's client coordinates.
        Handles letterboxing/centering offset. Returns None if out of bounds or no stream.
        """
        if not self.capturer:
            return None
        if self._last_src_w is None or self._last_src_h is None or self._last_draw_rect is None:
            return None

        x0, y0, pw, ph = self._last_draw_rect

        # If click is in the black bars, ignore
        if lx < x0 or ly < y0 or lx >= x0 + pw or ly >= y0 + ph:
            return None

        # 1. Normalize (0.0 to 1.0) relative to the drawn image
        nx = (lx - x0) / float(pw)
        ny = (ly - y0) / float(ph)

        # 2. Map to source window resolution
        cx = int(nx * (self._last_src_w - 1))
        cy = int(ny * (self._last_src_h - 1))

        # 3. Clamp to window bounds (safety)
        cx = max(0, min(cx, self._last_src_w - 1))
        cy = max(0, min(cy, self._last_src_h - 1))

        return cx, cy

    # -------------------------------------------------------------------------
    # Input Event Handlers
    # -------------------------------------------------------------------------

    def on_stream_press(self, lx: int, ly: int, button: str):
        coords = self._map_coords(lx, ly)
        if coords:
            try:
                _post_mouse_down(self.capturer.hwnd, coords[0], coords[1], button)
            except Exception as e:
                print(f"Mouse Down Failed: {e}")

    def on_stream_release(self, lx: int, ly: int, button: str):
        coords = self._map_coords(lx, ly)
        # Even if mouse is outside bounds now, release where it is (clamped) to finish drags
        if coords:
            try:
                _post_mouse_up(self.capturer.hwnd, coords[0], coords[1], button)
            except Exception as e:
                print(f"Mouse Up Failed: {e}")

    def on_stream_hover(self, lx: int, ly: int):
        self._hover_pending = (lx, ly)

    def on_stream_leave(self):
        self._hover_pending = None
        if self.capturer:
            try:
                _post_mouse_leave(self.capturer.hwnd)
            except Exception:
                pass

    def _flush_hover(self):
        if not self.capturer or self._hover_pending is None:
            return

        coords = self._map_coords(self._hover_pending[0], self._hover_pending[1])
        if coords:
            try:
                _post_mouse_move(self.capturer.hwnd, coords[0], coords[1])
            except Exception:
                pass

    def on_stream_wheel(self, lx: int, ly: int, delta_y: int, delta_x: int):
        if not self.capturer:
            return

        coords = self._map_coords(lx, ly)
        if coords is None:
            return

        cx, cy = coords
        hwnd = self.capturer.hwnd

        try:
            # 1. Activate the window "silently" without stealing focus from your GUI
            # MA_NOACTIVATE eats the click but tells the window to process mouse events
            win32gui.SendMessage(hwnd, win32con.WM_MOUSEACTIVATE, 0, 0)

            # 2. WM_MOUSEWHEEL requires SCREEN coordinates in the lParam, NOT Client coords
            screen_point = win32gui.ClientToScreen(hwnd, (cx, cy))
            sx, sy = screen_point

            # Pack coordinates: (Y << 16) | X
            # Note: Python handles large integers automatically, but we mask for safety
            lparam = ((sy & 0xFFFF) << 16) | (sx & 0xFFFF)

            # 3. Keys state (MK_CONTROL, MK_SHIFT etc). 0 is fine for basic scrolling.
            keys = 0

            # 4. Send Vertical Scroll
            if delta_y != 0:
                # delta is in multiples of 120. High word is delta, low word is keys.
                wparam = ((delta_y & 0xFFFF) << 16) | (keys & 0xFFFF)
                win32gui.SendMessage(hwnd, win32con.WM_MOUSEWHEEL, wparam, lparam)

            # 5. Send Horizontal Scroll
            if delta_x != 0:
                wparam = ((delta_x & 0xFFFF) << 16) | (keys & 0xFFFF)
                # WM_MOUSEHWHEEL = 0x020E
                win32gui.SendMessage(hwnd, 0x020E, wparam, lparam)

        except Exception as e:
            print(f"Scroll Error: {e}")
    # -------------------------------------------------------------------------
    # Frame Update
    # -------------------------------------------------------------------------

    def update_frame(self):
        if not self.capturer:
            return

        try:
            frame = self.capturer.grab()
            if frame is None:
                return

            processed = apply_pipeline(frame, self.active_pipeline, self.stage_params)

            rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w

            self._last_src_w = w
            self._last_src_h = h

            qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img)

            label_w = max(1, self.display.width())
            label_h = max(1, self.display.height())

            scaled = pixmap.scaled(
                label_w, label_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            self.display.setPixmap(scaled)

            # Store rect for coordinate mapping
            x0 = (label_w - scaled.width()) // 2
            y0 = (label_h - scaled.height()) // 2
            self._last_draw_rect = (x0, y0, scaled.width(), scaled.height())

        except Exception as e:
            print(f"Frame Error: {e}")
            self.disconnect_stream()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StreamGUI()
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec())