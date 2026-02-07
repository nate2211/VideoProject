from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from collections import deque

from block import BaseBlock, block_param
from registry import BLOCKS


def _clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _screen_blend(base: np.ndarray, add: np.ndarray, amt: float) -> np.ndarray:
    """
    Screen blend: 1 - (1-a)(1-b). Nice for "glow".
    amt: 0..1 controls strength of screening.
    """
    a = base.astype(np.float32) / 255.0
    b = add.astype(np.float32) / 255.0
    scr = 1.0 - (1.0 - a) * (1.0 - b)
    out = a * (1.0 - amt) + scr * amt
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)

@block_param("length", int, 15, 2, 60)
@block_param("opacity", float, 0.9, 0.1, 1.0)  # High opacity for "solid" look
@block_param("threshold", int, 25, 1, 80)  # Sensitivity to movement
@block_param("inflate", float, 0.4, 0.0, 1.0)  # How much to "fatten" the ghost
@block_param("smooth", float, 0.5, 0.0, 1.0)  # Smooth edges
@block_param("red", int, 0, 0, 255)  # Ghost Color R
@block_param("green", int, 255, 0, 255)  # Ghost Color G
@block_param("blue", int, 255, 0, 255)  # Ghost Color B
@block_param("drift_x", float, 0.3, -1.0, 1.0)  # Horizontal float speed
@block_param("drift_y", float, -0.2, -1.0, 1.0)  # Vertical float (negative = up)
@dataclass
class GhostBlock(BaseBlock):
    """
    "Solid Spirit" effect:
      - Extracts solid silhouettes from motion.
      - Fills them with a flat color.
      - Detaches them from the body and floats them around.
    """
    prev: np.ndarray | None = field(default=None, init=False, repr=False)
    buffer: deque = field(default_factory=lambda: deque(maxlen=60), init=False, repr=False)
    frame_i: int = field(default=0, init=False)

    def execute(self, payload: np.ndarray, *, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict]:
        img = payload
        self.frame_i += 1
        h, w = img.shape[:2]

        # Params
        length = int(params.get("length", 15))
        opacity = float(params.get("opacity", 0.9))
        thr = int(params.get("threshold", 25))
        inflate = float(params.get("inflate", 0.4))
        smooth = float(params.get("smooth", 0.5))

        # Color params (BGR for OpenCV)
        c_b = int(params.get("blue", 255))
        c_g = int(params.get("green", 255))
        c_r = int(params.get("red", 0))

        # Float physics
        dx_base = float(params.get("drift_x", 0.3)) * 20.0
        dy_base = float(params.get("drift_y", -0.2)) * 20.0

        # Update buffer size dynamically
        if self.buffer.maxlen != length:
            self.buffer = deque(self.buffer, maxlen=length)

        if self.prev is None or self.prev.shape != img.shape:
            self.prev = img.copy()
            self.buffer.clear()
            # Fill buffer with current frame to avoid index errors at start
            for _ in range(length):
                self.buffer.append(np.zeros_like(img))  # Start with empty ghosts
            return img, {}

        # 1. Detect Motion (Current vs Previous)
        gray_now = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(self.prev, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_now, gray_prev)

        # 2. Threshold to get raw mask
        _, mask = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)

        # 3. Solidify: Dilate + Close + Fill Holes
        # This makes it look like a person, not just outlines.
        k_size = max(3, int(inflate * 20)) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

        # Dilate to merge limbs/body
        solid_mask = cv2.dilate(mask, kernel, iterations=2)
        # Close gaps
        solid_mask = cv2.morphologyEx(solid_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours and fill them to make it truly opaque
        contours, _ = cv2.findContours(solid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(solid_mask)
        cv2.drawContours(filled_mask, contours, -1, (255), thickness=cv2.FILLED)

        # 4. Soften edges (optional, looks better)
        if smooth > 0:
            kb = max(1, int(smooth * 15)) | 1
            filled_mask = cv2.GaussianBlur(filled_mask, (kb, kb), 0)

        # 5. Create the "Solid Color Person"
        colored_ghost = np.zeros_like(img)
        colored_ghost[:] = (c_b, c_g, c_r)  # Fill entire frame with color

        # Mask the color layer (keep color only where the person is)
        # We store the *mask* in the buffer so we can re-colorize it later if needed,
        # but storing the pre-masked image is faster.
        ghost_entity = cv2.bitwise_and(colored_ghost, colored_ghost, mask=filled_mask)

        # Add to trail buffer
        self.buffer.append(ghost_entity)

        # 6. Render: Pick an older ghost from the buffer and float it
        # We pick the oldest item to maximize the "lag" effect
        old_ghost = self.buffer[0]

        # Calculate floating drift (Sine wave + Linear drift)
        # This makes the ghost feel like it's bobbing in water/air
        t = self.frame_i * 0.05
        offset_x = dx_base + np.sin(t) * 10.0
        offset_y = dy_base + np.cos(t * 0.7) * 10.0

        M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        drifted_ghost = cv2.warpAffine(old_ghost, M, (w, h))

        # 7. Composite
        # Since 'drifted_ghost' is black everywhere except the body, we can just add/blend.
        # But we want 'opacity' to control how much it covers the background.

        # Create alpha mask from the drifted ghost image itself
        ghost_gray = cv2.cvtColor(drifted_ghost, cv2.COLOR_BGR2GRAY)
        _, alpha_mask = cv2.threshold(ghost_gray, 1, 255, cv2.THRESH_BINARY)

        alpha = (alpha_mask.astype(np.float32) / 255.0) * opacity

        # Standard Alpha Blend: Out = Ghost * Alpha + BG * (1 - Alpha)
        out = (drifted_ghost.astype(np.float32) * alpha[..., None] +
               img.astype(np.float32) * (1.0 - alpha[..., None]))

        self.prev = img.copy()
        return out.astype(np.uint8), {"type": "solid_spirit"}

@block_param("chance", float, 0.1, 0.0, 0.5)
@block_param("intensity", float, 0.5, 0.0, 1.0)
@dataclass
class GlitchBlock(BaseBlock):
    """
    Digital Artifacts: Randomly shifts rows and swaps color channels.
    """

    def execute(self, payload: np.ndarray, *, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict]:
        chance = float(params.get("chance", 0.1))
        intensity = float(params.get("intensity", 0.5))

        out = payload.copy()
        h, w = out.shape[:2]

        # Random Row Shift
        if np.random.random() < chance:
            num_glitches = int(5 * intensity + 1)
            for _ in range(num_glitches):
                y = np.random.randint(0, h)
                h_glitch = np.random.randint(5, int(20 * intensity + 6))
                shift = np.random.randint(-int(50 * intensity), int(50 * intensity))

                section = out[y:y + h_glitch, :]
                out[y:y + h_glitch, :] = np.roll(section, shift, axis=1)

        # Random Color Swap
        if np.random.random() < (chance * 0.5):
            b, g, r = cv2.split(out)
            out = cv2.merge([r, b, g])

        return out, {"type": "glitch"}


@block_param("decay", float, 0.8, 0.5, 0.99)
@block_param("zoom", float, 1.02, 1.0, 1.1)
@dataclass
class FeedbackBlock(BaseBlock):
    """
    Psychedelic Feedback: Trails that grow and fade, common in 90s videos.
    """
    last_frame: np.ndarray | None = field(default=None, init=False)

    def execute(self, payload: np.ndarray, *, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict]:
        decay = float(params.get("decay", 0.8))
        zoom = float(params.get("zoom", 1.02))

        if self.last_frame is None or self.last_frame.shape != payload.shape:
            self.last_frame = payload.astype(np.float32)
            return payload, {}

        # Slight zoom on the feedback buffer
        h, w = payload.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, zoom)
        feedback = cv2.warpAffine(self.last_frame, M, (w, h))

        # Blend
        blended = payload.astype(np.float32) * (1.0 - decay) + feedback * decay
        self.last_frame = np.clip(blended, 0, 255)

        return self.last_frame.astype(np.uint8), {"type": "feedback"}


@block_param("smooth", float, 0.6, 0.0, 1.0)  # Skin/surface smoothing
@block_param("lines", float, 0.4, 0.0, 1.0)  # Edge line strength
@block_param("sat", float, 0.3, 0.0, 1.0)  # Saturation boost
@block_param("brightness", float, 0.1, -0.5, 0.5)  # Brightness adjust
@dataclass
class AnimeBlock(BaseBlock):
    """
    Anime/Manga style filter:
      - Strong bilateral smoothing (skin texture removal)
      - Dark edge enhancement (pencil lines)
      - Vibrance/Saturation boost
    """

    def execute(self, payload: np.ndarray, *, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict]:
        img = payload

        # Parameters
        smooth_amt = float(params.get("smooth", 0.6))
        line_str = float(params.get("lines", 0.4))
        sat_boost = float(params.get("sat", 0.3))
        bright = float(params.get("brightness", 0.1))

        # 1. Bilateral Filter (Smooths colors but preserves edges)
        # d=9 is standard, sigmaColor higher = more "flat" looking colors
        sigma = 25 + int(smooth_amt * 100)
        filt = cv2.bilateralFilter(img, d=9, sigmaColor=sigma, sigmaSpace=75)

        # 2. Edge Lines (Adaptive Thresholding)
        # Convert to gray, median blur to reduce noise, then adaptive threshold
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 5)

        # Block size 9 or 11 usually works best for lines
        edges = cv2.adaptiveThreshold(
            gray_blur,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            blockSize=9,
            C=9
        )

        # Convert edges to BGR (0=black line, 255=white background)
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Mix edges: interpolate between original smooth image and image with black lines
        # line_str 0.0 -> no lines, 1.0 -> full black lines
        # We use bitwise_and to apply the black lines, then blend
        lined = cv2.bitwise_and(filt, edges_color)
        out = cv2.addWeighted(filt, 1.0 - line_str, lined, line_str, 0)

        # 3. Saturation & Brightness Boost
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Saturation (Channel 1)
        hsv[..., 1] = hsv[..., 1] * (1.0 + sat_boost * 0.8)

        # Value (Channel 2) - Brightness
        hsv[..., 2] = hsv[..., 2] * (1.0 + bright)

        # Clip and convert back
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)

        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return out, {"type": "anime"}


@block_param("sharpness", float, 0.5, 0.0, 1.0)
@block_param("contrast", float, 0.2, 0.0, 0.5)
@block_param("grain", float, 0.15, 0.0, 0.5)
@block_param("warmth", float, 0.1, -0.5, 0.5)
@block_param("vignette", float, 0.3, 0.0, 1.0)
@dataclass
class RealismBlock(BaseBlock):
    """
    Optimized Realism Enhancer:
      - Uses LUTs for instant color/contrast mapping.
      - Pre-allocated buffers for noise/vignette to stop lag.
      - Stays in uint8 mode (no slow float conversion).
    """
    # Cache state to avoid re-allocating heavy arrays
    _cache_w: int = 0
    _cache_h: int = 0
    _vig_map: np.ndarray | None = field(default=None, init=False, repr=False)
    _noise_buf: np.ndarray | None = field(default=None, init=False, repr=False)
    _lut: np.ndarray | None = field(default=None, init=False, repr=False)

    # Track params to only rebuild LUT when needed
    _last_params: Tuple[float, float] = field(default=(-99.0, -99.0), init=False, repr=False)

    def execute(self, payload: np.ndarray, *, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict]:
        img = payload  # Keep as uint8! Don't convert to float.
        h, w = img.shape[:2]

        sharp = float(params.get("sharpness", 0.5))
        contrast = float(params.get("contrast", 0.2))
        grain_amt = float(params.get("grain", 0.15))
        warmth = float(params.get("warmth", 0.1))
        vig_amt = float(params.get("vignette", 0.3))

        # --- 1. Buffer Management (Only runs when size changes) ---
        if (w, h) != (self._cache_w, self._cache_h):
            self._cache_w, self._cache_h = w, h

            # Pre-allocate noise buffer
            self._noise_buf = np.zeros((h, w), dtype=np.uint8)

            # Pre-calculate Vignette Map (fixed scale)
            # We create a 16-bit integer map to allow cleaner multiplication later
            X = np.linspace(-1, 1, w)[None, :]
            Y = np.linspace(-1, 1, h)[:, None]
            radius = np.sqrt(X ** 2 + Y ** 2)
            # Standard vignette curve
            vig_curve = 1.0 - np.clip((radius - 0.5) * 1.5, 0, 1)
            # Store as 0..255 fixed point
            self._vig_map = (vig_curve * 255).astype(np.uint8)
            # Broadcast to 3 channels for cv2.multiply
            self._vig_map = cv2.merge([self._vig_map, self._vig_map, self._vig_map])

        # --- 2. LUT Generation (Only runs when contrast/warmth change) ---
        current_lut_params = (contrast, warmth)
        if current_lut_params != self._last_params:
            self._last_params = current_lut_params

            # Generate look-up table for 0-255
            x = np.arange(256, dtype=np.float32) / 255.0

            # Contrast S-Curve
            if contrast > 0:
                s_curve = x * x * (3 - 2 * x)
                x = x * (1.0 - contrast) + s_curve * contrast

            # Apply to channels (B, G, R)
            # Warmth: Red +, Blue -
            r_curve = np.clip(x + (warmth * 0.2), 0, 1) * 255
            g_curve = np.clip(x, 0, 1) * 255
            b_curve = np.clip(x - (warmth * 0.2), 0, 1) * 255

            # Stack into shape (1, 256, 3) for cv2.LUT
            self._lut = np.dstack((b_curve, g_curve, r_curve)).astype(np.uint8)

        # --- 3. Fast Processing Pipeline ---

        # A. Detail (Unsharp Mask)
        if sharp > 0:
            # Low sigma blur is faster
            blurred = cv2.GaussianBlur(img, (0, 0), 3.0)
            # addWeighted calculates: src1*alpha + src2*beta + gamma
            img = cv2.addWeighted(img, 1.0 + sharp, blurred, -sharp, 0)

        # B. Color & Contrast (LUT)
        # Replaces all complex float math with a simple array lookup
        if self._lut is not None:
            img = cv2.LUT(img, self._lut)

        # C. Film Grain (Optimized)
        if grain_amt > 0:
            # Use OpenCV's optimized RNG filler on pre-allocated buffer
            cv2.randn(self._noise_buf, 0, grain_amt * 50)
            # Subtract 128 to make it signed-like, then add to image
            # Or simpler: just add noise (makes image slightly brighter but faster)
            # Fast way: Add weighted. img = img + noise
            # Reshape noise to 3 channels to match img?
            # Optimization: Add noise to V channel in HSV? Too slow.
            # Fast hack: Add single channel noise to all 3 channels
            noise_rgb = cv2.merge([self._noise_buf, self._noise_buf, self._noise_buf])
            # addWeighted handles clipping automatically
            img = cv2.addWeighted(img, 1.0, noise_rgb, 0.2, -15)

            # D. Vignette (Optimized)
        if vig_amt > 0 and self._vig_map is not None:
            # Blend vignette map based on strength
            # If vig_amt < 1.0, we interpolate the map towards white (255)
            # But calculating that every frame is slow.
            # Faster approach: multiply and mix original.

            # Fast fixed-point multiplication: (img * vig_map) / 255
            # cv2.multiply with scale factor handles this
            vignetted = cv2.multiply(img, self._vig_map, scale=1.0 / 255.0, dtype=cv2.CV_8U)

            # Blend based on strength
            img = cv2.addWeighted(img, 1.0 - vig_amt, vignetted, vig_amt, 0)

        return img, {"type": "realism_fast"}


BLOCKS.register("realism", RealismBlock)
BLOCKS.register("anime", AnimeBlock)
BLOCKS.register("ghost", GhostBlock)
BLOCKS.register("glitch", GlitchBlock)
BLOCKS.register("feedback", FeedbackBlock)

@block_param("tile", int, 16, 2, 128)
@block_param("softness", float, 0.15, 0.0, 1.0)
@dataclass
class MosaicBlock(BaseBlock):
    """Pixel mosaic (cv2-only)."""

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        if not isinstance(payload, np.ndarray):
            raise TypeError("MosaicBlock expects a cv2 frame.")

        tile = max(2, int(params.get("tile", 16)))
        softness = float(params.get("softness", 0.15))

        img = payload
        h, w = img.shape[:2]

        if softness > 0:
            k = max(1, int(softness * 10))
            k = k + 1 if (k % 2 == 0) else k
            img = cv2.GaussianBlur(img, (k, k), 0)

        dw, dh = max(1, w // tile), max(1, h // tile)
        small = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_LINEAR)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        return out, {"type": "mosaic"}


@block_param("k", int, 7, 1, 31)
@block_param("edges", int, 60, 0, 255)
@block_param("levels", int, 12, 2, 64)
@dataclass
class CartoonBlock(BaseBlock):
    """Cartoonize effect."""

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        k = int(params.get("k", 7))
        edges = int(params.get("edges", 60))
        levels = int(params.get("levels", 12))

        img = payload
        # Ensure k is odd for bilateralFilter
        k_odd = max(1, k if k % 2 != 0 else k + 1)
        smooth = cv2.bilateralFilter(img, d=k_odd, sigmaColor=75, sigmaSpace=75)

        gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
        edge = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)

        if edges > 0:
            edge = cv2.Canny(gray, edges, edges * 3)
            edge = cv2.bitwise_not(edge)

        q = max(2, levels)
        step = 256 // q
        quant = (smooth // step) * step + step // 2

        out = cv2.bitwise_and(quant, cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR))
        return out, {"type": "cartoon"}


@block_param("strength", float, 1.0, 0.0, 2.0)
@block_param("strobe", float, 0.35, 0.0, 1.0)
@block_param("split", float, 0.55, 0.0, 1.0)
@block_param("hue", float, 0.45, 0.0, 1.0)
@block_param("poster", float, 0.25, 0.0, 1.0)
@block_param("pulse", float, 0.35, 0.0, 1.0)
@block_param("blur", float, 0.20, 0.0, 1.0)
@block_param("energy_gain", float, 2.0, 0.0, 5.0)
@dataclass
class VibeBlock(BaseBlock):
    """Motion-reactive 'music video' effects."""
    prev_gray: np.ndarray | None = field(default=None, init=False, repr=False)
    energy: float = field(default=0.0, init=False)
    frame_i: int = field(default=0, init=False)

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        img = payload
        self.frame_i += 1

        # UI Params
        strength = float(params.get("strength", 1.0))
        strobe = float(params.get("strobe", 0.35))
        split = float(params.get("split", 0.55))
        hue_amt = float(params.get("hue", 0.45))
        poster = float(params.get("poster", 0.25))
        pulse = float(params.get("pulse", 0.35))
        blur = float(params.get("blur", 0.20))
        energy_gain = float(params.get("energy_gain", 2.0))
        energy_smooth = 0.9  # Hardcoded internal smoothing

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray.copy()

        diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray

        e = (float(np.mean(diff)) / 35.0) * energy_gain
        self.energy = (self.energy * energy_smooth) + (e * (1.0 - energy_smooth))
        E = np.clip(self.energy, 0, 1)

        out = img

        if blur > 0:
            k = max(1, int(blur * 20)) | 1
            out = cv2.GaussianBlur(out, (k, k), 0)

        if pulse > 0:
            amp = (pulse * strength) * (0.03 + 0.07 * E)
            scale = 1.0 + amp * (1.0 if (self.frame_i // 2) % 2 == 0 else -1.0)
            nw, nh = max(2, int(w * scale)), max(2, int(h * scale))
            resized = cv2.resize(out, (nw, nh))
            y0, x0 = (nh - h) // 2, (nw - w) // 2
            out = resized[y0:y0 + h, x0:x0 + w]

        if split > 0:
            px = int((2 + 10 * E) * split * strength)
            if px > 0:
                b, g, r = cv2.split(out)
                out = cv2.merge([np.roll(b, -px, 0), g, np.roll(r, px, 1)])

        if poster > 0:
            lvls = max(2, int(4 + (1.0 - poster) * 24))
            step = 256 // lvls
            out = ((out // step) * step + step // 2).astype(np.uint8)

        if hue_amt > 0:
            hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
            shift = int((20 + 120 * E) * hue_amt * strength)
            hsv[..., 0] = (hsv[..., 0].astype(np.int16) + shift) % 180
            out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        if strobe > 0:
            flash = int(255 * (strobe * strength) * max(0.0, E - 0.25) / 0.75)
            if flash > 0:
                out = cv2.add(out, np.full_like(out, flash, dtype=np.uint8))

        return out, {"type": "vibe"}


# Register all updated blocks
BLOCKS.register("mosaic", MosaicBlock)
BLOCKS.register("cartoon", CartoonBlock)
BLOCKS.register("vibe", VibeBlock)