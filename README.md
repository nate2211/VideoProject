Nate's Stream Processor is a high-performance, real-time computer vision pipeline designed for Windows. It allows you to capture any active window, apply a series of complex image processing "blocks," and interact with the target window directly through a processed preview.
<img width="1912" height="983" alt="Screenshot 2026-02-16 222205" src="https://github.com/user-attachments/assets/fe210db8-8f5b-42c9-9ea8-c611665541c9" />

The project is built for speed, utilizing pre-allocated memory buffers and OpenCV LUTs (Look-Up Tables) to minimize latency and CPU overhead.
🌟 Features

    Zero-Latency Window Capture: Uses a custom Win32ClientPrintCapturer with pre-allocated BGRA buffers and GDI+ to capture window contents without the memory leaks common in standard Python capture methods.

    Bi-Directional Interaction: Click, drag, scroll, and type directly on the processed preview; the engine maps coordinates back to the source window and injects the input natively.

    Modular Pipeline: Chain effects together using a simple pipe syntax (e.g., realism|anime|vibe).

    Hardware Accelerated Recording: Stream processed results directly to an MP4 file using FFmpeg with NVENC (NVIDIA), QSV (Intel), or AMF (AMD) hardware encoders.

    Pro-Grade FX Suite: Includes motion-reactive vibe effects, digital glitching, and optimized realism enhancers.

🎨 Available Effects (Blocks)
Block	Description	Key Parameters
Ghost	Extracts solid silhouettes from motion and floats them as "spirits" using a temporal buffer.	length, opacity, drift_x/y
Realism	Cinema-grade color grading using LUTs, high-speed unsharp masking, and film grain.	sharpness, warmth, vignette
Anime	Converts live footage to a manga/anime aesthetic using bilateral filtering and adaptive thresholding.	smooth, lines, sat
Vibe	A motion-reactive effect that pulses, shifts hue, and strobes based on the "energy" of movement in the frame.	pulse, hue, energy_gain
Glitch	Simulates digital transmission errors, row shifting, and channel swapping.	chance, intensity
Feedback	Creates psychedelic temporal trails with recursive zooming.	decay, zoom
Cartoon	Simplifies color palettes and adds heavy black outlines.	levels, edges
🚀 Getting Started
Prerequisites

    OS: Windows (Required for Win32 API capture and input injection).

    Python: 3.9+

    Dependencies: opencv-python, numpy, pywin32, PyQt6.

    FFmpeg: (Optional) Required for recording functionality.

Installation

    Clone the repository:
    Bash

    git clone https://github.com/yourusername/gemini-stream-processor.git
    cd gemini-stream-processor

    Install dependencies:
    Bash

    pip install -r requirements.txt

🛠 Usage
1. Launching the GUI

The GUI provides a real-time playground to add blocks, adjust sliders, and interact with the stream.
Bash

python gui.py

    Click Connect to Window.

    Select your target (e.g., a Video Player, Browser, or Game).

    Double-click Available Blocks to add them to your pipeline.

    Interact with the window by clicking/typing directly on the display.

2. Running via CLI

For headless processing or automated recording:
Bash

python main.py stream --window "VLC" --pipeline "realism|mosaic" --extra "mosaic.tile=32" --preview

3. Recording a Stream

To record the processed output using hardware acceleration:
Bash

python main.py stream --window "Browser" --pipeline "anime" --ffmpeg-bin "C:/ffmpeg/bin/ffmpeg.exe" --ffmpeg-out "output.mp4"

🏗 Project Structure

    gui.py: The PyQt6 dashboard, parameter logic, and coordinate mapping.

    stream.py: The engine room. Contains the Win32 capture logic and input injection code.

    blocks.py: The implementation of all image processing algorithms.

    registry.py: Logic for registering and dynamically instantiating blocks.

    block.py: Base classes and decorators for GUI metadata.

🛡 Disclaimer

This tool uses standard Windows API calls (PostMessage) to simulate input. Use responsibly. Some applications with anti-cheat or high-security measures may ignore injected inputs.
📜 License

MIT License. Free for personal and commercial use.
