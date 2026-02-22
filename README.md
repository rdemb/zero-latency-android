# zero-latency-android ⚡

[![Status: Active R&D](https://img.shields.io/badge/Status-Active_R%26D-orange.svg)]()
[![Platform: Android/Termux](https://img.shields.io/badge/Platform-Android%20%7C%20Termux-green.svg)]()
[![Architecture: ARM Cortex/Snapdragon](https://img.shields.io/badge/Architecture-ARM%20Cortex-blue.svg)]()

An experimental edge computing framework designed to force CPython into hard real-time constraints natively on Android mobile SoCs. It features a **Dual-Brain Architecture**: a sub-millisecond deterministic event loop (Fast Brain) running concurrently with a locally hosted, quantized LLM (Slow Brain).

> **Note:** This is an infrastructure and engineering spin-off from a proprietary High-Frequency Trading (HFT) project. It provides the bare-metal OS tuning, IPC, and memory management patterns required to achieve zero-latency telemetry and processing on constrained mobile hardware.

## 🧠 The Dual-Brain Paradigm
Running heavy probabilistic AI models alongside deterministic control loops on a smartphone leads to resource starvation. We solved this by splitting the architecture:

1. **The Fast Brain (Deterministic Core):** A pure-Python event loop executing in sub-milliseconds. It bypasses standard Python overhead by utilizing C-level byte arrays (`array.array`) and zero-copy ring buffers, defeating the Global Interpreter Lock (GIL).
2. **The Slow Brain (Local LLM Oracle):** A background worker running a quantized LLM (e.g., DeepSeek via `llama.cpp`). It processes heavy contextual data (RSS, logs, anomalies) and injects state vectors into the Fast Brain without blocking the primary loop.

## 🔬 Beating Android's Physical Limits (R&D Focus)
Achieving this on a commercial Android device requires bypassing severe OS and hardware bottlenecks. This framework addresses:

* **Float Boxing & GC Latency:** Python's GC causes fatal "Stop-The-World" spikes. We utilize `gc.disable()`, manual collection in "flat" regimes, and C-native `array.array` structures to prevent `PyFloatObject` memory fragmentation.
* **The Phantom Process Killer:** Android 12+ aggressively kills heavy background processes. This documentation includes the ADB overrides (`max_phantom_processes`) required to untether the CPU.
* **Thermal Throttling & Core Pinning:** Heavy LLM inference causes thermal throttling, destroying Fast Brain latency. We explore Linux `taskset` implementations to pin the deterministic loop to prime performance cores while restricting the LLM to efficiency cores.
* **Lock-Free IPC (WIP):** Moving away from `asyncio.Queue` overhead toward `multiprocessing.shared_memory` (LMAX Disruptor pattern) for true nanosecond inter-process communication between the LLM and the main loop.

## 📂 Repository Structure
* `core_loop.py` - The GC-disabled, sub-millisecond deterministic event loop.
* `memory_vault/` - Zero-copy ring buffers and C-array implementations.
* `llm_worker/` - Asynchronous local LLM inference wrappers.
* `android_tuning/` - ADB scripts and `taskset` utilities for OS optimization.

## 🤝 Use Cases
While originally stress-tested for algorithmic market making, this framework is universally applicable to:
* Autonomous Drone Telemetry & Control
* Edge-Robotics Sensor Fusion
* Real-time Medical IoT Anomaly Detection (with LLM diagnostics)

## License
MIT License. See `LICENSE` for details.
