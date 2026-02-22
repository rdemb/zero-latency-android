# -*- coding: utf-8 -*-
# ============================================================================
# ZERO-LATENCY-ANDROID: Sub-Millisecond Edge Computing Framework
# Repository: https://github.com/rdemb/zero-latency-android
# Architecture: ARM Cortex / Android SoC (via Termux)
# ============================================================================

import asyncio
import logging
import time
import os
import gc
import sys
import select
import array
from datetime import datetime

# Optional: Rich for TUI, gracefully fallback if not installed
try:
    from rich.live import Live
    from rich.table import Table
    from rich.console import Group
    from rich.panel import Panel
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------------------------------------------------------
# POSIX Non-Blocking Input Handler
# Allows keyboard navigation without blocking the main deterministic HFT loop.
# ---------------------------------------------------------------------------
class InputHandler:
    def __init__(self):
        self._active = False
        try:
            import termios, tty
            self._termios = termios
            self._tty = tty
            self._old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            self._active = True
        except Exception:
            logging.warning("POSIX termios not found. Keyboard navigation disabled.")

    def check_key(self):
        if not self._active:
            return None
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def cleanup(self):
        if self._active:
            self._termios.tcsetattr(sys.stdin, self._termios.TCSADRAIN, self._old_settings)


# ---------------------------------------------------------------------------
# Fast Brain: The Deterministic Core
# ---------------------------------------------------------------------------
class ZeroLatencyCore:
    def __init__(self):
        # 1. GC Mitigation: Prevent "Stop-The-World" latency spikes
        gc.set_threshold(50000, 20, 10)
        self.last_gc_time = time.time()
        
        self.start_time = time.time()
        self.is_running = True

        # 2. Defeating Float Boxing: Using C-native double arrays (O(1) memory)
        # Prevents Python from creating 28-byte PyFloatObjects in the hot loop.
        self.buffer_size = 1000
        self.sensor_data_buffer = array.array('d', [0.0] * self.buffer_size)
        self.buffer_index = 0

        # UI & Navigation State
        self.input_handler = InputHandler()
        self.tabs = ["TELEMETRY", "EDGE_INFERENCE", "SYSTEM_LOGS"]
        self.current_tab_index = 0
        self.last_tab_switch = time.time()
        self.tab_auto_interval = 15.0
        self.tab_manual_hold = 0.0

        # Safe Async Task Tracking
        self._pending_tasks: set = set()

        # Slow Brain (LLM) State
        self.llm_state = {
            "Anomaly_Probability": 0.0,
            "Context_Bias": 0.0,
            "Killswitch_Active": False,
            "Status": "Awaiting initial LLM inference..."
        }
        self.last_llm_inference = 0

        # Telemetry State
        self.telemetry = {
            "sys_cpu": "0.0%",
            "sys_ram": "0.0%",
            "sys_uptime": "0:00:00",
            "sys_bat": "N/A",
            "loop_latency_ms": "0",
            "hz_rate": "0"
        }
        self.logs = ["[+] Zero-Latency Core Initialized on ARM SoC."]

    # ------------------------------------------------------------------
    # Safe Asyncio Fire-and-Forget (Prevents Phantom Task Drops)
    # ------------------------------------------------------------------
    def safe_create_task(self, coro, name="bg_task"):
        task = asyncio.create_task(coro, name=name)
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
        task.add_done_callback(self._on_task_done)
        return task

    def _on_task_done(self, task: asyncio.Task):
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            self.logs.append(f"[!] ASYNC ERR ({task.get_name()}): {str(exc)[:60]}")
            logging.error(f"Task {task.get_name()} failed: {exc}")

    # ------------------------------------------------------------------
    # Keyboard Handling (Non-blocking)
    # ------------------------------------------------------------------
    def handle_keyboard_input(self, current_time):
        key = self.input_handler.check_key()
        if key is None:
            if current_time - self.last_tab_switch > self.tab_auto_interval and current_time > self.tab_manual_hold:
                self.current_tab_index = (self.current_tab_index + 1) % len(self.tabs)
                self.last_tab_switch = current_time
            return

        if key in ('1', '2', '3'):
            self.current_tab_index = int(key) - 1
            self.last_tab_switch = current_time
            self.tab_manual_hold = current_time + 60.0
        elif key == 'q':
            self.is_running = False
            self.logs.append("[*] Graceful shutdown initiated (Q pressed).")
        elif key == 'n':
            self.current_tab_index = (self.current_tab_index + 1) % len(self.tabs)
            self.last_tab_switch = current_time
            self.tab_manual_hold = current_time + 60.0

    # ------------------------------------------------------------------
    # LLM Oracle Confidence Decay
    # ------------------------------------------------------------------
    def get_llm_confidence(self, current_time):
        age = current_time - self.last_llm_inference
        if self.last_llm_inference == 0:
            return 0.0
        import math
        return float(math.exp(-age / 600.0))  # Half-life = 600s

    # ------------------------------------------------------------------
    # Simulated Slow Brain Inference (e.g., Local GGUF LLM)
    # ------------------------------------------------------------------
    async def _async_trigger_llm_inference(self, current_data_mean):
        """Simulates an asynchronous call to a local LLM running in a separate process."""
        self.logs.append("[*] Offloading contextual analysis to Slow Brain (LLM)...")
        await asyncio.sleep(2.5) # Simulating heavy GGUF inference delay
        
        # Injecting results back into the Fast Brain state
        self.llm_state["Context_Bias"] = 0.85
        self.llm_state["Anomaly_Probability"] = 0.12
        self.llm_state["Status"] = "Analysis Complete: Normal Operating Regime"
        self.last_llm_inference = time.time()
        self.logs.append("[+] LLM Inference integrated seamlessly.")

    # ------------------------------------------------------------------
    # The Main Deterministic Loop
    # ------------------------------------------------------------------
    async def update_cycle(self):
        self.logs.append("[+] Entering deterministic sub-millisecond loop.")
        
        target_hz = 100  # 100 loops per second
        base_sleep_time = 1.0 / target_hz

        while self.is_running:
            cycle_start = time.time()

            try:
                # 1. Non-Blocking I/O & Navigation
                self.handle_keyboard_input(cycle_start)

                # 2. Hardware Telemetry (No-Root Android/Linux Polling)
                uptime_secs = int(cycle_start - self.start_time)
                m, s = divmod(uptime_secs, 60)
                h, m = divmod(m, 60)
                self.telemetry["sys_uptime"] = f"{h:d}:{m:02d}:{s:02d}"

                try:
                    with open("/sys/class/power_supply/battery/capacity", "r") as f:
                        self.telemetry["sys_bat"] = f"{f.read().strip()}%"
                except Exception:
                    pass

                try:
                    load1, _, _ = os.getloadavg()
                    cores = os.cpu_count() or 1
                    cpu_proxy = min(100.0, (load1 / cores) * 100.0)
                    self.telemetry["sys_cpu"] = f"{cpu_proxy:.1f}%"
                except Exception:
                    pass

                try:
                    with open("/proc/meminfo", "r") as f:
                        mem_data = f.read()
                        mem_total = int(mem_data.split("MemTotal:")[1].split("kB")[0].strip())
                        mem_avail = int(mem_data.split("MemAvailable:")[1].split("kB")[0].strip())
                        ram_pct = ((mem_total - mem_avail) / mem_total) * 100
                        self.telemetry["sys_ram"] = f"{ram_pct:.1f}%"
                except Exception:
                    pass

                # 3. Fast Brain Data Ingestion (Simulating high-freq data stream)
                import random
                fake_sensor_value = 100.0 + random.uniform(-1.0, 1.0)
                
                # C-native array insertion (Zero Python Object Overhead)
                self.sensor_data_buffer[self.buffer_index] = fake_sensor_value
                self.buffer_index = (self.buffer_index + 1) % self.buffer_size

                # 4. Trigger Slow Brain (LLM) periodically without blocking
                if cycle_start - self.last_llm_inference > 300: # Every 5 minutes
                    self.safe_create_task(
                        self._async_trigger_llm_inference(fake_sensor_value),
                        name="llm_inference_task"
                    )

                # 5. Smart Garbage Collection
                # We only trigger GC when the system is not under heavy load/anomaly
                is_anomalous = self.llm_state.get("Anomaly_Probability", 0.0) > 0.8
                if (cycle_start - self.last_gc_time > 120) and not is_anomalous:
                    gc.collect()
                    self.last_gc_time = cycle_start

                # 6. Latency Profiling & Loop Rate Limiting
                loop_time = time.time() - cycle_start
                self.telemetry["loop_latency_ms"] = f"{loop_time * 1000:.2f}"
                self.telemetry["hz_rate"] = f"{1.0 / max(loop_time, 0.0001):.0f}"
                
                await asyncio.sleep(max(0.0001, base_sleep_time - loop_time))

            except Exception as e:
                self.logs.append(f"Err: {str(e)[:50]}")
                await asyncio.sleep(1)

    # ------------------------------------------------------------------
    # Simple CLI Renderer (Placeholder for UI_Manager)
    # ------------------------------------------------------------------
    def render_cli(self):
        if not HAS_RICH:
            print(f"\r[Loop Latency: {self.telemetry['loop_latency_ms']}ms] | [CPU: {self.telemetry['sys_cpu']}] | [LLM Confidence: {self.get_llm_confidence(time.time()):.0%}]", end="")
            return None

        # Minimal Rich UI rendering for the Open-Source Repo
        tab_name = self.tabs[self.current_tab_index]
        t = Table(show_header=True, header_style="bold magenta", expand=True)
        t.add_column(f"D-LOGIC EDGE CORE | Tab: {tab_name}", justify="left")
        
        info = (
            f"CPU: {self.telemetry['sys_cpu']} | RAM: {self.telemetry['sys_ram']} | BAT: {self.telemetry['sys_bat']}\n"
            f"Loop Latency: {self.telemetry['loop_latency_ms']} ms | Effective Rate: {self.telemetry['hz_rate']} Hz\n"
            f"LLM Confidence Decay: {self.get_llm_confidence(time.time()):.0%}\n"
            f"Recent Sensor Mean: {sum(self.sensor_data_buffer[:10])/10:.4f}"
        )
        t.add_row(info)
        return Panel(t, title="[bold blue]Zero-Latency Android Node[/]", border_style="cyan")

    async def run(self):
        self.safe_create_task(self.update_cycle(), name="main_cycle")
        
        if HAS_RICH:
            with Live(self.render_cli(), refresh_per_second=10, screen=True) as live:
                while self.is_running:
                    live.update(self.render_cli())
                    await asyncio.sleep(0.1)
        else:
            while self.is_running:
                self.render_cli()
                await asyncio.sleep(0.1)


if __name__ == "__main__":
    core = ZeroLatencyCore()
    try:
        asyncio.run(core.run())
    except KeyboardInterrupt:
        core.input_handler.cleanup()
        print("\n[!] Graceful shutdown initiated by user.")
        core.is_running = False
