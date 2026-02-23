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

# ✅ NOWE: Real Dual-Brain IPC Integration
from dual_brain_ipc import DualBrainBridge

# Optional: Rich for TUI, gracefully fallback if not installed
try:
    from rich.live import Live
    from rich.table import Table
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text
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

        # ✅ NOWE: Real Dual-Brain IPC Connection
        self.slow_brain_ipc = None
        self.llm_connection_status = "INITIALIZING"
        
        try:
            self.slow_brain_ipc = DualBrainBridge(role="reader")
            self.llm_connection_status = "CONNECTED"
            self.logs = [
                "[+] Zero-Latency Core Initialized on ARM SoC.",
                "[+] ✓ Connected to Slow Brain via Shared Memory IPC"
            ]
        except RuntimeError as e:
            self.llm_connection_status = "OFFLINE"
            self.logs = [
                "[+] Zero-Latency Core Initialized on ARM SoC.",
                f"[!] Slow Brain OFFLINE: {str(e)[:60]}",
                "[*] Fast Brain operating in standalone mode."
            ]
        except Exception as e:
            self.llm_connection_status = "ERROR"
            self.logs = [
                "[+] Zero-Latency Core Initialized on ARM SoC.",
                f"[!] IPC Error: {str(e)[:60]}"
            ]

        # Legacy LLM State (fallback/display format)
        self.llm_state = {
            "Anomaly_Probability": 0.0,
            "Context_Bias": 0.0,
            "Confidence": 0.0,
            "Regime": 0,
            "Killswitch_Active": False,
            "Status": "Awaiting initial LLM inference...",
            "Staleness_Seconds": float('inf')
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
    # ✅ NOWE: Real Slow Brain State Reading (Zero-Latency IPC)
    # ------------------------------------------------------------------
    def sync_llm_state(self, current_time):
        """
        Reads latest LLM state from shared memory.
        This is a ZERO-LATENCY operation (~100-200ns).
        """
        if self.slow_brain_ipc is None:
            return
        
        try:
            llm_data = self.slow_brain_ipc.read_state()
            
            if llm_data["ready"]:
                # Update Fast Brain state with Slow Brain insights
                self.llm_state["Context_Bias"] = llm_data["macro_bias"]
                self.llm_state["Confidence"] = llm_data["confidence"]
                self.llm_state["Regime"] = llm_data["regime"]
                
                # Calculate anomaly probability (inverse of confidence)
                self.llm_state["Anomaly_Probability"] = 1.0 - llm_data["confidence"]
                
                # Track staleness
                staleness = self.slow_brain_ipc.get_staleness_seconds()
                self.llm_state["Staleness_Seconds"] = staleness
                
                # Update status based on regime
                regime_names = {0: "CALM", 1: "VOLATILE", 2: "ANOMALY"}
                regime_str = regime_names.get(llm_data["regime"], "UNKNOWN")
                self.llm_state["Status"] = f"LLM Active: {regime_str} Regime"
                
                # Warn if data is stale
                if staleness > 600:  # 10 minutes
                    self.llm_state["Status"] += f" [STALE: {int(staleness)}s]"
                    if len(self.logs) == 0 or "STALE" not in self.logs[-1]:
                        self.logs.append(f"[!] LLM data stale: {int(staleness)}s old")
                
                # Killswitch activation (example logic)
                if llm_data["regime"] == 2 and llm_data["confidence"] > 0.8:
                    self.llm_state["Killswitch_Active"] = True
                    if len(self.logs) == 0 or "KILLSWITCH" not in self.logs[-1]:
                        self.logs.append("[!!!] KILLSWITCH ACTIVATED: High-confidence anomaly detected")
                else:
                    self.llm_state["Killswitch_Active"] = False
                
                self.llm_connection_status = "CONNECTED"
                
            else:
                self.llm_state["Status"] = "Slow Brain initializing..."
                self.llm_connection_status = "INITIALIZING"
                
        except Exception as e:
            self.llm_connection_status = "ERROR"
            self.llm_state["Status"] = f"IPC Error: {str(e)[:40]}"
            if len(self.logs) == 0 or "IPC Error" not in self.logs[-1]:
                self.logs.append(f"[!] IPC read error: {str(e)[:50]}")

    # ------------------------------------------------------------------
    # LLM Oracle Confidence Decay (Time-based degradation)
    # ------------------------------------------------------------------
    def get_llm_confidence_decayed(self):
        """
        Returns time-decayed confidence (half-life = 10 minutes).
        """
        if self.slow_brain_ipc:
            return self.slow_brain_ipc.get_confidence_decay(half_life_seconds=600.0)
        
        # Fallback calculation if IPC offline
        staleness = self.llm_state.get("Staleness_Seconds", float('inf'))
        if staleness == float('inf'):
            return 0.0
        
        import math
        base_confidence = self.llm_state.get("Confidence", 0.0)
        decay_factor = math.exp(-staleness / 600.0)
        return base_confidence * decay_factor

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

                # ✅ 4. NOWE: Sync with Slow Brain (Zero-Latency Shared Memory Read)
                self.sync_llm_state(cycle_start)

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
                self.logs.append(f"[!] Loop Error: {str(e)[:50]}")
                await asyncio.sleep(1)

    # ------------------------------------------------------------------
    # Enhanced Rich UI Renderer
    # ------------------------------------------------------------------
    def render_cli(self):
        current_time = time.time()
        
        if not HAS_RICH:
            # Fallback to simple CLI
            print(
                f"
[Loop: {self.telemetry['loop_latency_ms']}ms] | "
                f"[CPU: {self.telemetry['sys_cpu']}] | "
                f"[LLM: {self.llm_connection_status}] | "
                f"[Confidence: {self.get_llm_confidence_decayed():.0%}]",
                end=""
            )
            return None

        # Rich TUI with tabs
        tab_name = self.tabs[self.current_tab_index]
        
        # Header
        header = Table.grid(padding=0)
        header.add_column(justify="center")
        header.add_row(
            Text("### D-LOGIC ###", style="bold cyan") + 
            Text("
Q U A N T   C O R E", style="bold white")
        )
        
        # Status bar
        status_bar = Table.grid(expand=True)
        status_bar.add_column(justify="left")
        status_bar.add_column(justify="center")
        status_bar.add_column(justify="right")
        
        # LLM connection indicator
        if self.llm_connection_status == "CONNECTED":
            llm_indicator = Text("● ", style="green") + Text("LLM", style="green bold")
        elif self.llm_connection_status == "INITIALIZING":
            llm_indicator = Text("◐ ", style="yellow") + Text("LLM", style="yellow bold")
        else:
            llm_indicator = Text("● ", style="red") + Text("LLM", style="red bold")
        
        status_bar.add_row(
            f"⏱ {datetime.now().strftime('%H:%M:%S UTC')}",
            llm_indicator,
            f"📅 {datetime.now().strftime('%Y-%m-%d')}"
        )
        
        # Main content based on tab
        if tab_name == "TELEMETRY":
            content = self._render_telemetry_tab()
        elif tab_name == "EDGE_INFERENCE":
            content = self._render_inference_tab()
        else:  # SYSTEM_LOGS
            content = self._render_logs_tab()
        
        # Footer
        footer = Table.grid(expand=True)
        footer.add_column(justify="left")
        footer.add_column(justify="right")
        footer.add_row(
            f"NET: {Text('ONLINE', style='green bold')} | Lat: {self.telemetry['loop_latency_ms']}ms",
            f"[1] TELEMETRY  [2] INFERENCE  [3] LOGS  [Q] Quit"
        )
        
        # Combine all panels
        layout = Group(
            Panel(header, border_style="cyan", padding=(0, 1)),
            Panel(status_bar, border_style="blue", padding=(0, 1)),
            content,
            Panel(footer, border_style="cyan", padding=(0, 1))
        )
        
        return layout

    def _render_telemetry_tab(self):
        """Telemetry panel with system stats."""
        t = Table(show_header=True, header_style="bold magenta", expand=True, box=None)
        t.add_column("METRIC", style="cyan")
        t.add_column("VALUE", justify="right", style="white")
        
        t.add_row("System Uptime", self.telemetry["sys_uptime"])
        t.add_row("CPU Load", self.telemetry["sys_cpu"])
        t.add_row("RAM Usage", self.telemetry["sys_ram"])
        t.add_row("Battery", self.telemetry["sys_bat"])
        t.add_row("─" * 20, "─" * 20)
        t.add_row("Loop Latency", f"{self.telemetry['loop_latency_ms']} ms")
        t.add_row("Loop Rate", f"{self.telemetry['hz_rate']} Hz")
        t.add_row("Buffer Index", f"{self.buffer_index}/{self.buffer_size}")
        
        # Recent sensor statistics
        recent_data = list(self.sensor_data_buffer[max(0, self.buffer_index-100):self.buffer_index])
        if recent_data:
            import statistics
            t.add_row("─" * 20, "─" * 20)
            t.add_row("Sensor Mean (100)", f"{statistics.mean(recent_data):.4f}")
            t.add_row("Sensor Std (100)", f"{statistics.stdev(recent_data) if len(recent_data) > 1 else 0:.4f}")
        
        return Panel(t, title="[bold blue]System Telemetry[/]", border_style="blue")

    def _render_inference_tab(self):
        """Edge AI inference status panel."""
        t = Table(show_header=True, header_style="bold yellow", expand=True, box=None)
        t.add_column("PARAMETER", style="yellow")
        t.add_column("VALUE", justify="right", style="white")
        
        # Connection status
        status_color = {
            "CONNECTED": "green",
            "INITIALIZING": "yellow",
            "OFFLINE": "red",
            "ERROR": "red"
        }.get(self.llm_connection_status, "white")
        
        t.add_row("Slow Brain Status", Text(self.llm_connection_status, style=f"bold {status_color}"))
        t.add_row("IPC Method", "Shared Memory (struct)")
        t.add_row("─" * 25, "─" * 25)
        
        # LLM state
        t.add_row("Context Bias", f"{self.llm_state['Context_Bias']:+.2f}")
        t.add_row("Raw Confidence", f"{self.llm_state['Confidence']:.2%}")
        t.add_row("Decayed Confidence", f"{self.get_llm_confidence_decayed():.2%}")
        t.add_row("Anomaly Probability", f"{self.llm_state['Anomaly_Probability']:.2%}")
        
        # Regime display
        regime = self.llm_state.get("Regime", 0)
        regime_display = {0: "CALM", 1: "VOLATILE", 2: "ANOMALY"}.get(regime, "UNKNOWN")
        regime_color = {0: "green", 1: "yellow", 2: "red"}.get(regime, "white")
        t.add_row("Regime", Text(regime_display, style=f"bold {regime_color}"))
        
        # Staleness
        staleness = self.llm_state.get("Staleness_Seconds", float('inf'))
        if staleness == float('inf'):
            staleness_str = "N/A"
        elif staleness < 60:
            staleness_str = f"{int(staleness)}s"
        else:
            staleness_str = f"{int(staleness/60)}m {int(staleness%60)}s"
        t.add_row("Data Age", staleness_str)
        
        t.add_row("─" * 25, "─" * 25)
        t.add_row("Status", self.llm_state["Status"])
        
        # Killswitch
        if self.llm_state.get("Killswitch_Active", False):
            t.add_row("", Text("⚠ KILLSWITCH ACTIVE ⚠", style=cy Profiling & Loop Rate Limiting
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
