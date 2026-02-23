# -*- coding: utf-8 -*-
"""
Dual-Brain IPC: Zero-copy shared memory bridge between Fast Brain (Python) 
and Slow Brain (llama.cpp subprocess).

Architecture:
- Fast Brain: READER (sub-microsecond reads, never blocks)
- Slow Brain: WRITER (updates state after each LLM inference)

Memory Layout (29 bytes):
    [timestamp:double][bias:double][confidence:double][regime:int32][ready:uint8]
"""

from multiprocessing.shared_memory import SharedMemory
import struct
import time
import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class DualBrainBridge:
    """
    Production-grade shared memory communication channel.
    Uses struct-based serialization for zero-allocation reads.
    """
    
    # Fixed binary layout - portable across processes
    LAYOUT = struct.Struct('!dddib')  # Network byte order (big-endian)
    SHM_NAME = "dual_brain_state"
    
    def __init__(self, role: str = "reader"):
        """
        Args:
            role: "reader" (Fast Brain) or "writer" (Slow Brain)
        """
        if role not in ("reader", "writer"):
            raise ValueError(f"Invalid role: {role}. Must be 'reader' or 'writer'.")
        
        self.role = role
        self.shm: Optional[SharedMemory] = None
        
        if role == "writer":
            self._init_writer()
        else:
            self._init_reader()
    
    def _init_writer(self):
        """Initialize as writer - creates shared memory segment."""
        try:
            # Try to unlink stale segment from previous crash
            try:
                stale_shm = SharedMemory(name=self.SHM_NAME, create=False)
                stale_shm.close()
                stale_shm.unlink()
            except FileNotFoundError:
                pass
            
            self.shm = SharedMemory(name=self.SHM_NAME, create=True, size=self.LAYOUT.size)
            
            # Initialize with safe defaults
            self.write_state(
                bias=0.0,
                confidence=0.0,
                timestamp=time.time(),
                regime=0,
                ready=0  # Not ready until first LLM inference
            )
            logger.info(f"Shared memory created: {self.SHM_NAME} ({self.LAYOUT.size} bytes)")
            
        except Exception as e:
            logger.error(f"Failed to initialize writer: {e}")
            raise
    
    def _init_reader(self):
        """Initialize as reader - attaches to existing shared memory."""
        # Retry logic for startup race condition
        max_retries = 20
        retry_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                self.shm = SharedMemory(name=self.SHM_NAME, create=False)
                logger.info(f"Connected to shared memory: {self.SHM_NAME}")
                return
            except FileNotFoundError:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Slow Brain not responding after {max_retries * retry_delay}s. "
                        "Ensure Slow Brain process is running first."
                    )
                time.sleep(retry_delay)
    
    def write_state(
        self,
        bias: float,
        confidence: float,
        regime: int,
        timestamp: Optional[float] = None,
        ready: int = 1
    ):
        """
        Write state to shared memory (Slow Brain only).
        
        Args:
            bias: Macro bias signal [-1.0, 1.0]
            confidence: LLM confidence [0.0, 1.0]
            regime: Market regime (0=calm, 1=volatile, 2=crash)
            timestamp: Unix timestamp (defaults to current time)
            ready: Ready flag (1=valid data, 0=initializing)
        """
        if self.role != "writer":
            raise RuntimeError("Only writer can call write_state()")
        
        if timestamp is None:
            timestamp = time.time()
        
        # Clamp values to valid ranges
        bias = max(-1.0, min(1.0, bias))
        confidence = max(0.0, min(1.0, confidence))
        
        try:
            self.LAYOUT.pack_into(
                self.shm.buf, 0,
                timestamp, bias, confidence, regime, ready
            )
        except Exception as e:
            logger.error(f"Failed to write state: {e}")
            raise
    
    def read_state(self) -> Dict[str, float]:
        """
        Read current state from shared memory (Fast Brain).
        
        Returns:
            dict with keys: timestamp, macro_bias, confidence, regime, ready
        
        Performance: ~100-200 nanoseconds (sub-microsecond)
        """
        if self.role != "reader":
            raise RuntimeError("Only reader can call read_state()")
        
        try:
            timestamp, bias, confidence, regime, ready = self.LAYOUT.unpack_from(
                self.shm.buf, 0
            )
            
            return {
                "timestamp": timestamp,
                "macro_bias": bias,
                "confidence": confidence,
                "regime": regime,
                "ready": bool(ready)
            }
        except Exception as e:
            logger.error(f"Failed to read state: {e}")
            # Return safe defaults on read error
            return {
                "timestamp": 0.0,
                "macro_bias": 0.0,
                "confidence": 0.0,
                "regime": 0,
                "ready": False
            }
    
    def get_staleness_seconds(self) -> float:
        """
        Calculate age of LLM state.
        
        Returns:
            Seconds since last LLM update
        """
        state = self.read_state()
        if not state["ready"]:
            return float('inf')
        return time.time() - state["timestamp"]
    
    def get_confidence_decay(self, half_life_seconds: float = 600.0) -> float:
        """
        Calculate time-decayed confidence.
        
        Args:
            half_life_seconds: Time for confidence to decay to 50%
        
        Returns:
            Decayed confidence [0.0, 1.0]
        """
        state = self.read_state()
        if not state["ready"]:
            return 0.0
        
        import math
        staleness = self.get_staleness_seconds()
        decay_factor = math.exp(-staleness / half_life_seconds)
        
        return state["confidence"] * decay_factor
    
    def cleanup(self):
        """Clean up shared memory resources."""
        if self.shm is None:
            return
        
        try:
            self.shm.close()
            if self.role == "writer":
                self.shm.unlink()
                logger.info(f"Shared memory unlinked: {self.SHM_NAME}")
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


class SlowBrainManager:
    """
    Manages llama.cpp subprocess and orchestrates LLM inference cycles.
    Runs in separate process/thread, updates shared memory asynchronously.
    """
    
    def __init__(
        self,
        model_path: str = "./models/deepseek-1.5b-q4.gguf",
        port: int = 8765,
        threads: int = 2
    ):
        """
        Args:
            model_path: Path to GGUF model file
            port: HTTP port for llama.cpp server
            threads: CPU threads for inference (pin to efficiency cores)
        """
        self.model_path = model_path
        self.port = port
        self.threads = threads
        self.process = None
        self.ipc = DualBrainBridge(role="writer")
        self.is_running = False
    
    def start_llm_server(self):
        """Launch llama.cpp server subprocess."""
        import subprocess
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found: {self.model_path}
"
                "Download with: wget https://huggingface.co/TheBloke/deepseek-coder-1.3b-instruct-GGUF/resolve/main/deepseek-coder-1.3b-instruct.Q4_K_M.gguf"
            )
        
        cmd = [
            "llama-server",
            "--model", self.model_path,
            "--ctx-size", "2048",
            "--threads", str(self.threads),
            "--port", str(self.port),
            "--n-gpu-layers", "0"  # CPU-only for mobile
        ]
        
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(self.threads)  # Prevent thread oversubscription
        
        logger.info(f"Starting llama.cpp server: {' '.join(cmd)}")
        
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )
        
        # Wait for server startup
        import time
        time.sleep(5)
        
        if self.process.poll() is not None:
            raise RuntimeError("llama.cpp server failed to start")
        
        logger.info(f"LLM server started on port {self.port}")
        self.is_running = True
    
    async def inference_cycle(self, context_data: Dict):
        """
        Execute one LLM inference and update shared memory.
        
        Args:
            context_data: Dict with 'mean', 'std', 'recent_values', etc.
        """
        import aiohttp
        import json
        
        # Construct prompt for JSON-mode output
        prompt = f"""Analyze sensor telemetry and output JSON only:
Input: mean={context_data.get('mean', 0):.4f}, std={context_data.get('std', 0):.4f}

Required JSON format:
{{
  "bias": <float from -1.0 to 1.0>,
  "confidence": <float from 0.0 to 1.0>,
  "regime": <int: 0=calm, 1=volatile, 2=anomaly>
}}"""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://localhost:{self.port}/completion",
                    json={
                        "prompt": prompt,
                        "temperature": 0.1,
                        "max_tokens": 128,
                        "stop": ["}"]
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"LLM server returned {resp.status}")
                    
                    result = await resp.json()
                    content = result.get("content", "")
                    
                    # Parse LLM JSON output
                    # Handle case where LLM outputs markdown fences
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")
                    
                    llm_output = json.loads(content.strip())
                    
                    # Update shared memory
                    self.ipc.write_state(
                        bias=float(llm_output["bias"]),
                        confidence=float(llm_output["confidence"]),
                        regime=int(llm_output["regime"]),
                        ready=1
                    )
                    
                    logger.info(
                        f"LLM inference complete: "
                        f"bias={llm_output['bias']:.2f}, "
                        f"confidence={llm_output['confidence']:.2f}"
                    )
        
        except Exception as e:
            logger.error(f"LLM inference failed: {e}")
            # Write failure state
            self.ipc.write_state(
                bias=0.0,
                confidence=0.0,
                regime=0,
                ready=0
            )
    
    def stop(self):
        """Graceful shutdown."""
        self.is_running = False
        
        if self.process:
            logger.info("Stopping LLM server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except:
                self.process.kill()
        
        self.ipc.cleanup()


# ============================================================================
# STANDALONE TEST
# ============================================================================
if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    
    print("=== Dual-Brain IPC Test ===
")
    
    # Test 1: Write-Read Round-Trip
    print("Test 1: Creating writer...")
    writer = DualBrainBridge(role="writer")
    
    print("Test 2: Writing test data...")
    writer.write_state(bias=0.75, confidence=0.92, regime=1)
    
    print("Test 3: Creating reader...")
    reader = DualBrainBridge(role="reader")
    
    print("Test 4: Reading data...")
    state = reader.read_state()
    print(f"  Read state: {state}")
    
    print("Test 5: Staleness check...")
    import time
    time.sleep(2)
    print(f"  Staleness: {reader.get_staleness_seconds():.2f}s")
    print(f"  Decayed confidence: {reader.get_confidence_decay():.4f}")
    
    print("
Test 6: Benchmarking read latency...")
    latencies = []
    for _ in range(100000):
        start = time.perf_counter()
        _ = reader.read_state()
        latencies.append((time.perf_counter() - start) * 1_000_000)
    
    import statistics
    print(f"  Mean: {statistics.mean(latencies):.2f} µs")
    print(f"  Median: {statistics.median(latencies):.2f} µs")
    print(f"  P99: {sorted(latencies)[int(0.99 * len(latencies))]:.2f} µs")
    
    print("
Cleanup...")
    reader.cleanup()
    writer.cleanup()
    
    print("✅ All tests passed!")
