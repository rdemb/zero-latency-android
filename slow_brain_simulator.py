# -*- coding: utf-8 -*-
"""
Synthetic Slow Brain Simulator
Mimics llama.cpp behavior without requiring actual LLM installation.
Generates realistic inference latencies and contextual responses.
"""

import asyncio
import time
import random
import logging
import math
from dual_brain_ipc import DualBrainBridge, SlowBrainManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [SLOW_BRAIN] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SyntheticSlowBrain:
    """
    Realistic LLM simulator for testing without actual model installation.
    
    Features:
    - Simulates inference latency (2-5s based on thermal state)
    - Generates contextual responses based on Fast Brain telemetry
    - Models regime detection (calm/volatile/anomaly)
    - Confidence decay over time
    - Thermal throttling behavior
    """
    
    def __init__(
        self,
        base_inference_time: float = 3.0,
        thermal_throttle_multiplier: float = 1.5
    ):
        self.ipc = DualBrainBridge(role="writer")
        self.is_running = False
        
        # Simulation parameters
        self.base_inference_time = base_inference_time
        self.thermal_throttle_multiplier = thermal_throttle_multiplier
        
        # State tracking
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.current_thermal_state = "NORMAL"  # NORMAL, WARM, HOT
        
        # Statistical history for pattern detection
        self.value_history = []
        self.max_history = 1000
        
        logger.info("Synthetic Slow Brain initialized")
        logger.info(f"Base inference time: {base_inference_time}s")
    
    def _simulate_thermal_state(self) -> str:
        """
        Simulate thermal throttling based on inference count.
        On real hardware, this would read from thermal sensors.
        """
        # Every 10 inferences, increase thermal pressure
        thermal_pressure = (self.inference_count % 30) / 30.0
        
        if thermal_pressure < 0.3:
            return "NORMAL"
        elif thermal_pressure < 0.7:
            return "WARM"
        else:
            return "HOT"
    
    def _calculate_inference_latency(self) -> float:
        """
        Calculate realistic inference latency based on thermal state.
        """
        self.current_thermal_state = self._simulate_thermal_state()
        
        base = self.base_inference_time
        
        if self.current_thermal_state == "WARM":
            multiplier = 1.3
        elif self.current_thermal_state == "HOT":
            multiplier = self.thermal_throttle_multiplier
        else:
            multiplier = 1.0
        
        # Add random jitter (±20%)
        jitter = random.uniform(0.8, 1.2)
        
        return base * multiplier * jitter
    
    def _analyze_context(self, sensor_data: list) -> dict:
        """
        Simulate LLM contextual analysis.
        Uses statistical methods to mimic what an LLM would infer.
        
        Returns:
            dict: {bias, confidence, regime}
        """
        if len(sensor_data) < 10:
            return {
                "bias": 0.0,
                "confidence": 0.3,
                "regime": 0
            }
        
        # Calculate statistics
        mean_val = sum(sensor_data) / len(sensor_data)
        variance = sum((x - mean_val) ** 2 for x in sensor_data) / len(sensor_data)
        std_dev = math.sqrt(variance)
        
        # Detect trend (last 20 vs previous 20)
        if len(sensor_data) >= 40:
            recent = sensor_data[-20:]
            older = sensor_data[-40:-20]
            recent_mean = sum(recent) / len(recent)
            older_mean = sum(older) / len(older)
            trend = (recent_mean - older_mean) / (std_dev + 0.01)
        else:
            trend = 0.0
        
        # Regime detection
        volatility = std_dev / (abs(mean_val) + 1.0)
        
        if volatility < 0.01:
            regime = 0  # CALM
            confidence = 0.9
        elif volatility < 0.05:
            regime = 1  # VOLATILE
            confidence = 0.7
        else:
            regime = 2  # ANOMALY
            confidence = 0.85
        
        # Bias calculation (-1 to +1)
        # Positive bias = uptrend, Negative = downtrend
        bias = max(-1.0, min(1.0, trend))
        
        # Add realistic uncertainty
        confidence *= random.uniform(0.9, 1.0)
        
        return {
            "bias": bias,
            "confidence": confidence,
            "regime": regime
        }
    
    async def inference_cycle(self, context_data: dict):
        """
        Execute one simulated inference cycle.
        
        Args:
            context_data: Dict with sensor statistics from Fast Brain
        """
        self.inference_count += 1
        
        # Extract sensor data
        sensor_values = context_data.get("recent_values", [])
        self.value_history.extend(sensor_values)
        self.value_history = self.value_history[-self.max_history:]
        
        logger.info(f"Starting inference #{self.inference_count}")
        logger.info(f"Thermal state: {self.current_thermal_state}")
        
        # Calculate realistic latency
        inference_time = self._calculate_inference_latency()
        
        start_time = time.time()
        
        # Simulate heavy computation
        await asyncio.sleep(inference_time)
        
        # Perform contextual analysis
        analysis = self._analyze_context(self.value_history)
        
        actual_time = time.time() - start_time
        self.total_inference_time += actual_time
        
        # Write to shared memory
        self.ipc.write_state(
            bias=analysis["bias"],
            confidence=analysis["confidence"],
            regime=analysis["regime"],
            timestamp=time.time(),
            ready=1
        )
        
        # Log results
        logger.info(
            f"Inference complete in {actual_time:.2f}s | "
            f"Bias: {analysis['bias']:+.2f} | "
            f"Confidence: {analysis['confidence']:.0%} | "
            f"Regime: {analysis['regime']}"
        )
        
        avg_time = self.total_inference_time / self.inference_count
        logger.info(f"Average inference time: {avg_time:.2f}s")
    
    async def run(self, inference_interval: float = 30.0):
        """
        Main loop - runs periodic inferences.
        
        Args:
            inference_interval: Seconds between inferences
        """
        self.is_running = True
        logger.info(f"Starting inference loop (interval: {inference_interval}s)")
        
        # Initial state write
        self.ipc.write_state(
            bias=0.0,
            confidence=0.0,
            regime=0,
            ready=0  # Not ready yet
        )
        
        while self.is_running:
            try:
                # Simulate gathering context from Fast Brain
                # In real implementation, this would query Fast Brain's buffer
                context = {
                    "recent_values": [100.0 + random.uniform(-1.0, 1.0) for _ in range(100)],
                    "mean": 100.0,
                    "std": random.uniform(0.1, 2.0)
                }
                
                await self.inference_cycle(context)
                
                # Wait for next cycle
                await asyncio.sleep(inference_interval)
                
            except Exception as e:
                logger.error(f"Inference failed: {e}")
                await asyncio.sleep(5)
    
    def stop(self):
        """Graceful shutdown."""
        self.is_running = False
        logger.info("Stopping Slow Brain...")
        self.ipc.cleanup()
        logger.info(f"Total inferences: {self.inference_count}")
        logger.info(f"Total compute time: {self.total_inference_time:.1f}s")


# ============================================================================
# SCENARIO TESTING FRAMEWORK
# ============================================================================

class ScenarioTester:
    """
    Automated testing of different market/sensor scenarios.
    """
    
    def __init__(self):
        self.slow_brain = SyntheticSlowBrain(base_inference_time=2.0)
    
    async def test_calm_regime(self):
        """Test behavior during calm/stable conditions."""
        logger.info("\n" + "="*60)
        logger.info("SCENARIO 1: CALM REGIME (Low Volatility)")
        logger.info("="*60)
        
        context = {
            "recent_values": [100.0 + random.uniform(-0.1, 0.1) for _ in range(100)],
            "mean": 100.0,
            "std": 0.05
        }
        
        await self.slow_brain.inference_cycle(context)
        
        # Verify results
        state = self.slow_brain.ipc.read_state()
        assert state["regime"] == 0, "Expected CALM regime"
        logger.info("✅ CALM regime correctly detected")
    
    async def test_volatile_regime(self):
        """Test behavior during volatile conditions."""
        logger.info("\n" + "="*60)
        logger.info("SCENARIO 2: VOLATILE REGIME (Medium Volatility)")
        logger.info("="*60)
        
        context = {
            "recent_values": [100.0 + random.uniform(-2.0, 2.0) for _ in range(100)],
            "mean": 100.0,
            "std": 1.5
        }
        
        await self.slow_brain.inference_cycle(context)
        
        state = self.slow_brain.ipc.read_state()
        assert state["regime"] == 1, "Expected VOLATILE regime"
        logger.info("✅ VOLATILE regime correctly detected")
    
    async def test_anomaly_regime(self):
        """Test behavior during anomalous conditions."""
        logger.info("\n" + "="*60)
        logger.info("SCENARIO 3: ANOMALY REGIME (High Volatility)")
        logger.info("="*60)
        
        # Create anomalous pattern
        values = [100.0] * 50 + [110.0 + random.uniform(-5.0, 5.0) for _ in range(50)]
        
        context = {
            "recent_values": values,
            "mean": 105.0,
            "std": 4.5
        }
        
        await self.slow_brain.inference_cycle(context)
        
        state = self.slow_brain.ipc.read_state()
        assert state["regime"] == 2, "Expected ANOMALY regime"
        logger.info("✅ ANOMALY regime correctly detected")
    
    async def test_thermal_throttling(self):
        """Test thermal throttling behavior."""
        logger.info("\n" + "="*60)
        logger.info("SCENARIO 4: THERMAL THROTTLING")
        logger.info("="*60)
        
        latencies = []
        
        for i in range(3):
            self.slow_brain.inference_count = i * 10
            start = time.time()
            
            context = {
                "recent_values": [100.0] * 100,
                "mean": 100.0,
                "std": 0.5
            }
            
            await self.slow_brain.inference_cycle(context)
            latencies.append(time.time() - start)
        
        logger.info(f"Latencies: {[f'{l:.2f}s' for l in latencies]}")
        assert latencies[-1] > latencies[0], "Expected increasing latency with thermal pressure"
        logger.info("✅ Thermal throttling correctly simulated")
    
    async def run_all_scenarios(self):
        """Run complete test suite."""
        logger.info("\n" + "#"*60)
        logger.info("# SYNTHETIC SLOW BRAIN - SCENARIO TEST SUITE")
        logger.info("#"*60 + "\n")
        
        await self.test_calm_regime()
        await self.test_volatile_regime()
        await self.test_anomaly_regime()
        await self.test_thermal_throttling()
        
        logger.info("\n" + "="*60)
        logger.info("✅ ALL SCENARIOS PASSED")
        logger.info("="*60)
        
        self.slow_brain.stop()


# ============================================================================
# CLI INTERFACE
# ============================================================================

async def main():
    """Main entry point."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Run scenario tests
        tester = ScenarioTester()
        await tester.run_all_scenarios()
    else:
        # Run continuous simulation
        slow_brain = SyntheticSlowBrain(base_inference_time=3.0)
        
        try:
            await slow_brain.run(inference_interval=30.0)
        except KeyboardInterrupt:
            logger.info("\n[!] Shutdown signal received")
            slow_brain.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[!] Graceful shutdown")
