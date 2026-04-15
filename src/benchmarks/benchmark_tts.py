"""Benchmark: TTS backends para Windows (kokoro-onnx, edge-tts)."""

import platform
import sys
import time
import statistics
import numpy as np

# Test sentences of varying length
SENTENCES = {
    "short": "Hello, how are you doing today?",
    "medium": "I can see you're sitting at your desk. It looks like you're working on something interesting. How can I help you today?",
    "long": (
        "That's a really great question! Let me think about this for a moment. "
        "The history of artificial intelligence goes back to the 1950s, when Alan Turing "
        "first proposed the idea of machines that could think. Since then, we've made "
        "incredible progress, from simple rule-based systems to today's large language models "
        "that can understand and generate human-like text."
    ),
}

VOICE = "af_heart"
SPEED = 1.1
WARMUP = 2
RUNS = 5


def benchmark_kokoro_onnx():
    """Benchmark kokoro-onnx (ONNX Runtime, CPU)."""
    import kokoro_onnx
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download("fastrtc/kokoro-onnx", "kokoro-v1.0.onnx")
    voices_path = hf_hub_download("fastrtc/kokoro-onnx", "voices-v1.0.bin")

    print("Loading kokoro-onnx...")
    t0 = time.time()
    tts = kokoro_onnx.Kokoro(model_path, voices_path)
    print(f"  Loaded in {time.time() - t0:.2f}s")

    results = {}
    for label, text in SENTENCES.items():
        # Warmup
        for _ in range(WARMUP):
            tts.create(text, voice=VOICE, speed=SPEED)

        # Timed runs
        times = []
        audio_duration = None
        for _ in range(RUNS):
            t0 = time.time()
            pcm, sr = tts.create(text, voice=VOICE, speed=SPEED)
            elapsed = time.time() - t0
            times.append(elapsed)
            audio_duration = len(pcm) / sr

        results[label] = {
            "times": times,
            "mean": statistics.mean(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0,
            "min": min(times),
            "audio_sec": audio_duration,
            "rtf": statistics.mean(times) / audio_duration,
            "sample_rate": sr,
        }

    return results


def benchmark_edge_tts():
    """Benchmark edge-tts (Microsoft Azure voices, requires internet)."""
    import asyncio
    import edge_tts
    import io
    import soundfile as sf

    print("Loading edge-tts...")
    
    async def _test():
        communicate = edge_tts.Communicate("Hello", "en-US-GuyNeural")
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                break
    
    t0 = time.time()
    asyncio.get_event_loop().run_until_complete(_test())
    print(f"  Connection test: {time.time() - t0:.2f}s")

    results = {}
    for label, text in SENTENCES.items():
        async def _generate():
            communicate = edge_tts.Communicate(text, "en-US-GuyNeural", rate=f"{int((SPEED-1)*100)}%")
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            return audio_data

        # Warmup
        for _ in range(WARMUP):
            audio_data = asyncio.get_event_loop().run_until_complete(_generate())
        
        # Timed runs
        times = []
        audio_duration = None
        for _ in range(RUNS):
            t0 = time.time()
            audio_data = asyncio.get_event_loop().run_until_complete(_generate())
            elapsed = time.time() - t0
            times.append(elapsed)
            
            # Decode to get duration
            audio, sr = sf.read(io.BytesIO(audio_data))
            audio_duration = len(audio) / sr

        results[label] = {
            "times": times,
            "mean": statistics.mean(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0,
            "min": min(times),
            "audio_sec": audio_duration,
            "rtf": statistics.mean(times) / audio_duration,
            "sample_rate": sr,
        }

    return results


def print_results(name, results):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    for label, r in results.items():
        text = SENTENCES[label]
        print(f"\n  [{label}] ({len(text)} chars)")
        print(f"    Mean:   {r['mean']*1000:7.1f} ms  (+/-{r['stdev']*1000:.1f})")
        print(f"    Min:    {r['min']*1000:7.1f} ms")
        print(f"    Audio:  {r['audio_sec']:7.2f} s")
        print(f"    RTF:    {r['rtf']:7.3f}x  (< 1.0 = faster than real-time)")
        print(f"    SR:     {r['sample_rate']} Hz")


if __name__ == "__main__":
    is_windows = sys.platform == "win32"

    print("=" * 60)
    print("  TTS Benchmark: Windows Edition")
    print(f"  Warmup: {WARMUP} runs, Measured: {RUNS} runs")
    print("=" * 60)

    # Benchmark kokoro-onnx
    try:
        onnx_results = benchmark_kokoro_onnx()
        print_results("kokoro-onnx (ONNX Runtime, CPU)", onnx_results)
    except Exception as e:
        print(f"\n  kokoro-onnx failed: {e}")
        onnx_results = None

    # Benchmark edge-tts (requires internet)
    try:
        edge_results = benchmark_edge_tts()
        print_results("edge-tts (Microsoft Azure, requires internet)", edge_results)
    except Exception as e:
        print(f"\n  edge-tts failed: {e}")
        edge_results = None

    # Comparison
    if onnx_results and edge_results:
        print(f"\n{'=' * 60}")
        print(f"  Comparison: kokoro-onnx vs edge-tts")
        print(f"{'=' * 60}")
        for label in SENTENCES:
            onnx_mean = onnx_results[label]["mean"]
            edge_mean = edge_results[label]["mean"]
            ratio = onnx_mean / edge_mean
            print(f"  [{label}]  kokoro: {onnx_mean*1000:.0f}ms, edge: {edge_mean*1000:.0f}ms  ({ratio:.2f}x)")
