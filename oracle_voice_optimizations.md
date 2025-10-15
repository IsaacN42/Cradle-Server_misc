# Oracle Voice Pipeline - Low-Latency Optimization

## Target Performance
**End-to-end latency**: <1.5 seconds (wake word → spoken response)

This document preserves the original AI assistant optimization concepts for Oracle's voice interface.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  ALEXA GEN 2 HARDWARE (Edge Device)                         │
│  ┌──────────────────────┐                                   │
│  │  Wake Word Detection │  Porcupine: <100ms               │
│  └──────────┬───────────┘                                   │
└─────────────┼───────────────────────────────────────────────┘
              │ Audio Stream (RTP/UDP)
              ▼
┌─────────────────────────────────────────────────────────────┐
│  FATHOM (RTX 3060 Processing)                               │
│  ┌──────────────────────┐                                   │
│  │  Audio Streaming     │  Zero-copy buffers: <50ms        │
│  └──────────┬───────────┘                                   │
│  ┌──────────▼───────────┐                                   │
│  │  Speech-to-Text      │  Whisper quantized: 200-500ms    │
│  └──────────┬───────────┘                                   │
│  ┌──────────▼───────────┐                                   │
│  │  Intent Routing      │  Rule-based fast: <50ms          │
│  └──────────┬───────────┘                                   │
│             ├─────────────┐                                 │
│  ┌──────────▼───────────┐ │                                 │
│  │  Template Response   │ │  Instant: <10ms                │
│  └──────────┬───────────┘ │                                 │
│             │             │                                 │
│  ┌──────────▼───────────┐ │                                 │
│  │  Spider Query        │ │  Fast lookup: <100ms           │
│  └──────────┬───────────┘ │                                 │
│             │             │                                 │
│  ┌──────────▼───────────┐ │                                 │
│  │  LLM Generation      │ │  Llama 3.1 8B: 200-400ms       │
│  └──────────┬───────────┘ │                                 │
│             └─────────────┘                                 │
│  ┌──────────▼───────────┐                                   │
│  │  Text-to-Speech      │  Coqui streaming: 100-300ms      │
│  └──────────┬───────────┘                                   │
└─────────────┼───────────────────────────────────────────────┘
              │ Audio Return (RTP/UDP)
              ▼
┌─────────────────────────────────────────────────────────────┐
│  ALEXA GEN 2 HARDWARE (Speaker Output)                      │
│  Audio playback: <50ms                                      │
└─────────────────────────────────────────────────────────────┘

TOTAL LATENCY: 0.9 - 1.5 seconds
```

---

## Component Optimization

### 1. Wake Word Detection
**Technology**: Porcupine (Picovoice)  
**Target Latency**: <100ms  
**Deployment**: On-device (Alexa Gen 2 hardware)

**Optimization**:
- use ultra-low-latency mode
- optimize acoustic model size
- run with realtime priority on dedicated cpu core
- consider fpga/mcu accelerators for alexa hardware hack

**Alternative**: If Alexa hack infeasible, use USB mic + Porcupine on Fathom

---

### 2. Audio Streaming
**Target Latency**: <50ms  
**Protocol**: RTP/UDP with minimal jitter buffer

**Optimization**:
- zero-copy audio buffers (shared memory if co-located)
- use opus codec tuned for low delay
- minimize network hops (alexa → fathom direct over lan)
- rdma over lan if available
- consider usb audio forwarding for near-zero latency

**Implementation**:
```python
# low-latency audio streaming config
audio_config = {
    'codec': 'opus',
    'bitrate': 24000,  # lower bitrate for speed
    'frame_duration': 20,  # ms
    'complexity': 0,  # fastest encoding
    'buffer_size': 480  # minimal buffer
}
```

---

### 3. Speech-to-Text (ASR)
**Model**: Whisper (quantized)  
**Target Latency**: 200-500ms  
**Hardware**: RTX 3060 with TensorRT

**Optimization**:
- use whisper turbo or small model (not large)
- int8 or int4 quantization with minimal accuracy loss
- tensorrt optimization with kernel fusion
- streaming architecture (partial results)
- audio chunking with overlap-add
- beam search pruning (trade slight accuracy for speed)
- early-exit mechanisms

**Model Selection**:
```
whisper-tiny: fastest (~100ms) but lower accuracy
whisper-base: good balance (~200ms)
whisper-small: better accuracy (~350ms)
whisper-turbo: best for production (~200-300ms)
```

**Implementation**:
```python
# optimized whisper config
whisper_config = {
    'model': 'turbo',
    'device': 'cuda',
    'compute_type': 'int8',
    'beam_size': 1,  # no beam search for speed
    'best_of': 1,
    'temperature': 0,  # deterministic
    'vad_filter': True  # skip silence
}
```

---

### 4. Intent Parsing & Routing
**Target Latency**: <50ms  
**Method**: Rule-based + lightweight ML

**Fast Path (Template Responses)**:
```python
# instant responses for common queries
templates = {
    "what time is it": lambda: f"It's {current_time()}",
    "weather": lambda: weather_api_cached(),
    "server status": lambda: spider.quick_status(),
}
```

**Medium Path (Spider Queries)**:
```python
# fast system lookups
spider_intents = {
    "disk space": spider.get_disk_usage,
    "server tps": spider.get_minecraft_tps,
    "system temperature": spider.get_temps,
}
```

**Slow Path (LLM Generation)**:
```python
# complex queries requiring reasoning
llm_intents = {
    "explain": llama_generate,
    "research": research_db_query,
    "how do i": llama_generate,
}
```

**Optimization**:
- use distilled transformers for intent classification (tinybert)
- cache common intents
- prefer rule-based over ml when possible

---

### 5. LLM Response Generation
**Model**: Llama 3.1 8B (quantized)  
**Target Latency**: 200-400ms  
**Hardware**: RTX 3060 with optimized inference

**Optimization**:
- 4-bit quantization (gptq or awq)
- use exllama or llama.cpp with cuda
- context window pruning (only relevant context)
- batching when possible
- streaming generation (start tts before completion)
- fallback to cached responses for common queries

**Implementation**:
```python
# optimized llama config
llama_config = {
    'model': 'llama-3.1-8b-instruct',
    'quantization': '4bit-gptq',
    'max_tokens': 100,  # short voice responses
    'temperature': 0.7,
    'top_p': 0.9,
    'context_length': 512,  # minimal context for speed
}
```

**Smart Caching**:
```python
# cache frequent query patterns
cache = {
    "what's new in [topic]": recent_research_summary(topic),
    "spider system [query]": spider.cached_response(query),
}
```

---

### 6. Text-to-Speech (TTS)
**Model**: Coqui TTS or Piper  
**Target Latency**: 100-300ms  
**Hardware**: RTX 3060

**Optimization**:
- streaming tts (start speaking before full synthesis)
- use smaller or quantized models
- optimize buffer sizes
- parallel processing with llm (start tts while llm generating)

**Model Options**:
```
Piper: fastest, good quality (~100-200ms)
Coqui TTS: better quality (~200-300ms)
NVIDIA Riva: enterprise option, very fast
```

**Implementation**:
```python
# streaming tts config
tts_config = {
    'model': 'piper',
    'voice': 'en_US-lessac-medium',
    'streaming': True,
    'chunk_size': 512,  # small chunks for low latency
}
```

---

## System & Software Optimization

### Real-Time Priority
```bash
# run voice pipeline with rt priority
chrt -f 99 python oracle_voice.py

# dedicated cpu cores for voice
taskset -c 0-1 python oracle_voice.py
```

### Zero-Copy IPC
```python
# shared memory for audio buffers
import mmap
audio_buffer = mmap.mmap(-1, buffer_size)
```

### GPU Optimization
```python
# pre-load models at startup
models = {
    'whisper': load_whisper(),
    'llama': load_llama(),
    'tts': load_tts()
}
# keep in vram, no swapping
```

### Network Tuning
```bash
# optimize network stack for low latency
sysctl -w net.core.rmem_max=16777216
sysctl -w net.core.wmem_max=16777216
sysctl -w net.ipv4.tcp_low_latency=1
```

---

## Latency Budget Breakdown

| Component | Optimistic | Realistic | Pessimistic |
|-----------|------------|-----------|-------------|
| Wake word | 50ms | 100ms | 150ms |
| Audio transfer | 20ms | 50ms | 100ms |
| ASR (Whisper) | 150ms | 300ms | 500ms |
| Intent routing | 10ms | 50ms | 100ms |
| Response generation | 100ms | 250ms | 500ms |
| TTS | 100ms | 200ms | 400ms |
| Audio return | 20ms | 50ms | 100ms |
| **TOTAL** | **450ms** | **1000ms** | **1850ms** |

**Target**: Stay under 1.5s for 90% of queries

---

## Pipeline Parallelism

### Overlap Processing Stages
```
Traditional:
Wake → [Audio] → [ASR] → [LLM] → [TTS] → Speak
Total: Sequential sum

Optimized:
Wake → [Audio] → [ASR] → [LLM] → [TTS] → Speak
              ↓         ↓       ↓
           Buffer    Intent  Stream
           Next      Route   Start
           Audio     Cache   Speaking
Total: Overlapped, ~40% faster
```

### Streaming Generation
```python
# start tts before llm finishes
async def streaming_response(query):
    llm_stream = llama.generate_stream(query)
    tts_queue = asyncio.Queue()
    
    # parallel tasks
    asyncio.create_task(feed_tts(llm_stream, tts_queue))
    asyncio.create_task(play_audio(tts_queue))
```

---

## Hybrid Response Strategy

### Decision Tree
```
User query
    │
    ├─ Simple factual? → Template (instant)
    │
    ├─ System status? → Spider query (fast)
    │
    ├─ Research topic? → RAG + cache (medium)
    │
    └─ Complex reasoning? → Full LLM (slow)
```

### Example Routing
```python
def route_query(text):
    # instant responses
    if matches_template(text):
        return template_response(text)  # <10ms
    
    # fast spider queries
    if is_system_query(text):
        return spider.query(text)  # <100ms
    
    # cached research
    if in_research_db(text):
        return rag_lookup(text)  # <200ms
    
    # full llm generation
    return llama.generate(text)  # <400ms
```

---

## Hardware Acceleration

### GPU Optimization
- use nvidia tensorrt for model optimization
- kernel fusion for faster inference
- fp16 mixed precision for speed
- model quantization (int8/int4)

### CPU Optimization
- multi-threading for data processing
- simd instructions for audio processing
- numa-aware memory allocation
- cpu pinning for critical threads

### Storage Optimization
- nvme ssd for model loading (fathom's pm981a)
- mmap models for instant access
- ramdisk for temporary audio buffers

---

## Fallback & Error Handling

### Graceful Degradation
```
Primary: Full pipeline (target <1.5s)
Fallback 1: Simplified response (target <1s)
Fallback 2: "Let me look that up" + background processing
Emergency: "I'm having trouble, try again"
```

### Quality vs Speed Tradeoff
```python
# adaptive quality based on response time
if response_time > 2.0:
    switch_to_faster_model()
    reduce_context_length()
    use_more_templates()
```

---

## Testing & Benchmarking

### Latency Testing
```bash
# measure end-to-end latency
python benchmark_voice.py --queries test_queries.txt

# per-component profiling
python profile_pipeline.py --component whisper
```

### Quality Metrics
- word error rate (wer) for asr
- response accuracy for llm
- voice naturalness for tts
- user satisfaction (subjective)

### Continuous Monitoring
```python
# track latency percentiles
metrics = {
    'p50': 1.1s,  # median
    'p90': 1.4s,  # 90th percentile
    'p99': 1.8s   # 99th percentile
}
```

---

## Implementation Priority

### Phase 1: Core Pipeline
1. whisper asr integration
2. llama 3.1 8b inference
3. piper tts synthesis
4. basic audio i/o

### Phase 2: Optimization
1. model quantization
2. streaming generation
3. intent routing
4. template responses

### Phase 3: Hardware
1. alexa gen 2 wake word hack
2. low-latency audio streaming
3. real-time priority tuning
4. gpu optimization

### Phase 4: Polish
1. error handling
2. fallback strategies
3. continuous monitoring
4. adaptive quality

---

## Expected Performance Gains

| Technique | Latency Reduction |
|-----------|-------------------|
| Model quantization | 30-50% |
| Streaming generation | 20-30% |
| Intent routing | 40-60% (cached) |
| Pipeline parallelism | 30-40% |
| Template responses | 90%+ (instant) |

**Combined effect**: 2-3x faster than naive implementation

---

This pipeline design ensures Oracle can respond conversationally with minimal latency while maintaining high quality responses. The key is intelligent routing between fast and slow paths based on query complexity.
