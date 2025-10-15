# Oracle - Dual-Mode AI Assistant & Research System

## System Overview
Oracle is a unified AI system that operates in two modes:
- **Active Mode**: Low-latency voice assistant (when you're home/available)
- **Passive Mode**: Autonomous researcher (idle time, work/school hours)

The system learns and improves continuously, becoming more personalized through passive research and interaction tracking.

---

## Operating Modes

### Active Mode - Voice Assistant
**Trigger**: Wake word detected OR manual activation  
**Response Time Target**: <1.5s end-to-end  
**Hardware**: Alexa Gen 2 → Fathom (RTX 3060) → Alexa speaker

**Capabilities**:
- answer questions conversationally
- query spider for system diagnostics
- control home assistant devices
- provide research summaries from passive learning
- remember conversation context

**Pipeline**:
```
Alexa wake word → Whisper ASR → Llama 3.1 8B → Coqui TTS → Alexa speaker
                           ↓
                    queries spider for system info
                    accesses research knowledge base
```

### Passive Mode - AI Researcher
**Trigger**: System idle (30+ min) OR scheduled hours (work/school)  
**Resource Usage**: Full RTX 3060 + 6 CPU cores + 16GB RAM

**Process**:
- scrape reddit, twitter, hacker news, arxiv, news
- summarize and cluster information using vector embeddings
- build personalized knowledge base
- track topics you ask about most
- self-improvement: refine research focus based on interactions

**Output**:
- daily/weekly digest reports
- vector database (faiss embeddings)
- topic trend analysis
- ready-to-answer questions from research

---

## Self-Improvement Loop

### Learning from Voice Interactions
```
You: "Hey Oracle, what's new in LLM research?"
Oracle: [provides summary from passive research]
You: "Tell me more about that quantization technique"
Oracle: [detailed explanation]
→ System logs: user interested in quantization
→ Adjusts: future research prioritizes quantization topics
```

### Passive Research Refinement
- tracks what topics you ask about most frequently
- adjusts scraping keywords and sources dynamically
- identifies gaps in knowledge base
- experiments with different summarization approaches
- measures: answer helpfulness via conversation flow

### Knowledge Integration Architecture
```
Passive Research → Vector DB (FAISS embeddings)
                          ↓
Voice Query → RAG retrieval → Llama generates answer
                          ↓
Feedback Loop: Was answer helpful? Follow-up questions?
                          ↓
Refines: Research focus, summarization depth, source prioritization
```

---

## Resource Allocation with Minecraft

### Scenario 1: You're Away (Work/School) + No Minecraft Players
```
Oracle Passive Research: FULL POWER
- RTX 3060: 100% (research + summarization)
- CPU: 6-8 cores
- RAM: 20GB
Spider: minimal monitoring
```

### Scenario 2: You're Home + Voice Active
```
Oracle Voice Mode: HIGH PRIORITY
- RTX 3060: whisper + llama (real-time)
- CPU: 2-3 cores reserved
- RAM: 10-12GB
Passive research: PAUSED
Spider: available for oracle queries
```

### Scenario 3: Minecraft Active (2-4 players)
```
Minecraft: PRIORITY
- CPU: 4-6 cores
- RAM: 16-20GB
Oracle: BACKGROUND ONLY
- voice still responsive (uses remaining resources)
- passive research paused
Spider: essential monitoring
```

### Scenario 4: Gaming + Voice Command
```
You (while playing): "Hey Oracle, what's server TPS?"
Oracle: pauses any background tasks
     → queries spider (instant)
     → synthesizes quick response (<2s)
     → resumes background state
```

---

## Hardware Requirements

### Current Setup (Fathom)
- CPU: Ryzen 7 3700X (8C/16T) - sufficient
- GPU: RTX 3060 12GB - perfect for llama 3.1 8b + voice pipeline
- RAM: 16GB @ 3200MHz - **needs upgrade to 32GB for full research capacity**
- Storage: 256GB NVMe + HDDs - adequate

### RAM Upgrade Justification (32GB total)
- Llama 3.1 8B inference: 8-10GB
- Whisper ASR model: 2-3GB
- Vector database (FAISS): 2-4GB
- Research data processing: 4-6GB
- System + Minecraft buffer: 8-10GB
- **Total needed for full operation**: 24-33GB

### GPU Utilization (RTX 3060 12GB)
**Passive Mode**:
- Llama 3.1 8B: 8GB VRAM
- Future upgrade path: Llama 3.1 70B quantized (uses full 12GB, better quality)

**Active Mode (Voice)**:
- Whisper Turbo: 2GB
- Llama 3.1 8B: 8GB
- Coqui TTS: 1GB
- Remaining: 1GB buffer

---

## Intelligence Architecture

### Short-Term Memory (Voice Session)
- conversation context (last 10-20 exchanges)
- current task/query focus
- temporary spider system state
- cleared after session ends

### Long-Term Memory (Persistent Knowledge)
- research summaries (daily/weekly digests)
- user preference learning (topics of interest)
- spider system history (for diagnostics)
- successful solutions database (for future recommendations)

### Meta-Learning Layer
```python
# oracle tracks its own performance
performance_metrics = {
    'voice_response_time': '1.2s avg',
    'research_quality_score': 8.5/10,  # based on follow-up questions
    'answer_helpfulness': 85%,  # tracked via conversation flow
    'topic_prediction_accuracy': 78%  # what you ask vs what it researched
}
```

**Continuous improvement**:
- if topic_prediction_accuracy < 70%: broaden research scope
- if answer_helpfulness < 80%: improve summarization
- if voice_response_time > 2s: optimize inference pipeline

---

## Data Sources (Passive Research)

### Primary Sources
- **reddit**: using praw (machine learning, futurology, homelab, selfhosted)
- **twitter/x**: using api or snscrape (ai researchers, tech leaders)
- **hacker news**: official api (top stories, ask hn)
- **arxiv**: academic preprints via api (cs.ai, cs.lg)
- **news apis**: google news, bing news (tech section)

### Secondary Sources
- rss feeds from tech blogs
- github trending (ai/ml repositories)
- youtube transcripts (tech channels you follow)
- discord/matrix (if you join tech communities)

### Research Pipeline
```
1. ingestion: periodic api calls based on keywords
2. storage: raw data in sqlite
3. preprocessing: clean, normalize, deduplicate
4. summarization: llama 3.1 8b generates summaries
5. clustering: faiss vector embeddings for similarity
6. trend detection: identify novel insights
7. knowledge base update: add to rag database
```

---

## Implementation Roadmap

### Phase 1: Foundation (Months 1-2)
- basic passive research system (reddit/hn scraping)
- llama 3.1 8b local inference
- spider integration for system queries
- text-based chat interface for testing (gradio)
- 32gb ram upgrade for fathom

**Deliverable**: working research system with text chat

### Phase 2: Voice Interface (Months 3-4)
- porcupine wake word on alexa gen 2
- whisper asr integration (quantized for speed)
- coqui tts synthesis (streaming)
- sub-1.5s response pipeline optimization
- home assistant voice control integration

**Deliverable**: functional voice assistant

### Phase 3: Self-Improvement (Months 5-6)
- track conversation patterns and preferences
- adjust research focus based on query history
- a/b test summarization approaches
- build personalized knowledge graph
- predictive research (anticipate questions)

**Deliverable**: learning system that gets smarter over time

### Phase 4: Advanced Integration (Months 7+)
- proactive notifications ("found interesting research on X")
- multi-modal learning (images, videos, papers)
- cross-system orchestration (oracle + spider unified)
- advanced diagnostics (oracle asks spider complex questions)

---

## Integration with Spider

### Oracle Queries Spider
```
User: "Hey Oracle, why is Sanctum running hot?"
Oracle: queries spider.get_system_diagnostics('sanctum')
Spider: returns temps, processes, docker stats
Oracle: analyzes data + researches cooling solutions
Oracle: synthesizes response with diagnosis + fix
Voice: speaks answer
```

### Spider Informs Oracle
```
Spider: detects critical issue (disk 95% full)
Spider: logs to oracle's alert queue
Oracle: (next voice interaction) "by the way, sanctum disk is almost full"
User: "can you help me clean it up?"
Oracle: queries spider for large files, suggests cleanup
```

### Unified Intelligence
- spider provides facts (system state, metrics, diagnostics)
- oracle provides conversation (natural language, research, solutions)
- together: complete homelab intelligence system

---

## Success Metrics

### Voice Assistant Quality
- response latency: <1.5s (target), <2s (acceptable)
- answer accuracy: >90% for factual queries
- conversation naturalness: subjective but tracked via follow-ups
- spider integration reliability: >95% successful queries

### Research System Quality
- topic relevance: >80% of research aligns with interests
- information novelty: >60% of daily digest contains new info
- summarization quality: <3 follow-up questions per topic avg
- knowledge retention: can answer questions from past research

### Self-Improvement Metrics
- topic prediction accuracy: >75% (improves over time)
- research focus adjustment speed: <1 week to adapt to new interests
- solution database growth: +10 successful fixes per month
- user satisfaction: subjective but tracked via interaction patterns

---

## Key Differentiators

**vs generic voice assistants**:
- oracle learns YOUR interests through passive research
- personalized knowledge base, not generic search results
- gets smarter the longer it runs

**vs static ai systems**:
- continuous self-improvement through interaction tracking
- adapts research focus to your evolving interests
- builds historical context about your homelab

**vs separate voice + research systems**:
- unified intelligence: voice can access research instantly
- seamless mode switching: active ↔ passive
- shared learning: voice interactions improve research focus

---

## The Oracle Promise

When you ask "What's new in AI?", Oracle isn't searching in real-time—it already spent 8 hours while you were at work:
- scraping 50+ sources
- reading 200+ articles/papers
- summarizing key developments
- identifying trends relevant to YOUR interests
- organizing into digestible knowledge

**The result**: instant, personalized, comprehensive answers based on continuous learning.
