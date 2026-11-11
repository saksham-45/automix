---
title: automix — AI DJ Auto-Mixer
emoji: 🎛️
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Beat-matched, gapless DJ mixing — research demo
---

# automix — Automatic DJ Mixing (research project)

**A research prototype for automatic, beat-/phrase-aware DJ transitions.** Given a
set of tracks, automix renders **one continuous, gapless, club-style mix** in which
each consecutive pair is tempo-matched, phrase-locked, loudness-matched and
bass-swapped on a downbeat — the moves a human DJ makes, derived from the MIR
literature rather than hand-tuned rules.

## Showcase: automix × visualizzer

Mixes rendered end to end by automix. The visuals come from
[visualizzer](https://github.com/saksham-45/visualizzer), my real-time WebGL/Three.js
audio visualizer, reacting to the mix frame by frame. The on-screen HUD shows each
decision automix made: the move, its length in bars, both decks' BPM and key, and a
live crossfader.

<p align="center">
  <img src="docs/media/preview_chrome_morph.gif" width="49%" alt="Chrome Morph during a vocal-overlay handoff">
  <img src="docs/media/preview_liquid_metal.gif" width="49%" alt="Liquid Metal during an acapella overlay">
</p>

**▶ Full videos and mixes are on the [Showcase release](https://github.com/saksham-45/automix/releases/tag/showcase-2026-11).**

| | what | length |
|---|---|---|
| 🎬 | **Psychedelic rap — automix × visualizzer** (1080p60). I Wonder → Champion → Stronger → Passionfruit → The Less I Know The Better → Let It Happen | 4:07 |
| 🎬 | **House / EDM — automix × visualizzer** (1080p60, NCS tracks) | 2:55 |
| 🎧 | Psychedelic rap quick mix: the full render behind the video | 11:22 |
| 🎧 | House set: 8 NCS tracks, full songs | 18:58 |
| 🎧 | YouTube-playlist set | 21:06 |

The rap video runs in **quick-mix mode**. Every song plays continuously for a few
bars after it comes in, and then a **full 16-bar transition** takes it out. Nothing
inside a song is cut; the transitions do all the joining. Every transition is a
different move from the stem library:

| | transition | move |
|---|---|---|
| 0:10 | I Wonder → Champion | vocal overlay handoff: B's vocal over A's music, then B in full |
| 1:08 | Champion → Stronger | bass from A, beat from B |
| 1:53 | Stronger → Passionfruit | counter-melody: A's melody and vocal over B's rhythm section |
| 2:38 | Passionfruit → The Less I Know The Better | layered reveal: B comes in stem by stem |
| 3:19 | The Less I Know The Better → Let It Happen | acapella overlay: A's acapella over B's instrumental |

The video pipeline lives in [`video/`](video/README.md). It renders offline and
frame-accurately in headless Chrome, on a virtual clock with injected
AnalyserNode-equivalent audio data, at 1080p60 and 2× pixel density. Because it
doesn't render in real time, a slow GPU can't drop frames.

> **Scope & framing.** This is a personal research project on automatic mixing and
> music-information-retrieval (MIR). The web demo ships **only royalty-free,
> synthesized audio**. The house showcase uses NoCopyrightSounds releases (credits in
> the release notes). The rap showcase mixes commercial tracks for personal,
> non-commercial demonstration only; all rights belong to the artists and labels.
> YouTube ingestion is for local experimentation and is the user's responsibility
> under YouTube's Terms of Service and applicable copyright law.

### How it mixes (what a DJ does, in the order they do it)

**Listen** — every track is analysed once:

- **Beat grid accurate to a few ms.** The beat tracker only gives a rough tempo; the
  period is then sharpened by comb-folding a 3 ms spectral-flux onset envelope, the
  phase is taken from the kick (so the grid never locks to the off-beat hat), and slow
  drift is corrected per 32 beats. A 0.1 % tempo error is a 30 ms flam after 16 bars —
  this is the difference between a locked blend and a trainwreck.
- **Downbeats, 8-bar phrases and section boundaries** (intro end, breakdowns, outro
  start) from chord changes, low-band accents and bar-level spectral novelty.
- **Key** (Krumhansl profiles → Camelot wheel), **loudness** (BS.1770, K-weighted) and a
  **beat-confidence** score that says whether there is a steady beat to mix on at all.

**Decide** — where to leave A (a phrase line, ideally where its outro starts), where
to cue B (so the first point where B has real weight lands on the switch), and which
family of move fits the pair:

| family | when | what plays |
|--------|------|------------|
| **beat-matched** | tempos within ~18 % (incl. half/double time) | with demucs stems: the project's stem conversations and stem techniques, chosen from what is actually playing (who is singing, who has drums). Without stems: the full-mix techniques. |
| **cut** | tempos too far apart to match | backspin, drop on the one, drum roll, washout, loop roll + drop, spinback, tape stop, echo out — a hard drop only if B comes in with weight |
| **fade** | no steady beat (ambient, classical) | plain equal-power crossfade, nothing stretched |

The technique library (`src/technique_executor.py`, `src/stem_orchestrator.py`, run in
the live engine through `src/technique_bridge.py`; a few bar-timed extras in
`src/dj_techniques.py`):

- **Stem conversations** — layered reveal, counter-melody, vocal overlay handoff, bass
  from A + beat from B, melody A + drums & vocal B, call & response, interweave,
  progressive morph.
- **Stem techniques** — staggered stem mix, partial stem separation, vocal layering,
  acapella overlay, progressive stem morph, stem blend / swap, beat swap, vocal ride,
  vocal tease, drum trade.
- **Full-mix techniques** — long blend, bass swap, filter sweep, echo out, drop mix,
  double drop, energy build, loop transition, breakdown to build, back and forth, drum
  roll, washout, backspin, quick cut, drop on the one, EQ blend / filter / swap,
  build + drop, transform cuts.

The same move is never played twice in a row, and a rendered transition that leaves
seconds of dead air on the material at hand is re-done with another technique.

**Perform**

- **One master tempo for both decks.** Both tracks are warped (key-locked WSOLA) onto a
  single beat grid whose tempo glides from A's BPM to B's across the overlap — each
  track is at its own speed whenever it is heard alone, and track bodies are always the
  original, un-stretched audio. The grid is re-fitted locally around each transition.
- **Whole-track loudness normalisation** (−14 LUFS) so there is no level jump when a
  transition ends; overlaps pass through a look-ahead limiter.

References: Vande Veire & De Bie (2018); Zehren et al. (2022); Foote (2000) / Müller
*FMP*; Verhelst & Roelands (1993, WSOLA); Krumhansl (1990); ITU-R BS.1770.

### Live demo

The web app streams the mix **as it renders** (progressive Web-Audio playback, no
upfront wait) and shows each decision live (technique · bars · BPM · key). Click
**Sample** to mix three synthesized tracks — each with an intro, build, drop, breakdown
and outro, at 122/124/128 BPM. Tick **quick mix** to hear each song for only 4–8 bars
between full-length transitions (`play_bars=(4, 8)` in `build_continuous_set`).
Jobs are queued and cancellable, and uploads, health checks and a gunicorn config
are included. See **[DEPLOY.md](DEPLOY.md)** for a one-file deploy to a free
Hugging Face Spaces (Docker) Space.

```bash
# Run locally (CPU). Stems optional; the sample needs no network.
pip install -r requirements.txt
PORT=5005 python club_server.py     # then open http://localhost:5005
```

```bash
# Tests (synthetic audio with known tempo/structure — no downloads)
python tests/test_club_mixer.py
```

---

## (Legacy) AI DJ Mixing System

The sections below document the original learning-based pipeline (feature extraction,
NN/LSTM transition models, the superhuman engine). The deployed demo above uses the
self-contained, stem-optional **club mixer** (`src/club_mixer.py`); the learning
models are not required to run it.

## Table of Contents

- [Demo](#demo)
- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Complete Workflow](#complete-workflow)
- [Module Documentation](#module-documentation)
- [Technical Details](#technical-details)
- [Project Structure](#project-structure)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

### Philosophy

**Learn from the masters, don't reinvent the wheel.**

Instead of hardcoding mixing rules, this system:
1. **Analyzes real DJ mixes** (Boiler Room, Essential Mixes, live sets) to extract transition knowledge
2. **Trains AI models** on how professional DJs execute transitions
3. **Applies learned techniques** to create new professional-quality mixes

### Core Innovation

**Three-Stage "Superhuman" Architecture:**
- **Stage 1: Multi-Factor Analysis (Engine Core)** - Extracts 100+ spectral, temporal, and structural features.
- **Stage 2: Superhuman Creative Engine** - Uses **Monte Carlo Optimization** to simulate 50+ transition variations and select the perceptually optimal path.
- **Stage 3: Progressive Content Morphing** - Instead of simple volume fades, the system uses **Stem Morphing** to transform the actual audio data of one song into another.

This mimics how human DJs work: first decide the technique, then execute it precisely.

### What Makes This Different

- ✅ **Superhuman Mixing Engine** with Monte Carlo Optimization (test 50+ variations per mix)
- ✅ **Progressive Stem Morphing** - Warp the timbre and rhythm of Song A into Song B
- ✅ **84-second Ultra-Long Transitions** - Professional-grade long-form transformation standard
- ✅ **Micro-timing Alignment** - Sub-millisecond groove and transient negotiation
- ✅ **Spectral Intelligence** - Surgical frequency slotting to prevent "muddy" mixes
- ✅ **18 transition techniques** (including the flagship `progressive_morph`)
- ✅ **Mix Server** - Flask-based streaming server for dynamic YouTube playlist mixing

## Architecture

### High-Level System Flow

```
┌─────────────────────────────────────────────────────────────┐
│  1. INPUT & ANALYSIS                                        │
│     - Download/load audio files                             │
│     - Extract 100+ features (tempo, key, structure, etc.)   │
│     - Analyze song structure (sections, phrases, energy)    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  2. TRANSITION POINT FINDING                                │
│     - Find candidate "out" points in Song A (30-95%)        │
│     - Find candidate "in" points in Song B (0-95%)          │
│     - Evaluate 10×10 = 100 pairs with quality prediction    │
│     - Select best pair based on multiple factors            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  3. BEAT MATCHING & ALIGNMENT                               │
│     - Match tempos precisely                                │
│     - Align beat phases (kick drums hit together)           │
│     - Adjust transition points to downbeats                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  4. TECHNIQUE SELECTION                                     │
│     - Analyze harmonic compatibility                        │
│     - Determine energy flow direction                       │
│     - Predict frequency clash risk                          │
│     - Predict vocal overlap risk                            │
│     - Score all 17 techniques                               │
│     - Select best technique for context                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  5. STEM SEPARATION & PROCESSING                            │
│     - Separate drums, bass, vocals, other (using Demucs)    │
│     - Detect when vocals start in Song B                    │
│     - Apply aggressive fades to Song A vocals               │
│     - Fade out drums/bass from Song A                       │
│     - Recombine processed stems                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  6. TECHNIQUE EXECUTION                                     │
│     - Execute selected technique (phrase_match, energy_build,│
│       staggered_stem_mix, etc.)                             │
│     - Apply technique-specific volume/EQ curves             │
│     - Mix stems according to technique requirements         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  7. QUALITY VALIDATION                                      │
│     - Assess smoothness, clarity, harmonic tension          │
│     - Check frequency balance and energy continuity         │
│     - Generate overall quality score (0-1)                  │
│     - Warn if quality below threshold                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  8. FINAL ASSEMBLY                                          │
│     - Add 20s context before transition                     │
│     - Add 20s context after transition                      │
│     - Concatenate: [Context A] + [Mix] + [Context B]        │
│     - Save as WAV file (124 seconds total @ 84s Mix)        │
└─────────────────────────────────────────────────────────────┘
```

### Two-Stage AI Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  DecisionNN (Stage 1)                                       │
│  Input: Song features (tempo, key, energy, compatibility)   │
│  Output: Technique selection + basic parameters             │
│  Purpose: High-level decision making                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  CurveLSTM (Stage 2)                                        │
│  Input: Decision context + song features                    │
│  Output: Precise automation curves (volume, EQ, filters)    │
│  Purpose: Detailed execution curves                         │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Comprehensive Song Analysis

Extracts **100+ features** from songs:
- **Temporal:** Tempo (BPM), beat grid, microtiming, rhythm patterns
- **Harmonic:** Key detection (Camelot Wheel), chord progressions, pitch distribution
- **Spectral:** Frequency bands, brightness, warmth, spectral centroid, bandwidth
- **Structural:** Sections (intro/verse/chorus/breakdown/outro), phrase boundaries, mix points
- **Energy:** Loudness curves, RMS, dynamic range, energy trends
- **Perceptual:** Psychoacoustic features, frequency masking predictions

### 2. Intelligent Transition Finding

- **Multi-candidate evaluation:** Tests 100 pairs (10×10) before committing
- **Quality prediction:** Predicts transition quality before creating it
- **Full song exploration:** Analyzes entire Song B (0-95%), not just intros
- **Structural awareness:** Finds optimal points based on song sections
- **Beat alignment:** All candidates aligned to beat boundaries

### 3. Advanced Transition Techniques

**17 professional techniques:**
1. `long_blend` - Smooth 32+ bar equal-power crossfade
2. `quick_cut` - Fast 4-bar energetic cut at downbeat
3. `bass_swap` - Swap bass frequencies at midpoint
4. `filter_sweep` - High/low-pass filter automation
5. `echo_out` - Delay/reverb exit effect
6. `drop_mix` - Energy dip before transition, then build
7. `staggered_stem_mix` - Beat transitions first, vocals later
8. `partial_stem_separation` - Different stems transition at different times
9. `vocal_layering` - Song A vocals continue while Song B beat starts
10. `phrase_match` - Align to 8/16/32-bar phrase boundaries
11. `backspin` - Reverse/tape stop effect on outgoing track
12. `double_drop` - Time two drops together for energy peak
13. `acapella_overlay` - Layer acapella from A over instrumental B
14. `modulation` - Smooth key change during transition
15. `energy_build` - Progressive energy increase with filter/EQ
16. `loop_transition` - Use loops to create smooth mixing points
17. `breakdown_to_build` - Transition from breakdown to build section
18. `progressive_morph` - **[Flagship]** Content-aware audio transformation using stem morphing

### 4. Stem Separation & Intelligent Processing

- **Demucs integration:** Separates drums, bass, vocals, other instruments
- **Vocal detection:** Detects when vocals actually start in Song B
- **Aggressive vocal fading:** Multi-stage fade for Song A vocals (gradual → sudden drop)
- **Selective fading:** Drums/bass fade out fast to prevent clashes
- **Stem-aware techniques:** Some techniques require and use stems intelligently

### 5. Quality Assurance

- **Pre-transition prediction:** Predicts quality before committing
- **Post-transition validation:** Assesses actual quality after mixing
- **Multi-factor scoring:** Harmonic compatibility, energy flow, spectral clash, vocal overlap, beat alignment
- **Quality threshold:** Warns if score below 0.6 (configurable)

## Installation

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- FFmpeg (for audio processing with yt-dlp)

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd ai-dj-mixing-system
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `librosa>=0.10.0` - Audio analysis and feature extraction
- `numpy>=1.24.0` - Numerical computing
- `scipy>=1.10.0` - Signal processing
- `torch>=2.0.0` - Deep learning (PyTorch)
- `yt-dlp>=2023.0.0` - YouTube audio download
- `demucs>=4.0.0` - Stem separation
- `soundfile>=0.12.0` - Audio I/O

### Step 4: Install FFmpeg (for YouTube downloads)

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

**Windows:**
Download from [FFmpeg website](https://ffmpeg.org/download.html) and add to PATH

### Step 5: Verify Installation

```bash
python -c "import librosa, numpy, torch, demucs; print('✓ All dependencies installed')"
```

## Quick Start

### Run the web mixer (production-friendly)

```bash
pip install -r requirements.txt
cp .env.example .env   # optional; tweak values if needed
python3 mix_server.py
```

Then open http://127.0.0.1:5005 and paste a YouTube playlist.

Notes:
- `ENABLE_AGENTIC_DURATION=false` keeps predictable 30s overlaps.
- Cache and temp paths (`CACHE_DIR`, `TEMP_DIR`) are set via env and are gitignored.
- Heavy stem separation is optional; install the commented torch/demucs extras only when you have GPU capacity.

### Generate a Mix from YouTube URLs

```bash
python create_mix_from_youtube.py \
  "https://youtu.be/SONG_A_URL" \
  "https://youtu.be/SONG_B_URL" \
  --duration 120
```

**Example:**
```bash
python create_mix_from_youtube.py \
  "https://youtu.be/uxpDa-c-4Mc" \
  "https://youtu.be/Sh855LWd774" \
  --duration 120
```

**Output:**
- Downloads audio (120 seconds each)
- Analyzes both songs
- Finds optimal transition points
- Generates mix using AI models
- Saves to `ai_dj_mix_YYYYMMDD_HHMMSS.wav`
- Moves old mixes to `data/old_mixes/`

### Launch the High-Fidelity Mix Server

The system includes a production-grade Flask server that can mix entire YouTube playlists on the fly.

```bash
# Start the server (default port 5005)
python mix_server.py
```

**Server Features:**
- **On-the-fly Mixing:** Streams high-fidelity chunks directly to your browser.
- **Playlist Support:** Just paste a YouTube playlist URL.
- **Live Morphing Controls:** Toggle `progressive_morph` and adjust depth via the UI/API.
- **Superhuman Standard:** Defaults to **84-second** transitions with Monte Carlo optimization.

### Generate a Mix from Local Files

```bash
python create_mix.py \
  --song-a "path/to/song_a.wav" \
  --song-b "path/to/song_b.wav" \
  --duration 16.0
```

### Python API Usage

```python
from src.smart_mixer import SmartMixer

# Initialize mixer
mixer = SmartMixer()

# Create mix
mixed_audio = mixer.create_smooth_mix(
    'song_a.wav',
    'song_b.wav',
    transition_duration=30.0
)

# Save result
import soundfile as sf
sf.write('output_mix.wav', mixed_audio, mixer.sr)
```

## Complete Workflow

### Phase 1: Song Analysis

**Step 1: Load Audio**
```python
import librosa
y_a, sr = librosa.load('song_a.wav', sr=44100)
y_b, sr = librosa.load('song_b.wav', sr=44100)
```

**Step 2: Analyze Features**
```python
from src.smart_mixer import SmartMixer

mixer = SmartMixer()
analysis_a = mixer._analyze_song_fast(y_a)
analysis_b = mixer._analyze_song_fast(y_b)

# Extracts:
# - Key (e.g., "A#m", "G")
# - Tempo (BPM)
# - Structure (sections, phrases, mix points)
# - Energy (RMS, loudness curves)
```

### Phase 2: Transition Point Finding

**Step 1: Find Candidates**
```python
from src.smart_transition_finder import SmartTransitionFinder

finder = SmartTransitionFinder()
song_a_points = finder._find_transition_points(y_a, sr, is_outgoing=True)
song_b_points = finder._find_transition_points(y_b, sr, is_outgoing=False)

# Returns: Top 30 candidates per song, scored by quality
```

**Step 2: Evaluate Pairs with Quality Prediction**
```python
transition_pair = finder.find_best_transition_pair_intelligent(
    'song_a.wav',
    'song_b.wav',
    song_a_analysis=analysis_a,
    song_b_analysis=analysis_b
)

# Evaluates 10×10 = 100 pairs
# Predicts quality based on 7 factors:
# - Harmonic compatibility
# - Energy compatibility
# - Structural compatibility
# - Tempo/phase match
# - Spectral clash risk
# - Vocal overlap risk
# - Beat alignment quality
```

**Step 3: Get Best Pair**
```python
print(f"Song A @ {transition_pair.song_a_point.time_sec:.1f}s "
      f"({transition_pair.song_a_point.structural_label})")
print(f"Song B @ {transition_pair.song_b_point.time_sec:.1f}s "
      f"({transition_pair.song_b_point.structural_label})")
print(f"Predicted Quality: {transition_pair.compatibility_score:.3f}")
```

### Phase 3: Beat Matching

```python
from src.advanced_beatmatcher import AdvancedBeatMatcher

beatmatcher = AdvancedBeatMatcher(sr=sr)
beat_match = beatmatcher.match_beats(
    y_a_seg, y_b_seg,
    point_a_sec, point_b_sec
)

# Returns:
# - aligned_point_a_sec: Adjusted point in Song A
# - aligned_point_b_sec: Adjusted point in Song B
# - tempo_ratio: Speed adjustment needed
# - phase_shift: Phase alignment needed
```

### Phase 4: Technique Selection

```python
from src.transition_strategist import TransitionStrategist

strategist = TransitionStrategist()
technique = strategist.select_technique(
    key_a=analysis_a['key'],
    key_b=analysis_b['key'],
    tempo_a=analysis_a['tempo'],
    tempo_b=analysis_b['tempo'],
    section_a=transition_pair.song_a_point.structural_label,
    section_b=transition_pair.song_b_point.structural_label,
    energy_a=transition_pair.song_a_point.energy,
    energy_b=transition_pair.song_b_point.energy,
    clash_score=clash_analysis['clash_score'],
    vocal_overlap_risk=transition_pair.quality_factors['vocal_overlap_risk']
)

# Returns:
# {
#   'technique_name': 'phrase_match',
#   'duration_sec': 16.0,
#   'duration_bars': 24,
#   'confidence': 0.85
# }
```

### Phase 5: Stem Separation

```python
from src.stem_separator import StemSeparator

separator = StemSeparator(model_name='htdemucs')

seg_a_stems = separator.separate_segment(seg_a, sr)
seg_b_stems = separator.separate_segment(seg_b, sr)

# Returns:
# {
#   'drums': np.ndarray,
#   'bass': np.ndarray,
#   'vocals': np.ndarray,
#   'other': np.ndarray
# }

# Detect vocal start in Song B
vocal_start_ratio = mixer._detect_vocal_start_time(
    seg_b_stems.get('vocals'),
    len(seg_b)
)

# Apply aggressive vocal fade to Song A
vocal_fade = mixer.crossfade_engine.create_aggressive_vocal_fade(
    len(seg_a),
    vocal_start_time_ratio=vocal_start_ratio,
    aggressive_drop_ratio=0.9
)

seg_a_stems['vocals'] *= vocal_fade[:, np.newaxis]
```

### Phase 6: Technique Execution

```python
from src.technique_executor import TechniqueExecutor

executor = TechniqueExecutor(sr=sr)
mixed = executor.execute(
    technique['technique_name'],  # e.g., 'phrase_match'
    seg_a, seg_b,
    technique_params,
    seg_a_stems=seg_a_stems,
    seg_b_stems=seg_b_stems
)

# Each technique has custom logic:
# - long_blend: Equal-power crossfade
# - energy_build: Progressive energy increase with filter sweep
# - staggered_stem_mix: Beat first, vocals later
# - vocal_layering: Song A vocals + Song B beat
# etc.
```

### Phase 7: Quality Validation

```python
from src.quality_assessor import QualityAssessor

assessor = QualityAssessor(sr=sr)
quality = assessor.assess_transition_quality(
    mixed,
    y_a=seg_a,
    y_b=seg_b,
    key_a=analysis_a['key'],
    key_b=analysis_b['key']
)

# Returns:
# {
#   'overall_score': 0.78,
#   'smoothness': {'score': 0.82, 'analysis': 'smooth'},
#   'clarity': {'score': 0.75, 'analysis': 'clear'},
#   'harmonic_tension': 0.3,
#   'frequency_balance': {'score': 0.80, 'analysis': 'balanced'},
#   'energy_continuity': {'score': 0.76, 'analysis': 'continuous'},
#   'quality_rating': 'very_good'
# }
```

## Module Documentation

### Core Mixing Engine

**`src/smart_mixer.py`** - Main orchestrator
- `SmartMixer` class: Coordinates all modules
- `create_smooth_mix()`: Main entry point for creating mixes
- `_analyze_song_fast()`: Fast song analysis (key, tempo, structure)
- `_find_optimal_transition_points()`: Wrapper for transition finding
- `_extract_segments()`: Extracts transition segments from songs
- `_detect_vocal_start_time()`: Detects when vocals start in Song B

**`src/smart_transition_finder.py`** - Transition point finding
- `SmartTransitionFinder` class: Finds optimal transition points
- `find_best_transition_pair_intelligent()`: Multi-candidate evaluation
- `_find_transition_points()`: Finds candidates in a single song
- `_predict_transition_quality()`: Predicts quality before committing
- `_score_transition_pair()`: Scores a transition pair

**`src/transition_strategist.py`** - Technique selection
- `TransitionStrategist` class: Selects optimal technique
- `select_technique()`: Scores all 17 techniques, selects best
- `get_technique_parameters()`: Gets parameters for selected technique

**`src/technique_executor.py`** - Technique execution
- `TechniqueExecutor` class: Executes transition techniques
- `execute()`: Routes to technique-specific method
- `_execute_long_blend()`, `_execute_phrase_match()`, etc.: Individual technique implementations

### Signal Processing Modules

**`src/advanced_beatmatcher.py`** - Beat matching
- `AdvancedBeatMatcher` class: Precise beat matching and phase alignment
- `match_beats()`: Matches tempos and aligns phases

**`src/beat_aligner.py`** - Beat alignment
- `BeatAligner` class: Aligns beats to downbeats

**`src/harmonic_analyzer.py`** - Harmonic analysis
- `HarmonicAnalyzer` class: Key detection and harmonic compatibility
- `detect_key_camelot()`: Detects key using Camelot Wheel
- `score_transition_harmonics()`: Scores harmonic compatibility

**`src/structure_analyzer.py`** - Song structure
- `StructureAnalyzer` class: Identifies song sections
- `analyze_structure()`: Detects intro, verse, chorus, breakdown, outro

**`src/psychoacoustics.py`** - Psychoacoustic analysis
- `PsychoacousticAnalyzer` class: Predicts perceptual quality
- `predict_frequency_clash()`: Predicts frequency clashes

**`src/dynamic_processor.py`** - Dynamic processing
- `DynamicProcessor` class: EQ automation and frequency processing
- `analyze_frequency_clash()`: Analyzes frequency conflicts
- `create_bass_swap_automation()`: Creates bass swap curves

**`src/crossfade_engine.py`** - Volume curves
- `CrossfadeEngine` class: Generates volume automation curves
- `create_equal_power_crossfade()`: Standard equal-power crossfade
- `create_fast_fade()`: Fast fade for drums/bass
- `create_aggressive_vocal_fade()`: Multi-stage vocal fade
- `create_multi_stage_curve()`: Complex multi-stage curves

**`src/stem_separator.py`** - Stem separation
- `StemSeparator` class: Separates audio into stems
- `separate_segment()`: Separates drums, bass, vocals, other
- Uses Demucs (htdemucs model)

**`src/quality_assessor.py`** - Quality assessment
- `QualityAssessor` class: Evaluates mix quality
- `assess_transition_quality()`: Comprehensive quality assessment

### Analysis Modules

**`src/song_analyzer.py`** - Song analysis
- `SongAnalyzer` class: Comprehensive song feature extraction
- `analyze()`: Extracts 100+ features from a song

**`src/transition_detector.py`** - Transition detection
- `TransitionDetector` class: Detects transitions in DJ mixes
- `detect_transitions()`: Finds transition points in a mix

**`src/transition_analyzer.py`** - Transition analysis
- `TransitionAnalyzer` class: Analyzes transition execution
- `analyze_transition()`: Extracts transition details

**`src/deep_transition_analyzer.py`** - Deep transition analysis
- `DeepTransitionAnalyzer` class: Deep analysis of techniques
- `analyze_transition()`: Detailed technique analysis

**`src/mix_analyzer.py`** - Mix analysis
- `MixAnalyzer` class: Complete mix analysis pipeline
- `analyze_mix()`: Full analysis of a DJ mix

### AI Models

**`src/models/decision_nn.py`** - Decision network
- `DecisionNN` class: Predicts transition technique
- Input: Song features (tempo, key, energy, compatibility)
- Output: Technique class, parameters, duration

**`src/models/curve_lstm.py`** - Curve generation network
- `CurveLSTM` class: Generates automation curves
- Input: Decision context + song features
- Output: Time-series curves (volume, EQ, filters)

**`src/models/combined_model.py`** - End-to-end model
- `CombinedDJModel` class: Combines DecisionNN + CurveLSTM
- End-to-end fine-tuning

### Utility Modules

**`src/database.py`** - Database operations
- SQLite database for storing analysis results

**`src/schema_validator.py`** - JSON schema validation
- Validates analysis output formats

**`src/training_data_extractor.py`** - Training data extraction
- `TrainingDataExtractor` class: Extracts training examples from analyses

**`src/utils.py`** - Utility functions
- Various helper functions

## Technical Details

### Feature Extraction Pipeline

**Temporal Features:**
- Tempo detection (librosa beat tracking)
- Beat grid alignment
- Microtiming analysis
- Rhythm pattern extraction

**Harmonic Features:**
- Key detection (chromagram + pitch class profiles)
- Chord recognition
- Harmonic content analysis
- Camelot Wheel compatibility

**Spectral Features:**
- Spectral centroid (brightness)
- Spectral bandwidth (spread)
- Frequency band energy (bass, mids, highs)
- Spectral rolloff

**Structural Features:**
- Novelty-based segmentation
- Energy-based section detection
- Phrase boundary detection
- Mix point identification (best_mix_in_points, best_mix_out_points)

**Energy Features:**
- RMS (Root Mean Square) curves
- Loudness curves (LUFS)
- Dynamic range
- Energy trends (rising, falling, stable, dip)

**Perceptual Features:**
- Psychoacoustic loudness
- Frequency masking predictions
- Spectral clash risk
- Vocal overlap risk

### Quality Prediction Factors

The system predicts transition quality based on 7 factors:

1. **Harmonic Compatibility (20%)**
   - Key compatibility using Camelot Wheel
   - Harmonic interval relationships
   - Dissonance level

2. **Energy Compatibility (15%)**
   - Energy flow direction (outgoing fading, incoming rising)
   - Energy level match
   - Energy trend compatibility

3. **Structural Compatibility (15%)**
   - Section type pairing (breakdown→intro is good)
   - Structural context match

4. **Tempo/Phase Match (20%)**
   - Tempo difference
   - Phase alignment quality
   - Beat synchronization

5. **Spectral Clash Risk (15%)**
   - Frequency overlap
   - Spectral centroid similarity
   - Bandwidth overlap

6. **Vocal Overlap Risk (10%)**
   - Vocal presence in both songs
   - Vocal timing overlap

7. **Beat Alignment Quality (5%)**
   - Beat alignment at transition points
   - Downbeat alignment

### Transition Technique Scoring

Each technique is scored based on:
- **Harmonic requirements** (0.3 points): Technique requires compatible keys?
- **Energy direction match** (0.3 points): Technique matches energy flow?
- **Structure preference** (0.2 points): Technique works for these sections?
- **Frequency clash handling** (0.3 points): Technique handles clashes?
- **Vocal overlap handling** (0.4 points): Technique handles vocals well?
- **Special bonuses**: Technique-specific bonuses (e.g., double_drop for energy peaks)

### Stem Separation Details

**Model:** htdemucs (Hybrid Transformer Demucs)
- Separates audio into 4 stems: drums, bass, vocals, other
- State-of-the-art quality
- Requires GPU for best performance (CPU works but slower)

**Processing:**
1. Separate both segments (Song A and Song B)
2. Detect vocal start in Song B (energy-based detection)
3. Apply aggressive fade to Song A vocals (multi-stage curve)
4. Apply fast fade to Song A drums/bass
5. Recombine processed stems

**Vocal Fade Curve:**
- Phase 1: Gradual fade from start until Song B vocals begin
- Phase 2: Hold at reduced level
- Phase 3: Sudden aggressive drop in last 10% (exponential decay)

### Quality Assessment Metrics

**Smoothness (30%):**
- Energy continuity
- No sudden jumps
- Smooth volume curves

**Clarity (25%):**
- Spectral separation
- No frequency masking
- Clear instrument distinction

**Harmonic Tension (20%):**
- Key compatibility
- Harmonic dissonance level
- Smooth harmonic flow

**Frequency Balance (15%):**
- Balanced frequency spectrum
- No excessive bass/mid/high
- EQ distribution

**Energy Continuity (10%):**
- Energy flow direction
- No energy gaps
- Progressive energy change

## Project Structure

```
ai-dj-mixing-system/
├── src/                              # Core source code
│   ├── smart_mixer.py               # Main mixing orchestrator
│   ├── smart_transition_finder.py   # Transition point finding
│   ├── transition_strategist.py     # Technique selection
│   ├── technique_executor.py        # Technique execution
│   │
│   ├── advanced_beatmatcher.py      # Beat matching
│   ├── beat_aligner.py              # Beat alignment
│   ├── harmonic_analyzer.py         # Key detection & harmonic analysis
│   ├── structure_analyzer.py        # Song structure analysis
│   ├── psychoacoustics.py           # Psychoacoustic analysis
│   ├── dynamic_processor.py         # EQ automation
│   ├── crossfade_engine.py          # Volume curve generation
│   ├── stem_separator.py            # Stem separation (Demucs)
│   ├── quality_assessor.py          # Quality assessment
│   │
│   ├── song_analyzer.py             # Comprehensive song analysis
│   ├── transition_detector.py       # Transition detection
│   ├── transition_analyzer.py       # Transition analysis
│   ├── deep_transition_analyzer.py  # Deep transition analysis
│   ├── mix_analyzer.py              # Mix analysis pipeline
│   │
│   ├── models/                      # AI models
│   │   ├── decision_nn.py          # Decision network
│   │   ├── curve_lstm.py           # Curve generation LSTM
│   │   └── combined_model.py       # End-to-end model
│   │
│   ├── database.py                  # Database operations
│   ├── schema_validator.py          # JSON schema validation
│   ├── training_data_extractor.py   # Training data extraction
│   └── utils.py                     # Utility functions
│
├── scripts/                          # Executable scripts
│   ├── create_mix_from_youtube.py  # Create mix from YouTube URLs
│   ├── create_mix.py                # Create mix from local files
│   ├── analyze_song.py              # Analyze single song
│   ├── analyze_mix.py               # Analyze DJ mix
│   ├── train_model.py               # Train AI models
│   └── ...
│
├── config/                           # Configuration
│   └── config.yaml                  # System configuration
│
├── data/                             # Data storage
│   ├── songs/                       # Song analyses
│   ├── mixes/                       # Mix analyses
│   ├── transitions/                 # Transition analyses
│   ├── training_splits/             # Train/val/test splits
│   ├── old_mixes/                   # Old mix outputs
│   └── music_analysis.db            # SQLite database
│
├── models/                           # Trained model checkpoints
│   ├── decision_nn.pt               # Decision network weights
│   └── curve_lstm.pt                # Curve LSTM weights
│
├── club_server.py                    # Gapless playlist→mix web app (primary)
├── create_mix_from_youtube.py        # Two-URL CLI mixer (superhuman engine)
├── mix_server.py                     # Legacy superhuman streaming server
├── legacy/                           # Archived/superseded entry scripts
├── data/fixtures/                    # Transition JSON fixtures
├── scripts/                          # Tools, grouped:
│   ├── training/                     #   ML training/inference pipeline
│   ├── analysis/                     #   song/mix analysis utilities
│   └── dev/                          #   tests, debug, one-offs
├── docs/                             # All documentation & plans
│   ├── PROJECT_EXPLANATION.md        # Detailed project explanation
│   ├── EXECUTIVE_SUMMARY.md          # Executive summary
│   ├── HOW_TO_START.md               # Quick start guide
│   ├── BOILER_ROOM_MIXING_PLAN.md    # Club-mixing research + plan
│   ├── PRODUCTION_STREAMING_PLAN.md  # Streaming architecture
│   └── dj_techniques_knowledge_base.md
│
└── requirements.txt                  # Python dependencies
```

## Advanced Usage

### Custom Configuration

Edit `config/config.yaml`:

```yaml
analysis:
  stem_separation:
    enabled: true
    model: "htdemucs"
    separate_stems: ["drums", "bass"]
    drums_fade_ratio: 0.25
    bass_fade_ratio: 0.5
  structure_analysis:
    max_duration_sec: 60
  quality_threshold: 0.6
```

### Batch Processing

```python
from src.smart_mixer import SmartMixer

mixer = SmartMixer()

song_pairs = [
    ("song_a_1.wav", "song_b_1.wav"),
    ("song_a_2.wav", "song_b_2.wav"),
    # ...
]

for song_a, song_b in song_pairs:
    mixed = mixer.create_smooth_mix(song_a, song_b)
    # Save mixed audio
```

### Training Custom Models

```python
from scripts.train_model import train_decision_nn, train_curve_lstm

# Train DecisionNN
train_decision_nn(
    training_data_path='data/training_splits/train.json',
    epochs=100,
    batch_size=32
)

# Train CurveLSTM
train_curve_lstm(
    training_data_path='data/training_splits/train.json',
    epochs=100,
    batch_size=16
)
```

### Analyzing DJ Mixes

```python
from scripts.analyze_mix import analyze_dj_mix

analysis = analyze_dj_mix(
    'path/to/dj_mix.wav',
    save_analysis=True
)
```

## Troubleshooting

### Common Issues

**Issue: "demucs not found"**
```bash
pip install demucs
```

**Issue: "FFmpeg not found" (YouTube downloads)**
- Install FFmpeg: `brew install ffmpeg` (macOS) or `sudo apt-get install ffmpeg` (Linux)

**Issue: "Out of memory" (stem separation)**
- Use CPU mode (slower but less memory): Set `device='cpu'` in StemSeparator
- Reduce segment length: Use shorter transition durations

**Issue: "Low quality scores"**
- Check harmonic compatibility: Some keys don't mix well
- Try different transition points: The system explores multiple candidates
- Adjust quality threshold: Lower threshold in config if needed

**Issue: "Slow processing"**
- Stem separation is the bottleneck: Use GPU for Demucs if available
- Reduce analysis depth: Use `_analyze_song_fast()` instead of full analysis
- Cache analyses: Save song analyses to avoid re-computing

### Performance Tips

- **GPU acceleration:** Use GPU for stem separation (10x faster)
- **Parallel processing:** Analyze multiple songs in parallel
- **Caching:** Save song analyses to disk, reload when needed
- **Sample rate:** Lower sample rate (22050 Hz) for faster analysis

## Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make changes and test thoroughly
4. Commit with descriptive messages
5. Push to your fork: `git push origin feature/new-feature`
6. Create a pull request

### Code Style

- Follow PEP 8 Python style guide
- Use type hints where appropriate
- Document functions with docstrings
- Keep functions focused and small

### Testing

Run tests before committing:
```bash
python -m pytest tests/
```

### Adding New Techniques

1. Add technique definition to `src/transition_strategist.py`:
```python
'new_technique': TransitionTechnique(
    name='new_technique',
    duration_bars=16,
    description='Description here',
    energy_direction='up',
    harmonic_requirements='compatible',
    structure_preference=['verse', 'chorus']
)
```

2. Add scoring logic in `select_technique()` method

3. Implement execution in `src/technique_executor.py`:
```python
def _execute_new_technique(self, seg_a, seg_b, params, ...):
    # Implementation here
    return mixed
```

4. Add to method_map in `execute()` method

## Acknowledgments

- **Demucs** for state-of-the-art stem separation
- **librosa** for audio analysis features
- **PyTorch** for deep learning infrastructure
- **yt-dlp** for YouTube audio downloads

## Contact & Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact project maintainers

---

**Built with ❤️ for the DJ community**
