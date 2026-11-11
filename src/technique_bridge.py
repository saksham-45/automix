"""
Bridge: run the project's ORIGINAL technique library inside the live club engine.

The original pipeline (SmartMixer -> SuperhumanDJEngine) owns a large library —
23 transition techniques in `TechniqueExecutor` and 8 stem "conversations" in
`StemOrchestrator` — but almost never let it play: with stems present every choice
was overridden to `clash_safe`, and a quality gate then re-rendered most results as
a plain long blend. The techniques also ran on segments that were never beat-aligned.

Here they run for real: `club_mixer` hands over two decks that are already on one
beat grid (plus their demucs stems), this module calls the original implementation
by name, and joins the result seamlessly to the untouched audio on both sides.
"""
from __future__ import annotations

import contextlib
import io
from typing import Dict, Optional

import numpy as np

# name in the live engine -> conversation type understood by StemOrchestrator
CONVERSATIONS = {
    "conv_layered_reveal": "layered_reveal",
    "conv_counter_melody": "counter_melody",
    "conv_vocal_overlay_handoff": "vocal_overlay_handoff",
    "conv_bass_from_a_beat_from_b": "bass_from_a_beat_from_b",
    "conv_melody_a_drums_vocals_b": "melody_a_drums_vocals_b",
    "conv_call_response": "call_response",
    "conv_interweave": "interweave",
    "conv_progressive_morph": "progressive_morph",
}
# TechniqueExecutor techniques that need stems / work on the full mix
EXECUTOR_STEM = ("staggered_stem_mix", "partial_stem_separation", "vocal_layering",
                 "acapella_overlay", "progressive_morph")
EXECUTOR_FX = ("long_blend", "bass_swap", "filter_sweep", "echo_out", "drop_mix", "double_drop",
               "energy_build", "loop_transition", "breakdown_to_build", "back_and_forth",
               "drum_roll", "washout", "backspin", "quick_cut", "drop_on_the_one")
# original techniques that still make sense when the two tempos cannot be matched
ORIGINAL_CUTS = ("backspin", "drop_on_the_one", "quick_cut", "drum_roll", "washout")
ORIGINAL = tuple(CONVERSATIONS) + EXECUTOR_STEM + EXECUTOR_FX

LABELS = {
    "conv_layered_reveal": "layered reveal", "conv_counter_melody": "counter-melody",
    "conv_vocal_overlay_handoff": "vocal overlay handoff",
    "conv_bass_from_a_beat_from_b": "bass from A, beat from B",
    "conv_melody_a_drums_vocals_b": "melody A + drums & vocal B",
    "conv_call_response": "call & response", "conv_interweave": "interweave",
    "conv_progressive_morph": "progressive morph (layered)",
    "staggered_stem_mix": "staggered stem mix", "partial_stem_separation": "partial stem separation",
    "vocal_layering": "vocal layering", "acapella_overlay": "acapella overlay",
    "progressive_morph": "progressive stem morph", "long_blend": "long blend",
    "bass_swap": "bass swap", "filter_sweep": "filter sweep", "drop_mix": "drop mix",
    "double_drop": "double drop", "energy_build": "energy build",
    "loop_transition": "loop transition", "breakdown_to_build": "breakdown to build",
    "back_and_forth": "back and forth", "drum_roll": "drum roll", "backspin": "backspin",
    "quick_cut": "quick cut", "drop_on_the_one": "drop on the one",
}
DESCRIPTIONS = {
    "conv_layered_reveal": "next track revealed stem by stem",
    "conv_counter_melody": "this track's melody and vocal over the next track's rhythm section",
    "conv_vocal_overlay_handoff": "next track's vocal over this track's music, then its full track",
    "conv_bass_from_a_beat_from_b": "this track's bass keeps going while the next track's drums take over",
    "conv_melody_a_drums_vocals_b": "this track's instrumental hook with the next track's drums and vocal",
    "conv_call_response": "the two tracks' drums answer each other",
    "conv_interweave": "stems fade at different times and weave together",
    "conv_progressive_morph": "next track's stems introduced one by one",
    "staggered_stem_mix": "beat first, vocals later — drums hand off on the beat",
    "partial_stem_separation": "each stem transitions at its own time",
    "vocal_layering": "this track's vocal kept over the next track's beat",
    "acapella_overlay": "this track's acapella rides the next track's instrumental",
    "progressive_morph": "this track's stems are morphed toward the next track's",
    "long_blend": "long equal-power blend with a progressive bass cut",
    "bass_swap": "basslines swapped mid-blend", "filter_sweep": "outgoing track swept out with a filter",
    "drop_mix": "energy dips, then the next track builds in",
    "double_drop": "both tracks hit together, then the next one carries on",
    "energy_build": "filter and level build into the next track",
    "loop_transition": "a bar is looped, then released into the next track",
    "breakdown_to_build": "this track breaks down as the next one builds",
    "back_and_forth": "the two tracks alternate, then the next one stays",
    "drum_roll": "a beat is rolled with a rising filter, then the next track",
    "backspin": "the outgoing track is spun out as the next one comes in",
    "quick_cut": "cut on the beat", "drop_on_the_one": "next track dropped on the downbeat",
}

_EXECUTOR = None
_ORCHESTRATOR = None


def _executor(sr: int):
    global _EXECUTOR
    if _EXECUTOR is None or _EXECUTOR.sr != sr:
        from src.technique_executor import TechniqueExecutor
        _EXECUTOR = TechniqueExecutor(sr=sr)
    return _EXECUTOR


def _orchestrator(sr: int):
    global _ORCHESTRATOR
    if _ORCHESTRATOR is None or _ORCHESTRATOR.sr != sr:
        from src.stem_orchestrator import StemOrchestrator
        _ORCHESTRATOR = StemOrchestrator(sr=sr)
    return _ORCHESTRATOR


def _stereo(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = np.column_stack([x, x])
    elif x.shape[1] == 1:
        x = np.column_stack([x[:, 0], x[:, 0]])
    x = x[:n, :2]
    if len(x) < n:
        x = np.vstack([x, np.zeros((n - len(x), 2), dtype=np.float32)])
    return x


def _exact_stems(seg: np.ndarray, stems: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Copies of the four stems that sum back to the segment exactly (whatever the
    separator lost is given to 'other'), so a stem mix joins untouched audio cleanly."""
    n = len(seg)
    out = {k: _stereo(stems.get(k, np.zeros((n, 2))), n).copy() for k in ("drums", "bass", "other", "vocals")}
    out["other"] += seg - (out["drums"] + out["bass"] + out["other"] + out["vocals"])
    return out


def run_original(name: str, seg_a: np.ndarray, seg_b: np.ndarray, sr: int,
                 tempo_a: float, tempo_b: float,
                 stems_a: Optional[Dict] = None, stems_b: Optional[Dict] = None,
                 switch_ratio: float = 0.5, beat: Optional[int] = None,
                 verbose: bool = False) -> np.ndarray:
    """Render one transition with an ORIGINAL technique / stem conversation.

    seg_a / seg_b: equal-length stereo decks (beat-aligned for blends). Returns the
    transition, the same length, beginning as pure A and ending as pure B.
    """
    n = min(len(seg_a), len(seg_b))
    seg_a, seg_b = _stereo(seg_a, n), _stereo(seg_b, n)
    tempo = 0.5 * (tempo_a + tempo_b)
    sa = _exact_stems(seg_a, stems_a) if stems_a is not None else None
    sb = _exact_stems(seg_b, stems_b) if stems_b is not None else None
    log = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO() if not verbose else log):
        if name in CONVERSATIONS:
            if sa is None or sb is None:
                raise ValueError(f"{name} needs stems")
            orch = _orchestrator(sr)
            conv = orch.create_stem_conversation(
                sa, sb, CONVERSATIONS[name],
                phrase_data_a=orch.detect_vocal_phrases(sa["vocals"]),
                phrase_data_b=orch.detect_vocal_phrases(sb["vocals"]),
                segment_duration_sec=n / sr, tempo_a=tempo_a, tempo_b=tempo_b)
            out = orch.orchestrate_mix(sa, sb, conv, mix_at_level=True, finalize=False)
        else:
            params = {"tempo": tempo, "tempo_a": tempo_a, "tempo_b": tempo_b,
                      "cut_ratio": switch_ratio, "swap_point_ratio": switch_ratio,
                      "sync_point_ratio": switch_ratio}
            out = _executor(sr).execute(name, seg_a.copy(), seg_b.copy(), params, sa, sb)
    out = _stereo(out, n)
    if not np.isfinite(out).all():
        raise FloatingPointError(f"{name} produced non-finite audio")

    # Join: ease from the untouched A into the technique over one beat, and from the
    # technique into the untouched B over the last beat. A technique that already
    # starts as A / ends as B is left exactly as it rendered.
    k = int(min(n // 4, beat if beat else 0.4 * sr))
    if k > 8:
        r = np.linspace(0.0, 1.0, k, dtype=np.float32)[:, None]
        out[:k] = seg_a[:k] * (1.0 - r) + out[:k] * r
        out[-k:] = out[-k:] * (1.0 - r) + seg_b[-k:] * r
    return out
