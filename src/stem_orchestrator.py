"""
Stem Intelligence Orchestrator

Advanced stem-level processing that creates musical conversations:
- Intelligent stem orchestration (musical stem dialogues)
- Stem-level effect processing
- Vocal phrase awareness (never overlap phrases)
- Counter-melody creation (Song A melody over Song B)

This module treats stems as musical voices and orchestrates them together.
"""
import random
import numpy as np
import librosa
from typing import Dict, Tuple, Optional
from scipy import signal
from src.curve_smooth import gaussian_filter1d  # same curves, ~100x faster


class StemOrchestrator:
    """
    Intelligent stem orchestration for creative mixing.
    
    Human DJs fade stems. This engine creates musical conversations
    between stems of different songs.
    """
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
        
        # Stem categories and their characteristics
        self.stem_profiles = {
            'drums': {
                'frequency_range': (40, 15000),
                'rhythmic': True,
                'harmonic': False,
                'priority': 1,  # Most important for groove
                'fade_sensitivity': 'low'  # Can fade quickly
            },
            'bass': {
                'frequency_range': (30, 300),
                'rhythmic': True,
                'harmonic': True,
                'priority': 2,
                'fade_sensitivity': 'medium'
            },
            'vocals': {
                'frequency_range': (100, 8000),
                'rhythmic': False,
                'harmonic': True,
                'priority': 3,  # Important for identity
                'fade_sensitivity': 'high'  # Needs careful fading
            },
            'other': {
                'frequency_range': (100, 16000),
                'rhythmic': False,
                'harmonic': True,
                'priority': 4,
                'fade_sensitivity': 'medium'
            }
        }
    
    # ==================== CONVERSATION CREATION ====================
    
    def create_stem_conversation(self,
                                 stems_a: Dict[str, np.ndarray],
                                 stems_b: Dict[str, np.ndarray],
                                 conversation_type: str = 'call_response',
                                 phrase_data_a: Optional[Dict] = None,
                                 phrase_data_b: Optional[Dict] = None,
                                 segment_duration_sec: Optional[float] = None,
                                 tempo_a: float = 120.0,
                                 tempo_b: float = 120.0) -> Dict:
        """
        Create a musical conversation between stems of two songs.
        
        Instead of simply fading, this creates interplay:
        - Call-response: Song A stems, then Song B responds
        - Interweave: Alternating stems create dialogue
        - Layered reveal: Gradually reveal Song B stem by stem
        - Counter melody: Song A melody over Song B rhythm
        
        Args:
            stems_a: Dict with keys 'drums', 'bass', 'vocals', 'other'
            stems_b: Dict with keys 'drums', 'bass', 'vocals', 'other'
            conversation_type: Type of conversation to create
            phrase_data_a: Optional dict with safe_transition_points (sec from segment start)
            phrase_data_b: Optional dict with safe_transition_points for Song B
            segment_duration_sec: Duration of segment in seconds (n_samples / sr)
        
        Returns:
            Conversation specification with timing and volume curves
        """
        available_stems = set(stems_a.keys()) & set(stems_b.keys())
        tempo_avg = 0.5 * (tempo_a + tempo_b) if (tempo_a > 0 and tempo_b > 0) else 120.0
        phrase_ctx = {
            'phrase_data_a': phrase_data_a,
            'phrase_data_b': phrase_data_b,
            'segment_duration_sec': segment_duration_sec,
            'tempo_avg': tempo_avg,
        }
        
        if conversation_type == 'call_response':
            return self._create_call_response(stems_a, stems_b, available_stems, phrase_ctx)
        elif conversation_type == 'interweave':
            return self._create_interweave(stems_a, stems_b, available_stems, phrase_ctx)
        elif conversation_type == 'layered_reveal':
            return self._create_layered_reveal(stems_a, stems_b, available_stems, phrase_ctx)
        elif conversation_type == 'counter_melody':
            return self._create_counter_melody(stems_a, stems_b, available_stems, phrase_ctx)
        elif conversation_type == 'vocal_overlay_handoff':
            return self._create_vocal_overlay_handoff(stems_a, stems_b, available_stems, phrase_ctx)
        elif conversation_type == 'bass_from_a_beat_from_b':
            return self._create_bass_from_a_beat_from_b(stems_a, stems_b, available_stems, phrase_ctx)
        elif conversation_type == 'melody_a_drums_vocals_b':
            return self._create_melody_a_drums_vocals_b(stems_a, stems_b, available_stems, phrase_ctx)
        elif conversation_type == 'progressive_morph':
            return self._create_progressive_morph(stems_a, stems_b, available_stems, phrase_ctx)
        else:
            return self._create_layered_reveal(stems_a, stems_b, available_stems, phrase_ctx)
    
    def _get_phrase_fade_timing(self,
                                phrase_ctx: Dict,
                                n_samples: int,
                                stem_kind: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Get phrase-aware fade timing for vocals/other.
        Returns (fade_complete_by_sample for curve_a, fade_in_start_sample for curve_b)
        or (None, None) to use fallback ratios.
        """
        if stem_kind not in ('vocals', 'other'):
            return None, None
        seg_sec = phrase_ctx.get('segment_duration_sec')
        if seg_sec is None or seg_sec <= 0:
            return None, None
        min_fade_sec = 2.0
        
        # Outgoing: first safe point that allows at least min_fade_sec for the fade
        phrase_a = phrase_ctx.get('phrase_data_a')
        fade_complete_by_sample = None
        if phrase_a:
            safe = phrase_a.get('safe_transition_points') or []
            for sec in safe:
                if 0 <= sec <= seg_sec and sec >= min_fade_sec:
                    fade_complete_by_sample = int(sec * self.sr)
                    fade_complete_by_sample = min(fade_complete_by_sample, n_samples)
                    break
            if fade_complete_by_sample is None and safe:
                # Use last safe point before end
                in_seg = [s for s in safe if 0 <= s <= seg_sec]
                if in_seg:
                    sec = max(in_seg) if max(in_seg) >= min_fade_sec else in_seg[0]
                    fade_complete_by_sample = min(n_samples, int(sec * self.sr))
        
        # Incoming: first safe point as fade-in start
        phrase_b = phrase_ctx.get('phrase_data_b')
        fade_in_start_sample = None
        if phrase_b:
            safe = phrase_b.get('safe_transition_points') or []
            for sec in safe:
                if 0 <= sec < seg_sec:
                    fade_in_start_sample = int(sec * self.sr)
                    fade_in_start_sample = min(fade_in_start_sample, n_samples - 1)
                    break
        
        return fade_complete_by_sample, fade_in_start_sample
    
    def _create_call_response(self, 
                              stems_a: Dict, 
                              stems_b: Dict,
                              available_stems: set,
                              phrase_ctx: Optional[Dict] = None) -> Dict:
        """
        Create call-response pattern:
        - Song A drums play, then Song B drums
        - Creates conversational rhythm
        """
        # Find reference length
        ref_stem = list(stems_a.values())[0]
        n_samples = len(ref_stem)
        
        # Call-response pattern: 8-16 bars alternating
        bar_samples = int(2.0 * self.sr)  # Approximate 2 seconds per bar at 120 BPM
        phrase_samples = bar_samples * 4  # 4 bars per phrase
        
        curves = {stem: {} for stem in available_stems}
        
        for stem in available_stems:
            curve_a = np.zeros(n_samples)
            curve_b = np.zeros(n_samples)
            
            # Alternate phrases
            phrase_idx = 0
            for start in range(0, n_samples, phrase_samples):
                end = min(start + phrase_samples, n_samples)
                
                if phrase_idx % 2 == 0:
                    # Song A phrase
                    curve_a[start:end] = 1.0
                    # Song B very quiet underneath
                    curve_b[start:end] = 0.15
                else:
                    # Song B phrase
                    curve_b[start:end] = 1.0
                    # Song A quiet underneath
                    curve_a[start:end] = 0.15
                
                phrase_idx += 1
            
            # Smooth transitions. 100ms sigma (0.1*sr) blurred the 1.0<->0.15 phrase
            # edges across ~±300ms, bleeding phrases into each other; ~15ms de-clicks
            # without destroying the phrase boundaries.
            _sig = int(0.015 * self.sr)
            curve_a = gaussian_filter1d(curve_a, sigma=_sig)
            curve_b = gaussian_filter1d(curve_b, sigma=_sig)
            
            curves[stem] = {
                'a': curve_a.tolist(),
                'b': curve_b.tolist()
            }
        
        return {
            'type': 'call_response',
            'curves': curves,
            'description': 'Alternating phrases create musical dialogue'
        }
    
    def _create_interweave(self,
                           stems_a: Dict,
                           stems_b: Dict,
                           available_stems: set,
                           phrase_ctx: Optional[Dict] = None) -> Dict:
        """
        Create interweave pattern:
        - Different stems fade at different times
        - Creates complex texture
        """
        phrase_ctx = phrase_ctx or {}
        ref_stem = list(stems_a.values())[0]
        n_samples = len(ref_stem)
        
        # Staggered transition times
        transition_order = ['drums', 'bass', 'other', 'vocals']
        transition_ratios = [0.15, 0.35, 0.50, 0.65]
        
        curves = {}
        
        tempo = phrase_ctx.get('tempo_avg', 120.0)
        from src.crossfade_engine import CrossfadeEngine
        cf = CrossfadeEngine(sr=self.sr)
        
        for stem in available_stems:
            if stem in transition_order:
                idx = transition_order.index(stem)
                ratio = transition_ratios[idx]
            else:
                ratio = 0.5
            
            if stem == 'drums':
                curve_a, curve_b = cf.create_drum_handoff_curves(
                    n_samples, tempo, handoff_ratio=ratio, overlap_beats=0.0
                )
            else:
                transition_point = int(n_samples * ratio)
                transition_length = int(n_samples * 0.25)  # Slightly longer for gentler
                curve_a = np.ones(n_samples)
                curve_b = np.zeros(n_samples)
                start = max(0, transition_point - transition_length // 2)
                end = min(n_samples, transition_point + transition_length // 2)
                n_fade = end - start
                if n_fade > 0:
                    t = np.linspace(0, 1, n_fade)
                    if stem in ['vocals', 'other']:
                        fade_out = 0.5 * (1 + np.cos(np.pi * t))
                        fade_in = 0.5 * (1 - np.cos(np.pi * t))
                    else:
                        fade_out = np.sqrt(1 - t)
                        fade_in = np.sqrt(t)
                    curve_a[start:end] = fade_out
                    curve_a[end:] = 0.0
                    curve_b[:start] = 0.0
                    curve_b[start:end] = fade_in
                    curve_b[end:] = 1.0
                curve_a = gaussian_filter1d(curve_a, sigma=1000)
                curve_b = gaussian_filter1d(curve_b, sigma=1000)
            
            curves[stem] = {
                'a': curve_a.tolist(),
                'b': curve_b.tolist()
            }
        
        return {
            'type': 'interweave',
            'curves': curves,
            'description': 'Stems transition at different times for complex texture'
        }
        
    def _create_progressive_morph(self,
                                  stems_a: Dict,
                                  stems_b: Dict,
                                  available_stems: set,
                                  phrase_ctx: Optional[Dict] = None) -> Dict:
        """
        Creates volume curves for progressive content morphs in a layered way.
        
        To prevent the "empty" feeling, we introduce Song B stems one by one 
        throughout the transition (Layered Reveal style).
        
        Since StemMorpher is already transforming stems_a into stems_b, 
        the crossfade between them is naturally smooth as they converge in identity.
        """
        ref_stem = next((stems_a[s] for s in available_stems if len(stems_a[s]) > 0), 
                        list(stems_a.values())[0])
        n_samples = len(ref_stem)
        
        # Reveal order for layering Song B
        reveal_order = ['drums', 'bass', 'other', 'vocals']
        # When each stem of Song B starts to fade in (ratio of transition)
        reveal_starts = [0.05, 0.25, 0.45, 0.65] 
        # Duration of each individual stem's crossfade
        fade_duration_ratio = 0.30 
        
        curves = {}
        for stem in available_stems:
            curve_a = np.ones(n_samples)
            curve_b = np.zeros(n_samples)
            
            if stem in ['vocals', 'other']:
                # The user explicitly wants NO vocals/complex synths during the intense
                # middle section of the morph to prevent "128kbps mp3" smearing artifacts.
                # A fades out early (0.1 -> 0.3)
                # B fades in late (0.7 -> 0.9)
                start_a = int(n_samples * 0.1)
                end_a = int(n_samples * 0.3)
                if end_a > start_a:
                    t_out = np.linspace(0, 1, end_a - start_a)
                    curve_a[start_a:end_a] = 0.5 * (1 + np.cos(np.pi * t_out))
                    curve_a[end_a:] = 0.0
                
                start_b = int(n_samples * 0.7)
                end_b = int(n_samples * 0.9)
                if end_b > start_b:
                    t_in = np.linspace(0, 1, end_b - start_b)
                    curve_b[start_b:end_b] = 0.5 * (1 - np.cos(np.pi * t_in))
                    curve_b[end_b:] = 1.0
            else:
                # Find when this specific stem should transition
                if stem in reveal_order:
                    idx = reveal_order.index(stem)
                    start_ratio = reveal_starts[idx]
                else:
                    start_ratio = 0.5
                    
                start_sample = int(n_samples * start_ratio)
                fade_samples = int(n_samples * fade_duration_ratio)
                end_sample = min(n_samples, start_sample + fade_samples)
                
                if end_sample > start_sample:
                    t = np.linspace(0, 1, end_sample - start_sample)
                    # S-curve crossfade
                    fade_out = 0.5 * (1 + np.cos(np.pi * t))
                    fade_in = 0.5 * (1 - np.cos(np.pi * t))
                    
                    curve_a[start_sample:end_sample] = fade_out
                    curve_a[end_sample:] = 0.0
                    
                    curve_b[start_sample:end_sample] = fade_in
                    curve_b[end_sample:] = 1.0
            
            # Smooth out curves
            curve_a = gaussian_filter1d(curve_a, sigma=500)
            curve_b = gaussian_filter1d(curve_b, sigma=500)
            
            curves[stem] = {
                'a': curve_a.tolist(),
                'b': curve_b.tolist()
            }
            
        return {
            'type': 'progressive_morph',
            'curves': curves,
            'description': 'Layered reveal of Song B using stencils for morphing transformation'
        }

    
    def _create_layered_reveal(self,
                               stems_a: Dict,
                               stems_b: Dict,
                               available_stems: set,
                               phrase_ctx: Optional[Dict] = None) -> Dict:
        """
        Create layered reveal pattern:
        - Gradually introduce Song B stems one by one
        - Professional DJ technique
        """
        phrase_ctx = phrase_ctx or {}
        ref_stem = list(stems_a.values())[0]
        n_samples = len(ref_stem)
        seg_sec = phrase_ctx.get('segment_duration_sec') or (n_samples / self.sr)
        phrase_ctx = {**phrase_ctx, 'segment_duration_sec': seg_sec}
        
        reveal_order = ['drums', 'bass', 'other', 'vocals']
        reveal_points = [0.10, 0.30, 0.50, 0.70]
        fade_duration_sec = 2.5
        fade_duration_samples = min(n_samples, int(fade_duration_sec * self.sr))
        tempo = phrase_ctx.get('tempo_avg', 120.0)
        from src.crossfade_engine import CrossfadeEngine
        cf = CrossfadeEngine(sr=self.sr)
        
        curves = {}
        
        for stem in available_stems:
            if stem in reveal_order:
                idx = reveal_order.index(stem)
                start_ratio = reveal_points[idx]
            else:
                start_ratio = 0.5
            
            if stem == 'drums':
                curve_a, curve_b = cf.create_drum_handoff_curves(
                    n_samples, tempo, handoff_ratio=start_ratio, overlap_beats=0.0
                )
                curves[stem] = {'a': curve_a.tolist(), 'b': curve_b.tolist()}
                continue
            
            fade_complete_by, fade_in_start = self._get_phrase_fade_timing(phrase_ctx, n_samples, stem)
            
            if stem in ['vocals', 'other'] and (fade_complete_by is not None or fade_in_start is not None):
                # Phrase-aware timing and moderate (cosine) curves
                curve_a = np.ones(n_samples)
                curve_b = np.zeros(n_samples)
                # Outgoing: complete fade by phrase break
                end_sample = int(n_samples * 0.85) if fade_complete_by is None else fade_complete_by
                end_sample = max(1, min(end_sample, n_samples))
                start_sample = max(0, end_sample - fade_duration_samples)
                if end_sample > start_sample:
                    n_fade = end_sample - start_sample
                    t = np.linspace(0, 1, n_fade)
                    curve_a[start_sample:end_sample] = 0.5 * (1 + np.cos(np.pi * t))
                curve_a[end_sample:] = 0.0
                # Incoming: gentler fade-in from phrase break
                b_start = int(n_samples * start_ratio) if fade_in_start is None else fade_in_start
                b_start = max(0, min(b_start, n_samples - 1))
                b_end = min(n_samples, b_start + fade_duration_samples)
                if b_end > b_start:
                    n_fade = b_end - b_start
                    t = np.linspace(0, 1, n_fade)
                    curve_b[b_start:b_end] = 0.5 * (1 - np.cos(np.pi * t))
                curve_b[b_end:] = 1.0
            else:
                start_sample = int(n_samples * start_ratio)
                fade_length = int(n_samples * 0.25)
                curve_a = np.ones(n_samples)
                curve_b = np.zeros(n_samples)
                a_fade_start = int(n_samples * (start_ratio + 0.1))
                a_fade_end = min(n_samples, a_fade_start + fade_length)
                if a_fade_end > a_fade_start:
                    t = np.linspace(0, 1, a_fade_end - a_fade_start)
                    if stem in ['vocals', 'other']:
                        curve_a[a_fade_start:a_fade_end] = 0.5 * (1 + np.cos(np.pi * t))
                    else:
                        curve_a[a_fade_start:a_fade_end] = np.sqrt(1 - t)
                    curve_a[a_fade_end:] = 0.0
                b_fade_end = min(start_sample + fade_length, n_samples)
                if b_fade_end > start_sample:
                    t = np.linspace(0, 1, b_fade_end - start_sample)
                    if stem in ['vocals', 'other']:
                        curve_b[start_sample:b_fade_end] = 0.5 * (1 - np.cos(np.pi * t))
                    else:
                        curve_b[start_sample:b_fade_end] = np.sqrt(t)
                    curve_b[b_fade_end:] = 1.0
            
            curve_a = gaussian_filter1d(curve_a, sigma=500)
            curve_b = gaussian_filter1d(curve_b, sigma=500)
            curves[stem] = {'a': curve_a.tolist(), 'b': curve_b.tolist()}
        
        return {
            'type': 'layered_reveal',
            'curves': curves,
            'description': 'Gradually reveal Song B stems one by one'
        }
    
    def _create_counter_melody(self,
                               stems_a: Dict,
                               stems_b: Dict,
                               available_stems: set,
                               phrase_ctx: Optional[Dict] = None) -> Dict:
        """
        Create counter-melody effect:
        - Song A melody/vocals over Song B rhythm section
        - Creates unique hybrid
        - Phrase-aware: vocal/other fade-out completes at phrase break, fade-in starts at break
        """
        phrase_ctx = phrase_ctx or {}
        ref_stem = list(stems_a.values())[0]
        n_samples = len(ref_stem)
        seg_sec = phrase_ctx.get('segment_duration_sec') or (n_samples / self.sr)
        phrase_ctx = {**phrase_ctx, 'segment_duration_sec': seg_sec}
        fade_duration_samples = min(n_samples, int(2.5 * self.sr))
        
        curves = {}
        
        for stem in available_stems:
            curve_a = np.zeros(n_samples)
            curve_b = np.zeros(n_samples)
            
            if stem in ['vocals', 'other']:
                fade_complete_by, fade_in_start = self._get_phrase_fade_timing(phrase_ctx, n_samples, stem)
                counter_start = int(n_samples * 0.20)
                counter_end_ratio = 0.70
                counter_end = int(n_samples * counter_end_ratio)
                
                if fade_complete_by is not None and fade_complete_by >= fade_duration_samples:
                    end_sample = min(fade_complete_by, n_samples)
                    start_sample = max(0, end_sample - fade_duration_samples)
                else:
                    end_sample = counter_end
                    start_sample = max(0, end_sample - fade_duration_samples)
                
                curve_a[:start_sample] = 1.0
                if end_sample > start_sample:
                    n_fade = end_sample - start_sample
                    t = np.linspace(0, 1, n_fade)
                    curve_a[start_sample:end_sample] = 0.5 * (1 + np.cos(np.pi * t))
                curve_a[end_sample:] = 0.0
                
                b_start_ratio = counter_start / n_samples
                b_start = int(n_samples * b_start_ratio) if fade_in_start is None else fade_in_start
                b_start = max(0, min(b_start, n_samples - 1))
                b_end = min(n_samples, b_start + fade_duration_samples)
                curve_b[:b_start] = 0.0
                if b_end > b_start:
                    n_fade = b_end - b_start
                    t = np.linspace(0, 1, n_fade)
                    curve_b[b_start:b_end] = 0.5 * (1 - np.cos(np.pi * t))
                curve_b[b_end:] = 1.0
                
            elif stem == 'drums':
                from src.crossfade_engine import CrossfadeEngine
                cf = CrossfadeEngine(sr=self.sr)
                tempo = phrase_ctx.get('tempo_avg', 120.0)
                curve_a, curve_b = cf.create_drum_handoff_curves(
                    n_samples, tempo, handoff_ratio=0.15, overlap_beats=0.0
                )
            elif stem == 'bass':
                transition_ratio = 0.25
                transition_point = int(n_samples * transition_ratio)
                fade_length = int(n_samples * 0.20)
                fade_end = min(transition_point + fade_length, n_samples)
                curve_a[:transition_point] = 1.0
                if fade_end > transition_point:
                    t = np.linspace(0, 1, fade_end - transition_point)
                    curve_a[transition_point:fade_end] = 1.0 - t
                curve_a[fade_end:] = 0.0
                curve_b[:transition_point] = 0.0
                if fade_end > transition_point:
                    t = np.linspace(0, 1, fade_end - transition_point)
                    curve_b[transition_point:fade_end] = t
                curve_b[fade_end:] = 1.0
            
            if stem != 'drums':
                curve_a = gaussian_filter1d(curve_a, sigma=500)
                curve_b = gaussian_filter1d(curve_b, sigma=500)
            curves[stem] = {'a': curve_a.tolist(), 'b': curve_b.tolist()}
        
        return {
            'type': 'counter_melody',
            'curves': curves,
            'description': 'Song A melody plays over Song B rhythm'
        }
    
    def _create_vocal_overlay_handoff(self,
                                      stems_a: Dict,
                                      stems_b: Dict,
                                      available_stems: set,
                                      phrase_ctx: Optional[Dict] = None) -> Dict:
        """
        B vocals on A bed, then introduce B music (e.g. Hey Jude vocals over Yesterday's
        guitars, then bring in Hey Jude's full track). Layer incoming vocals on outgoing
        music, then hand off to incoming music for modulation-perfect feel.
        """
        phrase_ctx = phrase_ctx or {}
        ref_stem = next((stems_a[s] for s in available_stems if len(stems_a[s]) > 0), None)
        if ref_stem is None:
            ref_stem = next((stems_b[s] for s in available_stems if len(stems_b[s]) > 0), None)
        n_samples = len(ref_stem)
        handoff_start_ratio = 0.40   # First 40%: B vocals + A bed only; then crossfade bed to B
        handoff_fade_ratio = 0.35   # Fade over 35% of segment (smooth handoff)
        fade_duration_samples = max(1, int(n_samples * handoff_fade_ratio))
        handoff_start_sample = int(n_samples * handoff_start_ratio)
        handoff_end_sample = min(n_samples, handoff_start_sample + fade_duration_samples)
        
        curves = {}
        tempo = phrase_ctx.get('tempo_avg', 120.0)
        from src.crossfade_engine import CrossfadeEngine
        cf = CrossfadeEngine(sr=self.sr)
        
        for stem in available_stems:
            curve_a = np.ones(n_samples)
            curve_b = np.zeros(n_samples)
            
            if stem == 'vocals':
                curve_a[:] = 0.0
                curve_b[:] = 1.0
            elif stem == 'drums':
                curve_a, curve_b = cf.create_drum_handoff_curves(
                    n_samples, tempo, handoff_ratio=handoff_start_ratio, overlap_beats=0.0
                )
            else:
                # Bass, other: A full for first portion, then smooth crossfade to B
                curve_a[:handoff_start_sample] = 1.0
                curve_b[:handoff_start_sample] = 0.0
                if handoff_end_sample > handoff_start_sample:
                    n_fade = handoff_end_sample - handoff_start_sample
                    t = np.linspace(0, 1, n_fade)
                    curve_a[handoff_start_sample:handoff_end_sample] = 0.5 * (1 + np.cos(np.pi * t))
                    curve_b[handoff_start_sample:handoff_end_sample] = 0.5 * (1 - np.cos(np.pi * t))
                curve_a[handoff_end_sample:] = 0.0
                curve_b[handoff_end_sample:] = 1.0
            
            if stem != 'drums':
                curve_a = gaussian_filter1d(np.clip(curve_a, 0, 1), sigma=300)
                curve_b = gaussian_filter1d(np.clip(curve_b, 0, 1), sigma=300)
            curves[stem] = {'a': curve_a.tolist(), 'b': curve_b.tolist()}
        
        return {
            'type': 'vocal_overlay_handoff',
            'curves': curves,
            'description': 'B vocals on A bed, then introduce B music'
        }
    
    def _create_bass_from_a_beat_from_b(self,
                                        stems_a: Dict,
                                        stems_b: Dict,
                                        available_stems: set,
                                        phrase_ctx: Optional[Dict] = None) -> Dict:
        """
        Keep Song A's bass and bring in Song B's beat (drums). A's bass stays;
        B's drums replace A's drums; other/vocals crossfade A→B. Good for
        "same bass line, new groove" (e.g. New Person → Let It Happen).
        """
        phrase_ctx = phrase_ctx or {}
        ref_stem = next((stems_a[s] for s in available_stems if len(stems_a[s]) > 0), None)
        if ref_stem is None:
            ref_stem = next((stems_b[s] for s in available_stems if len(stems_b[s]) > 0), None)
        n_samples = len(ref_stem)
        # Beat (drums) and other/vocals: crossfade over second half
        beat_handoff_start_ratio = 0.25   # B's beat starts coming in at 25%
        beat_handoff_fade_ratio = 0.50    # Crossfade over 50% of segment
        start_sample = int(n_samples * beat_handoff_start_ratio)
        fade_samples = int(n_samples * beat_handoff_fade_ratio)
        end_sample = min(n_samples, start_sample + fade_samples)
        
        curves = {}
        for stem in available_stems:
            curve_a = np.ones(n_samples)
            curve_b = np.zeros(n_samples)
            
            if stem == 'bass':
                # Keep Song A bass the whole time
                curve_a[:] = 1.0
                curve_b[:] = 0.0
            elif stem == 'drums':
                # Beat-aligned drum handoff: A out, B in at downbeat (no overlap = no clash)
                from src.crossfade_engine import CrossfadeEngine
                cf = CrossfadeEngine(sr=self.sr)
                tempo = phrase_ctx.get('tempo_avg', 120.0)
                curve_a, curve_b = cf.create_drum_handoff_curves(
                    n_samples, tempo, handoff_ratio=0.25, overlap_beats=0.0
                )
            elif stem == 'vocals':
                # Phrase-aware vocal crossfade: A out, B in (avoid cutting mid-word)
                seg_sec = phrase_ctx.get('segment_duration_sec') or (n_samples / self.sr)
                phrase_ctx_v = {**phrase_ctx, 'segment_duration_sec': seg_sec}
                fade_complete_by, fade_in_start = self._get_phrase_fade_timing(
                    phrase_ctx_v, n_samples, 'vocals'
                )
                fade_duration_samples = min(n_samples, int(2.5 * self.sr))
                if fade_complete_by is not None and fade_in_start is not None:
                    end_a = min(fade_complete_by, n_samples)
                    start_a = max(0, end_a - fade_duration_samples)
                    curve_a[:start_a] = 1.0
                    if end_a > start_a:
                        t = np.linspace(0, 1, end_a - start_a)
                        curve_a[start_a:end_a] = 0.5 * (1 + np.cos(np.pi * t))
                    curve_a[end_a:] = 0.0
                    b_end = min(n_samples, fade_in_start + fade_duration_samples)
                    curve_b[:fade_in_start] = 0.0
                    if b_end > fade_in_start:
                        t = np.linspace(0, 1, b_end - fade_in_start)
                        curve_b[fade_in_start:b_end] = 0.5 * (1 - np.cos(np.pi * t))
                    curve_b[b_end:] = 1.0
                else:
                    # Fallback: smooth crossfade over middle 50%
                    curve_a[:start_sample] = 1.0
                    curve_b[:start_sample] = 0.0
                    if end_sample > start_sample:
                        t = np.linspace(0, 1, end_sample - start_sample)
                        curve_a[start_sample:end_sample] = 0.5 * (1 + np.cos(np.pi * t))
                        curve_b[start_sample:end_sample] = 0.5 * (1 - np.cos(np.pi * t))
                    curve_a[end_sample:] = 0.0
                    curve_b[end_sample:] = 1.0
            else:
                # Other: smooth crossfade A→B over same window
                curve_a[:start_sample] = 1.0
                curve_b[:start_sample] = 0.0
                if end_sample > start_sample:
                    n_fade = end_sample - start_sample
                    t = np.linspace(0, 1, n_fade)
                    curve_a[start_sample:end_sample] = 0.5 * (1 + np.cos(np.pi * t))
                    curve_b[start_sample:end_sample] = 0.5 * (1 - np.cos(np.pi * t))
                curve_a[end_sample:] = 0.0
                curve_b[end_sample:] = 1.0
            
            # Skip heavy smoothing for drums (already beat-aligned, minimal overlap)
            if stem != 'drums':
                curve_a = gaussian_filter1d(np.clip(curve_a, 0, 1), sigma=300)
                curve_b = gaussian_filter1d(np.clip(curve_b, 0, 1), sigma=300)
            curves[stem] = {'a': curve_a.tolist(), 'b': curve_b.tolist()}
        
        return {
            'type': 'bass_from_a_beat_from_b',
            'curves': curves,
            'description': 'Keep A bass, bring in B beat (drums); other/vocals crossfade'
        }
    
    def _create_melody_a_drums_vocals_b(self,
                                        stems_a: Dict,
                                        stems_b: Dict,
                                        available_stems: set,
                                        phrase_ctx: Optional[Dict] = None) -> Dict:
        """
        Keep Song A's melody (other) and bring in Song B's drums + B's vocals.
        So we hear A's instrumental hook with B's beat and B's vocal (e.g. Mask Off
        melody + Follow God drums + Follow God vocal — Kanye fits his own beat).
        """
        phrase_ctx = phrase_ctx or {}
        ref_stem = next((stems_a[s] for s in available_stems if len(stems_a[s]) > 0), None)
        if ref_stem is None:
            ref_stem = next((stems_b[s] for s in available_stems if len(stems_b[s]) > 0), None)
        n_samples = len(ref_stem)
        handoff_start_ratio = 0.20   # B drums + vocals start coming in at 20%
        handoff_fade_ratio = 0.55    # Crossfade over 55%
        start_sample = int(n_samples * handoff_start_ratio)
        fade_samples = int(n_samples * handoff_fade_ratio)
        end_sample = min(n_samples, start_sample + fade_samples)
        
        curves = {}
        for stem in available_stems:
            curve_a = np.ones(n_samples)
            curve_b = np.zeros(n_samples)
            
            if stem == 'other':
                # Keep A's melody (other) the whole time — Mask Off flute/synth under the transition
                curve_a[:] = 1.0
                curve_b[:] = 0.0
            elif stem == 'drums':
                # Beat-aligned drum handoff: A out, B in at downbeat (no overlap = no clash)
                from src.crossfade_engine import CrossfadeEngine
                cf = CrossfadeEngine(sr=self.sr)
                tempo = phrase_ctx.get('tempo_avg', 120.0)
                curve_a, curve_b = cf.create_drum_handoff_curves(
                    n_samples, tempo, handoff_ratio=0.20, overlap_beats=0.0
                )
            elif stem == 'vocals':
                # B's vocals only (incoming artist on their own beat)
                curve_a[:] = 0.0
                curve_b[:] = 1.0
            else:
                # bass: crossfade A→B so bass follows B's drums
                curve_a[:start_sample] = 1.0
                curve_b[:start_sample] = 0.0
                if end_sample > start_sample:
                    n_fade = end_sample - start_sample
                    t = np.linspace(0, 1, n_fade)
                    curve_a[start_sample:end_sample] = 0.5 * (1 + np.cos(np.pi * t))
                    curve_b[start_sample:end_sample] = 0.5 * (1 - np.cos(np.pi * t))
                curve_a[end_sample:] = 0.0
                curve_b[end_sample:] = 1.0
            
            if stem != 'drums':
                curve_a = gaussian_filter1d(np.clip(curve_a, 0, 1), sigma=300)
                curve_b = gaussian_filter1d(np.clip(curve_b, 0, 1), sigma=300)
            curves[stem] = {'a': curve_a.tolist(), 'b': curve_b.tolist()}
        
        return {
            'type': 'melody_a_drums_vocals_b',
            'curves': curves,
            'description': 'A melody (other) + B drums + B vocals (e.g. Mask Off melody, Follow God beat + vocal)'
        }
    
    # ==================== VOCAL PHRASE DETECTION ====================
    
    def detect_vocal_phrases(self, vocal_stem: np.ndarray) -> Dict:
        """
        Detect vocal phrases to avoid cutting in the middle of words.
        
        Human speech has natural pauses between phrases.
        We detect these to place transitions.
        """
        if len(vocal_stem) == 0:
            return {'phrases': [], 'phrase_boundaries': []}
        
        # Convert to mono if stereo
        if vocal_stem.ndim > 1:
            vocal_mono = np.mean(vocal_stem, axis=1)
        else:
            vocal_mono = vocal_stem
        
        # Calculate RMS energy with small window
        frame_length = int(0.05 * self.sr)  # 50ms frames
        hop_length = frame_length // 2
        
        rms = librosa.feature.rms(
            y=vocal_mono,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        # Smooth RMS
        rms_smooth = gaussian_filter1d(rms, sigma=3)
        
        # Find silence/low energy regions (phrase boundaries)
        threshold = np.mean(rms_smooth) * 0.3
        is_silence = rms_smooth < threshold
        
        # Find boundaries (transitions from sound to silence)
        boundaries = []
        in_phrase = False
        phrase_start = 0
        
        for i, silent in enumerate(is_silence):
            if not silent and not in_phrase:
                # Start of phrase
                phrase_start = i
                in_phrase = True
            elif silent and in_phrase:
                # End of phrase
                phrase_end = i
                boundaries.append({
                    'start_frame': phrase_start,
                    'end_frame': phrase_end,
                    'start_sec': float(phrase_start * hop_length / self.sr),
                    'end_sec': float(phrase_end * hop_length / self.sr),
                    'duration_sec': float((phrase_end - phrase_start) * hop_length / self.sr)
                })
                in_phrase = False
        
        # Handle phrase extending to end
        if in_phrase:
            boundaries.append({
                'start_frame': phrase_start,
                'end_frame': len(is_silence) - 1,
                'start_sec': float(phrase_start * hop_length / self.sr),
                'end_sec': float(len(vocal_mono) / self.sr),
                'duration_sec': float((len(is_silence) - 1 - phrase_start) * hop_length / self.sr)
            })
        
        # Find best transition points (silence gaps)
        silence_regions = []
        silence_start = None
        
        for i, silent in enumerate(is_silence):
            if silent and silence_start is None:
                silence_start = i
            elif not silent and silence_start is not None:
                silence_duration = i - silence_start
                if silence_duration > 5:  # At least 5 frames (~125ms)
                    silence_regions.append({
                        'center_sec': float((silence_start + i) / 2 * hop_length / self.sr),
                        'duration_sec': float(silence_duration * hop_length / self.sr)
                    })
                silence_start = None
        
        return {
            'phrases': boundaries,
            'phrase_boundaries': silence_regions,
            'phrase_count': len(boundaries),
            'safe_transition_points': [s['center_sec'] for s in silence_regions]
        }
    
    def find_safe_vocal_transition_point(self,
                                         current_point_sec: float,
                                         phrase_data: Dict,
                                         search_window_sec: float = 2.0) -> float:
        """
        Find a safe transition point that doesn't cut vocals mid-phrase.
        """
        safe_points = phrase_data.get('safe_transition_points', [])
        
        if len(safe_points) == 0:
            return current_point_sec
        
        # Find closest safe point within window
        candidates = [
            p for p in safe_points
            if abs(p - current_point_sec) <= search_window_sec
        ]
        
        if len(candidates) == 0:
            return current_point_sec
        
        # Return closest
        return min(candidates, key=lambda x: abs(x - current_point_sec))
    
    # ==================== STEM-LEVEL EFFECTS ====================
    
    def apply_stem_effect(self,
                          stem: np.ndarray,
                          effect_type: str,
                          params: Dict,
                          progress: float = 0.0) -> np.ndarray:
        """
        Apply effect to individual stem.
        
        Args:
            stem: Audio stem
            effect_type: 'filter', 'reverb', 'delay', 'pitch_shift'
            params: Effect-specific parameters
            progress: Transition progress (0-1)
        """
        if effect_type == 'filter':
            return self._apply_filter(stem, params, progress)
        elif effect_type == 'reverb':
            return self._apply_reverb(stem, params, progress)
        elif effect_type == 'delay':
            return self._apply_delay(stem, params, progress)
        elif effect_type == 'pitch_shift':
            return self._apply_pitch_shift(stem, params, progress)
        else:
            return stem
    
    def _apply_filter(self, 
                      stem: np.ndarray,
                      params: Dict,
                      progress: float) -> np.ndarray:
        """Apply dynamic filter to stem."""
        filter_type = params.get('type', 'lowpass')
        start_freq = params.get('start_freq', 20000)
        end_freq = params.get('end_freq', 200)
        
        # Calculate current frequency
        current_freq = start_freq + (end_freq - start_freq) * progress
        current_freq = max(20, min(20000, current_freq))
        
        nyq = self.sr / 2
        normalized_freq = current_freq / nyq
        normalized_freq = max(0.001, min(0.999, normalized_freq))
        
        try:
            if filter_type == 'lowpass':
                sos = signal.butter(4, normalized_freq, btype='low', output='sos')
            else:
                sos = signal.butter(4, normalized_freq, btype='high', output='sos')
            
            return signal.sosfilt(sos, stem)
        except:
            return stem
    
    def _apply_reverb(self,
                      stem: np.ndarray,
                      params: Dict,
                      progress: float) -> np.ndarray:
        """Apply simple reverb (convolution with impulse response)."""
        wet_mix = params.get('wet_mix', 0.3) * progress
        decay = params.get('decay', 0.5)
        
        # Create simple impulse response
        ir_length = int(decay * self.sr)
        t = np.linspace(0, 1, ir_length)
        ir = np.exp(-5 * t) * np.random.randn(ir_length) * 0.1
        
        # Convolve
        wet = signal.fftconvolve(stem, ir, mode='same')
        
        # Mix
        return stem * (1 - wet_mix) + wet * wet_mix
    
    def _apply_delay(self,
                     stem: np.ndarray,
                     params: Dict,
                     progress: float) -> np.ndarray:
        """Apply delay effect."""
        delay_ms = params.get('delay_ms', 500)
        feedback = params.get('feedback', 0.4)
        wet_mix = params.get('wet_mix', 0.5) * progress
        
        delay_samples = int(delay_ms * self.sr / 1000)
        
        if stem.ndim == 1:
            delayed = np.zeros_like(stem)
            if delay_samples < len(stem):
                delayed[delay_samples:] = stem[:-delay_samples] * feedback
            return stem * (1 - wet_mix) + delayed * wet_mix
        else:
            result = stem.copy()
            for ch in range(stem.shape[1]):
                delayed = np.zeros(len(stem))
                if delay_samples < len(stem):
                    delayed[delay_samples:] = stem[:-delay_samples, ch] * feedback
                result[:, ch] = stem[:, ch] * (1 - wet_mix) + delayed * wet_mix
            return result
    
    def _apply_pitch_shift(self,
                           stem: np.ndarray,
                           params: Dict,
                           progress: float) -> np.ndarray:
        """Apply pitch shift (semitones)."""
        semitones = params.get('semitones', 0)
        
        if semitones == 0:
            return stem
        
        try:
            return librosa.effects.pitch_shift(
                stem, sr=self.sr, n_steps=semitones * progress
            )
        except:
            return stem
    
    # ==================== ORCHESTRATED MIXING ====================
    
    def orchestrate_mix(self,
                        stems_a: Dict[str, np.ndarray],
                        stems_b: Dict[str, np.ndarray],
                        conversation: Dict,
                        apply_effects: bool = True,
                        effect_params: Optional[Dict] = None,
                        role_plan: Optional[Dict] = None,
                        mix_at_level: bool = True,
                        finalize: bool = True) -> np.ndarray:
        """
        Create the final mix from stem conversation specification.

        Args:
            stems_a: Dict of stems from Song A
            stems_b: Dict of stems from Song B
            conversation: Conversation spec from create_stem_conversation
            apply_effects: Whether to apply stem effects
            effect_params: Effect parameters per stem
            role_plan: Optional vocal/bed plan (mute one vocal, bias bed)
            mix_at_level: If True, use equal-power curves and normalize output so no volume dip

        Returns:
            Mixed audio
        """
        curves = conversation.get('curves', {})
        
        # Get reference length
        ref_stem = next((s for s in stems_a.values() if len(s) > 0), None)
        if ref_stem is None:
            ref_stem = next((s for s in stems_b.values() if len(s) > 0), None)
        
        if ref_stem is None:
            return np.zeros(44100)  # 1 second of silence
        
        n_samples = len(ref_stem)
        common_stems = set(stems_a.keys()) & set(stems_b.keys()) & set(curves.keys())
        
        # Output shape: stereo if any stem is stereo (avoid broadcast mismatch)
        out_stereo = False
        n_channels = 1
        for sn in common_stems:
            sa = stems_a.get(sn)
            sb = stems_b.get(sn)
            if sa is not None and sa.ndim == 2:
                out_stereo = True
                n_channels = sa.shape[1]
                break
            if sb is not None and sb.ndim == 2:
                out_stereo = True
                n_channels = sb.shape[1]
                break
        if out_stereo and ref_stem.ndim == 2:
            n_channels = ref_stem.shape[1]
        
        if out_stereo:
            mixed = np.zeros((n_samples, n_channels))
        else:
            mixed = np.zeros(n_samples)
        
        for stem_name in common_stems:
            stem_a = stems_a.get(stem_name, np.zeros(n_samples))
            stem_b = stems_b.get(stem_name, np.zeros(n_samples))
            
            # --- 1. HPF on non-bass stems (60-90Hz) ---
            if stem_name not in ['bass', 'drums']:
                try:
                    from scipy import signal as sig
                    nyq = self.sr / 2
                    sos_hp = sig.butter(2, 80.0 / nyq, btype='high', output='sos')
                    if stem_a.ndim == 2:
                        for ch in range(stem_a.shape[1]):
                            stem_a[:, ch] = sig.sosfiltfilt(sos_hp, stem_a[:, ch])
                            stem_b[:, ch] = sig.sosfiltfilt(sos_hp, stem_b[:, ch])
                    else:
                        stem_a = sig.sosfiltfilt(sos_hp, stem_a)
                        stem_b = sig.sosfiltfilt(sos_hp, stem_b)
                except Exception:
                    pass
            
            # --- 2. Control sub rumble: sidechain bass from drums ---
            if stem_name == 'bass':
                try:
                    drums_a = stems_a.get('drums', np.zeros_like(stem_a))
                    drums_b = stems_b.get('drums', np.zeros_like(stem_b))
                    
                    def apply_ducking(bass_arr, drum_arr):
                        if np.max(np.abs(drum_arr)) < 1e-4: return bass_arr
                        drum_env = np.abs(np.mean(drum_arr, axis=1) if drum_arr.ndim == 2 else drum_arr)
                        # smooth envelope
                        try:
                            nyq = self.sr / 2
                            sos_env = sig.butter(1, 20.0 / nyq, btype='low', output='sos')
                            drum_env = sig.sosfiltfilt(sos_env, drum_env)
                        except Exception:
                            pass
                        # heavy ducking on kick (loudest drum parts)
                        thresh = np.mean(drum_env) * 1.5
                        duck_gain = np.clip(1.0 - (np.maximum(0, drum_env - thresh) * 2.0), 0.3, 1.0)
                        if bass_arr.ndim == 2:
                            return bass_arr * duck_gain[:, np.newaxis]
                        return bass_arr * duck_gain
                        
                    stem_a = apply_ducking(stem_a, drums_a)
                    stem_b = apply_ducking(stem_b, drums_b)
                except Exception:
                    pass
            
            # Get curves
            curve_spec = curves.get(stem_name, {})
            curve_a = np.array(curve_spec.get('a', [1.0] * n_samples))
            curve_b = np.array(curve_spec.get('b', [0.0] * n_samples))
            
            # Interpolate curves to match audio length
            if len(curve_a) != n_samples:
                curve_a = np.interp(
                    np.linspace(0, len(curve_a) - 1, n_samples),
                    np.arange(len(curve_a)),
                    curve_a
                )
            if len(curve_b) != n_samples:
                curve_b = np.interp(
                    np.linspace(0, len(curve_b) - 1, n_samples),
                    np.arange(len(curve_b)),
                    curve_b
                )
            
            # Apply effects if requested
            if apply_effects and effect_params:
                stem_effects = effect_params.get(stem_name, {})
                for effect_type, eparams in stem_effects.items():
                    if stem_name in ['drums', 'bass']:
                        # Apply filter to outgoing
                        for i in range(10):
                            progress = i / 10
                            chunk_start = int(n_samples * i / 10)
                            chunk_end = int(n_samples * (i + 1) / 10)
                            stem_a[chunk_start:chunk_end] = self.apply_stem_effect(
                                stem_a[chunk_start:chunk_end],
                                effect_type, eparams, progress
                            )
            
            # Apply vocal/bed role plan if provided (skip for vocal_overlay_handoff; curves are explicit)
            conv_type = conversation.get('type', '')
            if role_plan is not None and conv_type != 'vocal_overlay_handoff':
                bed_src = role_plan.get('bed_source', 'a')
                vocal_src = role_plan.get('vocal_source', 'a')
                drum_keep = role_plan.get('bed_source', 'a')
                
                if stem_name == 'vocals':
                    # Only keep the chosen vocal source; mute the other.
                    if vocal_src == 'a':
                        curve_b = np.zeros_like(curve_b)
                    elif vocal_src == 'b':
                        curve_a = np.zeros_like(curve_a)
                elif stem_name == 'drums':
                    # Force single drum owner to avoid dual-kit clashes.
                    if drum_keep == 'a':
                        curve_b = np.zeros_like(curve_b)
                    elif drum_keep == 'b':
                        curve_a = np.zeros_like(curve_a)
                else:
                    # Non-vocal stems: gently bias towards the chosen bed source
                    if bed_src == 'a':
                        curve_b *= 0.3
                    elif bed_src == 'b':
                        curve_a *= 0.3

            # Mix at level: equal-power normalize ONLY where both sources are
            # genuinely active (the crossfade overlap). Normalizing everywhere
            # boosted a lone curve up to 1.0, erasing intentional ducks (e.g. the
            # 0.15 call-response bed) and role-plan mutes (curve=0 -> 1.0).
            if mix_at_level:
                power = np.sqrt(np.square(curve_a) + np.square(curve_b) + 1e-12)
                both_active = (curve_a > 0.05) & (curve_b > 0.05)
                scale = np.where(both_active & (power > 1e-8), 1.0 / power, 1.0)
                curve_a = curve_a * scale
                curve_b = curve_b * scale

            # Apply curves; ensure contributions match mixed shape (mono vs stereo)
            samples = min(len(stem_a), len(stem_b), n_samples)
            curve_a_s = curve_a[:samples]
            curve_b_s = curve_b[:samples]
            
            if out_stereo:
                # Mixed is (n_samples, n_channels); contributions must broadcast to that
                if stem_a.ndim == 1:
                    contribution_a = (stem_a[:samples] * curve_a_s)[:, np.newaxis]
                else:
                    ch_a = min(stem_a.shape[1], n_channels)
                    contribution_a = stem_a[:samples, :ch_a] * curve_a_s[:, np.newaxis]
                    if ch_a < n_channels:
                        contribution_a = np.column_stack([contribution_a] + [contribution_a[:, 0]] * (n_channels - ch_a))
                if stem_b.ndim == 1:
                    contribution_b = (stem_b[:samples] * curve_b_s)[:, np.newaxis]
                else:
                    ch_b = min(stem_b.shape[1], n_channels)
                    contribution_b = stem_b[:samples, :ch_b] * curve_b_s[:, np.newaxis]
                    if ch_b < n_channels:
                        contribution_b = np.column_stack([contribution_b] + [contribution_b[:, 0]] * (n_channels - ch_b))
            else:
                # Mixed is 1D
                if stem_a.ndim == 1:
                    contribution_a = stem_a[:samples] * curve_a_s
                else:
                    contribution_a = np.mean(stem_a[:samples], axis=1) * curve_a_s
                if stem_b.ndim == 1:
                    contribution_b = stem_b[:samples] * curve_b_s
                else:
                    contribution_b = np.mean(stem_b[:samples], axis=1) * curve_b_s
            
            mixed[:samples] += contribution_a + contribution_b
        
        
        # finalize=False: return the stem mix exactly as the curves define it. The live
        # engine (club_mixer) loudness-matches whole tracks and limits the overlap itself;
        # the ducking / -1.4 dB tail / peak-normalise / tanh stages below would put a
        # level step (and saturation) between the transition and the songs around it.
        if not finalize:
            return mixed

        # --- 3. Tame distortion/harshness: reduce crossfade gain where loud transients overlap ---
        try:
            from scipy import signal as sig
            energy = np.mean(mixed**2, axis=1) if mixed.ndim == 2 else mixed**2
            nyq = self.sr / 2
            sos_env = sig.butter(2, 10.0 / nyq, btype='low', output='sos')
            smooth_energy = sig.sosfiltfilt(sos_env, energy)
            threshold = np.mean(smooth_energy) * 3.0
            if threshold > 0:
                # Duck the mix slightly during massive transient pile-ups
                gain_reduction = np.clip(threshold / (smooth_energy + 1e-10), 0.7, 1.0)
                if mixed.ndim == 2:
                    mixed *= gain_reduction[:, np.newaxis]
                else:
                    mixed *= gain_reduction
        except Exception:
            pass

        # --- 4. Transition tails: lower crossfade depth slightly in the last ~5s ---
        tail_samples = int(5.0 * self.sr)
        if len(mixed) > tail_samples:
            tail_curve = np.linspace(1.0, 0.85, tail_samples)
            if mixed.ndim == 2:
                mixed[-tail_samples:] *= tail_curve[:, np.newaxis]
            else:
                mixed[-tail_samples:] *= tail_curve
        
        # --- 5. Soft-limit output + enforce headroom ---
        # Target: -2 dBTP (0.794) for hot transitions, -1 dBTP (0.891) for normal
        max_val = np.max(np.abs(mixed))
        if max_val > 1e-12:
            target = 0.79433 if mix_at_level else 0.89125
            if max_val > target:
                # Soft-limit via tanh — preserves transient shape vs hard normalization
                mixed = np.tanh(mixed / max_val * 1.5) * target
            else:
                # Normalize up to target safely
                mixed = mixed * (target / max_val)
        
        return mixed
    
    def analyze_stems_for_orchestration(self,
                                        stems_a: Dict[str, np.ndarray],
                                        stems_b: Dict[str, np.ndarray]) -> Dict:
        """
        Analyze stems to recommend orchestration strategy.
        """
        analysis = {'stems_a': {}, 'stems_b': {}}
        
        for name, stem in stems_a.items():
            if len(stem) > 0:
                energy = float(np.sqrt(np.mean(stem ** 2)))
                analysis['stems_a'][name] = {
                    'energy': energy,
                    'has_content': energy > 0.01
                }
        
        for name, stem in stems_b.items():
            if len(stem) > 0:
                energy = float(np.sqrt(np.mean(stem ** 2)))
                analysis['stems_b'][name] = {
                    'energy': energy,
                    'has_content': energy > 0.01
                }
        
        # Recommend conversation type
        has_vocals_a = analysis['stems_a'].get('vocals', {}).get('has_content', False)
        has_vocals_b = analysis['stems_b'].get('vocals', {}).get('has_content', False)
        has_bass_a = analysis['stems_a'].get('bass', {}).get('has_content', False)
        has_bass_b = analysis['stems_b'].get('bass', {}).get('has_content', False)
        has_drums_a = analysis['stems_a'].get('drums', {}).get('has_content', False)
        has_drums_b = analysis['stems_b'].get('drums', {}).get('has_content', False)
        
        if has_vocals_a and has_vocals_b:
            # Both have vocals: vary approach instead of always layering B on A bed
            # - bass_from_a_beat_from_b: smooth vocal crossfade A→B, A bass + B drums
            # - counter_melody: A vocals stay over B rhythm
            # - layered_reveal: gradual B stem reveal
            # - interweave: stems transition at different times
            # - progressive_morph: stems transform content A→B progressively
            pool = ['bass_from_a_beat_from_b', 'counter_melody', 'layered_reveal',
                    'interweave', 'progressive_morph']
            recommended = random.choice(pool)
        elif has_vocals_a and not has_vocals_b:
            # Song A vocals can play over Song B
            recommended = 'counter_melody'
        elif has_vocals_b and not has_vocals_a:
            # Reveal Song B vocals last
            recommended = 'layered_reveal'
        elif has_bass_a and has_bass_b and has_drums_a and has_drums_b:
            # Both have bass and drums: progressive morph or bass_from_a_beat_from_b
            pool = ['bass_from_a_beat_from_b', 'progressive_morph']
            recommended = random.choice(pool)
        else:
            # Instrumental - call and response works well
            recommended = 'call_response'
        
        analysis['recommended_conversation'] = recommended
        analysis['reasoning'] = f"Stems: A vocals={has_vocals_a}, B vocals={has_vocals_b} → {recommended}"
        
        return analysis
