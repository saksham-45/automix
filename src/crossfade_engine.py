"""
Perceptual Crossfade Engine

Perceptually-correct crossfading and volume automation:
- Equal-power crossfades
- LUFS-based loudness matching
- Adaptive curves based on frequency content
"""
import numpy as np
from typing import Dict, Tuple, List

from src.psychoacoustics import PsychoacousticAnalyzer


class CrossfadeEngine:
    """
    Perceptual crossfade engine using psychoacoustic principles.
    """
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
        self.psychoacoustics = PsychoacousticAnalyzer(sr=sr)
    
    def create_equal_power_crossfade(self, n_samples: int, curve_shape: str = 'smooth') -> Tuple[np.ndarray, np.ndarray]:
        """
        Create equal-power crossfade curves.
        
        Equal-power maintains perceived loudness (not linear energy).
        """
        t = np.linspace(0, 1, n_samples)
        
        if curve_shape == 'linear':
            # Linear power curves
            fade_out = np.sqrt(1 - t)  # Square root for equal power
            fade_in = np.sqrt(t)
        
        elif curve_shape == 'smooth':
            # Smooth S-curve using cosine
            fade_out = np.sqrt(0.5 * (1 + np.cos(np.pi * t)))
            fade_in = np.sqrt(0.5 * (1 - np.cos(np.pi * t)))
        
        elif curve_shape == 'fast':
            # Faster (steeper) but still EQUAL POWER: keep fade_out^2 + fade_in^2 == 1.
            # The old version was sqrt((1-t)^2)=1-t and sqrt(t^2)=t, i.e. LINEAR gains,
            # which sum to power 0.5 at the midpoint -> an audible ~3 dB dip.
            fade_out = (1 - t) ** 2
            fade_in = np.sqrt(np.clip(1.0 - fade_out ** 2, 0.0, 1.0))
        
        else:  # 'smooth' default
            fade_out = np.sqrt(0.5 * (1 + np.cos(np.pi * t)))
            fade_in = np.sqrt(0.5 * (1 - np.cos(np.pi * t)))
        
        return fade_out, fade_in
    
    def create_gradual_crossfade(self, n_samples: int, overlap_ratio: float = 0.75) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create extra-gradual, smooth crossfade curves to prevent instrument conflicts.
        
        Uses a very gentle, extended fade that ensures:
        - Outgoing track fades out subtly and gradually
        - Incoming track starts very quiet and picks up smoothly
        - No sudden volume changes that cause instrument conflicts
        
        Args:
            n_samples: Number of samples in transition
            overlap_ratio: How much overlap (0.75 = balanced gradual fade)
        """
        t = np.linspace(0, 1, n_samples)
        
        # Equal-power crossfade curves (maintains constant perceived loudness)
        # Outgoing: smooth cosine fade from 1.0 to 0.0
        fade_out = np.sqrt(0.5 * (1 + np.cos(np.pi * t)))
        
        # Incoming: smooth cosine fade from 0.0 to 1.0
        fade_in = np.sqrt(0.5 * (1 - np.cos(np.pi * t)))
        
        # Apply extra smoothing with Gaussian filter for ultra-smooth curves
        from src.curve_smooth import gaussian_filter1d
        sigma = min(400, max(100, n_samples / 30))  # Reasonable smoothing
        fade_out = gaussian_filter1d(fade_out, sigma=sigma)
        fade_in = gaussian_filter1d(fade_in, sigma=sigma)
        
        # Ensure smooth boundaries (don't let fade_out go to exactly 0, fade_in start at exactly 0)
        # Keep small tails for smoothness
        fade_out = np.maximum(fade_out, 0.005)  # Minimum 0.5% at end
        fade_in = np.maximum(fade_in, 0.005)    # Minimum 0.5% at start
        
        # Re-normalize to maintain equal-power after smoothing
        total_power = fade_out**2 + fade_in**2
        normalization = np.sqrt(np.maximum(total_power, 0.01))
        fade_out = fade_out / normalization
        fade_in = fade_in / normalization
        
        # Final smooth boundaries - very gradual tail-off
        tail_len = int(n_samples * 0.02)  # Last 2% very gradual
        if tail_len > 0:
            fade_out[-tail_len:] = np.linspace(fade_out[-tail_len], 0.01, tail_len)
            fade_in[:tail_len] = np.linspace(0.01, fade_in[tail_len], tail_len)
        
        # Final clip to [0, 1]
        fade_out = np.clip(fade_out, 0, 1)
        fade_in = np.clip(fade_in, 0, 1)
        
        return fade_out, fade_in
    
    def create_drum_handoff_curves(self,
                                   n_samples: int,
                                   tempo: float,
                                   handoff_ratio: float = 0.25,
                                   overlap_beats: float = 0.0,
                                   fade_a_beats: float = 1.0,
                                   fade_b_beats: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create beat-aligned drum handoff curves for seamless drum switch.
        
        A drums fade to 0 ending exactly at a downbeat. B drums start at that downbeat.
        No overlapping drum stems = no clash. Switch on downbeat = imperceptible.
        
        Args:
            n_samples: Total samples in transition
            tempo: BPM (used to compute beat grid)
            handoff_ratio: Approximate ratio (0-1) for when switch should occur
            overlap_beats: 0 = hard cutover (A fully out before B). 0.5 = 1-beat overlap for smoothness
            fade_a_beats: Beats over which A drums fade out (ending at switch point)
            fade_b_beats: Beats over which B drums fade in (starting at switch point)
        
        Returns:
            (curve_a, curve_b) - A drums out, B drums in, sequential not overlapping
        """
        if tempo <= 0 or tempo > 300:
            tempo = 120.0
        beat_duration_sec = 60.0 / tempo
        beat_samples = int(beat_duration_sec * self.sr)
        bar_samples = beat_samples * 4  # 4 beats per bar
        
        # Find next downbeat after handoff_ratio
        handoff_sample = int(n_samples * handoff_ratio)
        handoff_sample = max(bar_samples, min(handoff_sample, n_samples - bar_samples))
        # Snap to next bar boundary (downbeat)
        bars_from_start = (handoff_sample + bar_samples - 1) // bar_samples
        switch_sample = bars_from_start * bar_samples
        switch_sample = min(switch_sample, n_samples - 1)
        switch_sample = max(beat_samples, switch_sample)
        
        # A drums: fade out ending exactly at switch_sample
        fade_a_samples = min(int(fade_a_beats * beat_samples), switch_sample)
        a_fade_start = switch_sample - fade_a_samples
        
        curve_a = np.ones(n_samples)
        curve_a[switch_sample:] = 0.0
        if fade_a_samples > 0 and a_fade_start >= 0:
            t = np.linspace(0, 1, fade_a_samples)
            curve_a[a_fade_start:switch_sample] = 0.5 * (1 + np.cos(np.pi * t))
        # Minimal smoothing for drums only (avoid extending overlap)
        from src.curve_smooth import gaussian_filter1d
        curve_a = gaussian_filter1d(np.clip(curve_a, 0, 1), sigma=min(20, max(1, fade_a_samples // 10)))
        curve_a = np.clip(curve_a, 0, 1)
        
        # B drums: start at switch_sample (or slightly before if overlap_beats > 0)
        b_start_offset = int(overlap_beats * beat_samples) if overlap_beats > 0 else 0
        b_start_sample = max(0, switch_sample - b_start_offset)
        fade_b_samples = min(int(fade_b_beats * beat_samples), n_samples - b_start_sample)
        b_end_sample = min(n_samples, b_start_sample + fade_b_samples)
        
        curve_b = np.zeros(n_samples)
        curve_b[:b_start_sample] = 0.0
        curve_b[b_end_sample:] = 1.0
        if fade_b_samples > 0 and b_end_sample > b_start_sample:
            t = np.linspace(0, 1, fade_b_samples)
            curve_b[b_start_sample:b_end_sample] = 0.5 * (1 - np.cos(np.pi * t))
        curve_b = gaussian_filter1d(np.clip(curve_b, 0, 1), sigma=min(20, max(1, fade_b_samples // 10)))
        curve_b = np.clip(curve_b, 0, 1)
        return curve_a, curve_b
    
    def create_fast_fade(self, n_samples: int, fade_out_ratio: float = 0.25) -> np.ndarray:
        """
        Create aggressive fade curve that drops to near-zero quickly.
        
        Used for fading out drums/bass in first portion of transition.
        
        Args:
            n_samples: Number of samples
            fade_out_ratio: Ratio of samples to fade out (0.25 = fade in first 25%)
        
        Returns:
            Fade curve array (starts at 1.0, drops to ~0.01 in fade_out_ratio)
        """
        fade_out_samples = int(n_samples * fade_out_ratio)
        fade_out_samples = max(1, min(fade_out_samples, n_samples))
        
        # Create exponential fade for smooth but fast drop
        t = np.linspace(0, 1, fade_out_samples)
        # Exponential curve: starts at 1, drops to 0.01
        fade_curve = 0.01 + (0.99 * np.exp(-5 * t))  # Fast exponential decay
        
        # Create full curve
        full_curve = np.ones(n_samples) * 0.01  # Start at minimum
        full_curve[:fade_out_samples] = fade_curve
        
        # Smooth the transition point to avoid clicks
        if fade_out_samples < n_samples:
            # Smooth the last few samples of fade
            smooth_len = min(100, fade_out_samples // 4)
            if smooth_len > 0:
                smooth_region = np.linspace(
                    fade_curve[-smooth_len] if len(fade_curve) >= smooth_len else 0.01,
                    0.01,
                    smooth_len
                )
                full_curve[fade_out_samples-smooth_len:fade_out_samples] = smooth_region
        
        return np.clip(full_curve, 0.01, 1.0)
    
    def create_aggressive_vocal_fade(self, 
                                    n_samples: int, 
                                    vocal_start_time_ratio: float = 0.5,
                                    aggressive_drop_ratio: float = 0.9) -> np.ndarray:
        """
        Create aggressive vocal fade that:
        - Gradually fades from start until Song B vocals begin
        - Sudden drop at the very end (last 10%)
        
        Args:
            n_samples: Total number of samples
            vocal_start_time_ratio: When Song B vocals start (0.5 = 50% through transition)
            aggressive_drop_ratio: When to start sudden drop (0.9 = last 10%)
        
        Returns:
            Fade curve array (starts at 1.0, fades gradually, then drops suddenly)
        """
        vocal_start_idx = int(n_samples * vocal_start_time_ratio)
        drop_start_idx = int(n_samples * aggressive_drop_ratio)
        
        fade_curve = np.ones(n_samples)
        
        # Phase 1: Gradual fade until Song B vocals start
        if vocal_start_idx > 0:
            # Gradual exponential fade from 1.0 to 0.3
            t = np.linspace(0, 1, vocal_start_idx)
            gradual_fade = 0.3 + (0.7 * np.exp(-3 * t))  # Exponential decay
            fade_curve[:vocal_start_idx] = gradual_fade
        
        # Phase 2: Hold at reduced level (if there's space)
        if drop_start_idx > vocal_start_idx:
            fade_curve[vocal_start_idx:drop_start_idx] = 0.3
        
        # Phase 3: Sudden aggressive drop at very end
        if drop_start_idx < n_samples:
            drop_samples = n_samples - drop_start_idx
            t = np.linspace(0, 1, drop_samples)
            # Aggressive exponential drop (cubed for extra aggression)
            sudden_drop = 0.3 * (1 - t) ** 3
            fade_curve[drop_start_idx:] = sudden_drop
        
        # Smooth transitions between phases
        from src.curve_smooth import gaussian_filter1d
        fade_curve = gaussian_filter1d(fade_curve, sigma=max(10, n_samples / 200))
        
        return np.clip(fade_curve, 0.0, 1.0)
    
    def create_moderate_vocal_fade(self,
                                   n_samples: int,
                                   fade_complete_by_ratio: float = 0.85,
                                   fade_duration_ratio: float = 0.25) -> np.ndarray:
        """
        Create moderately steep vocal fade (outgoing). No sudden drop; fade completes
        by fade_complete_by_ratio using a cosine curve over fade_duration_ratio.
        
        Args:
            n_samples: Total number of samples
            fade_complete_by_ratio: Ratio (0-1) by which fade must reach 0
            fade_duration_ratio: Ratio of segment used for the fade (e.g. 0.25 = 25%)
        
        Returns:
            Fade curve array (starts at 1.0, reaches 0 by fade_complete_by_ratio)
        """
        fade_complete_sample = int(n_samples * fade_complete_by_ratio)
        fade_complete_sample = max(1, min(fade_complete_sample, n_samples))
        fade_duration_samples = int(n_samples * fade_duration_ratio)
        fade_duration_samples = max(1, min(fade_duration_samples, fade_complete_sample))
        fade_start_sample = max(0, fade_complete_sample - fade_duration_samples)
        
        fade_curve = np.ones(n_samples)
        fade_curve[fade_complete_sample:] = 0.0
        if fade_start_sample < fade_complete_sample:
            n_fade = fade_complete_sample - fade_start_sample
            t = np.linspace(0, 1, n_fade)
            # Moderately steep: cosine (0.5 * (1 + cos(pi*t))) goes 1 -> 0
            fade_curve[fade_start_sample:fade_complete_sample] = 0.5 * (1 + np.cos(np.pi * t))
        
        from src.curve_smooth import gaussian_filter1d
        fade_curve = gaussian_filter1d(fade_curve, sigma=max(5, n_samples / 400))
        return np.clip(fade_curve, 0.0, 1.0)
    
    def create_gentler_fade_in(self,
                               n_samples: int,
                               fade_start_ratio: float = 0.0,
                               fade_duration_ratio: float = 0.30) -> np.ndarray:
        """
        Create gentler fade-in (incoming). Rises with cosine over fade_duration_ratio
        so the rise is "a little less fast" than linear.
        
        Args:
            n_samples: Total number of samples
            fade_start_ratio: Ratio (0-1) at which fade-in starts
            fade_duration_ratio: Ratio of segment used for the rise (e.g. 0.30 = 30%)
        
        Returns:
            Fade curve array (0 until fade_start, then cosine rise to 1.0)
        """
        fade_start_sample = int(n_samples * fade_start_ratio)
        fade_start_sample = max(0, min(fade_start_sample, n_samples - 1))
        fade_duration_samples = int(n_samples * fade_duration_ratio)
        fade_end_sample = min(n_samples, fade_start_sample + fade_duration_samples)
        fade_end_sample = max(fade_start_sample + 1, fade_end_sample)
        
        fade_curve = np.zeros(n_samples)
        fade_curve[:fade_start_sample] = 0.0
        fade_curve[fade_end_sample:] = 1.0
        if fade_end_sample > fade_start_sample:
            n_fade = fade_end_sample - fade_start_sample
            t = np.linspace(0, 1, n_fade)
            # Cosine rise: 0.5 * (1 - cos(pi*t)) goes 0 -> 1
            fade_curve[fade_start_sample:fade_end_sample] = 0.5 * (1 - np.cos(np.pi * t))
        
        from src.curve_smooth import gaussian_filter1d
        fade_curve = gaussian_filter1d(fade_curve, sigma=max(5, n_samples / 400))
        return np.clip(fade_curve, 0.0, 1.0)
    
    def create_multi_stage_curve(self, 
                                 n_samples: int,
                                 stages: List[Dict]) -> np.ndarray:
        """
        Create multi-stage fade curve with different behaviors per stage.
        
        Args:
            n_samples: Total samples
            stages: List of stage dicts with:
                - 'start': start ratio (0-1)
                - 'end': end ratio (0-1)
                - 'fade_type': 'hold', 'linear', 'exponential', 'smooth', 'aggressive'
                - 'value': value for 'hold' type
                - 'end_value': target value for fade types
        
        Returns:
            Fade curve array
        """
        curve = np.zeros(n_samples)
        
        for stage in stages:
            start_idx = int(n_samples * stage['start'])
            end_idx = int(n_samples * stage['end'])
            stage_samples = end_idx - start_idx
            
            if stage_samples <= 0:
                continue
            
            fade_type = stage.get('fade_type', 'linear')
            
            if fade_type == 'hold':
                value = stage.get('value', 1.0)
                curve[start_idx:end_idx] = value
            
            elif fade_type == 'linear':
                end_value = stage.get('end_value', 0.0)
                start_value = curve[start_idx - 1] if start_idx > 0 else 1.0
                curve[start_idx:end_idx] = np.linspace(start_value, end_value, stage_samples)
            
            elif fade_type == 'exponential':
                end_value = stage.get('end_value', 0.0)
                start_value = curve[start_idx - 1] if start_idx > 0 else 1.0
                t = np.linspace(0, 1, stage_samples)
                curve[start_idx:end_idx] = start_value * np.exp(-3 * t) + end_value * (1 - np.exp(-3 * t))
            
            elif fade_type == 'smooth':
                end_value = stage.get('end_value', 1.0)
                start_value = curve[start_idx - 1] if start_idx > 0 else 0.0
                t = np.linspace(0, 1, stage_samples)
                # Smooth cosine curve
                curve[start_idx:end_idx] = start_value + (end_value - start_value) * 0.5 * (1 - np.cos(np.pi * t))
            
            elif fade_type == 'aggressive':
                end_value = stage.get('end_value', 0.0)
                start_value = curve[start_idx - 1] if start_idx > 0 else 1.0
                t = np.linspace(0, 1, stage_samples)
                # Aggressive cubic drop
                curve[start_idx:end_idx] = start_value * (1 - t) ** 3 + end_value * (1 - (1 - t) ** 3)
            
            elif fade_type == 'moderate':
                end_value = stage.get('end_value', 0.0)
                start_value = curve[start_idx - 1] if start_idx > 0 else 1.0
                t = np.linspace(0, 1, stage_samples)
                # Cosine: w = 0.5*(1+cos(pi*t)) goes 1->0 so curve = w*start + (1-w)*end
                w = 0.5 * (1 + np.cos(np.pi * t))
                curve[start_idx:end_idx] = w * start_value + (1 - w) * end_value
        
        # Smooth the entire curve
        from src.curve_smooth import gaussian_filter1d
        curve = gaussian_filter1d(curve, sigma=max(5, n_samples / 300))
        
        return np.clip(curve, 0.0, 1.0)
    
    def create_lufs_matched_curves(self,
                                  y_a: np.ndarray,
                                  y_b: np.ndarray,
                                  n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create volume curves that match LUFS (perceptual loudness).
        
        Maintains consistent perceived loudness throughout transition.
        """
        # Analyze LUFS for both signals
        lufs_a = self.psychoacoustics.analyze_loudness_lufs(y_a)
        lufs_b = self.psychoacoustics.analyze_loudness_lufs(y_b)
        
        # Target LUFS (average, or match to one)
        target_lufs = (lufs_a['integrated_lufs'] + lufs_b['integrated_lufs']) / 2
        
        # Calculate gain adjustments needed
        gain_a_db = target_lufs - lufs_a['integrated_lufs']
        gain_b_db = target_lufs - lufs_b['integrated_lufs']
        
        # Convert to linear gain
        gain_a = 10 ** (gain_a_db / 20)
        gain_b = 10 ** (gain_b_db / 20)
        
        # Create crossfade curves
        fade_out, fade_in = self.create_equal_power_crossfade(n_samples, 'smooth')
        
        # Apply gains
        vol_a = fade_out * gain_a
        vol_b = fade_in * gain_b
        
        return vol_a, vol_b
    
    def create_adaptive_crossfade(self,
                                 y_a: np.ndarray,
                                 y_b: np.ndarray,
                                 n_samples: int,
                                 frequency_aware: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create adaptive crossfade curves based on frequency content.
        
        Different fade rates for different frequency bands prevent masking.
        """
        # Base equal-power curves
        fade_out, fade_in = self.create_equal_power_crossfade(n_samples, 'smooth')
        
        if not frequency_aware:
            return fade_out, fade_in
        
        # Analyze frequency content
        # If signal A has strong bass, fade bass slower
        # If signal B has strong highs, fade highs in faster
        
        # Simplified: use overall spectral centroid
        import librosa
        
        centroid_a = np.mean(librosa.feature.spectral_centroid(y=y_a, sr=self.sr))
        centroid_b = np.mean(librosa.feature.spectral_centroid(y=y_b, sr=self.sr))
        
        # Adjust curves based on brightness
        # Brighter sounds (higher centroid) can be faded faster
        brightness_factor_a = min(1.5, centroid_a / 2000)  # Normalize
        brightness_factor_b = min(1.5, centroid_b / 2000)
        
        # Adjust fade rates
        # Brighter = faster fade
        fade_out_adj = fade_out ** (1 / brightness_factor_a)
        fade_in_adj = fade_in ** brightness_factor_b
        
        # Smooth the adjusted curves
        from src.curve_smooth import gaussian_filter1d
        fade_out_adj = gaussian_filter1d(fade_out_adj, sigma=n_samples / 100)
        fade_in_adj = gaussian_filter1d(fade_in_adj, sigma=n_samples / 100)
        
        # Normalize to prevent clipping
        fade_out_adj = np.clip(fade_out_adj, 0, 1)
        fade_in_adj = np.clip(fade_in_adj, 0, 1)
        
        return fade_out_adj, fade_in_adj
    
    def apply_crossfade(self,
                       y_a: np.ndarray,
                       y_b: np.ndarray,
                       vol_a: np.ndarray,
                       vol_b: np.ndarray) -> np.ndarray:
        """
        Apply crossfade curves to audio signals.
        
        Args:
            y_a: Outgoing audio
            y_b: Incoming audio
            vol_a: Volume curve for A
            vol_b: Volume curve for B
        
        Returns:
            Crossfaded audio
        """
        n_samples = min(len(y_a), len(y_b), len(vol_a), len(vol_b))
        
        # Ensure stereo
        if y_a.ndim == 1:
            y_a = y_a[:, np.newaxis]
        if y_b.ndim == 1:
            y_b = y_b[:, np.newaxis]
        
        # Resample volume curves if needed
        if len(vol_a) != n_samples:
            indices = np.linspace(0, len(vol_a) - 1, n_samples)
            vol_a = np.interp(indices, np.arange(len(vol_a)), vol_a)
        if len(vol_b) != n_samples:
            indices = np.linspace(0, len(vol_b) - 1, n_samples)
            vol_b = np.interp(indices, np.arange(len(vol_b)), vol_b)
        
        # Apply volumes
        if vol_a.ndim == 1:
            vol_a = vol_a[:, np.newaxis]
        if vol_b.ndim == 1:
            vol_b = vol_b[:, np.newaxis]
        
        
        mixed = y_a[:n_samples] * vol_a[:n_samples] + y_b[:n_samples] * vol_b[:n_samples]
        
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(mixed))
        if max_val > 0.95:
            mixed = mixed * (0.95 / max_val)
        
        
        return mixed
    
    def create_multi_stage_fade(self,
                               n_samples: int,
                               stages: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create multi-stage fade curves.
        
        Different fade rates for different frequency bands or time segments.
        
        Args:
            n_samples: Total samples
            stages: List of stage dicts with 'start_ratio', 'end_ratio', 'curve_shape'
        
        Returns:
            (fade_out, fade_in) curves
        """
        fade_out = np.ones(n_samples)
        fade_in = np.zeros(n_samples)
        
        for stage in stages:
            start_idx = int(stage['start_ratio'] * n_samples)
            end_idx = int(stage['end_ratio'] * n_samples)
            stage_samples = end_idx - start_idx
            
            if stage_samples <= 0:
                continue
            
            # Create curve for this stage
            stage_fade_out, stage_fade_in = self.create_equal_power_crossfade(
                stage_samples, stage.get('curve_shape', 'smooth')
            )
            
            # Insert into main curves
            fade_out[start_idx:end_idx] = stage_fade_out
            fade_in[start_idx:end_idx] = stage_fade_in
        
        return fade_out, fade_in

