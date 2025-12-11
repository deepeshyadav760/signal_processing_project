"""
DSP (Digital Signal Processing) Module
Handles all audio preprocessing and noise reduction
"""

import numpy as np
from scipy import signal
from scipy.fft import fft
import logging

logger = logging.getLogger(__name__)


class DSPProcessor:
    """Complete DSP pipeline for audio processing"""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.frame_length = 512

    def pre_emphasis(self, audio: np.ndarray, alpha: float = 0.97) -> np.ndarray:
        """
        Pre-emphasis filter to boost high frequencies
        This enhances high-frequency components which improves speech clarity
        """
        if len(audio) < 2:
            return audio
        return np.append(audio[0], audio[1:] - alpha * audio[:-1])

    def compute_stft(self, audio: np.ndarray, nperseg: int = None):
        """
        Compute Short-Time Fourier Transform
        Converts time-domain signal to frequency domain
        """
        if nperseg is None:
            nperseg = self.frame_length

        f, t, Zxx = signal.stft(audio, fs=self.sample_rate, nperseg=nperseg)
        return f, t, Zxx

    def estimate_noise_profile(self, audio: np.ndarray, noise_duration: float = 0.5):
        """
        Estimate noise profile from the first portion of audio
        Assumes first 0.5s contains mostly noise
        """
        noise_samples = int(noise_duration * self.sample_rate)
        noise_samples = min(noise_samples, len(audio) // 4)  # Use max 25% of audio

        if noise_samples < 100:
            noise_samples = min(100, len(audio))

        noise_segment = audio[:noise_samples]

        _, _, Zxx_noise = signal.stft(noise_segment, fs=self.sample_rate, nperseg=self.frame_length)
        noise_profile = np.mean(np.abs(Zxx_noise), axis=1)

        return noise_profile

    def spectral_subtraction(self, Zxx: np.ndarray, noise_profile: np.ndarray, alpha: float = 2.0):
        """
        Spectral subtraction for noise reduction
        Subtracts estimated noise spectrum from signal spectrum
        """
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)

        # Subtract noise profile with over-subtraction factor alpha
        clean_magnitude = magnitude - alpha * noise_profile[:, np.newaxis]

        # Apply spectral floor to avoid negative values (musical noise)
        clean_magnitude = np.maximum(clean_magnitude, 0.1 * magnitude)

        # Reconstruct complex spectrum
        return clean_magnitude * np.exp(1j * phase)

    def wiener_filter(self, Zxx: np.ndarray, noise_power: float):
        """
        Wiener filtering for optimal noise reduction
        Balances noise reduction with signal preservation
        """
        signal_power = np.abs(Zxx) ** 2

        # Calculate Wiener gain
        gain = signal_power / (signal_power + noise_power + 1e-10)

        # Apply gain to complex spectrum
        return Zxx * gain

    def voice_activity_detection(self, audio: np.ndarray, energy_threshold: float = 0.02):
        """
        Voice Activity Detection (VAD)
        Detects segments with speech vs silence/noise
        """
        frame_size = 512
        hop_size = 256

        # Calculate energy for each frame
        energy = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i+frame_size]
            frame_energy = np.sqrt(np.mean(frame ** 2))
            energy.append(frame_energy)

        energy = np.array(energy)

        # Adaptive threshold based on audio statistics
        if len(energy) > 10:
            adaptive_threshold = np.median(energy) * 3
            threshold = max(energy_threshold, adaptive_threshold)
        else:
            threshold = energy_threshold

        # Find voice active frames
        voice_frames = energy > threshold

        # Calculate voice activity ratio
        voice_ratio = np.sum(voice_frames) / len(voice_frames) if len(voice_frames) > 0 else 0

        return voice_ratio, voice_frames

    def apply_bandpass_filter(self, audio: np.ndarray, lowcut: float = 80, highcut: float = 7900):
        """
        Apply bandpass filter to focus on speech frequencies
        Human speech typically ranges from 80Hz to 8kHz
        Using 7900Hz to ensure we stay below Nyquist frequency (8000Hz)
        """
        try:
            nyquist = self.sample_rate / 2
            low = lowcut / nyquist
            high = highcut / nyquist

            # Validate filter parameters (must be 0 < freq < 1)
            if low <= 0 or low >= 0.99 or high <= 0 or high >= 0.99 or low >= high:
                logger.warning(f"Invalid filter params: low={low:.3f}, high={high:.3f}, skipping bandpass")
                return audio

            # Design Butterworth bandpass filter (4th order)
            b, a = signal.butter(4, [low, high], btype='band')

            # Apply filter (zero-phase filtering)
            filtered = signal.filtfilt(b, a, audio)

            logger.info(f"✅ Bandpass filter applied: {lowcut}Hz - {highcut}Hz")
            return filtered
        except Exception as e:
            logger.error(f"Bandpass filter error: {e}")
            return audio

    def process_audio(self, audio_data: np.ndarray, enable_dsp: bool = True) -> dict:
        """
        Complete DSP pipeline with all processing steps
        Returns processed audio and analysis results
        """
        if not enable_dsp or len(audio_data) < 1000:
            logger.info("DSP disabled or audio too short - bypassing processing")
            return {
                "audio": audio_data,
                "audio_original": audio_data,
                "voice_ratio": 1.0,
                "energy": float(np.mean(np.abs(audio_data))),
                "dsp_applied": False,
                "noise_level": 0
            }

        try:
            logger.info("=" * 60)
            logger.info("🔊 STARTING DSP PIPELINE")
            logger.info("=" * 60)

            # Original audio stats
            orig_energy = float(np.mean(np.abs(audio_data)))
            orig_max = float(np.max(np.abs(audio_data)))
            logger.info(f"📊 Original Audio - Energy: {orig_energy:.6f}, Max: {orig_max:.6f}")

            # Step 1: Voice Activity Detection
            voice_ratio, voice_frames = self.voice_activity_detection(audio_data)
            logger.info(f"✅ Step 1: VAD - Voice activity ratio: {voice_ratio:.2f} ({int(voice_ratio*100)}%)")

            # Step 2: Bandpass filtering for speech frequencies (80Hz - 7900Hz)
            audio_filtered = self.apply_bandpass_filter(audio_data)
            filtered_energy = float(np.mean(np.abs(audio_filtered)))
            logger.info(f"✅ Step 2: Bandpass - Energy after filtering: {filtered_energy:.6f}")

            # Step 3: Pre-emphasis filter (boost high frequencies)
            audio_preemph = self.pre_emphasis(audio_filtered)
            preemph_energy = float(np.mean(np.abs(audio_preemph)))
            logger.info(f"✅ Step 3: Pre-emphasis - Energy: {preemph_energy:.6f}")

            # Step 4: STFT (convert to frequency domain)
            f, t, Zxx = self.compute_stft(audio_preemph)
            logger.info(f"✅ Step 4: STFT - Frequency bins: {len(f)}, Time frames: {len(t)}")

            # Step 5: Noise estimation (estimate background noise)
            noise_profile = self.estimate_noise_profile(audio_data)
            noise_level = float(np.mean(noise_profile))
            logger.info(f"✅ Step 5: Noise Estimation - Average noise level: {noise_level:.6f}")

            # Step 6: Spectral subtraction (remove noise from spectrum)
            Zxx_clean = self.spectral_subtraction(Zxx, noise_profile, alpha=2.0)
            logger.info(f"✅ Step 6: Spectral Subtraction - Over-subtraction factor: 2.0")

            # Step 7: Wiener filtering (optimal signal recovery)
            noise_power = np.mean(noise_profile ** 2)
            Zxx_clean = self.wiener_filter(Zxx_clean, noise_power)
            logger.info(f"✅ Step 7: Wiener Filter - Noise power: {noise_power:.6f}")

            # Step 8: Inverse STFT to get clean audio (back to time domain)
            _, audio_clean = signal.istft(Zxx_clean, fs=self.sample_rate, nperseg=self.frame_length)
            logger.info(f"✅ Step 8: Inverse STFT - Reconstructed {len(audio_clean)} samples")

            # Normalize to prevent clipping
            max_val = np.max(np.abs(audio_clean))
            if max_val > 0:
                audio_clean = audio_clean / max_val * 0.9

            final_energy = float(np.mean(np.abs(audio_clean)))

            # Calculate noise reduction
            noise_reduction_db = 20 * np.log10((orig_energy + 1e-10) / (noise_level + 1e-10))

            # Calculate PSNR (Peak Signal-to-Noise Ratio)
            # PSNR measures quality of reconstructed signal vs original
            # Higher PSNR = better quality (less distortion)
            mse = np.mean((audio_data - audio_clean) ** 2)
            if mse > 0:
                max_signal = np.max(np.abs(audio_data))
                psnr = 20 * np.log10(max_signal / np.sqrt(mse))
            else:
                psnr = 100.0  # Perfect reconstruction

            # Calculate SNR (Signal-to-Noise Ratio)
            # SNR compares signal power to noise power
            signal_power = np.mean(audio_clean ** 2)
            noise_power = np.mean((audio_data - audio_clean) ** 2)
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
            else:
                snr = 100.0  # No noise

            logger.info(f"📊 Final Audio - Energy: {final_energy:.6f}")
            logger.info(f"📉 Noise Reduction: {noise_reduction_db:.1f} dB")
            logger.info(f"📈 PSNR (Peak Signal-to-Noise Ratio): {psnr:.1f} dB")
            logger.info(f"📡 SNR (Signal-to-Noise Ratio): {snr:.1f} dB")
            logger.info("=" * 60)
            logger.info("✅ DSP PIPELINE COMPLETE")
            logger.info("=" * 60)

            return {
                "audio": audio_clean,
                "audio_original": audio_data,  # Keep original for comparison
                "voice_ratio": float(voice_ratio),
                "energy": final_energy,
                "dsp_applied": True,
                "noise_level": noise_level,
                "noise_reduction_db": float(noise_reduction_db),
                "psnr": float(psnr),
                "snr": float(snr)
            }

        except Exception as e:
            logger.error(f"DSP processing error: {e}")
            return {
                "audio": audio_data,
                "voice_ratio": 1.0,
                "energy": float(np.mean(np.abs(audio_data))),
                "dsp_applied": False,
                "error": str(e)
            }
