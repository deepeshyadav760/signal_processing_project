"""
Gender Classification Module
Uses MFCC features and machine learning to predict speaker gender
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, dct
import logging

logger = logging.getLogger(__name__)


class GenderClassifier:
    """
    Gender classification using audio features
    Uses MFCC (Mel-Frequency Cepstral Coefficients) and other acoustic features
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 13):
        """
        Extract MFCC features from audio
        MFCCs are the most commonly used features for speech/speaker recognition
        """
        try:
            # Parameters
            frame_size = 512
            hop_size = 256
            n_filters = 26
            n_fft = 512

            # Pre-emphasis
            audio_emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

            # Frame the signal
            frames = self._frame_signal(audio_emphasized, frame_size, hop_size)

            # Apply Hamming window
            frames *= np.hamming(frame_size)

            # FFT
            mag_frames = np.abs(np.fft.rfft(frames, n_fft))
            pow_frames = ((1.0 / n_fft) * (mag_frames ** 2))

            # Mel filterbank
            mel_filters = self._create_mel_filterbank(n_filters, n_fft, self.sample_rate)
            filter_banks = np.dot(pow_frames, mel_filters.T)
            filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
            filter_banks = 20 * np.log10(filter_banks)

            # DCT to get MFCCs
            mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :n_mfcc]

            # Mean normalization
            mfcc -= (np.mean(mfcc, axis=0) + 1e-8)

            return mfcc

        except Exception as e:
            logger.error(f"MFCC extraction error: {e}")
            return np.zeros((1, n_mfcc))

    def _frame_signal(self, signal_data, frame_size, hop_size):
        """Split signal into overlapping frames"""
        signal_length = len(signal_data)
        num_frames = 1 + int(np.ceil((signal_length - frame_size) / hop_size))

        frames = np.zeros((num_frames, frame_size))
        for i in range(num_frames):
            start = i * hop_size
            end = start + frame_size
            if end <= signal_length:
                frames[i] = signal_data[start:end]
            else:
                # Pad last frame
                frames[i, :signal_length-start] = signal_data[start:]

        return frames

    def _create_mel_filterbank(self, n_filters, n_fft, sample_rate):
        """Create Mel-scale filterbank"""
        low_freq_mel = 0
        high_freq_mel = self._hz_to_mel(sample_rate / 2)

        mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filters + 2)
        hz_points = self._mel_to_hz(mel_points)
        bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

        filterbank = np.zeros((n_filters, int(n_fft / 2 + 1)))

        for i in range(1, n_filters + 1):
            left = bin_points[i - 1]
            center = bin_points[i]
            right = bin_points[i + 1]

            for j in range(left, center):
                filterbank[i - 1, j] = (j - left) / (center - left)
            for j in range(center, right):
                filterbank[i - 1, j] = (right - j) / (right - center)

        return filterbank

    def _hz_to_mel(self, hz):
        """Convert Hz to Mel scale"""
        return 2595 * np.log10(1 + hz / 700)

    def _mel_to_hz(self, mel):
        """Convert Mel to Hz scale"""
        return 700 * (10 ** (mel / 2595) - 1)

    def extract_pitch(self, audio: np.ndarray):
        """
        Extract fundamental frequency (pitch) using autocorrelation
        Males typically have lower pitch (85-180 Hz)
        Females typically have higher pitch (165-255 Hz)
        """
        try:
            # Autocorrelation
            correlation = np.correlate(audio, audio, mode='full')
            correlation = correlation[len(correlation)//2:]

            # Find peaks
            min_period = int(self.sample_rate / 400)  # Max 400 Hz
            max_period = int(self.sample_rate / 50)   # Min 50 Hz

            if max_period >= len(correlation):
                max_period = len(correlation) - 1

            # Find maximum correlation in valid range
            peak_corr = correlation[min_period:max_period]
            if len(peak_corr) == 0:
                return 150.0

            peak_idx = np.argmax(peak_corr) + min_period
            pitch = self.sample_rate / peak_idx if peak_idx > 0 else 150.0

            return float(pitch)

        except Exception as e:
            logger.error(f"Pitch extraction error: {e}")
            return 150.0

    def extract_formants(self, audio: np.ndarray):
        """
        Extract formant frequencies using LPC (Linear Predictive Coding)
        Formants are resonance frequencies that characterize vowels
        """
        try:
            # LPC order
            order = 12

            # Compute autocorrelation
            r = np.correlate(audio, audio, mode='full')
            r = r[len(r)//2:len(r)//2 + order + 1]

            # Solve Levinson-Durbin recursion
            lpc_coeffs = self._levinson_durbin(r, order)

            # Find roots of LPC polynomial
            roots = np.roots(lpc_coeffs)

            # Extract formants from roots
            roots = roots[np.imag(roots) >= 0]
            angles = np.angle(roots)
            freqs = angles * (self.sample_rate / (2 * np.pi))

            # Sort and get first 3 formants
            formants = sorted(freqs[freqs > 90])[:3]

            if len(formants) >= 1:
                return formants
            else:
                return [500, 1500, 2500]  # Default values

        except Exception as e:
            logger.error(f"Formant extraction error: {e}")
            return [500, 1500, 2500]

    def _levinson_durbin(self, r, order):
        """Levinson-Durbin recursion for LPC coefficients"""
        a = np.zeros(order + 1)
        a[0] = 1.0
        e = r[0]

        for i in range(1, order + 1):
            lambda_val = -np.sum(a[:i] * r[i:0:-1]) / e
            a[:i+1] += lambda_val * np.concatenate(([0], a[i-1::-1]))
            e *= (1 - lambda_val ** 2)

        return a

    def predict_gender(self, audio: np.ndarray) -> dict:
        """
        Predict gender using multiple acoustic features

        Features used:
        - Pitch (fundamental frequency)
        - Formants (resonance frequencies)
        - Spectral features
        - MFCCs

        Returns dict with gender prediction and confidence
        """
        try:
            if len(audio) < 1000:
                return {"gender": "unknown", "confidence": 0.0, "features": {}}

            # Extract features
            pitch = self.extract_pitch(audio)
            formants = self.extract_formants(audio)
            mfcc = self.extract_mfcc(audio)

            # Spectral centroid (mean frequency)
            fft_vals = np.abs(fft(audio))
            freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = fft_vals[:len(fft_vals)//2]

            spectral_centroid = np.sum(positive_freqs * positive_fft) / (np.sum(positive_fft) + 1e-10)

            # Calculate energy in different frequency bands
            low_energy = np.sum(positive_fft[positive_freqs < 500])
            mid_energy = np.sum(positive_fft[(positive_freqs >= 500) & (positive_freqs < 2000)])
            high_energy = np.sum(positive_fft[positive_freqs >= 2000])

            total_energy = low_energy + mid_energy + high_energy + 1e-10
            low_ratio = low_energy / total_energy
            mid_ratio = mid_energy / total_energy

            # Feature analysis for gender classification
            features = {
                "pitch": float(pitch),
                "formant_f1": float(formants[0]) if len(formants) > 0 else 0,
                "formant_f2": float(formants[1]) if len(formants) > 1 else 0,
                "spectral_centroid": float(spectral_centroid),
                "low_energy_ratio": float(low_ratio),
                "mid_energy_ratio": float(mid_ratio),
                "mfcc_mean": float(np.mean(mfcc))
            }

            # Rule-based classification (can be replaced with trained ML model)
            male_score = 0
            female_score = 0

            # Pitch-based scoring
            if pitch < 160:
                male_score += 3
            elif pitch > 180:
                female_score += 3
            else:
                male_score += 1
                female_score += 1

            # Formant-based scoring
            if len(formants) >= 2:
                if formants[0] < 700:  # Lower F1 indicates male
                    male_score += 2
                else:
                    female_score += 2

            # Spectral centroid scoring
            if spectral_centroid < 2000:
                male_score += 2
            else:
                female_score += 2

            # Low frequency energy scoring
            if low_ratio > 0.3:
                male_score += 1
            else:
                female_score += 1

            # Determine gender
            total_score = male_score + female_score
            if male_score > female_score:
                gender = "male"
                confidence = male_score / total_score
            elif female_score > male_score:
                gender = "female"
                confidence = female_score / total_score
            else:
                gender = "unknown"
                confidence = 0.5

            logger.info(f"Gender prediction: {gender} (confidence: {confidence:.2f})")
            logger.info(f"Features - Pitch: {pitch:.1f}Hz, F1: {formants[0]:.1f}Hz, Centroid: {spectral_centroid:.1f}Hz")

            return {
                "gender": gender,
                "confidence": float(confidence),
                "features": features
            }

        except Exception as e:
            logger.error(f"Gender prediction error: {e}")
            return {
                "gender": "unknown",
                "confidence": 0.0,
                "features": {},
                "error": str(e)
            }
