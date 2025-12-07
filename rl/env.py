"""Gym environment for denoising-policy search."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from gymnasium import Env, spaces
from pystoi import stoi as stoi_metric

try:
    import pywt
except ImportError:  # pragma: no cover
    pywt = None

from .dataset import FeatureDataset

EPS = 1e-9

METHODS = ("spectral_sub", "wiener", "spectral_gate", "wavelet")


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = reference - np.mean(reference)
    est = estimate - np.mean(estimate)
    if np.sum(ref**2) < EPS:
        return -30.0
    scale = np.dot(ref, est) / (np.sum(ref**2) + EPS)
    proj = scale * ref
    noise = est - proj
    return 10 * np.log10((np.sum(proj**2) + EPS) / (np.sum(noise**2) + EPS))


def spectral_sub(noisy: np.ndarray, sr: int, strength: float,
                 smoothing: float, band_mix: float,
                 n_fft: int, hop_length: int) -> np.ndarray:
    if len(noisy) == 0:
        return noisy
    stft = librosa.stft(noisy, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(stft), np.angle(stft)
    noise_floor = np.median(mag, axis=1, keepdims=True)
    bias = 1.0 + 2.0 * band_mix
    freq_weights = np.linspace(1.0, bias, noise_floor.shape[0])[:, None]
    mask = np.clip(1.0 - strength * freq_weights * noise_floor / (mag + EPS), 0.0, 1.0)
    mask = mask ** (1.0 + smoothing * 2.0)
    enhanced = (mag * mask) * np.exp(1j * phase)
    audio = librosa.istft(enhanced, hop_length=hop_length, length=len(noisy))
    return audio.astype(np.float32)


def wiener_filter(noisy: np.ndarray, sr: int, strength: float,
                  smoothing: float, band_mix: float,
                  frame_len: int = 1024, hop_len: int = 256) -> np.ndarray:
    if len(noisy) == 0:
        return noisy
    stft = librosa.stft(noisy, n_fft=frame_len, hop_length=hop_len)
    mag = np.abs(stft)
    freq_bias = np.linspace(1.0 - band_mix, 1.0 + band_mix, mag.shape[0])[:, None]
    noise_psd = (1 - strength) * np.mean(mag, axis=1, keepdims=True) * freq_bias
    gain = mag**2 / (mag**2 + noise_psd + EPS)
    denoised = librosa.istft(stft * gain, hop_length=hop_len, length=len(noisy))
    alpha = 0.3 + 0.7 * smoothing
    return (alpha * denoised + (1 - alpha) * noisy).astype(np.float32)


def spectral_gate(noisy: np.ndarray, sr: int, strength: float, smoothing: float,
                  band_mix: float, n_fft: int, hop_length: int) -> np.ndarray:
    if len(noisy) == 0:
        return noisy
    stft = librosa.stft(noisy, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(stft), np.angle(stft)
    noise_floor = np.percentile(mag, 30, axis=1, keepdims=True)
    threshold = (1.0 + smoothing * 4.0) * noise_floor * (1.0 + band_mix)
    gain = np.where(mag >= threshold, 1.0, np.clip(1.0 - strength, 0.0, 1.0))
    gain = librosa.decompose.nn_filter(gain, aggregate=np.median, metric="cosine")
    enhanced = (mag * gain) * np.exp(1j * phase)
    return librosa.istft(enhanced, hop_length=hop_length, length=len(noisy)).astype(np.float32)


def mmse_lsa(noisy: np.ndarray, sr: int, strength: float, smoothing: float, band_mix: float,
             n_fft: int, hop_length: int) -> np.ndarray:
    if len(noisy) == 0:
        return noisy
    stft = librosa.stft(noisy, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(stft), np.angle(stft)
    noise_psd = librosa.decompose.nn_filter(mag ** 2, aggregate=np.median)
    post_snr = (mag ** 2) / (noise_psd + EPS)
    prior_snr = smoothing * (post_snr - 1).clip(min=0) + (1 - smoothing)
    gain = prior_snr / (1 + prior_snr) * np.exp(0.5 * scipy_special_expint(prior_snr * post_snr / (1 + prior_snr)))
    gain = np.clip(gain, 0.0, 1.0) ** strength
    gain = gain * (1.0 - band_mix) + (gain ** 0.5) * band_mix
    enhanced = (mag * gain) * np.exp(1j * phase)
    return librosa.istft(enhanced, hop_length=hop_length, length=len(noisy)).astype(np.float32)


def scipy_special_expint(x: np.ndarray) -> np.ndarray:
    """Approximate exponential integral for MMSE-LSA."""
    # Use polynomial approximation for Ei-like behaviour
    return np.log(1 + x + EPS)


def nmf_denoise(noisy: np.ndarray, sr: int, strength: float, smoothing: float, band_mix: float,
                n_fft: int, hop_length: int, n_components: int = 32) -> np.ndarray:
    if len(noisy) == 0:
        return noisy
    stft = librosa.stft(noisy, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(stft), np.angle(stft)
    W, H = librosa.decompose.decompose(mag, n_components=n_components, sort=True)
    noise_components = int(max(1, n_components * (1 - strength)))
    mask = np.dot(W[:, -noise_components:], H[-noise_components:, :])
    mask = 1 - mask / (mag + EPS)
    emphasis = np.linspace(1 - band_mix, 1 + band_mix, mask.shape[0])[:, None]
    mask = np.clip(mask * emphasis, 0.0, 1.0) ** (1.0 + smoothing)
    enhanced = (mag * mask) * np.exp(1j * phase)
    return librosa.istft(enhanced, hop_length=hop_length, length=len(noisy)).astype(np.float32)


def wavelet_denoise(noisy: np.ndarray, strength: float, smoothing: float, band_mix: float) -> np.ndarray:
    if pywt is None or len(noisy) == 0:
        return noisy
    coeffs = pywt.wavedec(noisy, "db8", mode="periodization")
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    t = strength * sigma * np.sqrt(2 * np.log(len(noisy)))
    new_coeffs = [coeffs[0]]
    for level, detail in enumerate(coeffs[1:], start=1):
        new_detail = pywt.threshold(detail, t, mode="soft")
        blend = smoothing * (level / len(coeffs)) * (0.5 + band_mix / 2)
        new_coeffs.append((1 - blend) * detail + blend * new_detail)
    return pywt.waverec(new_coeffs, "db8", mode="periodization").astype(np.float32)[:len(noisy)]


class DenoiseEnv(Env):
    """Multi-step env: choose denoising method + strength sequentially."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset: FeatureDataset,
        reward_scale: float = 5.0,
        seed: int | None = None,
        n_fft: int = 1024,
        hop_length: int = 256,
        max_steps: int = 3,
        stoi_weight: float = 0.0,
        estoi_weight: float = 0.0,
        quality_interval: int = 0,
        diversity_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.reward_scale = reward_scale
        self.rng = np.random.default_rng(seed)
        self.n_fft = n_fft
        self.hop = hop_length
        self.max_steps = max_steps
        self.stoi_weight = stoi_weight
        self.estoi_weight = estoi_weight
        self.quality_interval = quality_interval
        self._audio_cache: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
        self.diversity_weight = diversity_weight

        base_dim = dataset.features.shape[1]
        extra_dim = len(METHODS) + 4  # previous action, strength, smoothing, bandmix, step fraction
        self.obs_dim = base_dim + extra_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([len(METHODS) - 1 + EPS, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._current_idx = 0
        self._clean = None
        self._current_audio = None
        self._current_sisdr = 0.0
        self._current_stoi = 0.0
        self._current_estoi = 0.0
        self._base_sisdr = 0.0
        self._base_stoi = 0.0
        self._base_estoi = 0.0
        self.step_count = 0
        self.prev_action_onehot = np.zeros(len(METHODS), dtype=np.float32)
        self.prev_strength = 0.0
        self.prev_smoothing = 0.0
        self.prev_bandmix = 0.0
        self.prev_method_idx: int | None = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._current_idx = self.rng.integers(0, self.dataset.num_samples)
        clean, noisy, sr = self._get_audio_triplet(self._current_idx)
        self._clean = clean
        self._sr = sr
        self._current_audio = noisy.copy()
        self._current_sisdr = si_sdr(self._clean, self._current_audio)
        self._base_sisdr = self._current_sisdr
        self._current_stoi = self._compute_stoi(self._current_audio, extended=False)
        self._current_estoi = self._compute_stoi(self._current_audio, extended=True)
        self._base_stoi = self._current_stoi
        self._base_estoi = self._current_estoi
        self.step_count = 0
        self.prev_action_onehot = np.zeros(len(METHODS), dtype=np.float32)
        self.prev_strength = 0.0
        self.prev_smoothing = 0.0
        self.prev_bandmix = 0.0
        self.prev_method_idx = None
        obs = self._build_obs()
        info = {"utt_id": self.dataset.meta.iloc[self._current_idx]["utt_id"]}
        return obs, info

    def _build_obs(self) -> np.ndarray:
        base_features = self.dataset.features[self._current_idx]
        step_fraction = self.step_count / self.max_steps
        extras = np.concatenate(
            [
                self.prev_action_onehot,
                np.array(
                    [self.prev_strength, self.prev_smoothing, self.prev_bandmix, step_fraction],
                    dtype=np.float32,
                ),
            ]
        )
        return np.concatenate([base_features, extras]).astype(np.float32)

    def _compute_stoi(self, audio: np.ndarray, extended: bool = False) -> float:
        try:
            return float(stoi_metric(self._clean, audio, self._sr, extended=extended))
        except Exception:
            return 0.0

    def step(self, action: np.ndarray):
        raw_method = float(action[0])
        method_idx = int(np.clip(np.round(raw_method), 0, len(METHODS) - 1))
        strength = float(np.clip(action[1], 0.0, 1.0))
        smoothing = float(np.clip(action[2], 0.0, 1.0))
        bandmix = float(np.clip(action[3], 0.0, 1.0))

        if METHODS[method_idx] == "spectral_sub":
            denoised = spectral_sub(self._current_audio, self._sr, strength, smoothing, bandmix, self.n_fft, self.hop)
        elif METHODS[method_idx] == "wiener":
            denoised = wiener_filter(self._current_audio, self._sr, strength, smoothing, bandmix, self.n_fft, self.hop)
        elif METHODS[method_idx] == "spectral_gate":
            denoised = spectral_gate(self._current_audio, self._sr, strength, smoothing, bandmix, self.n_fft, self.hop)
        else:
            denoised = wavelet_denoise(self._current_audio, strength, smoothing, bandmix)

        prev_sisdr = self._current_sisdr
        prev_stoi = self._current_stoi
        prev_estoi = self._current_estoi

        self._current_audio = denoised
        self._current_sisdr = si_sdr(self._clean, self._current_audio)
        gain = self._current_sisdr - prev_sisdr

        should_measure = False
        if self.stoi_weight or self.estoi_weight:
            if self.quality_interval <= 0:
                should_measure = True if self.step_count + 1 >= self.max_steps else False
            else:
                should_measure = ((self.step_count + 1) % self.quality_interval == 0) or (
                    self.step_count + 1 >= self.max_steps
                )

        if should_measure:
            self._current_stoi = self._compute_stoi(self._current_audio, extended=False)
            self._current_estoi = self._compute_stoi(self._current_audio, extended=True)
            stoi_gain = self._current_stoi - prev_stoi
            estoi_gain = self._current_estoi - prev_estoi
        else:
            stoi_gain = 0.0
            estoi_gain = 0.0
        
        reward = gain / self.reward_scale
        if self.stoi_weight:
            reward += self.stoi_weight * stoi_gain
        if self.estoi_weight:
            reward += self.estoi_weight * estoi_gain

        diversity_bonus = 0.0
        if self.prev_method_idx is not None and self.diversity_weight > 0:
            if method_idx != self.prev_method_idx:
                diversity_bonus = self.diversity_weight
            else:
                diversity_bonus = -0.5 * self.diversity_weight
        reward += diversity_bonus

        self.step_count += 1
        self.prev_action_onehot = np.zeros(len(METHODS), dtype=np.float32)
        self.prev_action_onehot[method_idx] = 1.0
        self.prev_strength = strength
        self.prev_smoothing = smoothing
        self.prev_bandmix = bandmix
        self.prev_method_idx = method_idx

        obs = self._build_obs()
        done = self.step_count >= self.max_steps
        info = {
            "utt_id": self.dataset.meta.iloc[self._current_idx]["utt_id"],
            "method": METHODS[method_idx],
            "strength": strength,
            "smoothing": smoothing,
            "bandmix": bandmix,
            "si_sdr": self._current_sisdr,
            "baseline_si_sdr": self._base_sisdr,
            "gain": gain,
            "stoi": self._current_stoi,
            "baseline_stoi": self._base_stoi,
            "stoi_gain": stoi_gain,
            "estoi": self._current_estoi,
            "baseline_estoi": self._base_estoi,
            "estoi_gain": estoi_gain,
            "diversity_bonus": diversity_bonus,
            "step": self.step_count,
        }
        return obs, reward, done, False, info

    def _get_audio_triplet(self, idx: int) -> tuple[np.ndarray, np.ndarray, int]:
        if idx in self._audio_cache:
            return self._audio_cache[idx]
        clean_path = self.dataset.meta.iloc[idx]["clean_path"]
        noisy_path = self.dataset.meta.iloc[idx]["noisy_path"]
        
        # Convert to Path and resolve relative paths
        clean_path = Path(clean_path)
        noisy_path = Path(noisy_path)
        
        # If relative path, try to resolve from current working directory
        if not clean_path.is_absolute():
            clean_path = Path.cwd() / clean_path
        if not noisy_path.is_absolute():
            noisy_path = Path.cwd() / noisy_path
        
        # Check if files exist
        if not clean_path.exists():
            raise FileNotFoundError(
                f"Clean audio file not found: {clean_path} (original: {self.dataset.meta.iloc[idx]['clean_path']})"
            )
        if not noisy_path.exists():
            raise FileNotFoundError(
                f"Noisy audio file not found: {noisy_path} (original: {self.dataset.meta.iloc[idx]['noisy_path']})"
            )
        
        clean, sr_clean = sf.read(str(clean_path))
        noisy, sr_noisy = sf.read(str(noisy_path))
        if sr_clean != sr_noisy:
            noisy = librosa.resample(noisy, orig_sr=sr_noisy, target_sr=sr_clean)
        sr = sr_clean
        min_len = min(len(clean), len(noisy))
        clean = clean[:min_len].astype(np.float32)
        noisy = noisy[:min_len].astype(np.float32)
        triplet = (clean, noisy, sr)
        if len(self._audio_cache) > 1024:
            self._audio_cache.pop(next(iter(self._audio_cache)))
        self._audio_cache[idx] = triplet
        return triplet

