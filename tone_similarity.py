"""
Tone Similarity Tool
Compare a spoken vowel to a Chao tone (1-5 scale).
Usage: python tone_similarity.py ref.wav perf.wav 214 output_plot.png(optional)
"""
import sys
import numpy as np
import parselmouth
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
# Constants
# N_points, Noise gat eratio arbitrary but not super important
N_POINTS = 50
DTW_SENSITIVITY = 1.0
PITCH_FLOOR = 75
PITCH_CEILING = 600
TIME_STEP = 0.01
CHAO_MIN = 1.0
CHAO_MAX = 5.0
NOISE_GATE_RATIO = 0.1 
def preprocess_audio(snd):
    samples = snd.values[0].copy()
    sample_rate = snd.sampling_frequency
    # pre-emphasis filter
    samples[1:] -= 0.97 * samples[:-1]
    #noise gate to kill echo
    window_size = int(sample_rate * 0.02)
    peak_rms = 0.0
    for i in range(0, len(samples) - window_size, window_size):
        rms = np.sqrt(np.mean(samples[i:i + window_size] ** 2))
        if rms > peak_rms: peak_rms = rms
    if peak_rms > 0:
        for i in range(0, len(samples) - window_size, window_size):
            rms = np.sqrt(np.mean(samples[i:i + window_size] ** 2))
            if rms < peak_rms * NOISE_GATE_RATIO:
                samples[i:i + window_size] = 0.0
    return parselmouth.Sound(samples, sampling_frequency=sample_rate)
def extract_f0(wav_path, label="", floor=PITCH_FLOOR, ceiling=PITCH_CEILING):
    snd = parselmouth.Sound(wav_path)
    snd = preprocess_audio(snd)
    pitch = snd.to_pitch(time_step=TIME_STEP, pitch_floor=floor, pitch_ceiling=ceiling)
    f0 = pitch.selected_array["frequency"].astype(float)
    f0[f0 == 0] = np.nan
    if np.all(np.isnan(f0)):
        raise Exception(f"{wav_path} has no pitch")
    print(f"[{label}] Got {len(f0)} frames, {int((~np.isnan(f0)).mean()*100)}% voiced")
    return f0
def voiced_span(f0):
    # end trimming
    voiced_indices = np.where(~np.isnan(f0))[0]
    if len(voiced_indices) == 0: return f0
    
    f0 = f0[voiced_indices[0]:voiced_indices[-1] + 1].copy()
    # interpolation
    nans = np.isnan(f0)
    if np.any(nans):
        idx = np.arange(len(f0))
        f0 = np.interp(idx, idx[~nans], f0[~nans])
    return gaussian_filter1d(f0, sigma=2)
def build_template_chao(chao_string):
    from scipy.interpolate import PchipInterpolator
    digits = [int(c) for c in chao_string if c.isdigit()]
    if len(digits) < 2: raise ValueError("Need at least 2 digits for a contour")
    
    x_keys = np.linspace(0, 1, len(digits))
    x_full = np.linspace(0, 1, N_POINTS)
    return PchipInterpolator(x_keys, digits)(x_full)
def dtw_distance(a, b):
    # Standard DTW implementation
    n, m = len(a), len(b)
    D = np.full((n, m), np.inf)
    cost = np.abs(a[:, None] - b[None, :])
    D[0, 0] = cost[0, 0]
    for i in range(1, n): D[i, 0] = D[i-1, 0] + cost[i, 0]
    for j in range(1, m): D[0, j] = D[0, j-1] + cost[0, j]
    
    for i in range(1, n):
        for j in range(1, m):
            D[i, j] = cost[i, j] + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    # Backtrack to get path length for normalization
    i, j, path_len = n-1, m-1, 1
    while i > 0 or j > 0:
        if i == 0: j -= 1
        elif j == 0: i -= 1
        else:
            step = np.argmin([D[i-1, j], D[i, j-1], D[i-1, j-1]])
            if step == 0: i -= 1
            elif step == 1: j -= 1
            else: i -= 1; j -= 1
        path_len += 1
    return float(D[n-1, m-1] / path_len)
def plot_contours(template, performance, score, output_path):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 4))
    plt.plot(template, 'k--', label='Target', alpha=0.6)
    plt.plot(performance, 'b-', linewidth=2, label='User')
    
    plt.title(f"Score: {score*100:.1f}%")
    plt.ylabel("Chao Level (1-5)")
    plt.xlabel("Time")
    plt.ylim(0.5, 5.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")
def main():
    if len(sys.argv) < 4:
        print("Usage: python script.py ref.wav perf.wav 214")
        return
    ref_path, perf_path, chao_str = sys.argv[1:4]
    
    # Get Speaker Range
    # 5 and 95 is arbitrary, can change it if needed, will be very impactful
    ref_f0 = extract_f0(ref_path, "Ref")
    s_min = np.percentile(ref_f0[~np.isnan(ref_f0)], 5)
    s_max = np.percentile(ref_f0[~np.isnan(ref_f0)], 95)
    
    # Get Performance
    perf_f0 = voiced_span(extract_f0(perf_path, "Perf", floor=s_min*0.8, ceiling=s_max*1.5))
    
    # Normalize to Chao
    log_f0 = np.log2(perf_f0)
    perf_chao = 1 + (log_f0 - np.log2(s_min)) / (np.log2(s_max) - np.log2(s_min)) * 4
    
    # Resample & Compare
    x_old = np.linspace(0, 1, len(perf_chao))
    x_new = np.linspace(0, 1, N_POINTS)
    perf_resampled = interp1d(x_old, perf_chao)(x_new)
    target = build_template_chao(chao_str)
    
    dist = dtw_distance(perf_resampled, target)
    score = np.exp(-dist / DTW_SENSITIVITY)
    # unclear if this is menaingful to provide to a user
    print(f"\nFinal Score: {score*100:.2f}% (Dist: {dist:.3f})")
    
    plot_path = sys.argv[4] if len(sys.argv) > 4 else "result.png"
    plot_contours(target, perf_resampled, score, plot_path)
if __name__ == "__main__":
    main()
