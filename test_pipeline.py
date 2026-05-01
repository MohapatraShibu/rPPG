"""
validate the rPPG signal processing pipeline with synthetic RGB traces.
simulates what the extractor would produce for a face at known HR/RR.
run: venv/Scripts/python test_pipeline.py
"""
import math
import time
import numpy as np
from rppg.signal_processing import chrom, estimate_bpm, estimate_rr

FPS = 30.0
DURATION = 60
CHUNK_SEC = 5.0
HR_HZ = 1.2 # 72 BPM
RR_HZ = 0.25 # 15 br/min

def synthetic_rgb(n_frames, fps, hr_hz, rr_hz, noise=0.02):
    # simulate mean RGB trace from a face with known HR and RR
    t = np.arange(n_frames) / fps
    pulse  = np.sin(2 * np.pi * hr_hz * t)
    breath = np.sin(2 * np.pi * rr_hz * t)
    noise_arr = np.random.randn(n_frames) * noise
    r = 170 + 12 * pulse + 3 * breath + noise_arr * 10
    g = 110 + 18 * pulse + 4 * breath + noise_arr * 15
    b =  90 +  4 * pulse + 2 * breath + noise_arr *  5
    return np.stack([r, g, b], axis=1).astype(np.float32)

np.random.seed(42)
total_frames = int(FPS * DURATION)
rgb_full = synthetic_rgb(total_frames, FPS, HR_HZ, RR_HZ)

chunk_frames = int(FPS * CHUNK_SEC)
chunks = [rgb_full[i:i+chunk_frames] for i in range(0, total_frames, chunk_frames)
          if i + chunk_frames <= total_frames]

print(f"\n{'='*55}")
print(f" Synthetic Pipeline Test  |  GT HR={HR_HZ*60:.0f} BPM  RR={RR_HZ*60:.0f} br/min")
print(f"{'='*55}")

chunk_bpms = []
t_pipeline = time.perf_counter()
for idx, chunk in enumerate(chunks):
    t0 = time.perf_counter()
    sig = chrom(chunk, FPS)
    bpm = estimate_bpm(sig, FPS)
    rr = estimate_rr(chunk, FPS)
    lat = (time.perf_counter() - t0) * 1000
    chunk_bpms.append(bpm)
    t_s, t_e = idx * CHUNK_SEC, (idx + 1) * CHUNK_SEC
    rr_str = f"{rr:.1f}" if not math.isnan(rr) else "n/a"
    print(f"  Chunk {idx+1:>2}  [{t_s:.0f}s–{t_e:.0f}s]  "
          f"HR: {bpm:5.1f} BPM  |  RR: {rr_str:>5} br/min  |  {lat:.2f} ms")

total_time = time.perf_counter() - t_pipeline
sig_full = chrom(rgb_full, FPS)
overall_bpm = estimate_bpm(sig_full, FPS)
overall_rr  = estimate_rr(rgb_full, FPS)

print(f"\n{'='*55}")
print(f"  OVERALL")
print(f"{'='*55}")
print(f"Full-signal BPM  : {overall_bpm:.1f}  (GT: {HR_HZ*60:.0f})")
print(f"Median chunk BPM : {np.median(chunk_bpms):.1f}  (GT: {HR_HZ*60:.0f})")
print(f"Respiratory Rate : {overall_rr:.1f} br/min  (GT: {RR_HZ*60:.0f})")
print(f"Total runtime    : {total_time*1000:.1f} ms for {DURATION}s of signal")
print(f"Real-time factor : {DURATION/total_time:.0f}x faster than real-time")
print(f"{'='*55}\n")
