"""
near real-time rPPG pipeline
Usage: python main.py <video_path> [--chunk 5]
outputs per-chunk BPM, overall BPM, respiratory rate, and runtime metrics.
algorithm: CHROM (de Haan & Jeanne, IEEE TBME 2013)
face ROI:  MediaPipe FaceMesh
"""
import argparse
import time
import json
import numpy as np

from rppg.video import open_video, iter_chunks
from rppg.signal_processing import chrom, estimate_bpm, estimate_rr

def parse_args():
    p = argparse.ArgumentParser(description="rPPG BPM estimator")
    p.add_argument("video", help="Path to input video (≥60 s recommended)")
    p.add_argument("--chunk", type=float, default=5.0,
                   help="Chunk size in seconds (default: 5)")
    p.add_argument("--json", action="store_true",
                   help="Also write results to results.json")
    return p.parse_args()

def main():
    args = parse_args()
    cap, meta = open_video(args.video)

    print(f"\n{'='*55}")
    print(f"  rPPG Pipeline  |  CHROM + MediaPipe FaceMesh")
    print(f"{'='*55}")
    print(f" Video: {args.video}")
    print(f" Resolution : {meta.width}x{meta.height}  |  FPS: {meta.fps:.1f}")
    print(f" Duration: {meta.duration_s:.1f}s  |  Chunk: {args.chunk}s")
    print(f" Resize: frames downscaled to max 640px wide for speed")
    print(f"{'='*55}\n")

    chunk_results = []
    all_rgb: list[np.ndarray] = []
    pipeline_start = time.perf_counter()

    for chunk_idx, rgb_trace in iter_chunks(cap, meta, chunk_sec=args.chunk):
        t0 = time.perf_counter()
        signal = chrom(rgb_trace, meta.fps)
        bpm = estimate_bpm(signal, meta.fps)
        rr = estimate_rr(rgb_trace, meta.fps)
        latency_ms = (time.perf_counter() - t0) * 1000

        t_start = chunk_idx * args.chunk
        t_end = t_start + len(rgb_trace) / meta.fps
        chunk_results.append({
            "chunk": chunk_idx + 1,
            "window_s": f"{t_start:.1f}–{t_end:.1f}",
            "bpm": round(bpm, 1),
            "rr_bpm": round(rr, 1) if not np.isnan(rr) else None,
            "frames": len(rgb_trace),
            "latency_ms": round(latency_ms, 2),
        })
        all_rgb.append(rgb_trace)

        rr_str = f"{rr:.1f}" if not np.isnan(rr) else "n/a"
        print(f"Chunk {chunk_idx+1:>2}  [{t_start:.0f}s–{t_end:.0f}s]"
              f"HR: {bpm:5.1f} BPM  |  RR: {rr_str:>5} br/min"
              f" |  latency: {latency_ms:.1f} ms")

    cap.release()

    if not chunk_results:
        print("\n[ERROR] No face detected in any chunk. Check video quality.")
        return

    # overall estimate
    full_rgb = np.concatenate(all_rgb, axis=0)
    full_signal = chrom(full_rgb, meta.fps)
    overall_bpm = estimate_bpm(full_signal, meta.fps)
    overall_rr = estimate_rr(full_rgb, meta.fps)
    total_time = (time.perf_counter() - pipeline_start)

    chunk_bpms = [c["bpm"] for c in chunk_results]
    median_bpm = float(np.median(chunk_bpms))

    print(f"\n{'='*55}")
    print(f"  OVERALL RESULTS")
    print(f"{'='*55}")
    print(f" Full-signal BPM  : {overall_bpm:.1f}")
    print(f" Median chunk BPM : {median_bpm:.1f}")
    rr_str = f"{overall_rr:.1f}" if not np.isnan(overall_rr) else "n/a"
    print(f" Respiratory Rate : {rr_str} br/min")
    print(f" Chunks processed : {len(chunk_results)}")
    print(f" Total runtime : {total_time:.2f}s  "
          f"(real-time factor: {meta.duration_s/total_time:.1f}x)")
    print(f"{'='*55}\n")

    if args.json:
        output = {
            "video": args.video,
            "fps": meta.fps,
            "duration_s": meta.duration_s,
            "chunk_sec": args.chunk,
            "chunks": chunk_results,
            "overall": {
                "full_signal_bpm": round(overall_bpm, 1),
                "median_chunk_bpm": round(median_bpm, 1),
                "respiratory_rate_brpm": round(overall_rr, 1) if not np.isnan(overall_rr) else None,
            },
            "runtime": {
                "total_s": round(total_time, 3),
                "realtime_factor": round(meta.duration_s / total_time, 2),
            },
        }
        with open("results.json", "w") as f:
            json.dump(output, f, indent=2)
        print("  Results saved -> results.json\n")

if __name__ == "__main__":
    main()
