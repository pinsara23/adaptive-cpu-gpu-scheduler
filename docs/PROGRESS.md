# HPC Project Progress

## Current status
- Current stage: CPU baseline block COMPLETE (Stages 1-4 + stabilization). Next: Stage 5 (GPU).
- Last updated: 2026-11-15
- Last code commit: b7a2a6c (official CPU baseline v2 measured on this commit)

## Stage status
| Stage | Status |
|---|---|
| 1 Environment | Done; all versions recorded below |
| 2 Sequential pipeline (A0) | Done; verified; reference saved (~/hpc_ref/A0_ref.md5, A0_ref.mp4) |
| 3 Timing and tracing | Done for A0-A5: frame_id, decode/copy/enqueue/stage/encode events, CSV export. video_timestamp not yet exported |
| 4 CPU algorithms (A1-A5) | Done; all bit-identical to A0; benchmarked in two modes |
| 5 CUDA/GPU | Partial (pre-existing): device query, naive 3x3 blur kernel, A6. Not yet aligned with CPU pipeline |
| 6 CPU-GPU overlap | Partial (pre-existing): A7-legacy, A8 on synchronous GPU path; unverified |
| 7-12 | Not started (correct per plan) |

## Algorithm status
| Algo | Status | Notes |
|---|---|---|
| A0 sequential | verified, reference | single processing thread; FFmpeg uses 20 decoder threads |
| A1 single core | verified | whole process (incl. decoder threads) pinned to CPU 0 |
| A2 single-core SMT | verified | `A2 1` / `A2 2`: workers on core 0 (CPU 0 / CPUs 0+1), main+decoder on CPU 2 |
| A3 static | verified | per-worker queues, frame_id % workers |
| A4 dynamic | verified | one shared bounded queue (20) |
| A5 pinned | verified | topology-aware pinning: main CPU 0, workers 2,4,6,8 (one per P-core first) |
| A6 | partial, unverified | blur only on GPU; 3x3 kernel != CPU 5x5; native resolution |
| A7 (spec: fixed multi-stream GPU) | MISSING | current "A7" is a 3-thread CPU-GPU pipeline; rename A7_legacy |
| A8 | partial, unverified | synchronous GPU path; own stage chain; encode in writer thread |
| A9, A10, A11 | not started | after GPU baselines |

## Environment
- Ubuntu 24.04.5 LTS, kernel 7.0.0-38-generic
- NVIDIA driver 595.91.07 (CUDA 13.2 reported); nvcc 12.0.140; g++ 13.3.0; CMake 3.28.3; OpenCV 4.6.0
- CPU: i7-12700H, 20 logical CPUs. P-cores with SMT: CPU pairs 0/1, 2/3, 4/5, 6/7, 8/9, 10/11 (max 4.6-4.7 GHz). E-cores: CPUs 12-19 (max 3.5 GHz). L2 11.5 MiB, L3 24 MiB
- GPU: RTX 3050 Laptop, compute capability 8.6, 16 SMs, 3770 MB usable VRAM, 80 W cap, concurrent kernels: yes
- Build: Release (-O3), CUDA arch 86 (sm_86 cubin confirmed), -Wall -Wextra
- Power during benchmarks: AC, platform_profile=performance, governor=powersave (intel_pstate)
- Repo on NTFS (ntfs3) volume; effect on I/O timing not tested
- Input: videos/input.mp4, h264, 768x432, 12.5 fps, 377 frames (not in repo; record sha256 in README)

## Implemented files
- include/: bounded_queue.hpp, config.hpp, cpu_pipeline.hpp, cpu_topology.hpp, filters.hpp, frame_reorder.hpp, frame_task.hpp, gpu_pipeline.hpp, pipeline.hpp, trace_manager.hpp
- src/: cpu_pipeline.cpp, cpu_topology.cpp, filters.cpp, frame_reorder.cpp, gpu_pipeline.cu, main.cpp, pipeline.cpp, trace_manager.cpp
- scripts/: verify_cpu.sh (md5 vs A0 reference, empty-output guard), bench_cpu.sh (warm-up, interleaved repeats, median/min/max, sys/faults/RSS, machine-state header, frame-count check, MODE=full|noenc)
- results/: cpu_bench_full_20261115_131917.csv, cpu_bench_noenc_20261115_132142.csv (official baseline v2) + earlier runs

## Verified functionality
- A0-A5 outputs bit-identical to A0 reference (decoded-frame md5, 377/377) after every change
- Every A0-A5 run reports Frames Emitted = 377 in both modes; reorder fails loudly on missing/stranded/duplicate frames
- --no-encode mode: same processing, no write, real outputs untouched (writes noencode_*.mp4)
- Allocator tuning: faults ~1.07M -> ~24k, sys ~1.0 s -> ~0.06 s, +3.5 MB peak RSS (A0)
- Topology helper (TOPO command) matches lscpu; pinning verified via ps snapshots (A2)

## Test results: official CPU baseline v2 (commit b7a2a6c, 5 repeats after warm-up, 377 frames)
Full mode (encode on):
| algo | wk | median s | min-max | speedup |
|---|---|---|---|---|
| A0 | - | 4.253 | 4.245-4.304 | 1.00 |
| A1 | - | 4.471 | 4.462-4.478 | 0.95 |
| A2 | 1 | 4.215 | 4.205-4.273 | 1.01 |
| A2 | 2 | 3.576 | 3.569-3.578 | 1.19 |
| A3 | 4 | 2.298 | 2.271-2.350 | 1.85 |
| A4 | 4 | 2.319 | 2.304-2.372 | 1.83 |
| A5 | 4 | 2.300 | 2.262-2.320 | 1.85 |

Compute-only (--no-encode):
| algo | wk | median s | min-max | speedup |
|---|---|---|---|---|
| A0 | - | 2.200 | 2.182-2.209 | 1.00 |
| A1 | - | 2.419 | 2.410-2.420 | 0.91 |
| A2 | 1 | 2.161 | 2.157-2.169 | 1.02 |
| A2 | 2 | 2.117 | 2.114-2.127 | 1.04 |
| A3 | 4 | 0.733 | 0.718-0.769 | 3.00 |
| A4 | 4 | 0.713 | 0.711-0.737 | 3.09 |
| A5 | 4 | 0.722 | 0.720-0.730 | 3.05 |

Other measurements:
- Run-to-run noise ~1-5%; differences inside overlapping ranges are not reported as findings
- A4 worker scaling, full: 1w 4.18 s, 2w 2.40, 3w 2.33, 4w 2.30, 6w 2.32, 8w 2.34 (plateau)
- A4 worker scaling, compute-only: 1w 2.19, 2w 1.32, 4w 0.73, 6w 0.55, 8w 0.49
- Per-frame stage means (A0 trace): resize ~0.86, blur ~1.07, clahe ~2.69, sobel ~2.04, sharpen ~0.39, encode ~5.6-5.9 ms
- A4 8 workers compute-only, main thread per frame: decode wait 0.40, copy 0.23, enqueue wait 0.60 ms
- RSS: ~143 MB (A0) to ~210-235 MB (4 workers)

## Findings (supported by data; mechanisms marked as hypotheses)
1. Serial encode caps full-pipeline speedup at ~1.85-1.9x: encode ~5.9 ms/frame under the reorder lock ~= 6.1 ms/frame achieved; ceiling lifts when encode is removed (compute-only scales to 4.5x at 8 workers)
2. Default glibc allocation cost ~1 s/run of kernel page-fault time; fixed with process-wide mallopt (HPC_NO_MALLOC_TUNE=1 restores default)
3. Static vs dynamic vs pinned scheduling: no measurable difference on this workload (both modes)
4. SMT: ~2% gain for filter-only work, 1.18x when encode and filters can overlap on the two hardware threads (mechanism = hypothesis; needs perf IPC)
5. Confining FFmpeg decoder threads to the processing CPU (A1) costs 5% (full) / 10% (compute-only); moving decode off-core gains ~2%
6. At 8 workers compute-only, workers (not the main thread) are the limit: main thread waits ~0.6 ms/frame on a full queue (cause untested: E-cores, memory bandwidth, or lock contention)
7. Original A5 penalty (~25%) came from main thread and worker 0 sharing one physical core; fixed by topology-aware pinning

## Design decisions
- Shared CPU stage chain: process_frame_cpu() (cpu_pipeline.cpp); A0-A5 all use it
- Working resolution: config::TARGET_WIDTH/HEIGHT = 1280x720 (A0-A5)
- Filters: GaussianBlur 5x5 sigma 0; CLAHE clip 2.0, 8x8 tiles (grayscale); Sobel 3x3 blended 0.5/0.5; sharpen 3x3 kernel; output grayscale mp4v
- Timing: steady_clock; Total Time = first read to writer released; setup and trace flush excluded; A0 also prints Loop Time
- Process-wide malloc tuning in main.cpp (applies to all algorithms)
- Two measurement modes: full (default; correctness and demo) and --no-encode (compute-only benchmarking)
- Benchmark protocol: one discarded warm-up pass, >=5 interleaved repeats, report median + min/max, record machine state
- Thread placement uses cpu_assignment_order(): one logical CPU per SMT core, then single-thread cores, then siblings
- cv::setNumThreads(1): OpenCV internal parallelism disabled; FFmpeg decoder threads remain (documented)
- Reorder buffer always admits the next-expected frame (no deadlock); completeness checked against decoded count

## Known issues / open items
- M1: GPU blur (3x3, border copy, truncation) differs from CPU blur (5x5, reflect, rounding)
- M2: A6-A8 use native 768x432, CPU path uses 1280x720
- M9: no CUDA streams/pinned memory/CUDA events; A7 mislabeled; only blur on GPU
- A6-A8 not on shared code path, no frames-emitted check, timing boundaries differ
- video_timestamp not exported to traces
- Encode trace label in FrameReorder is "CPU_MAIN" although executed by worker threads
- Tracing overhead not measured (--no-trace flag planned)
- NTFS vs ext4 effect on I/O not tested
- Input/output paths hardcoded; working resolution not a CLI option (needed for large-video tests)
- Trace keeps all events in RAM (problem for very long videos)
- A0 and A1 Frames Emitted = loop count (sanity warning vs container metadata added)

## Future evaluation goal (user)
- End-of-project tests on many videos, 100 to 1,000,000 frames, multiple resolutions
- Needs: path/resolution CLI args, --no-trace or streaming trace, per-video A0 references, generated long inputs
- 1M-frame full-mode output ~115 GB at 720p and ~100 min/run: plan full matrix up to 100k, reduced set (compute-only, best algorithms, 1-3 repeats, ext4 output) for 1M

## Next steps
1. Stage 5 decisions: GPU blur definition (match CPU 5x5 sigma-0 within documented tolerance) and one working resolution for all algorithms (M1, M2)
2. GPU correctness checker (tolerance-based, vs A0)
3. Stage 5 implementation: CUDA error checking, pinned host memory, CUDA events for timing, stream pool -> real A7 (fixed multi-stream); rename old A7 to A7_legacy
4. Bring A6/A8 onto shared code path + frames-emitted check; re-verify A8
5. Define A9 inputs/outputs/objective before coding