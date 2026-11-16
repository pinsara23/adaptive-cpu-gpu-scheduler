# HPC Project Progress

## Current status
- Current stage: Stage 5 (GPU) COMPLETE for the blur-only GPU path (5a, 5b, 5c verified and benchmarked, 2026-11-17). Next: user decision between Stage 6 (rebuild A8) and GPU roadmap step 2 (see Next steps)
- Last updated: 2026-11-17
- Official CPU baseline v2 measured on commit b7a2a6c; re-confirmed on f477a60 and f6f573b (medians within noise). Day-to-day machine drift ~5% (A0 noenc 2.18-2.29 s): compare only within one interleaved bench run
- Commits: de2736e + follow-up (CUDA optional build), f477a60 (5a), f6f573b (5b), 8030e35 (GPU setup time, extended CUDA_INFO), 46a308d (5c: A7 + GPU_STREAM_BENCH, old A7 removed), 9a49958 (bench/verify parsing fix)

## Stage status
| Stage | Status |
|---|---|
| 1 Environment | Done; all versions recorded below |
| 2 Sequential pipeline (A0) | Done; verified; reference saved (~/hpc_ref/A0_ref.md5, A0_ref.mp4) |
| 3 Timing and tracing | Done for A0-A7: frame_id, decode/copy/enqueue/stage/encode events, CSV export; A6/A7 also per-frame CUDA-event CSV; A7 traces gpu_wait. video_timestamp not yet exported |
| 4 CPU algorithms (A1-A5) | Done; all bit-identical to A0; benchmarked in two modes |
| 5 CUDA/GPU | Done (blur-only GPU path): exact GPU blur, error checks, BLUR_CHECK, PCIE_BENCH (5a); A6 single-stream (5b); A7 fixed multi-stream with slot pool and safe buffer reuse, GPU_STREAM_BENCH (5c). All GPU algorithms md5-identical to A0 |
| 6 CPU-GPU overlap | Partial (pre-existing): A8 on old synchronous 3x3 GPU path; unverified. Old A7 (3-thread pipeline) deleted in 46a308d |
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
| A6 single-stream GPU | verified, benchmarked | one thread, one stream, synchronous; blur on GPU (GpuBlur5), rest via process_tail_cpu; md5-identical to A0; 1.04x vs A0 both modes |
| A7 fixed multi-stream GPU | verified, benchmarked | `A7 M S`: A4 structure, M CPU workers share S GPU stream slots; md5-identical to A0 (1/1, 4/2, 8/8); compute-only 1.04-1.09x vs A4 with same M; full mode = A4 (encode ceiling) |
| A8 | partial, unverified | old 3x3 GPU path (gpu_pipeline.cu launch_cuda_blur); own stage chain; encode in writer thread; pinning uses raw CPU ids (SMT siblings shared) |
| A9, A10, A11 | not started | after GPU baselines |

## Environment
- Ubuntu 24.04.5 LTS, kernel 7.0.0-38-generic
- NVIDIA driver 595.91.07 (CUDA 13.2 reported); nvcc 12.0.140; g++ 13.3.0; CMake 3.28.3; OpenCV 4.6.0
- CPU: i7-12700H, 20 logical CPUs. P-cores with SMT: CPU pairs 0/1, 2/3, 4/5, 6/7, 8/9, 10/11 (max 4.6-4.7 GHz). E-cores: CPUs 12-19 (max 3.5 GHz). L2 11.5 MiB, L3 24 MiB
- GPU (CUDA_INFO, 2026-11-16): RTX 3050 Laptop, compute capability 8.6, 16 SMs, 3770 MB usable VRAM, 80 W cap, concurrent kernels yes, 2 async copy engines, L2 1536 KB, memory bus 128-bit, reported memory clock 6001 MHz (nominal 192 GB/s via the CUDA-samples formula 2 x clock x bus/8; not verified for GDDR6), reported max GPU clock 1740 MHz (observed 1965 MHz under load in nvidia-smi), 65536 registers/SM, 100 KB shared memory/SM (48 KB/block default), 1536 max threads/SM, 1024 max threads/block, warp 32
- PCIe (measured 2026-11-16): link is dynamic. Idle Gen 1 x8; under sustained transfer load Gen 4 x8. nvidia-smi 250 ms samples during PCIE_BENCH: 50 Gen1, 8 Gen2, 24 Gen4; width x8 in every sample (GPU reports max Gen4 x16, platform runs x8). After load ends the link stays Gen4 ~2 s, then steps down Gen2 -> Gen1
- Build: Release (-O3), CUDA arch 86 (sm_86 cubin confirmed), -Wall -Wextra (host code in .cu also), warning-free
- Power during benchmarks: AC, platform_profile=performance, governor=powersave (intel_pstate)
- Repo on NTFS (ntfs3) volume; effect on I/O timing not tested
- Input: videos/input.mp4, h264, 768x432, 12.5 fps, 377 frames (not in repo; record sha256 in README)

## Implemented files
- CMakeLists.txt: default Release; optional CUDA (HPC_ENABLE_CUDA, auto-detected via check_language); defines HPC_HAS_CUDA when CUDA is available; GPU sources (gpu_pipeline.cu, gpu_blur.cu, gpu_selftest.cpp) only in CUDA builds
- include/: bounded_queue.hpp, config.hpp, cpu_pipeline.hpp, cpu_topology.hpp, cuda_check.hpp, filters.hpp, frame_reorder.hpp, frame_task.hpp, gpu_blur.hpp, gpu_pipeline.hpp, gpu_selftest.hpp, gpu_slot_pool.hpp, pipeline.hpp, trace_manager.hpp
- src/: cpu_pipeline.cpp, cpu_topology.cpp, filters.cpp, frame_reorder.cpp, gpu_blur.cu, gpu_pipeline.cu (old GPU path for A8 + CUDA_INFO), gpu_selftest.cpp (GPU files CUDA builds only), main.cpp, pipeline.cpp (A6-A8 inside #ifdef HPC_HAS_CUDA), trace_manager.cpp
- Stage 5a: cuda_check.hpp (CUDA_CHECK throws, CUDA_WARN for cleanup); gpu_blur.hpp/.cu (GpuBlur5: separable [1 4 6 4 1] kernels blur5_h/blur5_v, 32x8 blocks, non-blocking stream, 4 CUDA events; run_pcie_bench); gpu_selftest.hpp/.cpp (BLUR_CHECK). main.cpp commands PCIE_BENCH and BLUR_CHECK [tol]
- Stage 5b: filters::resize_into; process_tail_cpu (CLAHE -> Sobel -> sharpen; process_frame_cpu = resize + blur + process_tail_cpu); run_a6_gpu_pipeline (traces/a6_trace.csv, traces/a6_gpu_timings.csv); A6 prints GPU Setup Time
- Stage 5c: PinnedBuffer (RAII pinned host memory); GpuBlur5 optional own host buffers (own_host_buffers=false), run_sync(h_in, h_out) with caller's pinned buffers, async enqueue()/wait(); gpu_slot_pool.hpp (GpuSlotPool: S x GpuBlur5 without host buffers, acquire/release counting semaphore); run_a7_multistream_pipeline (`A7 M S`, defaults 4 2; traces/a7_trace.csv incl. gpu_wait; traces/a7_gpu_timings.csv with worker, slot, wait, H2D/kernel/D2H/wall; prints GPU Setup Time, GPU medians, slot wait mean/median); run_gpu_stream_bench (`GPU_STREAM_BENCH [frames=200] [max_streams=8]`); old run_a7_concurrent_pipeline deleted; main.cpp validates A7 args (>= 1)
- CUDA_INFO prints copy engines, L2, memory bus/clock, GPU clock, registers/shared memory/threads per SM, warp size
- scripts/: verify_cpu.sh (md5 vs A0 reference; A0, A1, A2 1, A2 2, A3-A5, plus A6, A7 1 1, A7 4 2, A7 8 8 when the build has CUDA; prints the full spec), bench_cpu.sh (warm-up, interleaved repeats, median/min/max, sys/faults/RSS, machine-state header incl. mode and GPU name, frames-emitted check, MODE=full|noenc, specs with up to 3 words e.g. "A7 8 4" shown as wk 8/4, default list adds A6 and "A7 W 2" with CUDA, GPU=0 to skip, ONLY="..." subset; result lines parsed with anchored patterns ^Total Time:, ^Total Frames:, ^Frames Emitted:), stage_means.py (per-stage n/mean/median from trace CSVs)
- results/: baseline v2 CSVs (20261115_131917/132142); re-checks 20261116_0117/0119; f6f573b incl. A6 (20261116_013328/013619); A7 matrix cpu_bench_noenc_20261117_000557.csv and the full-mode run after it (9a49958); pcie_bench_20261116_*.txt
- Repo also has docs/PROGRESS.md (local copy of this file; keep in sync)

## Verified functionality
- A0-A5 outputs bit-identical to A0 reference (decoded-frame md5, 377/377) after every change, including all Stage 5 work
- A6 and A7 (1/1, 4/2, 8/8) outputs bit-identical to A0 reference (377/377); Frames Emitted = 377 in both modes. A7 8/8 is the buffer-race stress case
- Every A0-A7 run reports Frames Emitted = 377 in both modes; reorder fails loudly on missing/stranded/duplicate frames
- --no-encode mode: same processing, no write, real outputs untouched (writes noencode_*.mp4)
- Allocator tuning: faults ~1.07M -> ~24k, sys ~1.0 s -> ~0.06 s, +3.5 MB peak RSS (A0)
- Topology helper (TOPO command) matches lscpu; pinning verified via ps snapshots (A2)
- CPU-only build (-DHPC_ENABLE_CUDA=OFF) compiles and links with all Stage 5 code; GPU commands (CUDA_INFO, A6-A8, PCIE_BENCH, BLUR_CHECK, GPU_STREAM_BENCH) print "needs CUDA" messages
- OpenCV 4.6 getGaussianKernel(5, 0) = [1 4 6 4 1]/16 (measured)
- GPU blur (GpuBlur5) BIT-IDENTICAL to filters::apply_blur: synthetic 37x23, 61x17, 1280x720 at 1 and 3 channels; video 377/377 frames, 0 / 1,042,329,600 differing elements (BLUR_CHECK)

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
- Run-to-run noise ~1-5% under the benchmark protocol; single runs can be off by ~10%; day-to-day drift ~5%. Differences inside overlapping ranges are not reported as findings. A3 occasionally has a single outlier run (spread 16-18%); medians unaffected
- A4 worker scaling, full: 1w 4.18 s, 2w 2.40, 3w 2.33, 4w 2.30, 6w 2.32, 8w 2.34 (plateau)
- A4 worker scaling, compute-only: 1w 2.19, 2w 1.32, 4w 0.73, 6w 0.55, 8w 0.49
- Per-stage times, A0 compute-only trace (2026-11-16, f6f573b, one run, mean ms): decode 0.188, resize 0.795, blur 1.019, clahe 2.592 (incl. cvtColor/clone/createCLAHE), sobel 0.787, sharpen 0.375; sum 5.757 ms ~= measured 5.78 ms/frame. Shares: clahe 45%, blur 18%, resize 14%, sobel 14%, sharpen 7%, decode 3%
- Full-mode encode ~5.6-5.9 ms/frame (older trace)
- A4 8 workers compute-only, main thread per frame: decode wait 0.40, copy 0.23, enqueue wait 0.60 ms
- RSS: ~143 MB (A0) to ~210-235 MB (4 workers)

## Test results: Stage 5a GPU (2026-11-16)
BLUR_CHECK per-frame medians, 377 frames, 1280x720 BGR (2.76 MB), single stream, pinned buffers, synchronous:
| part | ms | share of GPU wall |
|---|---|---|
| H2D (CUDA events) | 0.334 | 43% |
| kernels h+v (CUDA events) | 0.133 | 17% |
| D2H (CUDA events) | 0.298 | 38% |
| GPU wall total (host clock) | 0.781 | - |
| CPU blur, cv::GaussianBlur, 1 thread | 1.090 | - |
- Kernel traffic estimate (arithmetic, not measured): ~16.6 MB/frame / 0.133 ms ~ 125 GB/s, ~65% of the nominal 192 GB/s -> likely memory-bound (hypothesis; confirm with Nsight Compute)

PCIE_BENCH (10 warm-up + 100 timed copies per row; wall = host clock around copy + sync; GB/s from wall median, decimal):
| size | host mem | dir | wall med ms | min-max | event med ms | GB/s |
|---|---|---|---|---|---|---|
| 0.92 MB | pageable | H2D | 0.119 | 0.110-0.338 | 0.115 | 7.72 |
| 0.92 MB | pageable | D2H | 0.176 | 0.165-0.413 | 0.172 | 5.23 |
| 0.92 MB | pinned | H2D | 0.078 | 0.078-0.838 | 0.074 | 11.76 |
| 0.92 MB | pinned | D2H | 0.075 | 0.075-0.107 | 0.071 | 12.25 |
| 2.76 MB | pageable | H2D | 0.372 | 0.319-1.221 | 0.366 | 7.44 |
| 2.76 MB | pageable | D2H | 0.432 | 0.392-0.823 | 0.428 | 6.40 |
| 2.76 MB | pinned | H2D | 0.267 | 0.259-1.173 | 0.262 | 10.37 |
| 2.76 MB | pinned | D2H | 0.289 | 0.280-0.651 | 0.284 | 9.57 |
| 64 MiB | pageable | H2D | 9.355 | 8.206-12.801 | 9.348 | 7.17 |
| 64 MiB | pageable | D2H | 9.737 | 8.842-14.863 | 9.733 | 6.89 |
| 64 MiB | pinned | H2D | 6.939 | 6.082-9.865 | 6.934 | 9.67 |
| 64 MiB | pinned | D2H | 6.920 | 6.643-8.273 | 6.910 | 9.70 |
- Event and wall medians agree within ~5 us for all rows (incl. pageable)

## Test results: Stage 5b A6 (commit f6f573b, 2026-11-16, 5 repeats after warm-up, interleaved with A0-A5)
| mode | A0 median (min-max) s | A6 median (min-max) s | diff | speedup |
|---|---|---|---|---|
| full | 4.296 (4.253-4.306) | 4.147 (4.112-4.193) | -0.149 s (-0.40 ms/frame) | 1.04 |
| compute-only | 2.179 (2.175-2.181) | 2.095 (2.090-2.099) | -0.084 s (-0.22 ms/frame) | 1.04 |
- Ranges do not overlap in either mode -> reportable
- Process resources (median, /usr/bin/time, whole process incl. setup): A6 sys 0.29 s full / 0.27 noenc (A0 0.06 / 0.05), faults ~30.5k / 28.5k (A0 ~24.2k / 22.3k), RSS 235 / 227 MB (A0 144 / 137)
- A6 single-run GPU medians (CUDA events): H2D 0.312, kernels 0.133, D2H 0.283-0.286, wall 0.739-0.740 ms
- GPU Setup Time, single runs across A6/A7: 0.105-0.22 s (typically ~0.11 s)

Per-stage, compute-only traces (one run each, mean ms):
| stage | A0 | A6 | diff |
|---|---|---|---|
| decode | 0.188 | 0.193 | ~0 |
| resize | 0.795 | 0.809 | +0.014 |
| blur | 1.019 | 0.736 | -0.283 |
| clahe | 2.592 | 2.671 | +0.079 |
| sobel | 0.787 | 0.788 | ~0 |
| sharpen | 0.375 | 0.376 | ~0 |
| sum | 5.757 | 5.574 | -0.183 |

## Test results: Stage 5c (commits 46a308d / 9a49958, 2026-11-17)
GPU_STREAM_BENCH 200 8 (GPU only, no CPU stages, 1280x720 BGR, 5 timed runs after warm-up, one host thread, round-robin):
| streams | total ms med (min-max) | ms/frame | frames/s | vs S=1 | H2D / kern / D2H event med ms |
|---|---|---|---|---|---|
| 1 | 136.80 (134.27-141.10) | 0.684 | 1462 | 1.00x | 0.262 / 0.124 / 0.282 |
| 2 | 99.32 (98.30-102.49) | 0.497 | 2014 | 1.38x | 0.373 / 0.131 / 0.326 |
| 4 | 80.14 (79.52-85.05) | 0.401 | 2495 | 1.71x | 0.619 / 0.151 / 0.395 |
| 8 | 79.61 (79.39-87.54) | 0.398 | 2512 | 1.72x | 1.033 / 0.158 / 0.386 |
- Per-op event times grow with S because they include waiting for a busy copy engine; throughput is the valid metric
- At S=8: 5.53 MB per frame (2.76 MB each direction) / 0.398 ms ~ 13.9 GB/s combined, vs ~9.6-10.4 GB/s single-direction pinned (PCIE_BENCH)

A7 benchmark, compute-only (9a49958, 5 repeats after warm-up, interleaved; machine ~5% slower than previous days, compare within run only):
| algo | M/S | median s | min-max | speedup vs A0 | vs A4 same M | sys s | RSS MB |
|---|---|---|---|---|---|---|---|
| A0 | - | 2.285 | 2.265-2.299 | 1.00 | - | 0.04 | 136 |
| A4 | 4 | 0.763 | 0.750-0.769 | 3.00 | - | 0.07 | 199 |
| A7 | 4/1 | 0.699 | 0.694-0.718 | 3.27 | 1.09x | 0.26 | 287 |
| A7 | 4/2 | 0.714 | 0.684-0.725 | 3.20 | 1.07x | 0.21 | 288 |
| A7 | 4/4 | 0.710 | 0.705-0.725 | 3.22 | 1.07x | 0.21 | 288 |
| A4 | 8 | 0.508 | 0.502-0.519 | 4.50 | - | 0.10 | 258 |
| A7 | 8/1 | 0.490 | 0.489-0.495 | 4.66 | 1.04x | 0.23 | 340 |
| A7 | 8/2 | 0.480 | 0.475-0.482 | 4.76 | 1.06x | 0.24 | 342 |
| A7 | 8/4 | 0.483 | 0.479-0.486 | 4.73 | 1.05x | 0.23 | 340 |
| A7 | 8/8 | 0.484 | 0.479-0.486 | 4.72 | 1.05x | 0.23 | 340 |
- A7 vs A4: ranges do not overlap at M=4 or M=8 -> reportable. At M=4, S=1/2/4 overlap (no difference). At M=8, S=1 vs S=2/4 do not overlap (S>=2 ~2% faster); S=2/4/8 overlap

A7 benchmark, full mode (same session):
| algo | M/S | median s | min-max | speedup vs A0 |
|---|---|---|---|---|
| A0 | - | 4.346 | 4.332-4.400 | 1.00 |
| A4 | 4 | 2.375 | 2.360-2.383 | 1.83 |
| A7 | 4/2 | 2.335 | 2.290-2.376 | 1.86 |
| A4 | 8 | 2.383 | 2.360-2.401 | 1.82 |
| A7 | 8/4 | 2.343 | 2.325-2.382 | 1.86 |
- Ranges overlap: A7 = A4 in full mode (encode ceiling, Finding 1)

A7 slot waits, compute-only single runs (46a308d): 4/1 mean 0.043 ms/frame (median 0), 4/2 0.003-0.012, 8/1 1.062 (median 0.981), 8/4 0.004

## Findings (supported by data; mechanisms marked as hypotheses)
1. Serial encode caps full-pipeline speedup at ~1.85-1.9x: encode ~5.9 ms/frame under the reorder lock ~= 6.1 ms/frame achieved; ceiling lifts when encode is removed (compute-only scales to 4.5x at 8 workers). Also caps all GPU algorithms in full mode (A7 = A4)
2. Default glibc allocation cost ~1 s/run of kernel page-fault time; fixed with process-wide mallopt (HPC_NO_MALLOC_TUNE=1 restores default)
3. Static vs dynamic vs pinned scheduling: no measurable difference on this workload (both modes)
4. SMT: ~2% gain for filter-only work, 1.18x when encode and filters can overlap on the two hardware threads (mechanism = hypothesis; needs perf IPC)
5. Confining FFmpeg decoder threads to the processing CPU (A1) costs 5% (full) / 10% (compute-only); moving decode off-core gains ~2%
6. At 8 workers compute-only, workers (not the main thread) are the limit: main thread waits ~0.6 ms/frame on a full queue (cause untested: E-cores, memory bandwidth, or lock contention)
7. Original A5 penalty (~25%) came from main thread and worker 0 sharing one physical core; fixed by topology-aware pinning
8. An integer separable GPU implementation ([1 4 6 4 1], REFLECT_101, (S+128)>>8) reproduces OpenCV 4.6's 8-bit GaussianBlur(5x5, sigma 0) exactly; GPU correctness for blur needs no tolerance. Consequence: A6 and A7 end-to-end outputs are md5-identical to A0
9. PCIe link is power-managed: Gen1 x8 idle, Gen4 x8 under load; pinned 64 MiB reaches ~9.7 GB/s (~62% of Gen4 x8 theoretical 15.75 GB/s). Pinned is ~1.3-1.5x faster than pageable at frame sizes
10. For the blur-only GPU path, transfers are ~81% of GPU wall time per frame (kernels 0.133 ms). Reducing transferred bytes matters more than kernel speed
11. A6 (blur on GPU, one stream, synchronous) is 1.04x faster than A0 in both modes. The gain is small because blur is ~18% of compute and its GPU time is mostly transfer
12. In A6 the blur stage drops 1.019 -> 0.736 ms but CLAHE rises 2.592 -> 2.671 ms (+0.079). Hypothesis: blurred frame arrives by DMA into pinned memory and is cold in CPU caches. Test later with perf cache-miss counters
13. GPU algorithms use more kernel time (+~0.15-0.22 s sys), page faults (+5-8k) and memory (+~80-90 MB RSS) than their CPU counterparts; measured process-wide, most likely CUDA context/runtime and pinned-buffer setup (outside Total Time)
14. GPU setup (~0.11 s, up to 0.22 s) is about as large as or larger than the GPU gain on this 377-frame video (A6 0.08-0.15 s; A7 0.02-0.06 s compute-only). Including setup, GPU offload does not pay off on this short video; break-even for A6 estimated ~270-830 frames. Relevant to A10 (video length) and to the evaluation plan (longer videos)
15. In A6 the GPU is busy ~13% of each frame; one synchronous host thread cannot feed multiple streams
16. GPU-only stream scaling (GPU_STREAM_BENCH): 1.38x at S=2, 1.71-1.72x at S=4-8 (plateau ~0.40 ms/frame, ~2500 frames/s). Copy/compute overlap across streams is real but partial: combined copy throughput ~13.9 GB/s vs ~10 GB/s single-direction, i.e. H2D and D2H overlap only ~1.35-1.4x instead of 2x. Cause untested (hypotheses: bidirectional PCIe/host-memory limits, single-thread issue overhead); Nsight Systems timeline would show it
17. A7 is faster than A4 in compute-only mode at both M: M=4 1.07-1.09x, M=8 1.04-1.06x (non-overlapping ranges). Best overall compute-only speedup so far: A7 8/2, 4.76x vs A0 (A4 8: 4.50x). The relative gain at M=4 is larger than A6 vs A0 (1.04x) although the same CPU blur is removed. Hypothesis: CPU blur is slower under 4-worker memory-bandwidth contention, so offloading it saves more; test by comparing per-stage blur time in a4 vs a0 traces
18. Stream count barely matters for the blur-only pipeline: M=4 S=1/2/4 indistinguishable; M=8 S=1 ~2% slower than S>=2 (mean slot wait 1.06 ms/frame vs ~0.004 ms at S=4), S=2/4/8 equal. The GPU is far from saturated: the pipeline needs ~785 frames/s, one stream delivers ~1460 frames/s, S>=4 ~2500 frames/s. CPU stages are the limit. Open: why 1 ms wait/frame at 8/1 costs only ~2%
19. Consequence for A9: with blur-only GPU work there is almost no stream-count tuning space (S=2 suffices). A meaningful A9 study needs a workload where GPU configuration matters (more stages on GPU, higher resolution, or both)

## Stage 5 decisions (2026-11-15)
- M1 GPU blur definition: exact integer equivalent of cv::GaussianBlur(5x5, sigma 0): separable [1 4 6 4 1], BORDER_REFLECT_101, output (S + 128) >> 8. CONFIRMED: bit-identical, tolerance 0 (Finding 8)
- Correctness checks: (a) per-stage: BLUR_CHECK; (b) end-to-end: GPU algorithms md5 vs A0 reference via verify_cpu.sh (A6, A7 confirmed). In-process --check remains the plan for later stages that are not bit-exact (resize, CLAHE)
- M2: GPU algorithms use the same working resolution as the CPU path (1280x720). Done for A6, A7
- Placement: R1 = blur only on GPU, BGR in/out (2.76 MB each way). Later options decided after measurements: R1b fuse BGR->gray into blur, R2 resize+blur, R3 +CLAHE (optional), R4 all stages
- GPU code structure must allow changing which stages run on the GPU without a rewrite (needed by A10). Done for the blur/tail split via process_tail_cpu
- GPU code keeps OpenCV out of .cu files; .cpp code uses gpu_blur.hpp via forward-declared CUDA handle types
- Timing rule for GPU algorithms: GPU setup (CUDA context, buffers, one warm-up per stream) before the timer and reported as "GPU Setup Time"; GPU sub-stage times from CUDA events; Total Time boundary same as the matching CPU algorithm
- Buffer ownership: A6 reuses one pinned in/out pair (safe because sequential). A7: device buffers belong to a slot, pinned host buffers belong to a worker (one frame at a time per worker), so no two in-flight frames share memory

## A7 definition (implemented in 46a308d; user chose option B + C on 2026-11-16)
- Inputs: video, M CPU workers, S GPU stream slots (`A7 M S`, defaults 4 2; both >= 1). Output: same video as A0 (md5-identical), traces, per-frame GPU timings, GPU setup time, slot-wait stats
- Structure = A4: main thread decodes + clones into one shared bounded queue (20); M workers; FrameReorder encodes in order (inside push_and_write). Timer starts after worker launch, like A4
- Per frame (worker): resize_into worker's pinned input -> acquire slot (gpu_wait) -> H2D + kernels + D2H on the slot's stream, wait -> release -> process_tail_cpu reads worker's pinned output -> reorder
- Decision logic: none (fixed S, M); adaptive selection is A9
- Constraints: frames on GPU <= min(M, S); VRAM ~11 MB per slot; pinned host ~5.5 MB per worker
- Overheads: GPU setup (separate), slot waits (traced), per-frame launch/sync; worker threads may spin while waiting on the stream (default CUDA scheduling; cudaDeviceScheduleBlockingSync is an optional experiment)
- Worker error -> message + exit code 2 (a lost frame would stall the reorder buffer)
- Option C: GPU_STREAM_BENCH (GPU-only stream scaling; auxiliary test, not an algorithm)

## GPU stage roadmap (plan, 2026-11-16; each step pending the previous step's measurements)
- Goal: GPU covers more than blur. Blur-only is the starting point, not the final design: blur is ~18% of compute and needs a 2.76 MB round trip; A10 needs real stage-placement choices; A9 needs GPU-heavy work to tune (Finding 19)
- Constraint: stage order resize -> blur -> CLAHE -> Sobel -> sharpen; GPU work must be contiguous. Splits: "front on GPU" (resize+blur+gray on GPU; ~1.0 MB up at 768x432 BGR, 0.92 MB down gray; ~32%+ of compute), "all on GPU" (needs CUDA CLAHE), "GPU-CPU-GPU" (not planned)
- Step 1 (DONE, Stage 5): blur on GPU, A6 and A7
- Step 2: add BGR->gray (R1b), then resize -> "front on GPU". Gray likely exact (to verify); resize likely needs a documented tolerance
- Step 3: add Sobel and sharpen (3x3 stencils, likely exact with careful rounding/saturation; to verify); vehicle for the optional global-memory vs shared-memory tiling study
- Step 4 (optional): CLAHE on GPU, only if measurements justify it (largest stage, ~45% of compute; largest development cost; tolerance expected)
- Every added stage gets its own per-stage correctness check and CUDA-event timing before the next step
- Custom CUDA kernels, not OpenCV's cv::cuda module (believed absent from the Ubuntu apt OpenCV 4.6 build)

## Design decisions
- Shared CPU stage chain: process_frame_cpu() = resize + blur + process_tail_cpu() (cpu_pipeline.cpp); A0-A5 use process_frame_cpu, A6/A7 use process_tail_cpu after the GPU blur
- Working resolution: config::TARGET_WIDTH/HEIGHT = 1280x720 (A0-A7)
- Filters: GaussianBlur 5x5 sigma 0; CLAHE clip 2.0, 8x8 tiles (grayscale); Sobel 3x3 blended 0.5/0.5; sharpen 3x3 kernel; output grayscale mp4v
- Timing: steady_clock; Total Time = first read to writer released; setup and trace flush excluded; GPU algorithms print GPU Setup Time
- Process-wide malloc tuning in main.cpp (applies to all algorithms)
- Two measurement modes: full (default; correctness and demo) and --no-encode (compute-only benchmarking)
- Benchmark protocol: one discarded warm-up pass, >=5 interleaved repeats, report median + min/max, record machine state; compare only within one run (day-to-day drift ~5%)
- Thread placement uses cpu_assignment_order(): one logical CPU per SMT core, then single-thread cores, then siblings (A5; A4/A7 unpinned)
- cv::setNumThreads(1): OpenCV internal parallelism disabled; FFmpeg decoder threads remain (documented)
- CPU parallelism is hand-written (std::thread, bounded queues, reorder buffer, pthread affinity); OpenMP not used (optional extension)
- Reorder buffer always admits the next-expected frame (no deadlock); completeness checked against decoded count
- CMake option HPC_ENABLE_CUDA (auto-detected); on other GPUs pass -DCMAKE_CUDA_ARCHITECTURES=<cc> (e.g. 75 expected for RTX 2060; confirm with CUDA_INFO)
- GPU host buffers are pinned; GPU timing from CUDA events, plus host wall clock for what the calling thread pays
- No extra host copies on the GPU path: resize writes into pinned input, CLAHE reads pinned output
- Script output parsing uses anchored patterns (^Total Time: etc.); any new output line containing these words must not start with them

## Known issues / open items
- A8 (Stage 6): old 3x3 kernel at native 768x432 via gpu_pipeline.cu launch_cuda_blur (no CUDA error checks, pageable synchronous copies on the default stream); own stage chain; not on shared code path; no frames-emitted check; allocates inside the GPU thread after start_time; pinning uses raw CPU indices (SMT siblings shared); FrameReorder gets total_frames from container metadata; reads worker count from argv[3]
- A7 M=8 S=1: 1 ms mean slot wait per frame costs only ~2%; cause open
- A7 vs A4 relative gain larger than A6 vs A0 (Finding 17); hypothesis untested (run stage_means.py on a4_trace.csv vs a7_trace.csv vs sequential_trace.csv)
- GPU stream overlap plateau (Finding 16) cause untested; Nsight Systems not yet used (check `which nsys`)
- Spin vs blocking CUDA sync in worker threads untested (optional experiment)
- A6 full-mode saving per frame (0.40 ms) larger than compute-only (0.22 ms); cause untested
- CLAHE slower in A6 than A0 (Finding 12); cause hypothesis untested
- CUDA_INFO "max" GPU clock (1740 MHz) is below the observed 1965 MHz under load; use nvidia-smi for actual clocks
- A0 hard-codes 1280x720 instead of config::TARGET_WIDTH/HEIGHT (same values)
- H2D for 2.76 MB: ~0.31-0.44 ms inside pipelines vs 0.262 ms (event) in PCIE_BENCH; cause untested
- 0.92 MB pinned D2H (12.25 GB/s) faster than 64 MiB pinned D2H (9.70 GB/s); hypothesis: small D2H stays in CPU LLC; untested
- video_timestamp not exported to traces
- Encode trace label in FrameReorder is "CPU_MAIN" although executed by worker threads
- Tracing overhead not measured (--no-trace flag planned)
- NTFS vs ext4 effect on I/O not tested
- Input/output paths hardcoded; working resolution not a CLI option (needed for large-video tests)
- Trace keeps all events in RAM (~2M events at 100k frames; likely fits in 16 GB but adds overhead)
- A0 and A1 Frames Emitted = loop count (sanity warning vs container metadata added)
- Topology helper ignores L3/CCX groups (relevant on multi-CCD AMD Ryzen)
- "CLAHE" stage time includes cvtColor, clone and createCLAHE per frame
- Worker-count argument not validated for A3/A4/A5: 0 crashes A3/A5 (modulo by zero) and hangs A4; non-numeric input crashes (stoi). (A7 validates.) Optional fix
- Fixed in 9a49958: bench_cpu.sh/verify_cpu.sh matched "Total Time" inside the "GPU Setup Time ... (excluded from Total Time)" line (bench FAIL with frames='')

## Other machines (planned / optional)
- Available: RTX 2060 desktop; AMD Ryzen PCs; one PC without NVIDIA GPU (CPU-only build, A0-A5)
- Per machine: same input (check sha256), record lscpu/versions/power settings, run TOPO and compare with lscpu -e=CPU,CORE, make that machine's own A0 reference, verify_cpu.sh, bench both modes; keep result files tagged by machine. GPU machines: also CUDA_INFO, BLUR_CHECK, PCIE_BENCH with nvidia-smi link monitor, GPU_STREAM_BENCH
- Compare patterns (encode ceiling, SMT effect, scheduler ties, GPU transfer share, stream overlap) more than raw times; software versions must be recorded

## Future evaluation goal (user decision 2026-11-15: max 100,000 frames)
- End-of-project tests on several videos at 100, 1,000, 10,000 and 100,000 frames, multiple resolutions (e.g. 640x480, 1280x720, 1920x1080, 3840x2160 where feasible). 1,000,000 frames dropped (disk/time cost)
- Test inputs: generate with ffmpeg (scale for resolution, -stream_loop for length; verify real frame count with ffprobe -count_frames); include at least one different real video, not only the looped input
- Estimated cost at 720p (extrapolated from 377-frame runs; thermals may change it): output ~11.6 GB per 100k-frame full-mode run; A0 full ~19 min, 4 workers full ~10 min, A0 compute-only ~10 min, 4 workers compute-only ~3 min per 100k-frame run. Full CPU matrix (7 configs x 2 modes x 5 repeats) at 100k ~12-13 h per resolution
- Plan: full matrix (all algorithms, 5 repeats) for 100-10,000 frames; reduced set at 100,000 (best algorithm per family, 3 repeats, mostly compute-only, outputs on ext4)
- Needs before Stage 11: input-path and working-resolution CLI args, --no-trace (or periodic trace flush), per-video A0 references and frame counts in scripts, GPU VRAM check for large resolutions (~25 MB per 4K BGR frame). Longer videos also amortize GPU setup (Finding 14); higher resolutions increase GPU work (Finding 19)

## Next steps
1. User decision: (a) Stage 6 next (rebuild A8 as a stage-pipelined CPU-GPU overlap baseline on GpuBlur5 at 1280x720, shared code, topology-aware pinning, frames-emitted check), keeping blur-only GPU work; or (b) GPU roadmap step 2 first (gray + resize on GPU) so A6-A8 and later A9 have more GPU work. Recommendation: (a) then (b) before A9
2. Optional quick checks: stage_means.py on a0/a4/a7 compute-only traces (Finding 17 hypothesis); Nsight Systems timeline of A7 8 4 and GPU_STREAM_BENCH if nsys is available (Finding 16)
3. Before A9: define its inputs/outputs/objective; ensure a workload where GPU configuration matters (Finding 19)