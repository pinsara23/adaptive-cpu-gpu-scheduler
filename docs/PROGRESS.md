# HPC Project Progress

## Current status
- Current stage: Stage 5 (GPU) IN PROGRESS. Stage 5a VERIFIED (2026-11-16). Next: commit 5a, re-run CPU bench, then Stage 5b (A6 on GpuBlur5)
- Last updated: 2026-11-16
- Official CPU baseline v2 measured on commit b7a2a6c
- Later commits (no change to CPU results): de2736e "build: CUDA optional (CPU-only builds run A0-A5)" + follow-up "build: guard A6-A8 in pipeline.cpp for CPU-only builds"
- Stage 5a code verified but not yet committed at time of writing (record hash here after commit)

## Stage status
| Stage | Status |
|---|---|
| 1 Environment | Done; all versions recorded below |
| 2 Sequential pipeline (A0) | Done; verified; reference saved (~/hpc_ref/A0_ref.md5, A0_ref.mp4) |
| 3 Timing and tracing | Done for A0-A5: frame_id, decode/copy/enqueue/stage/encode events, CSV export. video_timestamp not yet exported |
| 4 CPU algorithms (A1-A5) | Done; all bit-identical to A0; benchmarked in two modes |
| 5 CUDA/GPU | In progress. Decisions done. 5a DONE + verified: CUDA error checks, exact 5x5 GPU blur (bit-identical), BLUR_CHECK, PCIE_BENCH. 5b (A6 on GpuBlur5 at 1280x720) next |
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
| A6 | partial, unverified | still on old 3x3 launch_cuda_blur at native 768x432; to be moved to GpuBlur5 at 1280x720 in 5b |
| A7 (spec: fixed multi-stream GPU) | MISSING | current "A7" is a 3-thread CPU-GPU pipeline; rename A7_legacy |
| A8 | partial, unverified | synchronous GPU path; own stage chain; encode in writer thread; pinning uses raw CPU ids (SMT siblings shared) |
| A9, A10, A11 | not started | after GPU baselines |

## Environment
- Ubuntu 24.04.5 LTS, kernel 7.0.0-38-generic
- NVIDIA driver 595.91.07 (CUDA 13.2 reported); nvcc 12.0.140; g++ 13.3.0; CMake 3.28.3; OpenCV 4.6.0
- CPU: i7-12700H, 20 logical CPUs. P-cores with SMT: CPU pairs 0/1, 2/3, 4/5, 6/7, 8/9, 10/11 (max 4.6-4.7 GHz). E-cores: CPUs 12-19 (max 3.5 GHz). L2 11.5 MiB, L3 24 MiB
- GPU: RTX 3050 Laptop, compute capability 8.6, 16 SMs, 3770 MB usable VRAM, 80 W cap, concurrent kernels: yes
- PCIe (measured 2026-11-16): link is dynamic. Idle Gen 1 x8; under sustained transfer load Gen 4 x8 (graphics clock 1965 MHz). nvidia-smi 250 ms samples during PCIE_BENCH: 50 Gen1, 8 Gen2, 24 Gen4; width x8 in every sample (GPU reports max Gen4 x16, platform runs x8). After load ends the link stays Gen4 ~2 s, then steps down Gen2 -> Gen1
- Build: Release (-O3), CUDA arch 86 (sm_86 cubin confirmed), -Wall -Wextra (host code in .cu also built with -Wall -Wextra)
- Power during benchmarks: AC, platform_profile=performance, governor=powersave (intel_pstate)
- Repo on NTFS (ntfs3) volume; effect on I/O timing not tested
- Input: videos/input.mp4, h264, 768x432, 12.5 fps, 377 frames (not in repo; record sha256 in README)

## Implemented files
- CMakeLists.txt: default Release; optional CUDA (HPC_ENABLE_CUDA, auto-detected via check_language); defines HPC_HAS_CUDA when CUDA is available; GPU sources (gpu_pipeline.cu, gpu_blur.cu, gpu_selftest.cpp) only in CUDA builds
- include/: bounded_queue.hpp, config.hpp, cpu_pipeline.hpp, cpu_topology.hpp, cuda_check.hpp, filters.hpp, frame_reorder.hpp, frame_task.hpp, gpu_blur.hpp, gpu_pipeline.hpp, gpu_selftest.hpp, pipeline.hpp, trace_manager.hpp
- src/: cpu_pipeline.cpp, cpu_topology.cpp, filters.cpp, frame_reorder.cpp, gpu_blur.cu, gpu_pipeline.cu, gpu_selftest.cpp (GPU files CUDA builds only), main.cpp, pipeline.cpp (A6-A8 inside #ifdef HPC_HAS_CUDA), trace_manager.cpp
- Stage 5a details: cuda_check.hpp (CUDA_CHECK throws, CUDA_WARN for cleanup); gpu_blur.hpp/.cu (GpuBlur5: separable [1 4 6 4 1] kernels blur5_h/blur5_v, 32x8 blocks, pinned host staging, non-blocking stream, 4 CUDA events; run_pcie_bench); gpu_selftest.hpp/.cpp (BLUR_CHECK: synthetic odd-size tests + all video frames vs filters::apply_blur). main.cpp commands PCIE_BENCH and BLUR_CHECK [tol] (CUDA builds only; CPU-only build prints "needs CUDA", exit 255)
- scripts/: verify_cpu.sh (md5 vs A0 reference, empty-output guard; A0, A1, A2 1, A2 2, A3-A5), bench_cpu.sh (warm-up, interleaved repeats, median/min/max, sys/faults/RSS, machine-state header incl. mode, frames-emitted check, MODE=full|noenc)
- results/: cpu_bench_full_20261115_131917.csv, cpu_bench_noenc_20261115_132142.csv (official baseline v2) + earlier runs; pcie_bench_20261116_*.txt (Stage 5a)
- Repo also has docs/PROGRESS.md (local copy of this file; keep in sync)

## Verified functionality
- A0-A5 outputs bit-identical to A0 reference (decoded-frame md5, 377/377) after every change, including after Stage 5a
- Every A0-A5 run reports Frames Emitted = 377 in both modes; reorder fails loudly on missing/stranded/duplicate frames
- --no-encode mode: same processing, no write, real outputs untouched (writes noencode_*.mp4)
- Allocator tuning: faults ~1.07M -> ~24k, sys ~1.0 s -> ~0.06 s, +3.5 MB peak RSS (A0)
- Topology helper (TOPO command) matches lscpu; pinning verified via ps snapshots (A2)
- CPU-only build (-DHPC_ENABLE_CUDA=OFF) compiles and links with Stage 5a present; CUDA_INFO, A6-A8, PCIE_BENCH and BLUR_CHECK print "needs CUDA" messages
- OpenCV 4.6 getGaussianKernel(5, 0) = [0.0625, 0.25, 0.375, 0.25, 0.0625] = [1 4 6 4 1]/16 (measured)
- GPU blur (GpuBlur5) BIT-IDENTICAL to filters::apply_blur: synthetic 37x23, 61x17, 1280x720 at 1 and 3 channels all 0 differing; video 377/377 frames, 0 / 1,042,329,600 differing elements, max abs diff 0 (BLUR_CHECK, exit 0)

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
- Stage means sum ~7.05 ms but A0 compute-only is ~5.84 ms/frame incl. decode: stage means likely from a full-mode trace; re-measure in a --no-encode trace before GPU break-even decisions
- A4 8 workers compute-only, main thread per frame: decode wait 0.40, copy 0.23, enqueue wait 0.60 ms
- RSS: ~143 MB (A0) to ~210-235 MB (4 workers)

## Test results: Stage 5a GPU (2026-11-16, 5a code on top of de2736e + follow-up, uncommitted)
BLUR_CHECK per-frame medians, 377 frames, 1280x720 BGR (2.76 MB), single stream, pinned buffers, synchronous:
| part | ms | share of GPU wall |
|---|---|---|
| H2D (CUDA events) | 0.334 | 43% |
| kernels h+v (CUDA events) | 0.133 | 17% |
| D2H (CUDA events) | 0.298 | 38% |
| GPU wall total (host clock) | 0.781 | - |
| CPU blur, cv::GaussianBlur, 1 thread | 1.090 | - |
- Not included in GPU wall: host copy of the resized frame into the pinned buffer (copyTo, outside the timed region). A6 must account for it
- Kernel traffic estimate (arithmetic, not measured): ~16.6 MB/frame (8-bit in, 16-bit intermediate written + read, 8-bit out) / 0.133 ms ~ 125 GB/s -> likely memory-bound (hypothesis; confirm with Nsight Compute)

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

verify_cpu.sh after 5a (single run each, no warm-up, run right after GPU benchmarks; NOT benchmark-protocol numbers): all identical, 377 frames. Total times A0 4.64, A1 5.17, A2(1) 4.59, A2(2) 3.77, A3 2.36, A4 2.44, A5 2.32 s

## Findings (supported by data; mechanisms marked as hypotheses)
1. Serial encode caps full-pipeline speedup at ~1.85-1.9x: encode ~5.9 ms/frame under the reorder lock ~= 6.1 ms/frame achieved; ceiling lifts when encode is removed (compute-only scales to 4.5x at 8 workers)
2. Default glibc allocation cost ~1 s/run of kernel page-fault time; fixed with process-wide mallopt (HPC_NO_MALLOC_TUNE=1 restores default)
3. Static vs dynamic vs pinned scheduling: no measurable difference on this workload (both modes)
4. SMT: ~2% gain for filter-only work, 1.18x when encode and filters can overlap on the two hardware threads (mechanism = hypothesis; needs perf IPC)
5. Confining FFmpeg decoder threads to the processing CPU (A1) costs 5% (full) / 10% (compute-only); moving decode off-core gains ~2%
6. At 8 workers compute-only, workers (not the main thread) are the limit: main thread waits ~0.6 ms/frame on a full queue (cause untested: E-cores, memory bandwidth, or lock contention)
7. Original A5 penalty (~25%) came from main thread and worker 0 sharing one physical core; fixed by topology-aware pinning
8. An integer separable GPU implementation ([1 4 6 4 1], REFLECT_101, (S+128)>>8) reproduces OpenCV 4.6's 8-bit GaussianBlur(5x5, sigma 0) exactly (0 differing elements over synthetic and all video frames); GPU correctness for blur needs no tolerance
9. PCIe link is power-managed: Gen1 x8 idle, Gen4 x8 under load; all measured rows exceed Gen2 x8 max (4 GB/s) and pinned rows exceed Gen3 x8 max (7.9 GB/s), so copies ran on a faster link than the idle reading. Pinned 64 MiB reaches ~9.7 GB/s (~62% of Gen4 x8 theoretical 15.75 GB/s). Pinned is ~1.3-1.5x faster than pageable at frame sizes
10. For the blur-only GPU path, transfers are ~81% of GPU wall time per frame (H2D + D2H 0.63 of 0.78 ms; kernels 0.133 ms). Reducing transferred bytes matters more than kernel speed (supports GPU roadmap step 2)

## Stage 5 decisions (2026-11-15)
- M1 GPU blur definition: exact integer equivalent of cv::GaussianBlur(5x5, sigma 0): separable [1 4 6 4 1] (horizontal pass -> uint16, vertical pass -> S <= 256*255), BORDER_REFLECT_101 (OpenCV default; filters.cpp passes no border type), output (S + 128) >> 8. CONFIRMED 2026-11-16: bit-identical, tolerance 0 (Finding 8)
- Correctness checks: (a) per-stage: GPU blur vs filters::apply_blur on synthetic odd sizes (37x23, 61x17, 1280x720; 1 and 3 channels) and all 377 resized frames, reporting max/mean abs diff and differing fraction (implemented: BLUR_CHECK); (b) end-to-end: because blur is bit-exact, A6 with the same CPU stages after blur is EXPECTED to be md5-identical to A0 (prediction; verify in 5b with verify_cpu.sh-style md5 check). In-process --check remains the plan for stages that are not bit-exact
- M2: GPU algorithms use the same working resolution as the CPU path (config::TARGET_WIDTH/HEIGHT = 1280x720)
- Placement: R1 first = blur only on GPU, BGR in/out (blur input is 3-channel; gray conversion happens inside apply_clahe), 2.76 MB each way. Options recorded for later decision after CUDA-event measurements: R1b fuse BGR->gray into blur (D2H 0.92 MB; needs exact cvtColor match), R2 resize+blur (resize not bit-exact, tolerance), R3 +CLAHE (optional, large dev cost), R4 all stages (one round trip)
- GPU code structure must allow changing which stages run on the GPU without a rewrite (needed by A10)
- GPU code keeps OpenCV out of .cu files; .cpp code uses gpu_blur.hpp via forward-declared CUDA handle types
- Planned timing rule for all GPU algorithms: force CUDA context creation (cudaFree(0)) before the timer; GPU sub-stage times from CUDA events; Total Time boundary same as CPU (first read to writer released)

## GPU stage roadmap (plan, 2026-11-16; each step pending the previous step's measurements)
- Goal: GPU covers more than blur. Blur-only is the starting point (simplest exact match + builds GPU infrastructure), not the final design: blur is ~1.07 of ~7 ms/frame and needs a 2.76 MB round trip, and A10 needs real stage-placement choices
- Constraint: stage order resize -> blur -> CLAHE -> Sobel -> sharpen; every CPU<->GPU switch costs a transfer, so GPU work must be contiguous. Sensible splits: "front on GPU" (resize+blur+gray on GPU, CLAHE+Sobel+sharpen on CPU; ~1.0 MB up at 768x432 BGR, 0.92 MB down gray), "all on GPU" (needs CUDA CLAHE), "GPU-CPU-GPU" (two round trips; expected poor, not planned)
- Step 1 (now, 5a/5b): blur, then A6 using it; measure H2D, kernel and D2H with CUDA events. 5a measured: transfers ~81% of GPU blur wall time (Finding 10)
- Step 2: add BGR->gray (R1b), then resize -> completes the "front on GPU" split (hybrid option for A8 and A10). Gray likely exact (to verify); resize likely needs a documented tolerance
- Step 3: add Sobel and sharpen (3x3 stencils, likely exact with careful rounding/saturation; to verify); also the vehicle for the optional global-memory vs shared-memory tiling study
- Step 4 (optional): CLAHE on GPU, only if measurements show CPU CLAHE is the bottleneck; largest development cost; tolerance expected
- Every added stage gets its own per-stage correctness check (like BLUR_CHECK) and CUDA-event timing before the next step
- Custom CUDA kernels, not OpenCV's cv::cuda module (believed absent from the Ubuntu apt OpenCV 4.6 build; would need building OpenCV from source; custom kernels keep kernel-level analysis in the project)

## Design decisions
- Shared CPU stage chain: process_frame_cpu() (cpu_pipeline.cpp); A0-A5 all use it
- Working resolution: config::TARGET_WIDTH/HEIGHT = 1280x720 (A0-A5; GPU algorithms to follow, M2)
- Filters: GaussianBlur 5x5 sigma 0; CLAHE clip 2.0, 8x8 tiles (grayscale); Sobel 3x3 blended 0.5/0.5; sharpen 3x3 kernel; output grayscale mp4v
- Timing: steady_clock; Total Time = first read to writer released; setup and trace flush excluded; A0 also prints Loop Time
- Process-wide malloc tuning in main.cpp (applies to all algorithms)
- Two measurement modes: full (default; correctness and demo) and --no-encode (compute-only benchmarking)
- Benchmark protocol: one discarded warm-up pass, >=5 interleaved repeats, report median + min/max, record machine state
- Thread placement uses cpu_assignment_order(): one logical CPU per SMT core, then single-thread cores, then siblings
- cv::setNumThreads(1): OpenCV internal parallelism disabled; FFmpeg decoder threads remain (documented)
- CPU parallelism is hand-written (std::thread, bounded queues, reorder buffer, pthread affinity); OpenMP not used (optional extension: OpenMP static/dynamic variant vs A3/A4)
- Reorder buffer always admits the next-expected frame (no deadlock); completeness checked against decoded count
- CMake option HPC_ENABLE_CUDA (auto-detected): CPU-only builds compile A0-A5; A6-A8 and CUDA_INFO print a clear "needs CUDA" message. On other GPUs pass -DCMAKE_CUDA_ARCHITECTURES=<cc> (e.g. 75 expected for RTX 2060; confirm with CUDA_INFO)
- GPU host buffers are pinned (Finding 9); GPU timing from CUDA events, plus host wall clock for what the calling thread pays

## Known issues / open items
- M1: resolved for GpuBlur5 (bit-exact); A6-A8 still use the old 3x3 kernel until 5b
- M2: decided (1280x720); A6-A8 still use native 768x432 until 5b
- M9: no CUDA streams/pinned memory/CUDA events in A6-A8; A7 mislabeled; only blur on GPU (GpuBlur5 has them; A6-A8 adopt in 5b/Stage 6)
- Old GPU code (gpu_pipeline.cu): no CUDA error checks; pageable synchronous cudaMemcpy on default stream; kernel time = host clock around launch + cudaDeviceSynchronize
- Timing boundaries differ: A6 allocates VRAM before start_time (context creation excluded) and stops timer before writer release; A7/A8 allocate inside the GPU thread after start_time (included)
- A8 pinning uses raw CPU indices: reader CPU 0 / writer CPU 1 and GPU thread CPU 2 / worker 0 CPU 3 are SMT siblings (same pattern as old A5 penalty); fix with cpu_assignment_order() in Stage 6
- A8 FrameReorder gets total_frames from container metadata; check against current FrameReorder interface (decoded count) in Stage 6
- A8 reads worker count from argv[3] (A2-A5 use argv[2]): `A8 4` silently runs 8 workers; fix in Stage 6
- A6-A8 not on shared code path, no frames-emitted check
- verify_cpu.sh single-run times after 5a: A0 4.64 s / A1 5.17 s vs baseline medians 4.25 / 4.47 (+9-16%); A3-A5 unchanged. Not benchmark-protocol (single run, no warm-up, right after GPU benchmarks); CPU code unchanged by 5a. Cause untested (thermal/turbo state suspected). Re-run bench_cpu.sh both modes before any GPU vs CPU comparison
- H2D for 2.76 MB: 0.334 ms in BLUR_CHECK vs 0.262 ms (event) in PCIE_BENCH; cause untested (possibly link/clock state with ~1 ms CPU work between copies)
- 0.92 MB pinned D2H (12.25 GB/s) faster than 64 MiB pinned D2H (9.70 GB/s); hypothesis: small D2H stays in CPU LLC; untested
- BLUR_CHECK GPU timings exclude the host copy into the pinned input buffer; A6 design should resize directly into pinned memory to avoid it
- video_timestamp not exported to traces
- Encode trace label in FrameReorder is "CPU_MAIN" although executed by worker threads
- Tracing overhead not measured (--no-trace flag planned)
- NTFS vs ext4 effect on I/O not tested
- Input/output paths hardcoded; working resolution not a CLI option (needed for large-video tests)
- Trace keeps all events in RAM (~2M events at 100k frames; likely fits in 16 GB but adds overhead)
- A0 and A1 Frames Emitted = loop count (sanity warning vs container metadata added)
- Topology helper ignores L3/CCX groups (relevant on multi-CCD AMD Ryzen)
- "CLAHE ~2.69 ms" includes cvtColor, clone and createCLAHE per frame
- Worker-count argument not validated: 0 crashes A3/A5 (modulo by zero) and hangs A4; non-numeric input crashes (stoi). Optional fix: reject values < 1 with a clear message

## Other machines (planned / optional)
- Available: RTX 2060 desktop; AMD Ryzen PCs; one PC without NVIDIA GPU (CPU-only build, A0-A5)
- Per machine: same input (check sha256), record lscpu/versions/power settings, run TOPO and compare with lscpu -e=CPU,CORE, make that machine's own A0 reference, verify_cpu.sh, bench both modes; keep result files tagged by machine. GPU machines: also BLUR_CHECK and PCIE_BENCH with nvidia-smi link monitor
- Compare patterns (encode ceiling, SMT effect, scheduler ties) more than raw times; software versions must be recorded

## Future evaluation goal (user decision 2026-11-15: max 100,000 frames)
- End-of-project tests on several videos at 100, 1,000, 10,000 and 100,000 frames, multiple resolutions (e.g. 640x480, 1280x720, 1920x1080, 3840x2160 where feasible). 1,000,000 frames dropped (disk/time cost)
- Test inputs: generate with ffmpeg (scale for resolution, -stream_loop for length; verify real frame count with ffprobe -count_frames); include at least one different real video, not only the looped input
- Estimated cost at 720p (extrapolated from 377-frame runs; thermals may change it): output ~11.6 GB per 100k-frame full-mode run; A0 full ~19 min, 4 workers full ~10 min, A0 compute-only ~10 min, 4 workers compute-only ~3 min per 100k-frame run. Full CPU matrix (7 configs x 2 modes x 5 repeats) at 100k ~12-13 h per resolution
- Plan: full matrix (all algorithms, 5 repeats) for 100-10,000 frames; reduced set at 100,000 (best algorithm per family, 3 repeats, mostly compute-only, outputs on ext4)
- Needs before Stage 11: input-path and working-resolution CLI args, --no-trace (or periodic trace flush), per-video A0 references and frame counts in scripts, GPU VRAM check for large resolutions (~25 MB per 4K BGR frame)

## Next steps
1. Commit Stage 5a (code + results/pcie_bench_*.txt + docs/PROGRESS.md); record hash above
2. Re-run bench_cpu.sh (full and noenc) on the current commit after the laptop cools; confirm A0 ~4.25 s full / ~2.20 s noenc, else investigate before GPU comparisons
3. Stage 5b: A6 on GpuBlur5 at 1280x720: resize directly into pinned input, CLAHE reads pinned output, cudaFree(0) before timer, same Total Time boundary as CPU, frames-emitted check, CUDA-event sub-stage traces; verify md5-identical to A0 (expected from Finding 8); benchmark A6 in both modes with the CPU protocol
4. Re-measure per-stage CPU times in a --no-encode trace; decide R1b/R2/R4 from CUDA-event H2D/kernel/D2H data (GPU stage roadmap step 2)
5. Real A7 (fixed multi-stream, 1/2/4/8/16 where feasible); rename old A7 to A7_legacy
6. Bring A8 onto shared code path, topology-aware pinning, frames-emitted check; re-verify
7. Define A9 inputs/outputs/objective before coding