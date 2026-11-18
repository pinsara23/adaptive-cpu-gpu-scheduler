#!/usr/bin/env bash
# Runs A0-A5 (and A6-A8 when the build has CUDA) and compares decoded-frame md5s
# against the saved A0 reference.
# Usage (from repo root): bash scripts/verify_cpu.sh
set -u
REF="$HOME/hpc_ref/A0_ref.md5"
md5_of() { ffmpeg -loglevel error -i "$1" -f framemd5 - | grep -v '^#' | awk -F, '{print $NF}'; }

cd build || exit 1
specs=("A0" "A1" "A2 1" "A2 2" "A3 4" "A4 4" "A5 4")
has_cuda=1
if ./video_pipeline CUDA_INFO | grep -q "Built without CUDA"; then
  echo "(CPU-only build: skipping GPU algorithms)"
  has_cuda=0
else
  specs+=("A6" "A7 1 1" "A7 4 2" "A7 8 8" "A8 1 1" "A8 6 1" "A8 8 2")
fi

fail=0
check() {   # args: label log algo
  md5_of "../output/output_$3.mp4" > "/tmp/$3.md5"
  local n; n=$(wc -l < "/tmp/$3.md5")
  if [ "$n" -gt 0 ] && cmp -s "$REF" "/tmp/$3.md5"; then
    echo "$1: identical ($n frames)  $(grep '^Total Time' "$2")"
  else
    echo "$1: DIFFERENT ($n frames)"; fail=1
  fi
}

for spec in "${specs[@]}"; do
  set -- $spec
  ./video_pipeline "$@" > "/tmp/run_$1.log" 2>&1
  check "$spec" "/tmp/run_$1.log" "$1"
done

# A8 stress case: no spare pinned pairs (K = R + S + T), maximum buffer reuse.
if [ "$has_cuda" = 1 ]; then
  HPC_A8_SLACK=0 ./video_pipeline A8 8 2 > /tmp/run_A8_stress.log 2>&1
  check "A8 8 2 (HPC_A8_SLACK=0)" /tmp/run_A8_stress.log A8
fi
exit $fail