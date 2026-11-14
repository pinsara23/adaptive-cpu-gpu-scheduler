#!/usr/bin/env bash
# Runs A0-A5 and compares decoded-frame md5s against the saved A0 reference.
# Usage (from repo root): bash scripts/verify_cpu.sh
set -u
REF="$HOME/hpc_ref/A0_ref.md5"
md5_of() { ffmpeg -loglevel error -i "$1" -f framemd5 - | grep -v '^#' | awk -F, '{print $NF}'; }

cd build || exit 1
fail=0
for spec in "A0" "A1" "A2" "A3 4" "A4 4" "A5 4"; do
  set -- $spec
  ./video_pipeline "$@" > "/tmp/run_$1.log" 2>&1
  md5_of "../output/output_$1.mp4" > "/tmp/$1.md5"
  n=$(wc -l < "/tmp/$1.md5")
  if [ "$n" -gt 0 ] && cmp -s "$REF" "/tmp/$1.md5"; then
    echo "$1: identical ($n frames)  $(grep 'Total Time' /tmp/run_$1.log)"
  else
    echo "$1: DIFFERENT ($n frames)"; fail=1
  fi
done
exit $fail