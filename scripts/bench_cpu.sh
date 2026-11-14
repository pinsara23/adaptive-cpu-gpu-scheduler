#!/usr/bin/env bash
# CPU benchmark: one discarded warm-up pass, then N interleaved repeats.
# Usage (from repo root): bash scripts/bench_cpu.sh [repeats=5] [workers=4]
# Env: EXPECTED=377 (frames the input must produce), HPC_NO_MALLOC_TUNE=1 (default allocator)
set -u
REPEATS=${1:-5}
W=${2:-4}
EXPECTED=${EXPECTED:-377}
ALGOS=("A0" "A1" "A2" "A3 $W" "A4 $W" "A5 $W")

mkdir -p results
OUT="results/cpu_bench_$(date +%Y%m%d_%H%M%S).csv"
TMP=$(mktemp)
: > /tmp/bench_err.log

commit=$(git rev-parse --short HEAD)
dirty=$(git status --porcelain -- src include CMakeLists.txt | wc -l)
gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
prof=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo na)
ac=$(cat /sys/class/power_supply/AC*/online 2>/dev/null | head -1)
tune=on; [ -n "${HPC_NO_MALLOC_TUNE:-}" ] && tune=off
btype=$(grep '^CMAKE_BUILD_TYPE' build/CMakeCache.txt | cut -d= -f2)
echo "# commit=$commit uncommitted_files=$dirty governor=$gov profile=$prof ac=${ac:-na} malloc_tune=$tune build=$btype" | tee "$OUT"
echo "algo,workers,run,total_s,sys_s,faults,maxrss_kb" >> "$OUT"

run_one() {   # args: algo [workers] -> prints: total frames sys faults rss
  local out total frames sys faults rss
  out=$(cd build && /usr/bin/time -o "$TMP" -f "%S %R %M" ./video_pipeline $1 $2 2>>/tmp/bench_err.log)
  total=$(echo "$out"  | awk '/Total Time/   {print $3}')
  frames=$(echo "$out" | awk '/Total Frames/ {print $3}')
  read -r sys faults rss < "$TMP"
  echo "$total $frames $sys $faults $rss"
}

echo "warm-up pass (discarded)..."
for spec in "${ALGOS[@]}"; do read -r a w <<< "$spec"; run_one "$a" "${w:-}" > /dev/null; done

for r in $(seq 1 "$REPEATS"); do
  for spec in "${ALGOS[@]}"; do
    read -r a w <<< "$spec"
    read -r total frames sys faults rss < <(run_one "$a" "${w:-}")
    if [ -z "$total" ] || [ "$frames" != "$EXPECTED" ]; then
      echo "FAIL: $a run $r (total='$total' frames='$frames', expected $EXPECTED). See /tmp/bench_err.log"
      exit 1
    fi
    echo "$a,${w:--},$r,$total,$sys,$faults,$rss" >> "$OUT"
    echo "  run $r  $a ${w:-}: ${total}s"
  done
done
rm -f "$TMP"

python3 - "$OUT" <<'PY'
import csv, statistics as st, sys
from collections import OrderedDict
rows = list(csv.DictReader(l for l in open(sys.argv[1]) if not l.startswith('#')))
g = OrderedDict()
for r in rows: g.setdefault((r['algo'], r['workers']), []).append(r)
base = None
print(f"\n{'algo':<5}{'wk':<3}{'median_s':>9}{'min':>8}{'max':>8}{'spread%':>9}{'speedup':>9}{'sys_s':>7}{'faults':>9}{'rss_MB':>8}")
for (a, w), rs in g.items():
    t = [float(r['total_s']) for r in rs]
    med = st.median(t)
    base = base or med
    print(f"{a:<5}{w:<3}{med:9.3f}{min(t):8.3f}{max(t):8.3f}{(max(t)-min(t))/med*100:9.1f}{base/med:9.2f}"
          f"{st.median(float(r['sys_s']) for r in rs):7.2f}"
          f"{st.median(int(r['faults']) for r in rs):9.0f}"
          f"{st.median(int(r['maxrss_kb']) for r in rs)/1024:8.0f}")
print(f"\nraw data: {sys.argv[1]}")
PY