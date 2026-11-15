#!/usr/bin/env python3
"""Per-stage durations (ms) from trace CSVs (time_ms,frame_id,stage,event,resource).
Usage: python3 scripts/stage_means.py traces/sequential_trace.csv traces/a6_trace.csv"""
import csv, statistics as st, sys
from collections import OrderedDict

def load(path):
    starts, durs = {}, OrderedDict()
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) < 4:
                continue
            try:
                t, fid = float(row[0]), int(row[1])
            except ValueError:
                continue                      # header line
            stage, ev = row[2], row[3]
            key = (fid, stage)
            if ev == "start":
                starts[key] = t
            elif ev == "end" and key in starts:
                durs.setdefault(stage, []).append(t - starts.pop(key))
    return durs

for path in sys.argv[1:]:
    d = load(path)
    print(f"\n{path}")
    print(f"  {'stage':<9}{'n':>6}{'mean':>9}{'median':>9}")
    total = 0.0
    for stage, v in d.items():
        m = st.mean(v); total += m
        print(f"  {stage:<9}{len(v):>6}{m:9.3f}{st.median(v):9.3f}")
    print(f"  {'sum of means':<15}{total:9.3f} ms/frame")