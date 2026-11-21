# Performance results — partitioned convolution

Record of the measurement arc of 2026-11-20: Apache's FFT in isolation,
the hand-rolled FFT head to head, the winning arm, the non-uniform
re-test under it, and the GC switch. Numbers are medians unless noted.

## Conditions

- Machine: Nathan's Mac Studio (Apple Silicon, NEON — `SPECIES_PREFERRED`
  is 2 doubles per vector). Java 28, `--enable-preview`, Vector API.
- Workload: 8192-tap kernel, 256-sample blocks @ 48 kHz, 5.333 ms
  per-block deadline. End-to-end reps are 10 convolutions each;
  summaries are medians across 15 reps (3 outer runs x 5 reps).
  Per-block samples: 2820 (188 blocks x 15 reps).
- **Era note:** runs before 21:57 CST ran under G1 (the default — the
  Surefire argLine set no GC flags). Runs from 22:02 ran under
  Generational ZGC (`-XX:+UseZGC` added to the Surefire argLine).
  Absolute medians move a little between eras and between runs
  (per-block medians move ~±1 µs); within-run comparisons are the
  valid ones. GC counts are total collections across the timed reps.

## The arms (glossary)

- **Overlap-save** — bulk overlap-save, the throughput baseline.
- **SoA** — uniform P=256 partitioned convolver, split-complex spectra,
  Vector API multiply-accumulate, Apache FFT through `apply`
  (the `Complex[]` wrapper).
- **InPlace** — same as SoA, but Apache's `transformInPlace` called
  directly on the split arrays: no `Complex[]` anywhere.
- **RealFft** — same as InPlace, but both FFT directions are the
  hand-rolled *real* FFT (`HandRolledFft`): the block going in is real
  and the result coming out is real, so each direction runs one
  half-size complex FFT plus O(N) packing instead of a full complex
  transform. No Apache anywhere in the class.
- **Non-uniform / Wefers** — the kernel split into groups of same-size
  partitions (e.g. Wefers 128^4 512^15), each group one uniform stage.
  The re-test runs them with RealFft stages.

## FFT round trip in isolation (µs, medians — G1 era, 21:37 CST)

`FftPerformanceTest`, 1000 calls x 5 reps x 3 runs = 15000 samples per
path per size. The 21:28 run (without the hand paths) agreed within
~0.5 µs on the three Apache paths.

| FFT size | apply | in-place | in-place reused | hand complex | hand real |
|----------|-------|----------|-----------------|--------------|-----------|
| 256 | 5.125 | 3.083 | 3.000 | 3.833 | **2.167** |
| 512 | 8.333 | 6.375 | 6.292 | 8.292 | **4.542** |
| 1024 | 19.333 | 14.375 | 14.291 | 17.959 | **9.667** |

- The `apply` wrapper costs ~2.5 µs per round trip at 512 (28%); reused
  buffers save only ~0.1 µs — the `double[]` allocation is free, the
  `Complex` objects + conversions are the cost. Maxes: apply ~1.5 ms vs
  reused ~49 µs at 512.
- Hand complex **loses** to Apache in place by ~30% at every size — the
  full-twiddle-table hypothesis is falsified for this loop shape
  (likely Apache's 4-point add-only base case + strided table access).
- Hand real **wins** by 28–32% over the best Apache path at every size.
  The FFT-512 apply round trip is ~47% of a SoA block.

## End to end, RealFft vs the field (ms, medians — ZGC era, 22:02 CST)

`RealFftSoAConvolutionPerformanceTest`. RealFft beats bulk overlap-save
at all four sizes, including 64-tap, which InPlace lost at 0.78x.

| Case | Overlap-save | SoA | InPlace | RealFft | RealFft vs OS |
|------|-------------|-----|---------|---------|---------------|
| 32768/64 | 8.42 | 13.42 | 9.39 | 7.39 | 1.14x |
| 32768/512 | 11.27 | 13.36 | 9.64 | 7.11 | 1.59x |
| 48000/1024 | 15.85 | 21.16 | 15.38 | 11.79 | 1.34x |
| 48000/8192 | 36.09 | 40.61 | 30.87 | 27.67 | 1.30x |

GC (total collections): RealFft 0 at all four cases (InPlace 0/0/6/4).
For reference, the G1-era InPlace run (21:33) gave 48000/8192:
OS 38.21 / SoA 42.13 / InPlace 33.49.

## Per block, 48000/8192 (µs — ZGC era, 22:02 CST)

| Arm | median | p95 | max | flush median |
|-----|--------|-----|-----|--------------|
| SoA | 23.17 | 30.29 | 66.46 | 0.740 ms |
| InPlace | 13.42 | 16.25 | 145.96 | 0.438 ms |
| RealFft | 12.50 | 17.13 | 145.42 | 0.386 ms |

0/2820 blocks over the 5.333 ms deadline on every arm. The per-block
drop from InPlace is 0.92 µs, about half the 1.75 µs the FFT harness
implied — the harness round trip overstates the arm saving (packing
against real memory traffic, the transforms' internal half-size
scratch, ±1 µs median noise between runs).

## Non-uniform under hand-real (ZGC era, 22:06 CST)

`RealFftNonUniformConvolutionPerformanceTest`, 48000/8192 only.
End to end, medians (ms): Overlap-save 37.89 / Uniform RealFft 27.93
/ Wefers 128^4 512^15 30.95 (0.90x uniform) / 256^2 512^15 30.46
(0.92x) / 256^4 1024^7 28.53 (0.98x, ties) / Uniform 256 control
27.77 (1.01x uniform — the control matches, as it must).

Per-block medians (µs): uniform 12.38 / Wefers 14.88 / 256^2 512^15
12.17 / 256^4 1024^7 7.25 / control 13.17. p95: 15.54 / 27.67 /
26.50 / 43.54 / 17.13. Max: 37.96 / 38.96 / 38.25 / 129.79 / 136.54.

**Verdict: the layout is dead for this kernel size.** Wefers still
loses 0.90x with the FFT bill halved, so the FFT was not the reason —
the FFT doubling is structural to the layout (Wefers runs ~2x the FFT
bins per block) and the multiply saving never covered it. The one
nuance: 256^4 1024^7 ties end to end with the lowest median (7.25 µs)
but the worst p95/max (43.5/129.8 µs vs uniform's 15.5/38 µs) — it
buys a cheaper typical block at the cost of a 3.4x worse worst case,
and ties on average, so it wins nothing for real-time work.

## GC: G1 vs Generational ZGC (21:45–21:57 CST)

Medians flat (FftPerformanceTest 17.0 µs ZGC vs 17.5–19.2 µs G1;
Apache per-block 0.19–0.21 ms). The tail is gone: worst timed max
across three ZGC runs 0.713 ms (SoA per-block) vs G1's 4.812 ms GC
pause, which cross-checked exactly onto a 4.901 ms FftPerformanceTest
`apply` max — the multi-ms spikes were GC pauses in allocating code,
not the FFT code. All 2,820 blocks under 0.72 ms, 0 misses against
the 5.333 ms deadline. **Verdict: switch** — `-XX:+UseZGC` in the
Surefire argLine (JDK 28 refuses `ZGenerational`; generational is
implied). Caveat carried: the bar asked for median-of-3 log maxes
and G1 was n=1 with the ZGC log max unparsed — called anyway on the
6.7x gap. The G1 baseline log `gc-g1-run1-836.log` is the only one in
existence; G1 re-runs need a pom revert. The switch covers only the
forked test JVM; a real-time demo process run directly would need the
flag separately.

## Reproduce

```bash
./mvnw test -Dtest=FftPerformanceTest
./mvnw test -Dtest=RealFftSoAConvolutionPerformanceTest
./mvnw test -Dtest=RealFftNonUniformConvolutionPerformanceTest
./mvnw test   # full suite: 136 tests, 0 failures, 5 skipped (2026-11-20)
```
