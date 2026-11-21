package dev.nathanlively.overlap_save_demo;

import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/**
 * End-to-end version of the multiply experiment: the full convolution
 * (FFT, multiply, IFFT) with the FFT and IFFT identical across arms and only
 * the multiply stage varying — Apache {@code Complex}, value
 * {@code ComplexValue}, or one interleaved flat {@code double[]}. Each of the
 * value and flat arms pays two O(N) conversions around the multiply, so this
 * measures whether the multiply win (or loss) survives the boundary cost.
 *
 * <p>Sizes are signal/kernel pairs whose result length is a power of two,
 * so the FFT size is exact: 256/257 -&gt; 512, 2048/2049 -&gt; 4096,
 * 8192/8193 -&gt; 16384. Data is deterministic ({@code Random(42)}).
 *
 * <p>Like {@link ValueObjectPerformanceTest}, this logs times and asserts
 * only correctness. Timing is median / p95 / max across all reps
 * (3 outer runs x 5 reps = 15 samples per arm per size).
 */
class FrequencyStagePerformanceTest {
    private static final Logger log = LoggerFactory.getLogger(FrequencyStagePerformanceTest.class);

    private static final int WARMUP_ITERATIONS = 10;
    private static final int INNER_ITERATIONS = 20;
    private static final int TIMED_REPS = 5;
    private static final int OUTER_RUNS = 3;

    private static volatile Object blackhole;

    private record Case(int signalLength, int kernelLength, double[] signal, double[] kernel) {
        int fftSize() {
            return signalLength + kernelLength - 1;
        }
    }

    private static List<Case> cases() {
        int[][] lengths = {{256, 257}, {2048, 2049}, {8192, 8193}};
        List<Case> cases = new ArrayList<>();
        for (int[] pair : lengths) {
            Random random = new Random(42L);
            double[] signal = random.doubles(pair[0], -1.0, 1.0).toArray();
            double[] kernel = random.doubles(pair[1], -1.0, 1.0).toArray();
            cases.add(new Case(pair[0], pair[1], signal, kernel));
        }
        return cases;
    }

    @Test
    void allFrequencyAdapters_agreeWithApacheAdapter() {
        Convolution reference = new ApacheAdapter();
        Convolution[] adapters = {
                new FrequencyDomainAdapter(),
                new ValueFrequencyDomainAdapter(),
                new InterleavedFrequencyDomainAdapter()
        };
        for (Case testCase : cases()) {
            double[] expected = reference.with(testCase.signal(), testCase.kernel());
            for (Convolution adapter : adapters) {
                double[] actual = adapter.with(testCase.signal(), testCase.kernel());
                assertThat(actual.length).isEqualTo(expected.length);
                for (int i = 0; i < expected.length; i++) {
                    // FFT round-trip error grows with size; 1e-6 is loose
                    // against the ~1e-11 expected here and tight enough to
                    // catch a wrong conversion.
                    assertThat(actual[i]).isCloseTo(expected[i], within(1e-6));
                }
            }
        }
    }

    @Test
    void frequencyStage_endToEnd_reportsMedianP95Max() {
        List<Case> cases = cases();
        Convolution apache = new FrequencyDomainAdapter();
        Convolution value = new ValueFrequencyDomainAdapter();
        Convolution interleaved = new InterleavedFrequencyDomainAdapter();
        Convolution[] adapters = {apache, value, interleaved};
        String[] names = {"Apache", "Value", "Interleaved"};

        log.info("End-to-end frequency stage: {} outer runs x {} reps, each rep = {} convolutions",
                OUTER_RUNS, TIMED_REPS, INNER_ITERATIONS);

        // Pooled rep times (ns) and GC deltas per adapter per case.
        List<List<List<Long>>> times = new ArrayList<>();
        List<List<List<Long>>> gcRuns = new ArrayList<>();
        for (int a = 0; a < adapters.length; a++) {
            times.add(newTimesTable(cases.size()));
            gcRuns.add(newGcTable(cases.size()));
        }

        double sink = 0.0;
        for (int run = 0; run < OUTER_RUNS; run++) {
            log.info("--- End-to-end run {}/{} ---", run + 1, OUTER_RUNS);
            for (int c = 0; c < cases.size(); c++) {
                Case testCase = cases.get(c);
                log.info("FFT {} | Apache med (max) | Value med (max) | Interleaved med (max)", testCase.fftSize());
                double[] medians = new double[adapters.length];
                double[] maxes = new double[adapters.length];
                long[] gcs = new long[adapters.length];
                for (int a = 0; a < adapters.length; a++) {
                    final Convolution adapter = adapters[a];
                    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
                        blackhole = adapter.with(testCase.signal(), testCase.kernel());
                    }
                    long gcBefore = gcCount();
                    List<Long> reps = collectRepTimes(() -> adapter.with(testCase.signal(), testCase.kernel()));
                    gcs[a] = gcCount() - gcBefore;
                    times.get(a).get(c).addAll(reps);
                    gcRuns.get(a).get(c).add(gcs[a]);
                    medians[a] = percentileMs(reps, 50);
                    maxes[a] = percentileMs(reps, 100);
                    sink += medians[a];
                }
                log.info("{} | {} ({}) | {} ({}) | {} ({}) | GC {}/{}/{}",
                        testCase.fftSize(),
                        medians[0], maxes[0], medians[1], maxes[1], medians[2], maxes[2],
                        gcs[0], gcs[1], gcs[2]);
            }
        }

        log.info("=== End-to-end summary: median / p95 / max across all {} reps, GC = total collections ===",
                OUTER_RUNS * TIMED_REPS);
        log.info("FFT | Apache | Value | Interleaved | Value vs Apache (median) | Interleaved vs Apache (median) | GC Apache/Value/Interleaved");
        for (int c = 0; c < cases.size(); c++) {
            double apacheMed = percentileMs(times.get(0).get(c), 50);
            double valueMed = percentileMs(times.get(1).get(c), 50);
            double interleavedMed = percentileMs(times.get(2).get(c), 50);
            log.info("{} | {}/{}/{} | {}/{}/{} | {}/{}/{} | {}x | {}x | {}/{}/{}",
                    cases.get(c).fftSize(),
                    apacheMed, percentileMs(times.get(0).get(c), 95), percentileMs(times.get(0).get(c), 100),
                    valueMed, percentileMs(times.get(1).get(c), 95), percentileMs(times.get(1).get(c), 100),
                    interleavedMed, percentileMs(times.get(2).get(c), 95), percentileMs(times.get(2).get(c), 100),
                    apacheMed / valueMed, apacheMed / interleavedMed,
                    sumOf(gcRuns.get(0).get(c)), sumOf(gcRuns.get(1).get(c)), sumOf(gcRuns.get(2).get(c)));
        }
        log.info("End-to-end sink: {}", sink);
    }

    private List<Long> collectRepTimes(java.util.function.Supplier<Object> work) {
        List<Long> nanos = new ArrayList<>();
        for (int rep = 0; rep < TIMED_REPS; rep++) {
            long start = System.nanoTime();
            for (int i = 0; i < INNER_ITERATIONS; i++) {
                blackhole = work.get();
            }
            nanos.add(System.nanoTime() - start);
        }
        assertThat(blackhole).isNotNull();
        return nanos;
    }

    private double percentileMs(List<Long> nanos, double p) {
        List<Long> sorted = new ArrayList<>(nanos);
        Collections.sort(sorted);
        int index = Math.min(sorted.size() - 1, (int) Math.ceil(p / 100.0 * sorted.size()) - 1);
        return sorted.get(Math.max(0, index)) / 1_000_000.0;
    }

    private long gcCount() {
        long total = 0;
        for (GarbageCollectorMXBean bean : ManagementFactory.getGarbageCollectorMXBeans()) {
            total += bean.getCollectionCount();
        }
        return total;
    }

    private long sumOf(List<Long> values) {
        long total = 0;
        for (long value : values) {
            total += value;
        }
        return total;
    }

    private List<List<Long>> newTimesTable(int sizes) {
        List<List<Long>> table = new ArrayList<>();
        for (int i = 0; i < sizes; i++) {
            table.add(new ArrayList<>());
        }
        return table;
    }

    private List<List<Long>> newGcTable(int sizes) {
        List<List<Long>> table = new ArrayList<>();
        for (int i = 0; i < sizes; i++) {
            table.add(new ArrayList<>());
        }
        return table;
    }
}
