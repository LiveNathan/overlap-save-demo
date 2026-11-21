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
 * Overlap-save version of the multiply experiment: long signal, short-to-
 * medium kernel — the shape overlap-save exists for, and the shape closest
 * to real-time audio (one long stream, one fixed impulse response). The
 * kernel FFT is computed once by both arms; only the per-block multiply
 * differs (Apache {@code Complex} vs one interleaved flat {@code double[]}),
 * plus the per-block conversion of the block transform to and from
 * interleaved form. The kernel conversion is paid once by the flat arm, so
 * this measures the multiply win with the kernel side amortised away.
 *
 * <p>Reports median / max across all reps (3 outer runs x 5 reps = 15
 * samples). No p95 column: with 15 samples p95 is the max by construction,
 * as the end-to-end test showed.
 */
class OverlapSavePerformanceTest {
    private static final Logger log = LoggerFactory.getLogger(OverlapSavePerformanceTest.class);

    private static final int WARMUP_ITERATIONS = 5;
    private static final int INNER_ITERATIONS = 10;
    private static final int TIMED_REPS = 5;
    private static final int OUTER_RUNS = 3;

    private static volatile Object blackhole;

    private record Case(String name, int signalLength, int kernelLength, double[] signal, double[] kernel) {
    }

    private static List<Case> cases() {
        Object[][] specs = {
                {"32768/64", 32768, 64},
                {"32768/512", 32768, 512},
                {"48000/1024 (1s @ 48kHz)", 48000, 1024}
        };
        List<Case> cases = new ArrayList<>();
        for (Object[] spec : specs) {
            Random random = new Random(42L);
            int signalLength = (int) spec[1];
            int kernelLength = (int) spec[2];
            double[] signal = random.doubles(signalLength, -1.0, 1.0).toArray();
            double[] kernel = random.doubles(kernelLength, -1.0, 1.0).toArray();
            cases.add(new Case((String) spec[0], signalLength, kernelLength, signal, kernel));
        }
        return cases;
    }

    @Test
    void interleavedOverlapSave_agreesWithApacheAdapter() {
        Convolution reference = new ApacheAdapter();
        Convolution apache = new OverlapSaveAdapter();
        Convolution interleaved = new InterleavedOverlapSaveAdapter();
        for (Case testCase : cases()) {
            double[] expected = reference.with(testCase.signal(), testCase.kernel());
            for (Convolution adapter : new Convolution[]{apache, interleaved}) {
                double[] actual = adapter.with(testCase.signal(), testCase.kernel());
                assertThat(actual.length).isEqualTo(expected.length);
                for (int i = 0; i < expected.length; i++) {
                    assertThat(actual[i]).isCloseTo(expected[i], within(1e-6));
                }
            }
        }
    }

    @Test
    void overlapSave_endToEnd_reportsMedianAndMax() {
        List<Case> cases = cases();
        Convolution apache = new OverlapSaveAdapter();
        Convolution interleaved = new InterleavedOverlapSaveAdapter();
        Convolution[] adapters = {apache, interleaved};

        log.info("Overlap-save: {} outer runs x {} reps, each rep = {} convolutions",
                OUTER_RUNS, TIMED_REPS, INNER_ITERATIONS);

        List<List<List<Long>>> times = new ArrayList<>();
        List<List<List<Long>>> gcRuns = new ArrayList<>();
        for (int a = 0; a < adapters.length; a++) {
            times.add(newTimesTable(cases.size()));
            gcRuns.add(newGcTable(cases.size()));
        }

        double sink = 0.0;
        for (int run = 0; run < OUTER_RUNS; run++) {
            log.info("--- Overlap-save run {}/{} ---", run + 1, OUTER_RUNS);
            for (int c = 0; c < cases.size(); c++) {
                Case testCase = cases.get(c);
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
                log.info("{} | Apache {} ({}) | Interleaved {} ({}) | Interleaved vs Apache {}x | GC {}/{}",
                        testCase.name(), medians[0], maxes[0], medians[1], maxes[1],
                        medians[0] / medians[1], gcs[0], gcs[1]);
            }
        }

        log.info("=== Overlap-save summary: median / max across all {} reps, GC = total collections ===",
                OUTER_RUNS * TIMED_REPS);
        log.info("Case | Apache | Interleaved | Interleaved vs Apache (median) | GC Apache/Interleaved");
        for (int c = 0; c < cases.size(); c++) {
            double apacheMed = percentileMs(times.get(0).get(c), 50);
            double interleavedMed = percentileMs(times.get(1).get(c), 50);
            log.info("{} | {}/{} | {}/{} | {}x | {}/{}",
                    cases.get(c).name(),
                    apacheMed, percentileMs(times.get(0).get(c), 100),
                    interleavedMed, percentileMs(times.get(1).get(c), 100),
                    apacheMed / interleavedMed,
                    sumOf(gcRuns.get(0).get(c)), sumOf(gcRuns.get(1).get(c)));
        }
        log.info("Overlap-save sink: {}", sink);
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
