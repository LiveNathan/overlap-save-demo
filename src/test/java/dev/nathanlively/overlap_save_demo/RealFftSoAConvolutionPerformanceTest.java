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
import java.util.function.Function;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * The SoA arm with the hand-rolled real FFT, against overlap-save, SoA
 * and InPlace — end to end and per block. Same FDL / overlap-add loop
 * and same Vector API multiply everywhere; only the FFT call differs,
 * so this isolates what the real-FFT halving buys. FftPerformanceTest
 * measured the FFT-512 round trip at 4.54 us hand real vs 6.29 us in
 * place with reused buffers, so the per-block median should drop by
 * roughly 1.75 us from InPlace's ~16.7 us.
 *
 * <p>Reports median / max across all reps (3 outer runs x 5 reps = 15
 * samples) for the end-to-end comparison. The per-block test collects
 * 188 blocks x 15 reps = 2820 samples per arm, so its p95 is meaningful.
 */
class RealFftSoAConvolutionPerformanceTest {
    private static final Logger log = LoggerFactory.getLogger(RealFftSoAConvolutionPerformanceTest.class);

    private static final int PARTITION_SIZE = 256;
    private static final int SAMPLE_RATE = 48000;
    private static final double BLOCK_DEADLINE_MS = PARTITION_SIZE * 1000.0 / SAMPLE_RATE;

    private static final int WARMUP_ITERATIONS = 5;
    private static final int INNER_ITERATIONS = 10;
    private static final int TIMED_REPS = 5;
    private static final int OUTER_RUNS = 3;

    private static volatile Object blackhole;

    private record Case(String name, int signalLength, int kernelLength, double[] signal, double[] kernel) {
    }

    private record Arm(String name, Function<double[], StreamingConvolver> factory) {
    }

    private static List<Arm> convolverArms() {
        return List.of(
                new Arm("SoA", kernel -> new SoAPartitionedConvolver(kernel, PARTITION_SIZE)),
                new Arm("InPlace", kernel -> new InPlaceSoAPartitionedConvolver(kernel, PARTITION_SIZE)),
                new Arm("RealFft", kernel -> new RealFftSoAPartitionedConvolver(kernel, PARTITION_SIZE)));
    }

    private static List<Case> cases() {
        Object[][] specs = {
                {"32768/64", 32768, 64},
                {"32768/512", 32768, 512},
                {"48000/1024 (1s @ 48kHz)", 48000, 1024},
                {"48000/8192 (1s @ 48kHz)", 48000, 8192}
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
    void realFft_agreesWithApacheAdapter() {
        Convolution reference = new ApacheAdapter();
        Convolution adapter = new RealFftSoAPartitionedConvolutionAdapter();
        for (Case testCase : cases()) {
            double[] expected = reference.with(testCase.signal(), testCase.kernel());
            double[] actual = adapter.with(testCase.signal(), testCase.kernel());
            assertThat(actual.length).isEqualTo(expected.length);
            int mismatches = 0;
            int firstMismatch = -1;
            for (int i = 0; i < expected.length; i++) {
                if (Math.abs(actual[i] - expected[i]) > 1e-6) {
                    if (firstMismatch < 0) {
                        firstMismatch = i;
                    }
                    mismatches++;
                }
            }
            if (mismatches > 0) {
                log.info("MISMATCH case={} adapter=RealFft firstIndex={} actual={} expected={} totalMismatches={}/{}",
                        testCase.name(), firstMismatch,
                        actual[firstMismatch], expected[firstMismatch], mismatches, expected.length);
            }
            assertThat(mismatches)
                    .as("case %s adapter RealFft", testCase.name())
                    .isZero();
        }
    }

    @Test
    void realFft_endToEnd_reportsMedianAndMax() {
        List<Case> cases = cases();
        Convolution[] adapters = {
                new OverlapSaveAdapter(),
                new SoAPartitionedConvolutionAdapter(),
                new InPlaceSoAPartitionedConvolutionAdapter(),
                new RealFftSoAPartitionedConvolutionAdapter()};

        log.info("RealFft (P={}) vs overlap-save, SoA and InPlace: {} outer runs x {} reps, each rep = {} convolutions",
                PARTITION_SIZE, OUTER_RUNS, TIMED_REPS, INNER_ITERATIONS);

        List<List<List<Long>>> times = new ArrayList<>();
        List<List<List<Long>>> gcRuns = new ArrayList<>();
        for (int a = 0; a < adapters.length; a++) {
            times.add(newTimesTable(cases.size()));
            gcRuns.add(newGcTable(cases.size()));
        }

        double sink = 0.0;
        for (int run = 0; run < OUTER_RUNS; run++) {
            log.info("--- RealFft run {}/{} ---", run + 1, OUTER_RUNS);
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
                log.info("{} | Overlap-save {} ({}) | SoA {} ({}) | InPlace {} ({}) | RealFft {} ({}) | vs overlap-save {}x/{}x/{}x | GC {}/{}/{}/{}",
                        testCase.name(), medians[0], maxes[0], medians[1], maxes[1],
                        medians[2], maxes[2], medians[3], maxes[3],
                        medians[0] / medians[1], medians[0] / medians[2], medians[0] / medians[3],
                        gcs[0], gcs[1], gcs[2], gcs[3]);
            }
        }

        log.info("=== RealFft summary: median / max across all {} reps, GC = total collections ===",
                OUTER_RUNS * TIMED_REPS);
        log.info("Case | Overlap-save | SoA | InPlace | RealFft | vs overlap-save (median) | GC");
        for (int c = 0; c < cases.size(); c++) {
            double[] med = new double[adapters.length];
            double[] max = new double[adapters.length];
            for (int a = 0; a < adapters.length; a++) {
                med[a] = percentileMs(times.get(a).get(c), 50);
                max[a] = percentileMs(times.get(a).get(c), 100);
            }
            log.info("{} | {}/{} | {}/{} | {}/{} | {}/{} | {}x/{}x/{}x | {}/{}/{}/{}",
                    cases.get(c).name(),
                    med[0], max[0], med[1], max[1], med[2], max[2], med[3], max[3],
                    med[0] / med[1], med[0] / med[2], med[0] / med[3],
                    sumOf(gcRuns.get(0).get(c)), sumOf(gcRuns.get(1).get(c)),
                    sumOf(gcRuns.get(2).get(c)), sumOf(gcRuns.get(3).get(c)));
        }
        log.info("RealFft sink: {}", sink);
    }

    @Test
    void realFft_perBlockTimesAgainstDeadline() {
        Case testCase = cases().get(3); // 48000/8192 (1s @ 48kHz)
        int blocksPerRep = (testCase.signalLength() + PARTITION_SIZE - 1) / PARTITION_SIZE;
        log.info("Per-block deadline: P={} @ {} Hz = {} ms, {} blocks per rep, {} reps per arm",
                PARTITION_SIZE, SAMPLE_RATE, BLOCK_DEADLINE_MS, blocksPerRep, OUTER_RUNS * TIMED_REPS);

        for (Arm arm : convolverArms()) {
            for (int i = 0; i < WARMUP_ITERATIONS; i++) {
                StreamingConvolver warmup = arm.factory().apply(testCase.kernel());
                for (int b = 0; b < blocksPerRep; b++) {
                    blackhole = warmup.processBlock(blockSlice(testCase.signal(), b));
                }
                blackhole = warmup.flush();
            }

            List<Long> blockNanos = new ArrayList<>();
            List<Long> flushNanos = new ArrayList<>();
            for (int run = 0; run < OUTER_RUNS; run++) {
                for (int rep = 0; rep < TIMED_REPS; rep++) {
                    StreamingConvolver convolver = arm.factory().apply(testCase.kernel());
                    for (int b = 0; b < blocksPerRep; b++) {
                        double[] block = blockSlice(testCase.signal(), b);
                        long start = System.nanoTime();
                        blackhole = convolver.processBlock(block);
                        blockNanos.add(System.nanoTime() - start);
                    }
                    long flushStart = System.nanoTime();
                    blackhole = convolver.flush();
                    flushNanos.add(System.nanoTime() - flushStart);
                }
            }
            assertThat(blackhole).isNotNull();

            long overDeadline = blockNanos.stream()
                    .filter(nanos -> nanos / 1_000_000.0 > BLOCK_DEADLINE_MS)
                    .count();
            log.info("{} per-block ({} samples): median {} ms | p95 {} ms | max {} ms | over {} ms deadline: {}/{} | flush median {} ms",
                    arm.name(), blockNanos.size(),
                    percentileMs(blockNanos, 50), percentileMs(blockNanos, 95), percentileMs(blockNanos, 100),
                    BLOCK_DEADLINE_MS, overDeadline, blockNanos.size(),
                    percentileMs(flushNanos, 50));
        }
    }

    private double[] blockSlice(double[] signal, int blockIndex) {
        double[] block = new double[PARTITION_SIZE];
        int copyLength = Math.min(PARTITION_SIZE, signal.length - blockIndex * PARTITION_SIZE);
        if (copyLength > 0) {
            System.arraycopy(signal, blockIndex * PARTITION_SIZE, block, 0, copyLength);
        }
        return block;
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
