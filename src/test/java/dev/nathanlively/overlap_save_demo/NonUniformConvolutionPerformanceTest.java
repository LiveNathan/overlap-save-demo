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
 * Non-uniform partitioned convolution (Wefers layouts) against overlap-save
 * and uniform interleaved partitions, end to end and per block. The layout
 * is the variable here; the multiply representation is fixed at interleaved
 * flat {@code double[]} (the per-block winner for uniform partitions).
 *
 * <p>Layouts, all B=256 and all summing to 8192 taps:
 * <ul>
 *   <li>Wefers 128^4 512^15 — the table optimum as quoted in the Sheta slides
 *       (whose block size is unstated, so it is tested, not trusted).</li>
 *   <li>256^2 512^15 — same tail, first stage matched to the block size.</li>
 *   <li>256^4 1024^7 — coarser second stage, runs once per 4 blocks.</li>
 *   <li>Uniform 256 (control) — one group, i.e. uniform P=256 through this
 *       driver; it must match the uniform interleaved adapter. (A single
 *       512 group is not a valid layout here: its first partition exceeds
 *       the block size, so no output exists after the first block.)</li>
 * </ul>
 *
 * <p>End to end and per block run only on the 48000/8192 case — the case
 * the layouts are built for. On shorter kernels the convolver zero-pads to
 * the layout total, so those timings would measure the fixed layout, not
 * the kernel; correctness still covers all cases.
 *
 * <p>Reports median / max across all reps (3 outer runs x 5 reps = 15
 * samples) for the end-to-end comparison. The per-block test collects
 * 188 blocks x 15 reps = 2820 samples per arm, so its p95 is meaningful —
 * and for these layouts the median is the cheap block while the max is
 * the block where a big stage runs, which is exactly the peak Sheta warns
 * about.
 */
class NonUniformConvolutionPerformanceTest {
    private static final Logger log = LoggerFactory.getLogger(NonUniformConvolutionPerformanceTest.class);

    private static final int BLOCK_SIZE = 256;
    private static final int SAMPLE_RATE = 48000;
    private static final double BLOCK_DEADLINE_MS = BLOCK_SIZE * 1000.0 / SAMPLE_RATE;

    private static final int WARMUP_ITERATIONS = 5;
    private static final int INNER_ITERATIONS = 10;
    private static final int TIMED_REPS = 5;
    private static final int OUTER_RUNS = 3;

    private static volatile Object blackhole;

    private record Case(String name, int signalLength, int kernelLength, double[] signal, double[] kernel) {
    }

    private record Layout(String name, int[][] groups) {
    }

    private record NamedAdapter(String name, Convolution adapter) {
    }

    private record Arm(String name, Function<double[], StreamingConvolver> factory) {
    }

    private static List<Layout> layouts() {
        return List.of(
                new Layout("Wefers 128^4 512^15", new int[][]{{4, 128}, {15, 512}}),
                new Layout("256^2 512^15", new int[][]{{2, 256}, {15, 512}}),
                new Layout("256^4 1024^7", new int[][]{{4, 256}, {7, 1024}}),
                // Control: one group, i.e. uniform P=256 through this driver.
                // It must match the uniform interleaved adapter. A single 512
                // group is NOT a valid control: its first partition is bigger
                // than the block, so no output exists after the first block
                // and the zero-latency contract cannot be met (P0 = B).
                new Layout("Uniform 256 (control)", new int[][]{{32, 256}}));
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
    void nonUniform_agreesWithApacheAdapter() {
        Convolution reference = new ApacheAdapter();
        List<NamedAdapter> adapters = new ArrayList<>();
        for (Layout layout : layouts()) {
            adapters.add(new NamedAdapter(layout.name(),
                    new NonUniformPartitionedConvolutionAdapter(BLOCK_SIZE, layout.groups())));
        }
        for (Case testCase : cases()) {
            double[] expected = reference.with(testCase.signal(), testCase.kernel());
            for (NamedAdapter named : adapters) {
                double[] actual = named.adapter().with(testCase.signal(), testCase.kernel());
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
                    log.info("MISMATCH case={} adapter={} firstIndex={} actual={} expected={} totalMismatches={}/{}",
                            testCase.name(), named.name(), firstMismatch,
                            actual[firstMismatch], expected[firstMismatch], mismatches, expected.length);
                }
                assertThat(mismatches)
                        .as("case %s adapter %s", testCase.name(), named.name())
                        .isZero();
            }
        }
    }

    @Test
    void nonUniform_endToEnd_reportsMedianAndMax() {
        Case testCase = cases().get(3); // 48000/8192 (1s @ 48kHz)
        List<NamedAdapter> adapters = new ArrayList<>();
        adapters.add(new NamedAdapter("Overlap-save", new OverlapSaveAdapter()));
        adapters.add(new NamedAdapter("Uniform interleaved", new InterleavedPartitionedConvolutionAdapter()));
        for (Layout layout : layouts()) {
            adapters.add(new NamedAdapter(layout.name(),
                    new NonUniformPartitionedConvolutionAdapter(BLOCK_SIZE, layout.groups())));
        }

        log.info("Non-uniform vs overlap-save / uniform interleaved, {}: {} outer runs x {} reps, each rep = {} convolutions",
                testCase.name(), OUTER_RUNS, TIMED_REPS, INNER_ITERATIONS);

        List<List<Long>> times = new ArrayList<>();
        List<List<Long>> gcRuns = new ArrayList<>();
        for (int a = 0; a < adapters.size(); a++) {
            times.add(new ArrayList<>());
            gcRuns.add(new ArrayList<>());
        }

        double sink = 0.0;
        for (int run = 0; run < OUTER_RUNS; run++) {
            log.info("--- Non-uniform run {}/{} ---", run + 1, OUTER_RUNS);
            for (int a = 0; a < adapters.size(); a++) {
                NamedAdapter named = adapters.get(a);
                for (int i = 0; i < WARMUP_ITERATIONS; i++) {
                    blackhole = named.adapter().with(testCase.signal(), testCase.kernel());
                }
                long gcBefore = gcCount();
                List<Long> reps = collectRepTimes(() -> named.adapter().with(testCase.signal(), testCase.kernel()));
                long gc = gcCount() - gcBefore;
                times.get(a).addAll(reps);
                gcRuns.get(a).add(gc);
                double median = percentileMs(reps, 50);
                double max = percentileMs(reps, 100);
                sink += median;
                log.info("{}: median {} ms | max {} ms | GC {}", named.name(), median, max, gc);
            }
        }

        log.info("=== Non-uniform summary ({}): median / max across all {} reps, GC = total collections ===",
                testCase.name(), OUTER_RUNS * TIMED_REPS);
        for (int a = 0; a < adapters.size(); a++) {
            log.info("{} | {}/{} | vs overlap-save {}x | GC {}",
                    adapters.get(a).name(),
                    percentileMs(times.get(a), 50), percentileMs(times.get(a), 100),
                    percentileMs(times.get(0), 50) / percentileMs(times.get(a), 50),
                    sumOf(gcRuns.get(a)));
        }
        log.info("Non-uniform sink: {}", sink);
    }

    @Test
    void nonUniform_perBlockTimesAgainstDeadline() {
        Case testCase = cases().get(3); // 48000/8192 (1s @ 48kHz)
        int blocksPerRep = (testCase.signalLength() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        log.info("Per-block deadline: B={} @ {} Hz = {} ms, {} blocks per rep, {} reps per arm",
                BLOCK_SIZE, SAMPLE_RATE, BLOCK_DEADLINE_MS, blocksPerRep, OUTER_RUNS * TIMED_REPS);

        List<Arm> arms = new ArrayList<>();
        arms.add(new Arm("Uniform interleaved", kernel -> new InterleavedPartitionedConvolver(kernel, BLOCK_SIZE)));
        for (Layout layout : layouts()) {
            arms.add(new Arm(layout.name(),
                    kernel -> new NonUniformPartitionedConvolver(kernel, BLOCK_SIZE, layout.groups())));
        }

        for (Arm arm : arms) {
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
        double[] block = new double[BLOCK_SIZE];
        int copyLength = Math.min(BLOCK_SIZE, signal.length - blockIndex * BLOCK_SIZE);
        if (copyLength > 0) {
            System.arraycopy(signal, blockIndex * BLOCK_SIZE, block, 0, copyLength);
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
}
