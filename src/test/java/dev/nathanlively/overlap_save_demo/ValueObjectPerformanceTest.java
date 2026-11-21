package dev.nathanlively.overlap_save_demo;

import org.apache.commons.numbers.complex.Complex;
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
 * First slice of the Java 28 value-object comparison: the multiply kernel only.
 *
 * <p>Why start here, not end-to-end? {@link SignalTransformer#multiply} is the one
 * place in the FFT path that allocates a fresh object per element, per block. If a
 * {@code value record ComplexValue} helps anywhere, it helps here first — and if it
 * does not help here, an end-to-end adapter built on it will not help either.
 *
 * <p>Three arms, same math, different representation:
 * <ul>
 *   <li>Apache {@code Complex[]} — the current baseline, one heap object per element.</li>
 *   <li>{@code ComplexValue[]} — the value-object arm (two doubles, 128 bits, so
 *       expect scalarization in the hot loop, not necessarily array flattening
 *       in this first JEP 401 preview).</li>
 *   <li>Two parallel {@code double[]} (re/im) — the structure-of-arrays control.
 *       If value only matches SoA, the win was "stop allocating objects", not
 *       value objects specifically.</li>
 * </ul>
 *
 * <p>Deliberately no "must be faster" assertion: this test logs numbers for the
 * blog post. A perf assertion would fail on a cold CI box and tell us nothing.
 */
class ValueObjectPerformanceTest {
    private static final Logger log = LoggerFactory.getLogger(ValueObjectPerformanceTest.class);
    private static final double PRECISION = 1e-12;
    private static final int WARMUP_ITERATIONS = 20;
    private static final int INNER_ITERATIONS = 100;
    private static final int TIMED_REPS = 5;
    // Whole-sweep repetitions: the sweep runs this many times in one JVM so the
    // summary can show run-to-run spread instead of a single median.
    private static final int OUTER_RUNS = 5;

    @Test
    void valueMultiply_matchesApacheMultiply() {
        TestData data = generate(8, 42L);

        Complex[] apache = multiplyApache(data.apacheA(), data.apacheB());
        ComplexValue[] value = multiplyValue(data.valueA(), data.valueB());
        double[][] soa = multiplySoA(data.aRe(), data.aIm(), data.bRe(), data.bIm());

        assertThat(value).hasSize(apache.length);
        for (int i = 0; i < apache.length; i++) {
            assertThat(value[i].real()).isCloseTo(apache[i].getReal(), within(PRECISION));
            assertThat(value[i].imaginary()).isCloseTo(apache[i].getImaginary(), within(PRECISION));
            assertThat(soa[0][i]).isCloseTo(apache[i].getReal(), within(PRECISION));
            assertThat(soa[1][i]).isCloseTo(apache[i].getImaginary(), within(PRECISION));
        }
    }

    @Test
    void valueObjects_haveValueSemantics() {
        // The demo for the blog post: == compares fields, not identity.
        // Two separately constructed values with the same fields are ==.
        // NB: == here is the whole point, not a mistake — for a value object it
        // compares fields, for an ordinary record it would compare identity and
        // fail. Do not let the IDE "fix" this to .equals(): equals() would pass
        // for a plain record too, so it would prove nothing about value semantics.
        assertThat(new ComplexValue(1.0, 2.0) == new ComplexValue(1.0, 2.0)).isTrue();
        assertThat(new ComplexValue(1.0, 2.0) == new ComplexValue(1.0, 3.0)).isFalse();
    }

    @Test
    void valueMultiply_performance_vsApacheAndSoA() {
        int[] sizes = {256, 4096, 16384};
        // Same inputs for every outer run, so runs are directly comparable.
        List<TestData> datasets = new ArrayList<>();
        for (int size : sizes) {
            datasets.add(generate(size, 42L));
        }

        log.info("Multiply kernel: {} outer runs, each run = median of {} reps, each rep = {} multiplies",
                OUTER_RUNS, TIMED_REPS, INNER_ITERATIONS);

        // One median (and one GC delta) per outer run, per size, per arm.
        List<List<Double>> apacheRuns = newRunsTable(sizes.length);
        List<List<Double>> valueRuns = newRunsTable(sizes.length);
        List<List<Double>> soaRuns = newRunsTable(sizes.length);
        List<List<Long>> apacheGcRuns = newGcTable(sizes.length);
        List<List<Long>> valueGcRuns = newGcTable(sizes.length);
        List<List<Long>> soaGcRuns = newGcTable(sizes.length);

        double sink = 0.0; // checksum sink: keeps the JIT from deleting the warmup multiply
        for (int run = 0; run < OUTER_RUNS; run++) {
            log.info("--- Run {}/{} ---", run + 1, OUTER_RUNS);
            log.info("Size | Apache (ms) | Value (ms) | SoA (ms) | Value vs Apache | GC Apache/Value/SoA");
            for (int s = 0; s < sizes.length; s++) {
                TestData data = datasets.get(s);

                // Warmup: same data, same code paths as the timed section.
                for (int i = 0; i < WARMUP_ITERATIONS; i++) {
                    sink += checksum(multiplyApache(data.apacheA(), data.apacheB()));
                    sink += checksum(multiplyValue(data.valueA(), data.valueB()));
                    sink += checksum(multiplySoA(data.aRe(), data.aIm(), data.bRe(), data.bIm()));
                }

                // Correctness first, on the warmed code: all three must agree.
                if (run == 0) {
                    assertAllAgree(data);
                }

                long gcBefore = gcCount();
                double apacheMs = medianMs(() -> multiplyApache(data.apacheA(), data.apacheB()));
                long apacheGc = gcCount() - gcBefore;

                gcBefore = gcCount();
                double valueMs = medianMs(() -> multiplyValue(data.valueA(), data.valueB()));
                long valueGc = gcCount() - gcBefore;

                gcBefore = gcCount();
                double soaMs = medianMs(() -> multiplySoA(data.aRe(), data.aIm(), data.bRe(), data.bIm()));
                long soaGc = gcCount() - gcBefore;

                apacheRuns.get(s).add(apacheMs);
                valueRuns.get(s).add(valueMs);
                soaRuns.get(s).add(soaMs);
                apacheGcRuns.get(s).add(apacheGc);
                valueGcRuns.get(s).add(valueGc);
                soaGcRuns.get(s).add(soaGc);

                log.info("{} | {} | {} | {} | {}x | {}/{}/{}",
                        sizes[s], apacheMs, valueMs, soaMs, apacheMs / valueMs, apacheGc, valueGc, soaGc);
                sink += apacheMs + valueMs + soaMs;
            }
        }

        log.info("=== Summary across {} runs: median (min-max), GC = total collections ===", OUTER_RUNS);
        log.info("Size | Apache (ms) | Value (ms) | SoA (ms) | Value vs Apache | GC Apache/Value/SoA");
        for (int s = 0; s < sizes.length; s++) {
            double apacheMed = medianOf(apacheRuns.get(s));
            double valueMed = medianOf(valueRuns.get(s));
            double soaMed = medianOf(soaRuns.get(s));
            log.info("{} | {} ({}-{}) | {} ({}-{}) | {} ({}-{}) | {}x | {}/{}/{}",
                    sizes[s],
                    apacheMed, Collections.min(apacheRuns.get(s)), Collections.max(apacheRuns.get(s)),
                    valueMed, Collections.min(valueRuns.get(s)), Collections.max(valueRuns.get(s)),
                    soaMed, Collections.min(soaRuns.get(s)), Collections.max(soaRuns.get(s)),
                    apacheMed / valueMed,
                    sumOf(apacheGcRuns.get(s)), sumOf(valueGcRuns.get(s)), sumOf(soaGcRuns.get(s)));
        }
        log.info("Checksum sink (prevents dead-code elimination): {}", sink);
    }

    private List<List<Double>> newRunsTable(int sizes) {
        List<List<Double>> table = new ArrayList<>();
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

    private double medianOf(List<Double> values) {
        List<Double> sorted = new ArrayList<>(values);
        Collections.sort(sorted);
        return sorted.get(sorted.size() / 2);
    }

    private long sumOf(List<Long> values) {
        long sum = 0;
        for (long v : values) {
            sum += v;
        }
        return sum;
    }

    /** Total GC collections across all collectors (-1, "undefined", excluded). */
    private long gcCount() {
        return ManagementFactory.getGarbageCollectorMXBeans().stream()
                .mapToLong(GarbageCollectorMXBean::getCollectionCount)
                .filter(c -> c >= 0)
                .sum();
    }

    private void assertAllAgree(TestData data) {
        Complex[] apache = multiplyApache(data.apacheA(), data.apacheB());
        ComplexValue[] value = multiplyValue(data.valueA(), data.valueB());
        double[][] soa = multiplySoA(data.aRe(), data.aIm(), data.bRe(), data.bIm());
        for (int i = 0; i < apache.length; i++) {
            assertThat(value[i].real()).isCloseTo(apache[i].getReal(), within(1e-9));
            assertThat(value[i].imaginary()).isCloseTo(apache[i].getImaginary(), within(1e-9));
            assertThat(soa[0][i]).isCloseTo(apache[i].getReal(), within(1e-9));
            assertThat(soa[1][i]).isCloseTo(apache[i].getImaginary(), within(1e-9));
        }
    }

    // Hand-rolled blackhole: storing each result in a volatile field forces it to
    // escape, so the JIT cannot delete the multiply. Cheaper and less biased than
    // checksumming the whole result inside the timed loop (whose getReal() calls
    // would have penalised the Apache arm specifically).
    private static volatile Object blackhole = new Object();

    /** Runs the work {@link #INNER_ITERATIONS} times per rep, {@link #TIMED_REPS} reps, returns the median rep time in ms. */
    private double medianMs(java.util.function.Supplier<Object> work) {
        List<Long> nanos = new ArrayList<>();
        for (int rep = 0; rep < TIMED_REPS; rep++) {
            long start = System.nanoTime();
            for (int i = 0; i < INNER_ITERATIONS; i++) {
                blackhole = work.get();
            }
            nanos.add(System.nanoTime() - start);
        }
        Collections.sort(nanos);
        // Read it back: the field is assigned in the timed loop, and this read
        // both silences the "assigned but never accessed" warning and keeps the
        // stored values live from the JIT's point of view. (A plain null check
        // here would be flagged as always-false, since the work never returns
        // null — an assertion reads the field without stating a condition.)
        assertThat(blackhole).isNotNull();
        return nanos.get(nanos.size() / 2) / 1_000_000.0;
    }

    // The three arms. Each allocates its result fresh per call, like SignalTransformer.multiply does.

    private Complex[] multiplyApache(Complex[] a, Complex[] b) {
        Complex[] result = new Complex[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i].multiply(b[i]);
        }
        return result;
    }

    private ComplexValue[] multiplyValue(ComplexValue[] a, ComplexValue[] b) {
        ComplexValue[] result = new ComplexValue[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i].multiply(b[i]);
        }
        return result;
    }

    private double[][] multiplySoA(double[] aRe, double[] aIm, double[] bRe, double[] bIm) {
        double[] outRe = new double[aRe.length];
        double[] outIm = new double[aRe.length];
        for (int i = 0; i < aRe.length; i++) {
            outRe[i] = aRe[i] * bRe[i] - aIm[i] * bIm[i];
            outIm[i] = aRe[i] * bIm[i] + aIm[i] * bRe[i];
        }
        return new double[][]{outRe, outIm};
    }

    private double checksum(Complex[] values) {
        double sum = 0.0;
        for (Complex v : values) {
            sum += v.getReal() + v.getImaginary();
        }
        return sum;
    }

    private double checksum(ComplexValue[] values) {
        double sum = 0.0;
        for (ComplexValue v : values) {
            sum += v.real() + v.imaginary();
        }
        return sum;
    }

    private double checksum(double[][] soa) {
        double sum = 0.0;
        for (double v : soa[0]) {
            sum += v;
        }
        for (double v : soa[1]) {
            sum += v;
        }
        return sum;
    }

    private TestData generate(int size, long seed) {
        Random random = new Random(seed); // deterministic: same inputs every run
        double[] aRe = new double[size];
        double[] aIm = new double[size];
        double[] bRe = new double[size];
        double[] bIm = new double[size];
        Complex[] apacheA = new Complex[size];
        Complex[] apacheB = new Complex[size];
        ComplexValue[] valueA = new ComplexValue[size];
        ComplexValue[] valueB = new ComplexValue[size];
        for (int i = 0; i < size; i++) {
            aRe[i] = random.nextDouble() * 2 - 1;
            aIm[i] = random.nextDouble() * 2 - 1;
            bRe[i] = random.nextDouble() * 2 - 1;
            bIm[i] = random.nextDouble() * 2 - 1;
            apacheA[i] = Complex.ofCartesian(aRe[i], aIm[i]);
            apacheB[i] = Complex.ofCartesian(bRe[i], bIm[i]);
            valueA[i] = new ComplexValue(aRe[i], aIm[i]);
            valueB[i] = new ComplexValue(bRe[i], bIm[i]);
        }
        return new TestData(apacheA, apacheB, valueA, valueB, aRe, aIm, bRe, bIm);
    }

    private record TestData(Complex[] apacheA, Complex[] apacheB,
                            ComplexValue[] valueA, ComplexValue[] valueB,
                            double[] aRe, double[] aIm, double[] bRe, double[] bIm) {
    }
}
