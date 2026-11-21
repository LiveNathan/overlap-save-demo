package dev.nathanlively.overlap_save_demo;

import org.apache.commons.numbers.complex.Complex;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
        assertThat(new ComplexValue(1.0, 2.0).equals(new ComplexValue(1.0, 2.0))).isTrue();
        assertThat(new ComplexValue(1.0, 2.0).equals(new ComplexValue(1.0, 3.0))).isFalse();
    }

    @Test
    void valueMultiply_performance_vsApacheAndSoA() {
        int[] sizes = {256, 4096, 16384};

        log.info("Multiply kernel: median of {} reps, each rep = {} multiplies", TIMED_REPS, INNER_ITERATIONS);
        log.info("Size | Apache Complex (ms) | ComplexValue (ms) | SoA double[] (ms) | Value vs Apache");
        log.info("-----|---------------------|-------------------|-------------------|----------------");

        double sink = 0.0; // checksum sink: keeps the JIT from deleting the multiply
        for (int size : sizes) {
            TestData data = generate(size, 42L);

            // Warmup: same data, same code paths as the timed section.
            for (int i = 0; i < WARMUP_ITERATIONS; i++) {
                sink += checksum(multiplyApache(data.apacheA(), data.apacheB()));
                sink += checksum(multiplyValue(data.valueA(), data.valueB()));
                sink += checksum(multiplySoA(data.aRe(), data.aIm(), data.bRe(), data.bIm()));
            }

            // Correctness first, on the warmed code: all three must agree.
            assertAllAgree(data);

            double apacheMs = medianMs(() -> multiplyApache(data.apacheA(), data.apacheB()));
            double valueMs = medianMs(() -> multiplyValue(data.valueA(), data.valueB()));
            double soaMs = medianMs(() -> multiplySoA(data.aRe(), data.aIm(), data.bRe(), data.bIm()));

            log.info("{} | {} | {} | {} | {}x", size, apacheMs, valueMs, soaMs, apacheMs / valueMs);
            sink += apacheMs + valueMs + soaMs;
        }
        log.info("Checksum sink (prevents dead-code elimination): {}", sink);
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
    private static volatile Object blackhole;

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
