package dev.nathanlively.overlap_save_demo;

import org.apache.commons.math4.transform.FastFourierTransform;
import org.apache.commons.numbers.complex.Complex;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Apache's FFT in isolation, at the sizes the partitioned arms use,
 * against the hand-rolled core.
 *
 * <p>The SoA arm's per-block work is one forward FFT-512 plus one inverse
 * FFT-512 plus a cheap Vector API multiply — so the question this test
 * answers is how much of a block is the two transforms, and how much of
 * that is Apache's {@code apply} wrapper (Complex[] allocations and the
 * split/merge conversions around it) versus the butterfly core
 * ({@code transformInPlace} on split-complex arrays).
 *
 * <p>Five paths per size, timed per call (1000 calls x 5 reps x 3 runs):
 * <ul>
 *   <li>{@code apply} — exactly what the SoA arm calls today:
 *       {@code SignalTransformer.fft(pad(block))} then {@code ifft}.</li>
 *   <li>{@code in-place} — the same Apache core, called directly on
 *       split-complex {@code double[2][n]} with fresh buffers per call.</li>
 *   <li>{@code in-place reused} — same, with one pair of buffers reused
 *       across calls (refilled each call).</li>
 *   <li>{@code hand complex} — the hand-rolled core ({@link HandRolledFft}),
 *       same split-complex shape, full twiddle table, buffers reused.</li>
 *   <li>{@code hand real} — the hand-rolled real variants: one half-size
 *       complex FFT plus O(N) packing per direction, buffers reused.</li>
 * </ul>
 * Plus a same-run SoA {@code processBlock} median, so the round-trip time
 * can be read as a share of a block. Times are microseconds: a round trip
 * is tens of microseconds, a block is tens of microseconds, and the
 * deadline is 5333 microseconds.
 */
class FftPerformanceTest {
    private static final Logger log = LoggerFactory.getLogger(FftPerformanceTest.class);

    private static final int PARTITION_SIZE = 256;
    private static final int SAMPLE_RATE = 48000;
    private static final double BLOCK_DEADLINE_MS = PARTITION_SIZE * 1000.0 / SAMPLE_RATE;

    private static final int[] FFT_SIZES = {256, 512, 1024};

    private static final int WARMUP_CALLS = 2000;
    private static final int CALLS_PER_REP = 1000;
    private static final int TIMED_REPS = 5;
    private static final int OUTER_RUNS = 3;

    private static final FastFourierTransform FORWARD =
            new FastFourierTransform(FastFourierTransform.Norm.STD);
    private static final FastFourierTransform INVERSE =
            new FastFourierTransform(FastFourierTransform.Norm.STD, true);

    private static volatile Object blackhole;

    @Test
    void fftRoundTrip_agreesWithSignalTransformer() {
        int fftSize = 512;
        double[] block = deterministicBlock(fftSize / 2);
        double[] padded = SignalTransformer.pad(block, fftSize);
        double[][] reused = {new double[fftSize], new double[fftSize]};

        double[] viaApply = roundTripApply(block, fftSize);
        double[] viaInPlace = roundTripInPlace(block, fftSize);
        double[] viaReused = roundTripInPlaceReused(block, fftSize, reused);

        assertThat(maxAbsDiff(viaApply, padded)).isLessThan(1e-9);
        assertThat(maxAbsDiff(viaInPlace, padded)).isLessThan(1e-9);
        assertThat(maxAbsDiff(viaReused, padded)).isLessThan(1e-9);
        assertThat(maxAbsDiff(viaApply, viaInPlace)).isLessThan(1e-12);
        assertThat(maxAbsDiff(viaApply, viaReused)).isLessThan(1e-12);

        // Hand-rolled paths agree with Apache, forward and round trip.
        double[][] handComplex = {new double[fftSize], new double[fftSize]};
        double[] handRe = new double[fftSize];
        double[] handIm = new double[fftSize];
        double[] handBack = new double[fftSize];
        double[] viaHandComplex = roundTripHandComplex(block, fftSize, handComplex);
        double[] viaHandReal = roundTripHandReal(padded, handRe, handIm, handBack);
        assertThat(maxAbsDiff(viaHandComplex, padded)).isLessThan(1e-9);
        assertThat(maxAbsDiff(viaHandReal, padded)).isLessThan(1e-9);
        assertThat(maxAbsDiff(viaHandComplex, viaApply)).isLessThan(1e-9);
        assertThat(maxAbsDiff(viaHandReal, viaApply)).isLessThan(1e-9);

        Complex[] apacheSpectrum = SignalTransformer.fft(padded);
        double[][] handSpectrum = {new double[fftSize], new double[fftSize]};
        System.arraycopy(padded, 0, handSpectrum[0], 0, fftSize);
        HandRolledFft.forward(handSpectrum[0], handSpectrum[1]);
        assertThat(maxSpectrumDiff(handSpectrum[0], handSpectrum[1], apacheSpectrum)).isLessThan(1e-9);
        HandRolledFft.forwardReal(padded, handRe, handIm);
        assertThat(maxSpectrumDiff(handRe, handIm, apacheSpectrum)).isLessThan(1e-9);
    }

    @Test
    void fftRoundTrip_reportsMedianP95MaxAndGc() {
        log.info("FFT round trip: {} calls x {} reps x {} runs = {} samples per path per size",
                CALLS_PER_REP, TIMED_REPS, OUTER_RUNS, CALLS_PER_REP * TIMED_REPS * OUTER_RUNS);

        double applyMedian512 = Double.NaN;
        for (int fftSize : FFT_SIZES) {
            int blockLength = fftSize / 2;
            double[] block = deterministicBlock(blockLength);
            double[][] reused = {new double[fftSize], new double[fftSize]};
            double[][] handComplex = {new double[fftSize], new double[fftSize]};
            double[] handRe = new double[fftSize];
            double[] handIm = new double[fftSize];
            double[] handBack = new double[fftSize];
            double[] handPadded = SignalTransformer.pad(block, fftSize);

            for (int i = 0; i < WARMUP_CALLS; i++) {
                blackhole = roundTripApply(block, fftSize);
                blackhole = roundTripInPlace(block, fftSize);
                blackhole = roundTripInPlaceReused(block, fftSize, reused);
                blackhole = roundTripHandComplex(block, fftSize, handComplex);
                blackhole = roundTripHandReal(handPadded, handRe, handIm, handBack);
            }

            List<Long> applyNanos = new ArrayList<>();
            List<Long> inPlaceNanos = new ArrayList<>();
            List<Long> reusedNanos = new ArrayList<>();
            List<Long> handComplexNanos = new ArrayList<>();
            List<Long> handRealNanos = new ArrayList<>();
            long gcBefore = gcCount();
            for (int run = 0; run < OUTER_RUNS; run++) {
                for (int rep = 0; rep < TIMED_REPS; rep++) {
                    applyNanos.addAll(timeRoundTrips(() -> roundTripApply(block, fftSize)));
                    inPlaceNanos.addAll(timeRoundTrips(() -> roundTripInPlace(block, fftSize)));
                    reusedNanos.addAll(timeRoundTrips(() -> roundTripInPlaceReused(block, fftSize, reused)));
                    handComplexNanos.addAll(timeRoundTrips(() -> roundTripHandComplex(block, fftSize, handComplex)));
                    handRealNanos.addAll(timeRoundTrips(() -> roundTripHandReal(handPadded, handRe, handIm, handBack)));
                }
            }
            long gcDelta = gcCount() - gcBefore;

            double applyMedian = percentileMicros(applyNanos, 50);
            if (fftSize == 512) {
                applyMedian512 = applyMedian;
            }
            log.info("FFT-{} round trip | apply: median {} us | p95 {} us | max {} us | "
                            + "in-place: median {} us | p95 {} us | max {} us | "
                            + "in-place reused: median {} us | p95 {} us | max {} us | "
                            + "hand complex: median {} us | p95 {} us | max {} us | "
                            + "hand real: median {} us | p95 {} us | max {} us | GC {}",
                    fftSize,
                    applyMedian, percentileMicros(applyNanos, 95), percentileMicros(applyNanos, 100),
                    percentileMicros(inPlaceNanos, 50), percentileMicros(inPlaceNanos, 95), percentileMicros(inPlaceNanos, 100),
                    percentileMicros(reusedNanos, 50), percentileMicros(reusedNanos, 95), percentileMicros(reusedNanos, 100),
                    percentileMicros(handComplexNanos, 50), percentileMicros(handComplexNanos, 95), percentileMicros(handComplexNanos, 100),
                    percentileMicros(handRealNanos, 50), percentileMicros(handRealNanos, 95), percentileMicros(handRealNanos, 100),
                    gcDelta);
        }

        // Same-run SoA processBlock median, as the denominator: what share
        // of a block is the FFT round trip?
        Random random = new Random(42L);
        double[] signal = random.doubles(48000, -1.0, 1.0).toArray();
        double[] kernel = random.doubles(8192, -1.0, 1.0).toArray();
        int blocksPerRep = (signal.length + PARTITION_SIZE - 1) / PARTITION_SIZE;
        List<Long> blockNanos = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            StreamingConvolver warmup = new SoAPartitionedConvolver(kernel, PARTITION_SIZE);
            for (int b = 0; b < blocksPerRep; b++) {
                blackhole = warmup.processBlock(blockSlice(signal, b));
            }
            blackhole = warmup.flush();
        }
        for (int run = 0; run < OUTER_RUNS; run++) {
            for (int rep = 0; rep < TIMED_REPS; rep++) {
                StreamingConvolver convolver = new SoAPartitionedConvolver(kernel, PARTITION_SIZE);
                for (int b = 0; b < blocksPerRep; b++) {
                    double[] block = blockSlice(signal, b);
                    long start = System.nanoTime();
                    blackhole = convolver.processBlock(block);
                    blockNanos.add(System.nanoTime() - start);
                }
                blackhole = convolver.flush();
            }
        }
        assertThat(blackhole).isNotNull();
        double blockMedianMicros = percentileMicros(blockNanos, 50);
        log.info("SoA processBlock 48000/8192 ({} samples): median {} us | p95 {} us | max {} us | "
                        + "FFT-512 apply round trip is {}% of a block | deadline {} ms",
                blockNanos.size(),
                blockMedianMicros, percentileMicros(blockNanos, 95), percentileMicros(blockNanos, 100),
                100.0 * applyMedian512 / blockMedianMicros, BLOCK_DEADLINE_MS);
    }

    private double[] roundTripApply(double[] block, int fftSize) {
        Complex[] spectrum = SignalTransformer.fft(SignalTransformer.pad(block, fftSize));
        return SignalTransformer.ifft(spectrum);
    }

    private double[] roundTripInPlace(double[] block, int fftSize) {
        double[][] dataRI = {new double[fftSize], new double[fftSize]};
        System.arraycopy(block, 0, dataRI[0], 0, block.length);
        FORWARD.transformInPlace(dataRI);
        INVERSE.transformInPlace(dataRI);
        return dataRI[0];
    }

    private double[] roundTripInPlaceReused(double[] block, int fftSize, double[][] dataRI) {
        System.arraycopy(block, 0, dataRI[0], 0, block.length);
        Arrays.fill(dataRI[0], block.length, fftSize, 0.0);
        Arrays.fill(dataRI[1], 0.0);
        FORWARD.transformInPlace(dataRI);
        INVERSE.transformInPlace(dataRI);
        return dataRI[0];
    }

    private double[] roundTripHandComplex(double[] block, int fftSize, double[][] dataRI) {
        System.arraycopy(block, 0, dataRI[0], 0, block.length);
        Arrays.fill(dataRI[0], block.length, fftSize, 0.0);
        Arrays.fill(dataRI[1], 0.0);
        HandRolledFft.forward(dataRI[0], dataRI[1]);
        HandRolledFft.inverse(dataRI[0], dataRI[1]);
        return dataRI[0];
    }

    private double[] roundTripHandReal(double[] padded, double[] outRe, double[] outIm, double[] back) {
        HandRolledFft.forwardReal(padded, outRe, outIm);
        HandRolledFft.inverseReal(outRe, outIm, back);
        return back;
    }

    private List<Long> timeRoundTrips(Supplier<double[]> work) {
        List<Long> nanos = new ArrayList<>(CALLS_PER_REP);
        for (int i = 0; i < CALLS_PER_REP; i++) {
            long start = System.nanoTime();
            blackhole = work.get();
            nanos.add(System.nanoTime() - start);
        }
        assertThat(blackhole).isNotNull();
        return nanos;
    }

    private double[] deterministicBlock(int length) {
        return new Random(42L).doubles(length, -1.0, 1.0).toArray();
    }

    private double[] blockSlice(double[] signal, int blockIndex) {
        double[] block = new double[PARTITION_SIZE];
        int copyLength = Math.min(PARTITION_SIZE, signal.length - blockIndex * PARTITION_SIZE);
        if (copyLength > 0) {
            System.arraycopy(signal, blockIndex * PARTITION_SIZE, block, 0, copyLength);
        }
        return block;
    }

    private double maxAbsDiff(double[] a, double[] b) {
        assertThat(a.length).isEqualTo(b.length);
        double max = 0.0;
        for (int i = 0; i < a.length; i++) {
            max = Math.max(max, Math.abs(a[i] - b[i]));
        }
        return max;
    }

    private double maxSpectrumDiff(double[] re, double[] im, Complex[] spectrum) {
        assertThat(re.length).isEqualTo(spectrum.length);
        assertThat(im.length).isEqualTo(spectrum.length);
        double max = 0.0;
        for (int i = 0; i < spectrum.length; i++) {
            max = Math.max(max, Math.abs(re[i] - spectrum[i].getReal()));
            max = Math.max(max, Math.abs(im[i] - spectrum[i].getImaginary()));
        }
        return max;
    }

    private double percentileMicros(List<Long> nanos, double p) {
        List<Long> sorted = new ArrayList<>(nanos);
        Collections.sort(sorted);
        int index = Math.min(sorted.size() - 1, (int) Math.ceil(p / 100.0 * sorted.size()) - 1);
        return sorted.get(Math.max(0, index)) / 1_000.0;
    }

    private long gcCount() {
        long total = 0;
        for (GarbageCollectorMXBean bean : ManagementFactory.getGarbageCollectorMXBeans()) {
            total += bean.getCollectionCount();
        }
        return total;
    }
}
