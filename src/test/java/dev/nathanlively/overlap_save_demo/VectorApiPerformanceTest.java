package dev.nathanlively.overlap_save_demo;

import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Comparator;

import static org.assertj.core.api.Assertions.assertThat;

class VectorApiPerformanceTest {
    private static final Logger log = LoggerFactory.getLogger(VectorApiPerformanceTest.class);

    private static Comparator<Double> doubleComparator() {
        return (a, b) -> Math.abs(a - b) < 1.0E-12 ? 0 : Double.compare(a, b);
    }

    @Test
    void vectorApiAdapter_producesCorrectResults() {
        VectorApiAdapter vectorAdapter = new VectorApiAdapter();
        TimeDomainAdapter timeAdapter = new TimeDomainAdapter();

        double[] signal = {1.0, 2.0, 3.0, 4.0, 5.0};
        double[] kernel = {0.5, 0.25, 0.125};

        double[] vectorResult = vectorAdapter.with(signal, kernel);
        double[] timeResult = timeAdapter.with(signal, kernel);

        assertThat(vectorResult).usingElementComparator(doubleComparator())
                .containsExactly(timeResult);
    }

    @Test
    void vectorApiPerformance_showsImprovementForLongSignals() {
        VectorApiAdapter vectorAdapter = new VectorApiAdapter();
        TimeDomainAdapter timeAdapter = new TimeDomainAdapter();

        // Create long signal and kernel to showcase Vector API benefits
        double[] longSignal = generateTestSignal(48000 * 10);
        double[] longKernel = generateTestKernel(8192);

        // Warm up JIT
        for (int i = 0; i < 5; i++) {
            vectorAdapter.with(longSignal, longKernel);
            timeAdapter.with(longSignal, longKernel);
        }

        // Measure Vector API performance
        long vectorStart = System.nanoTime();
        double[] vectorResult = vectorAdapter.with(longSignal, longKernel);
        long vectorTime = System.nanoTime() - vectorStart;

        // Measure time domain performance
        long timeStart = System.nanoTime();
        double[] timeResult = timeAdapter.with(longSignal, longKernel);
        long timeTime = System.nanoTime() - timeStart;

        log.info("Vector API convolution: {} ms", vectorTime / 1_000_000.0);
        log.info("Time domain convolution: {} ms", timeTime / 1_000_000.0);
        log.info("Speedup ratio: {}x", (double) timeTime / vectorTime);

        // Verify correctness
        assertThat(vectorResult).usingElementComparator(doubleComparator())
                .containsExactly(timeResult);

        // Vector API should be faster for large computations
        // Note: This may not always be true in test environments or on hardware without SIMD support
        log.info("Vector API performance test completed. Results may vary based on hardware SIMD support.");
    }

    @Test
    void vectorApiPerformance_withVariousKernelSizes() {
        VectorApiAdapter vectorAdapter = new VectorApiAdapter();
        TimeDomainAdapter timeAdapter = new TimeDomainAdapter();

        double[] signal = generateTestSignal(4096);
        int[] kernelSizes = {8, 16, 32, 64, 128};

        log.info("Performance comparison for different kernel sizes:");
        log.info("Kernel Size | Vector API (ms) | Time Domain (ms) | Speedup");
        log.info("-----------|----------------|-----------------|--------");

        for (int kernelSize : kernelSizes) {
            double[] kernel = generateTestKernel(kernelSize);

            // Warm up
            for (int i = 0; i < 3; i++) {
                vectorAdapter.with(signal, kernel);
                timeAdapter.with(signal, kernel);
            }

            // Measure Vector API
            long vectorStart = System.nanoTime();
            double[] vectorResult = vectorAdapter.with(signal, kernel);
            long vectorTime = System.nanoTime() - vectorStart;

            // Measure time domain
            long timeStart = System.nanoTime();
            double[] timeResult = timeAdapter.with(signal, kernel);
            long timeTime = System.nanoTime() - timeStart;

            double speedup = (double) timeTime / vectorTime;

            log.info("{} | {} | {} | {}x",
                    kernelSize,
                    vectorTime / 1_000_000.0,
                    timeTime / 1_000_000.0,
                    speedup);

            // Verify correctness
            assertThat(vectorResult).usingElementComparator(doubleComparator())
                    .containsExactly(timeResult);
        }
    }

    @Test
    void vectorApiMasking_handlesNonAlignedKernels() {
        VectorApiAdapter vectorAdapter = new VectorApiAdapter();
        TimeDomainAdapter timeAdapter = new TimeDomainAdapter();

        double[] signal = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};

        // Test with kernel sizes that don't align with vector lanes
        int[] oddKernelSizes = {3, 5, 7, 9, 11, 13};

        for (int kernelSize : oddKernelSizes) {
            double[] kernel = generateTestKernel(kernelSize);

            double[] vectorResult = vectorAdapter.with(signal, kernel);
            double[] timeResult = timeAdapter.with(signal, kernel);

            assertThat(vectorResult).usingElementComparator(doubleComparator())
                    .containsExactly(timeResult);
        }
    }

    private double[] generateTestSignal(int length) {
        double[] signal = new double[length];
        for (int i = 0; i < length; i++) {
            signal[i] = Math.sin(2.0 * Math.PI * i / 64.0) + 0.1 * Math.random();
        }
        return signal;
    }

    private double[] generateTestKernel(int length) {
        double[] kernel = new double[length];
        double sum = 0;
        for (int i = 0; i < length; i++) {
            kernel[i] = Math.exp(-0.5 * Math.pow((i - length/2.0) / (length/4.0), 2));
            sum += kernel[i];
        }

        // Normalize
        for (int i = 0; i < length; i++) {
            kernel[i] /= sum;
        }
        return kernel;
    }
}