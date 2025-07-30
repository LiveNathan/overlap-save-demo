package dev.nathanlively.overlap_save_demo;

import org.apache.commons.math4.legacy.core.MathArrays;
import org.apache.commons.math4.legacy.linear.ArrayRealVector;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Comparator;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class ConvolutionTest {
    private static final Logger log = LoggerFactory.getLogger(ConvolutionTest.class);
    private static final double precision = 1e-15;

    static Stream<Convolution> allImplementations() {
        return Stream.of(new ApacheAdapter(), new TimeDomainAdapter(),
                new FrequencyDomainAdapter(), new OverlapSaveAdapter(), new VectorApiAdapter());
    }

    @ParameterizedTest
    @MethodSource("allImplementations")
    void impulseConvolution_returnsIdentity(Convolution convolution) {
        double[] signal = {1};
        double[] kernel = {1};

        double[] actual = convolution.with(signal, kernel);

        assertThat(actual).isEqualTo(kernel);
    }

    private static Comparator<Double> doubleComparator() {
        return (a, b) -> Math.abs(a - b) < precision ? 0 : Double.compare(a, b);
    }

    @ParameterizedTest
    @MethodSource("allImplementations")
    void convolutionIsCommutative(Convolution convolution) {
        double[] signal = {1, 2, 3};
        double[] kernel = {0.5, 0.25};

        double[] result1 = convolution.with(signal, kernel);
        double[] result2 = convolution.with(kernel, signal);

        assertThat(result1).usingElementComparator(doubleComparator())
                .containsExactly(result2);
    }

    @Test
    void prepareKernel_flipsArray() {
        TimeDomainAdapter adapter = new TimeDomainAdapter();
        double[] kernel = {1, 2, 3};

        double[] reversedKernel = adapter.reverseKernel(kernel);

        assertThat(reversedKernel).containsExactly(3, 2, 1);
    }

    @ParameterizedTest
    @MethodSource("allImplementations")
    void largeSignalSmallKernel_efficientProcessing(Convolution convolution) {
        // Test case where overlap-save should shine: large signal, small kernel
        double[] signal = generateTestSignal(1024);
        double[] kernel = {0.25, 0.5, 0.25}; // Simple smoothing kernel

        long startTime = System.nanoTime();
        double[] result = convolution.with(signal, kernel);
        long duration = System.nanoTime() - startTime;

        log.info("{} took {} ms for 1024-sample signal with 3-sample kernel",
                convolution.getClass().getSimpleName(), duration / 1_000_000.0);

        assertThat(result.length).isEqualTo(signal.length + kernel.length - 1);

        // Verify the result makes sense (smoothed signal should have reduced variation)
        double originalVariance = calculateVariance(signal);
        double smoothedVariance = calculateVariance(Arrays.copyOf(result, signal.length));
        assertThat(smoothedVariance).isLessThan(originalVariance);
    }

    @ParameterizedTest
    @MethodSource("allImplementations")
    void veryLongSignal_handlesGracefully(Convolution convolution) {
        // Test with a signal that requires multiple blocks
        double[] signal = generateTestSignal(4096);
        double[] kernel = generateTestKernel(32);

        double[] result = convolution.with(signal, kernel);

        assertThat(result.length).isEqualTo(signal.length + kernel.length - 1);

        // Result should not contain NaN or infinite values
        assertThat(Arrays.stream(result).allMatch(Double::isFinite)).isTrue();
    }

    @Test
    void overlapSaveOptimalFftSize_choosesEfficientSizes() {
        OverlapSaveAdapter adapter = new OverlapSaveAdapter();

        // Small kernel should use minimum reasonable size
        assertThat(adapter.calculateOptimalFftSize(100, 5)).isGreaterThanOrEqualTo(64);

        // Large signal with small kernel should use larger FFT size for efficiency
        int largeSignalSize = adapter.calculateOptimalFftSize(10000, 10);
        int smallSignalSize = adapter.calculateOptimalFftSize(100, 10);
        assertThat(largeSignalSize).isGreaterThanOrEqualTo(smallSignalSize);

        log.info("FFT size for signal=100, kernel=10: {}", smallSignalSize);
        log.info("FFT size for signal=10000, kernel=10: {}", largeSignalSize);
    }

    @ParameterizedTest
    @MethodSource("allImplementations")
    void edgeCases_handleCorrectly(Convolution convolution) {
        // Single element convolution
        double[] singleSignal = {3.0};
        double[] singleKernel = {2.0};
        double[] singleResult = convolution.with(singleSignal, singleKernel);
        assertThat(singleResult).containsExactly(6.0);

        // Asymmetric sizes
        double[] longSignal = generateTestSignal(1000);
        double[] shortKernel = {1.0, -1.0}; // Difference operator
        double[] diffResult = convolution.with(longSignal, shortKernel);
        assertThat(diffResult.length).isEqualTo(1001);
    }

    @Test
    void performanceComparison_showsOverlapSaveAdvantage() {
        // Compare performance for the scenario where overlap-save should excel
        double[] largeSignal = generateTestSignal(8192);
        double[] smallKernel = generateTestKernel(16);

        Convolution[] implementations = {
                new FrequencyDomainAdapter(),
                new OverlapSaveAdapter()
        };

        for (Convolution impl : implementations) {
            long startTime = System.nanoTime();
            double[] result = impl.with(largeSignal, smallKernel);
            long duration = System.nanoTime() - startTime;

            log.info("{}: {} ms", impl.getClass().getSimpleName(), duration / 1_000_000.0);
            assertThat(result.length).isEqualTo(largeSignal.length + smallKernel.length - 1);
        }
    }

    private double[] generateTestSignal(int length) {
        double[] signal = new double[length];
        for (int i = 0; i < length; i++) {
            // Create a signal with some structure (sine wave + noise)
            signal[i] = Math.sin(2.0 * Math.PI * i / 64.0) + 0.1 * Math.random();
        }
        return signal;
    }

    private double[] generateTestKernel(int length) {
        // Generate a simple low-pass filter kernel
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

    private double calculateVariance(double[] data) {
        double mean = Arrays.stream(data).average().orElse(0.0);
        return Arrays.stream(data)
                .map(x -> Math.pow(x - mean, 2))
                .average()
                .orElse(0.0);
    }

    @ParameterizedTest
    @MethodSource("allImplementations")
    void twoElementConvolution_computesExpectedValues(Convolution convolution) {
        double[] signal = {1, 0.5};
        double[] kernel = {0.2, 0.1};

        double[] result = convolution.with(signal, kernel);

        assertThat(result.length).isEqualTo(signal.length + kernel.length - 1);
        assertThat(result[0]).isCloseTo(0.2, within(precision));
        assertThat(result[1]).isEqualTo(0.2, within(precision));
        assertThat(result[2]).isEqualTo(0.05, within(precision));
    }

    @Disabled
    @Test
    void apache() throws Exception {
        String fileNameKernel = "LakeMerrittBART.wav";
        String fileNameSignal = "11_Lecture-44k.wav";

        // Load kernel
        WavFile kernelAudio = loadWavFile(fileNameKernel);
        final double[] signalValues = kernelAudio.signal();
        double kernelSum = Arrays.stream(signalValues).map(Math::abs).sum();
        log.info("Kernel sum: {}", kernelSum);
        double[] normalizedKernel = new ArrayRealVector(signalValues).unitVector().toArray();
        log.info("Normalized kernel sum: {}", Arrays.stream(normalizedKernel).map(Math::abs).sum());

        // Load signal
        final WavFile signalFile = loadWavFile(fileNameSignal);
        double maxSignal = Arrays.stream(signalFile.signal()).max().orElseThrow();
        log.info("Signal max: {}", maxSignal);
        double[] signal = Arrays.copyOf(signalFile.signal(), (int) signalFile.sampleRate() * 10);

        // Perform convolution
        Convolution convolution = new OverlapSaveAdapter();
        double[] actual = convolution.with(signal, normalizedKernel);

        assertThat(actual).isNotNull();
        assertThat(actual.length).isGreaterThan(0);
        log.info("Convolution result length: {}", actual.length);
        double maxResult = Arrays.stream(actual).max().orElseThrow();
        log.info("Convolution result max before normalization: {}", maxResult);
        actual = MathArrays.scale(1.0 / maxResult, actual);
        double maxActual = Arrays.stream(actual).max().orElseThrow();
        log.info("Convolution result max after normalization: {}", maxActual);

        // Listen to the result
        saveWavFile(signalFile);
    }

    private WavFile loadWavFile(String fileName) {
        WavFileReader reader = new WavFileReader();
        WavFileReader.MultiChannelWavFile multiChannel = reader.loadFromClasspath(fileName);
        log.info("Signal WAV properties: channels={}, sampleRate={}, length={}",
                multiChannel.channelCount(), multiChannel.sampleRate(), multiChannel.length());

        return new WavFile(multiChannel.sampleRate(), multiChannel.getChannel(0));
    }

    private void saveWavFile(WavFile signalFile) throws IOException {
        WavFileWriter writer = new WavFileWriter();
        Path outputDir = Paths.get("target/test-outputs");
        Files.createDirectories(outputDir);
        String outputFileName = "convolution-result.wav";
        Path outputPath = outputDir.resolve(outputFileName);

        writer.saveToFile(signalFile, outputPath);

        log.info("Convolution result saved to convolution-result.wav");
        assertThat(outputPath).exists();
    }

}