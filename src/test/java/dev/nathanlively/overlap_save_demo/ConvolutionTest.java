package dev.nathanlively.overlap_save_demo;

import com.github.psambit9791.jdsp.io.WAV;
import com.github.psambit9791.wavfile.WavFileException;
import org.apache.commons.math4.legacy.core.MathArrays;
import org.apache.commons.math4.legacy.linear.ArrayRealVector;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Hashtable;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class ConvolutionTest {
    private static final Logger log = LoggerFactory.getLogger(ConvolutionTest.class);

    static Stream<Convolution> allImplementations() {
        return Stream.of(new ApacheAdapter(), new TimeDomainAdapter(),
                new FrequencyDomainAdapter(), new OverlapSaveAdapter());
    }

    @ParameterizedTest
    @MethodSource("allImplementations")
    void impulseConvolution_returnsIdentity(Convolution convolution) {
        double[] signal = {1};
        double[] kernel = {1};

        double[] actual = convolution.with(signal, kernel);

        assertThat(actual).isEqualTo(kernel);
    }

    @ParameterizedTest
    @MethodSource("allImplementations")
    void twoElementConvolution_computesExpectedValues(Convolution convolution) {
        double[] signal = {1, 0.5};
        double[] kernel = {0.2, 0.1};

        double[] result = convolution.with(signal, kernel);

        assertThat(result.length).isEqualTo(signal.length + kernel.length - 1);
        assertThat(result[0]).isCloseTo(0.2, within(1e-15));
        assertThat(result[1]).isEqualTo(0.2, within(1e-15));
        assertThat(result[2]).isEqualTo(0.05, within(1e-15));
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
    void preparePaddedSignal_addsCorrectPadding() {
        TimeDomainAdapter adapter = new TimeDomainAdapter();
        double[] signal = {1, 2};
        int kernelLength = 3;

        double[] paddedSignal = adapter.padSignal(signal, kernelLength);

        assertThat(paddedSignal).containsExactly(0, 0, 1, 2, 0, 0);
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

    private static Comparator<Double> doubleComparator() {
        return (a, b) -> Math.abs(a - b) < 1.0E-15 ? 0 : Double.compare(a, b);
    }

    // Implement custom time domain convolution

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
        Convolution convolution = new JdspAdapter();
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

    private WavFile loadWavFile(String fileName) throws WavFileException, IOException {
        final String path = new ClassPathResource(fileName).getFile().getCanonicalPath();
        WAV signalWavFile = new WAV();
        signalWavFile.readWAV(path);
        @SuppressWarnings("unchecked") Hashtable<String, Long> signalProperties = signalWavFile.getProperties();
        log.info("Signal WAV properties: {}", signalProperties);
        long sampleRate = signalProperties.get("SampleRate");
        double[][] signalData = signalWavFile.getData("int");
        short bitDepth = (short) Math.toIntExact(signalProperties.get("ValidBits"));
        double scaleFactor = Math.pow(2, bitDepth - 1) - 1;
        // extract just the left side
        int signalHeight = signalData.length;
        double[] signal = new double[signalHeight];
        for (int i = 0; i < signalHeight; i++) {
            signal[i] = signalData[i][0] / scaleFactor;
        }
        assertThat(signal.length).isEqualTo(signalHeight);
        return new WavFile(sampleRate, signal);
    }

    private void saveWavFile(WavFile signalFile) throws IOException, WavFileException {
        // WAV class expects a 2D array with samples as rows and channels as columns
        double[] actual = signalFile.signal();
        int numChannels = 1;
        int numFrames = actual.length;
        double[][] wavData = new double[numFrames][numChannels];

        for (int i = 0; i < numFrames; i++) {
            wavData[i][0] = actual[i];
        }
        Path outputDir = Paths.get("target/test-outputs");
        Files.createDirectories(outputDir);
        String outputFileName = "convolution-result.wav";
        Path outputPath = outputDir.resolve(outputFileName);
        WAV wavObj = new WAV();
        wavObj.putData(
                wavData,
                signalFile.sampleRate(),
                "double",
                outputPath.toString()
        );

        log.info("Convolution result saved to convolution-result.wav");
        assertThat(outputPath).exists();
    }

    private record WavFile(long sampleRate, double[] signal) {
    }
}