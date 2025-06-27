package dev.nathanlively.overlap_save_demo;

import com.github.psambit9791.jdsp.io.WAV;
import com.github.psambit9791.wavfile.WavFileException;
import org.apache.commons.math4.legacy.core.MathArrays;
import org.apache.commons.math4.legacy.linear.ArrayRealVector;
import org.apache.commons.numbers.complex.Complex;
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

    // Characterize Apache Commons time domain convolution

    static Stream<Convolution> convolutionImplementations() {
        return Stream.of(new ApacheAdapter(), new TimeDomainAdapter(), new FrequencyDomainAdapter());
    }

    @ParameterizedTest
    @MethodSource("convolutionImplementations")
    void impulseConvolution_returnsIdentity(Convolution convolution) {
        double[] signal = {1};
        double[] kernel = {1};

        double[] actual = convolution.with(signal, kernel);

        assertThat(actual).isEqualTo(kernel);
    }

    @ParameterizedTest
    @MethodSource("convolutionImplementations")
    void twoElementConvolution_computesExpectedValues(Convolution convolution) {
        double[] signal = {1, 0.5};
        double[] kernel = {0.2, 0.1};

        double[] result = convolution.with(signal, kernel);

        assertThat(result.length).isEqualTo(signal.length + kernel.length - 1);
        assertThat(result[0]).isCloseTo(0.2, within(1e-15));
        assertThat(result[1]).isEqualTo(0.2);
        assertThat(result[2]).isEqualTo(0.05);
    }

    @ParameterizedTest
    @MethodSource("convolutionImplementations")
    void convolutionIsCommutative(Convolution convolution) {
        double[] signal = {1, 2, 3};
        double[] kernel = {0.5, 0.25};

        double[] result1 = convolution.with(signal, kernel);
        double[] result2 = convolution.with(kernel, signal);

        assertThat(result1).isEqualTo(result2);
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

    @Test
    void transform_computesFFTForPowerOfTwoSignal() {
        FrequencyDomainAdapter adapter = new FrequencyDomainAdapter();
        double[] signal = {1, 2};

        Complex[] transform = adapter.transform(signal);

        assertThat(transform).hasSize(2);
        // FFT of [1, 2] should be [3+0i, -1+0i]
        assertThat(transform[0].getReal()).isEqualTo(3.0);
        assertThat(transform[0].getImaginary()).isEqualTo(0.0);
        assertThat(transform[1].getReal()).isEqualTo(-1.0);
        assertThat(transform[1].getImaginary()).isEqualTo(0.0);
    }

    private static Comparator<Double> doubleComparator() {
        return (a, b) -> Math.abs(a - b) < 1.0E-15 ? 0 : Double.compare(a, b);
    }

    @Test
    void transformRoundTrip_preservesOriginalSignal() {
        FrequencyDomainAdapter adapter = new FrequencyDomainAdapter();
        double[] original = {1, 2, 3, 4};

        Complex[] transformed = adapter.transform(adapter.padArray(original, 8));
        double[] roundTrip = adapter.inverseTransformRealOnly(transformed);

        assertThat(roundTrip).usingElementComparator(doubleComparator())
                .containsExactly(1, 2, 3, 4, 0, 0, 0, 0);
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