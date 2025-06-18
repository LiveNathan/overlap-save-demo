package dev.nathanlively.overlap_save_demo;

import com.github.psambit9791.jdsp.io.WAV;
import com.github.psambit9791.wavfile.WavFileException;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.util.MathArrays;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Hashtable;

import static org.assertj.core.api.Assertions.assertThat;

class ConvolutionTest {
    private static final Logger log = LoggerFactory.getLogger(ConvolutionTest.class);

    @Test
    void givenSignalImpulseAndKernelImpulse_whenConvolving_thenReturnImpulse() {
        Convolution convolution = new ApacheAdapter();
        double[] signal = {1};
        double[] kernel = {1};

        double[] actual = convolution.with(signal, kernel);

        assertThat(actual).isEqualTo(kernel);
    }

    @Test
    void givenSignalImpulseAndKernelImpulse_whenConvolving_thenReturnImpulse2() {
        Convolution convolution = new ApacheAdapter();
        double[] signal = {1, 0.5};
        double[] kernel = {0.2, 0.1};

        double[] result = convolution.with(signal, kernel);

        assertThat(result.length).isEqualTo(signal.length + kernel.length - 1); // 2 + 2 - 1 = 3
        // result[0] = signal[0] * kernel[0] = 1 * 0.2 = 0.2
        assertThat(result[0]).isEqualTo(0.2);  // 1 * 0.2
        // result[1] = signal[0] * kernel[1] + signal[1] * kernel[0]
        // result[1] = (1 * 0.1) + (0.5 * 0.2) = 0.2
        assertThat(result[1]).isEqualTo(0.2);  // 1 * 0.1 + 0.5 * 0.2
        // result[2] = signal[1] * kernel[1]
        // result[2] = 0.5 * 0.1 = 0.05
        assertThat(result[2]).isEqualTo(0.05); // 0.5 * 0.1
    }

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
        Convolution convolution = new JdspPort();
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