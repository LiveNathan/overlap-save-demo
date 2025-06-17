package dev.nathanlively.overlap_save_demo;

import com.github.psambit9791.jdsp.io.WAV;
import com.github.psambit9791.wavfile.WavFileException;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.util.MathArrays;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
        double[] impulse = {1};

        double[] actual = convolution.with(impulse, impulse);

        assertThat(actual).isEqualTo(impulse);
    }

    @Test
    void apache() throws Exception {
        String filePathKernel = "src/test/resources/LakeMerrittBART.wav";
        String filePathSignal = "src/test/resources/11_Lecture-44k.wav";

        // Load kernel
        WavFile kernelAudio = loadWavFile(filePathKernel);
        final double[] signalValues = kernelAudio.signal();
        double kernelSum = Arrays.stream(signalValues).map(Math::abs).sum();
        log.info("Kernel sum: {}", kernelSum);
        double[] normalizedKernel = new ArrayRealVector(signalValues).unitVector().toArray();
        log.info("Normalized kernel sum: {}", Arrays.stream(normalizedKernel).map(Math::abs).sum());

        // Load signal
        final WavFile signalFile = loadWavFile(filePathSignal);
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
        saveWavFile(actual, signalFile);
    }

    private WavFile loadWavFile(String filePathSignal) throws WavFileException, IOException {
        WAV signalWavFile = new WAV();
        signalWavFile.readWAV(filePathSignal);
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

    private void saveWavFile(double[] actual, WavFile signalFile) throws IOException, WavFileException {
        // WAV class expects a 2D array with samples as rows and channels as columns
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