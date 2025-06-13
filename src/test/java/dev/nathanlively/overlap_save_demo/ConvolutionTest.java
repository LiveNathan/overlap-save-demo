package dev.nathanlively.overlap_save_demo;

import com.github.psambit9791.jdsp.io.WAV;
import org.apache.commons.math3.util.MathArrays;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Arrays;
import java.util.Hashtable;

import static org.assertj.core.api.Assertions.assertThat;

class ConvolutionTest {
    private static final Logger log = LoggerFactory.getLogger(ConvolutionTest.class);

    @Test
    void apache() throws Exception {
        Convolution convolution = new ApachePort();
        String filePathKernel = "src/test/resources/LakeMerrittBART.wav";
        String filePathSignal = "src/test/resources/11_Lecture-44k.wav";
        
        // Load kernel (impulse response)
        WAV wavFile = new WAV();
        wavFile.readWAV(filePathKernel);
        @SuppressWarnings("unchecked") Hashtable<String, Long> kernelProperties = wavFile.getProperties();
        log.info("Kernel WAV properties: {}", kernelProperties);
        double[][] echoBridge = wavFile.getData("int");
        // extract just the left side
        int height = echoBridge.length;
        double[] kernel = new double[height];
        for (int i = 0; i < height; i++) {
            kernel[i] = echoBridge[i][0];
        }
        assertThat(kernel.length).isEqualTo(height);
        double kernelSum = Arrays.stream(kernel).sum();
        log.info("Kernel sum: {}", kernelSum);
        double[] normalizedKernel = MathArrays.normalizeArray(kernel, 1.0);
        log.info("Normalized kernel sum: {}", Arrays.stream(normalizedKernel).sum());

        // Load signal
        WAV signalWavFile = new WAV();
        signalWavFile.readWAV(filePathSignal);
        @SuppressWarnings("unchecked") Hashtable<String, Long> signalProperties = signalWavFile.getProperties();
        log.info("Signal WAV properties: {}", signalProperties);
        long sampleRate = signalProperties.get("SampleRate");
        double[][] signalData = signalWavFile.getData("int");
        // extract just the left side
        int signalHeight = signalData.length;
        double[] signal = new double[signalHeight];
        for (int i = 0; i < signalHeight; i++) {
            signal[i] = signalData[i][0];
        }
        assertThat(signal.length).isEqualTo(signalHeight);
        signal = Arrays.copyOf(signal, (int) sampleRate);
        double maxSignal = Arrays.stream(signal).max().orElseThrow();
        log.info("Signal max: {}", maxSignal);
        double[] scaledSignal = MathArrays.scale(1.0 / maxSignal, signal);
        log.info("Scaled signal max: {}", Arrays.stream(scaledSignal).max().orElseThrow());

        // Perform convolution
        double[] actual = convolution.with(scaledSignal, normalizedKernel);

        assertThat(actual).isNotNull();
        assertThat(actual.length).isGreaterThan(0);
        log.info("Convolution result length: {}", actual.length);
        double maxResult = Arrays.stream(actual).max().orElseThrow();
        log.info("Convolution result max before normalization: {}", maxResult);
        actual = MathArrays.scale(1.0 / maxResult, actual);
        double maxActual = Arrays.stream(actual).max().orElseThrow();
        log.info("Convolution result max after normalization: {}", maxActual);

        // Listen to the result
        // WAV class expects a 2D array with samples as rows and channels as columns
        int numChannels = 1;
        int numFrames = actual.length;
        double[][] wavData = new double[numFrames][numChannels];

        for (int i = 0; i < numFrames; i++) {
            wavData[i][0] = actual[i];
        }
        File outputDir = new File("target/test-outputs");
        assertThat(outputDir.mkdirs() || outputDir.exists()).isTrue();

        String filePath = outputDir.getPath() + "/convolution-result.wav";
        WAV wavObj = new WAV();
        wavObj.putData(
                wavData,
                sampleRate,
                "double",
                filePath
        );
        
        log.info("Convolution result saved to convolution-result.wav");
        
        // Verify the output file was created
        assertThat(new File(filePath)).exists();
    }
}