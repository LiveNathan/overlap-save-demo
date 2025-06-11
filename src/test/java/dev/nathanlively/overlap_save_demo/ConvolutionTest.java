package dev.nathanlively.overlap_save_demo;

import com.github.psambit9791.jdsp.io.WAV;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Hashtable;

import static org.assertj.core.api.Assertions.*;

class ConvolutionTest {
    private static final Logger log = LoggerFactory.getLogger(ConvolutionTest.class);

    @Test
    void apache() throws Exception {
        Convolution convolution = new ApachePort();
        String filePath = "src/test/resources/EchoBridge.wav";
        WAV wavFile = new WAV();
        wavFile.readWAV(filePath);
        @SuppressWarnings("unchecked") Hashtable<String, Long> properties = wavFile.getProperties();
        log.debug("WAV properties: {}", properties);
        double[][] echoBridge = wavFile.getData("int");
        int height = echoBridge.length;
        double[] kernel = new double[height];
        for (int i = 0; i < height; i++) {
            kernel[i] = echoBridge[i][0];
        }
        assertThat(kernel.length).isEqualTo(height);

        double[] actual = convolution.convolve(new double[]{1, 2, 3}, kernel);

        assertThat(actual).isNotNull();
    }
}