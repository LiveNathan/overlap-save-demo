package dev.nathanlively.overlap_save_demo;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class ConvolutionTest {

    @Test
    void apache() throws Exception {
        Convolution convolution = new ApachePort();

        double[] actual = convolution.convolve(new double[]{1, 2, 3}, new double[]{1, 2});

        assertThat(actual).isNotNull();
    }
}