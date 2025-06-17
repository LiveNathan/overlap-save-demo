package dev.nathanlively.overlap_save_demo;

import org.apache.commons.math3.util.MathArrays;

public class ApacheAdapter implements Convolution {
    @Override
    public double[] with(double[] signal, double[] kernel) {
        return MathArrays.convolve(signal, kernel);
    }
}
