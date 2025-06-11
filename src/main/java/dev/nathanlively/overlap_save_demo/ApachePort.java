package dev.nathanlively.overlap_save_demo;

import org.apache.commons.math3.util.MathArrays;

public class ApachePort implements Convolution {
    @Override
    public double[] convolve(double[] signal, double[] kernel) {
        return MathArrays.convolve(signal, kernel);
    }
}
