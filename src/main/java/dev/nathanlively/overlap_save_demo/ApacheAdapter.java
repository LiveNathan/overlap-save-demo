package dev.nathanlively.overlap_save_demo;

import org.apache.commons.math4.legacy.core.MathArrays;

public class ApacheAdapter implements Convolution {
    @Override
    public double[] with(double[] signal, double[] kernel) {
        return MathArrays.convolve(signal, kernel);
    }
}
