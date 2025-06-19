package dev.nathanlively.overlap_save_demo;

public class JdspAdapter implements Convolution {
    @Override
    public double[] with(double[] signal, double[] kernel) {
        com.github.psambit9791.jdsp.signal.Convolution con1 = new com.github.psambit9791.jdsp.signal.Convolution(signal, kernel);
        return con1.convolve("same");
    }
}
