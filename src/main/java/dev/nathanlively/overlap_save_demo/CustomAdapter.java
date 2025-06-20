package dev.nathanlively.overlap_save_demo;

import org.apache.commons.math3.exception.NoDataException;
import org.apache.commons.math3.util.MathUtils;

public class CustomAdapter implements Convolution {
    @Override
    public double[] with(double[] signal, double[] kernel) {
        MathUtils.checkNotNull(signal);
        MathUtils.checkNotNull(kernel);

        final int signalLength = signal.length;
        final int kernelLength = kernel.length;

        if (signalLength == 0 || kernelLength == 0) {
            throw new NoDataException();
        }
        final int resultLength = signalLength + kernelLength - 1;
        final double[] result = new double[resultLength];
        final int padding = kernelLength - 1;
        final int paddedLength = signalLength + 2 * padding;
        final double[] paddedSignal = new double[paddedLength];
        System.arraycopy(signal, 0, paddedSignal, padding, signalLength);

        double[] flippedKernel = new double[kernelLength];
        for (int i = 0; i < kernelLength; i++) {
            flippedKernel[i] = kernel[kernelLength - 1 - i];
        }

        for (int resultIndex = 0; resultIndex < resultLength; resultIndex++) {
            double sum = 0;
            int startSignalIndex = resultIndex - kernelLength + 1 + padding;

            for (int i = 0; i < kernelLength; i++) {
                sum += paddedSignal[startSignalIndex + i] * flippedKernel[i];
            }
            result[resultIndex] = sum;
        }
        return result;
    }
}
