package dev.nathanlively.overlap_save_demo;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.exception.NoDataException;
import org.apache.commons.math3.util.MathUtils;

public class TimeDomainAdapter implements Convolution {
    @Override
    public double[] with(double[] signal, double[] kernel) {
        validateInputs(signal, kernel);

        final double[] paddedSignal = padSignal(signal, kernel.length);
        final double[] reversedKernel = reverseKernel(kernel);

        return computeConvolution(paddedSignal, reversedKernel, signal.length);
    }

    double[] padSignal(double[] signal, int kernelLength) {
        final int padding = kernelLength - 1;
        final int paddedLength = signal.length + 2 * padding;
        final double[] paddedSignal = new double[paddedLength];
        System.arraycopy(signal, 0, paddedSignal, padding, signal.length);
        return paddedSignal;
    }

    double[] reverseKernel(double[] kernel) {
        final double[] flippedKernel = ArrayUtils.clone(kernel);
        ArrayUtils.reverse(flippedKernel);
        return flippedKernel;
    }

    private double[] computeConvolution(double[] paddedSignal, double[] reversedKernel, int signalLength) {
        int kernelLength = reversedKernel.length;
        final int resultLength = signalLength + kernelLength - 1;
        final double[] result = new double[resultLength];
        final int padding = kernelLength - 1;

        for (int outputPos = 0; outputPos < resultLength; outputPos++) {
            result[outputPos] = computeWindowConvolution(paddedSignal, reversedKernel,
                    outputPos, padding, kernelLength);
        }

        return result;
    }

    private double computeWindowConvolution(double[] paddedSignal, double[] preparedKernel,
                                            int outputPos, int padding, int kernelLength) {
        int windowStartPos = outputPos + padding - kernelLength + 1;
        double sum = 0;

        for (int i = 0; i < kernelLength; i++) {
            sum += paddedSignal[windowStartPos + i] * preparedKernel[i];
        }

        return sum;
    }

    private void validateInputs(double[] signal, double[] kernel) {
        MathUtils.checkNotNull(signal);
        MathUtils.checkNotNull(kernel);

        if (signal.length == 0 || kernel.length == 0) {
            throw new NoDataException();
        }
    }
}
