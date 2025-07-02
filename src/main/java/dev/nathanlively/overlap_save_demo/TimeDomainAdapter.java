package dev.nathanlively.overlap_save_demo;

import org.apache.commons.lang3.ArrayUtils;

public class TimeDomainAdapter implements Convolution {
    @Override
    public double[] with(double[] signal, double[] kernel) {
        SignalTransformer.validate(signal, kernel);

        final double[] paddedSignal = SignalTransformer.padSymmetric(signal, kernel.length - 1);
        final double[] reversedKernel = reverseKernel(kernel);

        return computeConvolution(paddedSignal, reversedKernel, signal.length);
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

}
