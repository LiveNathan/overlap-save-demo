package dev.nathanlively.overlap_save_demo;

import org.apache.arrow.memory.util.CommonUtil;
import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.exception.NoDataException;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.TransformType;
import org.apache.commons.math3.util.MathUtils;

public class FrequencyDomainAdapter implements Convolution {
    @Override
    public double[] with(double[] signal, double[] kernel) {
        validateInputs(signal, kernel);

        final Complex[] signalTransform = transform(signal);
        final Complex[] kernelTransform = transform(kernel);

        return computeConvolution(signalTransform, kernelTransform, signal.length);
    }

    Complex[] transform(double[] signal) {
        final double[] paddedSignal = zeroPadToNextPowerOfTwo(signal);
        FastFourierTransformer fft = new FastFourierTransformer(DftNormalization.STANDARD);
        return fft.transform(paddedSignal, TransformType.FORWARD);
    }

    private double[] zeroPadToNextPowerOfTwo(double[] signal) {
        int paddedLength = CommonUtil.nextPowerOfTwo(signal.length);
        double[] paddedSignal = new double[paddedLength];
        System.arraycopy(signal, 0, paddedSignal, 0, signal.length);
        return paddedSignal;
    }

    private double[] computeConvolution(Complex[] paddedSignal, Complex[] reversedKernel, int signalLength) {
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

    private double computeWindowConvolution(Complex[] paddedSignal, Complex[] preparedKernel,
                                            int outputPos, int padding, int kernelLength) {
        int windowStartPos = outputPos + padding - kernelLength + 1;
        double sum = 0;

        for (int i = 0; i < kernelLength; i++) {
//            sum += paddedSignal[windowStartPos + i] * preparedKernel[i];
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
