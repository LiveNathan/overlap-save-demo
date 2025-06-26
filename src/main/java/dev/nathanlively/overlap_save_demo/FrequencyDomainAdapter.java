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

        int convolutionLength = signal.length + kernel.length - 1;
        int paddedLength = CommonUtil.nextPowerOfTwo(convolutionLength);

        final double[] paddedSignal = padArray(signal, paddedLength);
        final double[] paddedKernel = padArray(kernel, paddedLength);
        final Complex[] signalTransform = transform(paddedSignal);
        final Complex[] kernelTransform = transform(paddedKernel);

        final Complex[] productTransform = multiplyTransforms(signalTransform, kernelTransform);
        final double[] convolutionResult = inverseTransformRealOnly(productTransform);

        return extractValidPortion(convolutionResult, convolutionLength);
    }

    double[] padArray(double[] array, int targetLength) {
        double[] padded = new double[targetLength];
        System.arraycopy(array, 0, padded, 0, array.length);
        return padded;
    }

    Complex[] transform(double[] signal) {
        FastFourierTransformer fft = new FastFourierTransformer(DftNormalization.STANDARD);
        return fft.transform(signal, TransformType.FORWARD);
    }

    private Complex[] multiplyTransforms(Complex[] transform1, Complex[] transform2) {
        Complex[] result = new Complex[transform1.length];
        for (int i = 0; i < transform1.length; i++) {
            result[i] = transform1[i].multiply(transform2[i]);
        }
        return result;
    }

    double[] inverseTransformRealOnly(Complex[] transform) {
        FastFourierTransformer fft = new FastFourierTransformer(DftNormalization.STANDARD);
        Complex[] result = fft.transform(transform, TransformType.INVERSE);

        double[] realResult = new double[result.length];
        for (int i = 0; i < result.length; i++) {
            realResult[i] = result[i].getReal();
        }
        return realResult;
    }

    private double[] extractValidPortion(double[] paddedResult, int validLength) {
        double[] result = new double[validLength];
        System.arraycopy(paddedResult, 0, result, 0, validLength);
        return result;
    }

    private void validateInputs(double[] signal, double[] kernel) {
        MathUtils.checkNotNull(signal);
        MathUtils.checkNotNull(kernel);

        if (signal.length == 0 || kernel.length == 0) {
            throw new NoDataException();
        }
    }
}