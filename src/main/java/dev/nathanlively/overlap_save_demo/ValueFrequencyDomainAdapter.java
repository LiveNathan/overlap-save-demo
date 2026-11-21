package dev.nathanlively.overlap_save_demo;

import org.apache.arrow.memory.util.CommonUtil;
import org.apache.commons.numbers.complex.Complex;

/**
 * {@link FrequencyDomainAdapter} with the multiply stage over value records:
 * the FFT results are converted to {@link ComplexValue}[], multiplied with
 * {@link ComplexValue#multiply}, and converted back for the IFFT. The FFT and
 * IFFT themselves are unchanged, so the only difference from the Apache arm
 * is the multiply stage plus its two O(N) conversions.
 */
public class ValueFrequencyDomainAdapter implements Convolution {
    @Override
    public double[] with(double[] signal, double[] kernel) {
        SignalTransformer.validate(signal, kernel);

        int resultLength = signal.length + kernel.length - 1;
        int paddedLength = CommonUtil.nextPowerOfTwo(resultLength);

        final double[] paddedSignal = SignalTransformer.pad(signal, paddedLength);
        final double[] paddedKernel = SignalTransformer.pad(kernel, paddedLength);
        final Complex[] signalTransform = SignalTransformer.fft(paddedSignal);
        final Complex[] kernelTransform = SignalTransformer.fft(paddedKernel);

        final ComplexValue[] signalValue = SignalTransformer.toComplexValue(signalTransform);
        final ComplexValue[] kernelValue = SignalTransformer.toComplexValue(kernelTransform);
        final ComplexValue[] productValue = SignalTransformer.multiplyValue(signalValue, kernelValue);
        final Complex[] productTransform = SignalTransformer.toComplex(productValue);
        final double[] convolutionResult = SignalTransformer.ifft(productTransform);

        return extractValidPortion(convolutionResult, resultLength);
    }

    private double[] extractValidPortion(double[] paddedResult, int validLength) {
        double[] result = new double[validLength];
        System.arraycopy(paddedResult, 0, result, 0, validLength);
        return result;
    }
}
