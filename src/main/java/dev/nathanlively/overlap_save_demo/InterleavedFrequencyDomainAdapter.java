package dev.nathanlively.overlap_save_demo;

import org.apache.arrow.memory.util.CommonUtil;
import org.apache.commons.numbers.complex.Complex;

/**
 * {@link FrequencyDomainAdapter} with the multiply stage over one interleaved
 * flat array ({@code [re, im, re, im, ...]}): the FFT results are converted to
 * interleaved {@code double[]}s, multiplied with no per-element object, and
 * converted back for the IFFT. The FFT and IFFT themselves are unchanged, so
 * the only difference from the Apache arm is the multiply stage plus its two
 * O(N) conversions.
 */
public class InterleavedFrequencyDomainAdapter implements Convolution {
    @Override
    public double[] with(double[] signal, double[] kernel) {
        SignalTransformer.validate(signal, kernel);

        int resultLength = signal.length + kernel.length - 1;
        int paddedLength = CommonUtil.nextPowerOfTwo(resultLength);

        final double[] paddedSignal = SignalTransformer.pad(signal, paddedLength);
        final double[] paddedKernel = SignalTransformer.pad(kernel, paddedLength);
        final Complex[] signalTransform = SignalTransformer.fft(paddedSignal);
        final Complex[] kernelTransform = SignalTransformer.fft(paddedKernel);

        final double[] signalFlat = SignalTransformer.toInterleaved(signalTransform);
        final double[] kernelFlat = SignalTransformer.toInterleaved(kernelTransform);
        final double[] productFlat = SignalTransformer.multiplyInterleaved(signalFlat, kernelFlat);
        final Complex[] productTransform = SignalTransformer.fromInterleaved(productFlat);
        final double[] convolutionResult = SignalTransformer.ifft(productTransform);

        return extractValidPortion(convolutionResult, resultLength);
    }

    private double[] extractValidPortion(double[] paddedResult, int validLength) {
        double[] result = new double[validLength];
        System.arraycopy(paddedResult, 0, result, 0, validLength);
        return result;
    }
}
