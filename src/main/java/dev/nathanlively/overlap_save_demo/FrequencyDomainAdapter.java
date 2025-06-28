package dev.nathanlively.overlap_save_demo;

import org.apache.arrow.memory.util.CommonUtil;
import org.apache.commons.numbers.complex.Complex;

public class FrequencyDomainAdapter implements Convolution {
    @Override
    public double[] with(double[] signal, double[] kernel) {
        SignalTransformer.validateInputs(signal, kernel);

        int convolutionLength = signal.length + kernel.length - 1;
        int paddedLength = CommonUtil.nextPowerOfTwo(convolutionLength);

        final double[] paddedSignal = SignalTransformer.padArray(signal, paddedLength);
        final double[] paddedKernel = SignalTransformer.padArray(kernel, paddedLength);
        final Complex[] signalTransform = SignalTransformer.transform(paddedSignal);
        final Complex[] kernelTransform = SignalTransformer.transform(paddedKernel);

        final Complex[] productTransform = SignalTransformer.multiplyTransforms(signalTransform, kernelTransform);
        final double[] convolutionResult = SignalTransformer.inverseTransformRealOnly(productTransform);

        return extractValidPortion(convolutionResult, convolutionLength);
    }

    private double[] extractValidPortion(double[] paddedResult, int validLength) {
        double[] result = new double[validLength];
        System.arraycopy(paddedResult, 0, result, 0, validLength);
        return result;
    }

}