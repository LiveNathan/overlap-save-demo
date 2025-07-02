package dev.nathanlively.overlap_save_demo;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.apache.commons.lang3.ArrayUtils;

public class VectorApiAdapter implements Convolution {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    @Override
    public double[] with(double[] signal, double[] kernel) {
        SignalTransformer.validate(signal, kernel);

        final double[] paddedSignal = SignalTransformer.padSymmetric(signal, kernel.length - 1);
        final double[] reversedKernel = reverseKernel(kernel);

        return computeVectorizedConvolution(paddedSignal, reversedKernel, signal.length);
    }

    private double[] computeVectorizedConvolution(double[] paddedSignal, double[] reversedKernel, int signalLength) {
        int kernelLength = reversedKernel.length;
        final int resultLength = signalLength + kernelLength - 1;
        final double[] result = new double[resultLength];
        final int padding = kernelLength - 1;

        // Pre-load kernel into vectors for reuse
        final int vectorLength = SPECIES.length();

        for (int outputPos = 0; outputPos < resultLength; outputPos++) {
            result[outputPos] = computeVectorizedWindowConvolution(paddedSignal, reversedKernel,
                    outputPos, padding, kernelLength, vectorLength);
        }

        return result;
    }

    private double computeVectorizedWindowConvolution(double[] paddedSignal, double[] preparedKernel,
                                                      int outputPos, int padding, int kernelLength, int vectorLength) {
        int windowStartPos = outputPos + padding - kernelLength + 1;
        double sum = 0.0;

        for (int i = 0; i < kernelLength; i += vectorLength) {
            int remainingElements = Math.min(vectorLength, kernelLength - i);
            VectorMask<Double> mask = SPECIES.indexInRange(0, remainingElements);

            DoubleVector signalVec = DoubleVector.fromArray(SPECIES, paddedSignal, windowStartPos + i, mask);
            DoubleVector kernelVec = DoubleVector.fromArray(SPECIES, preparedKernel, i, mask);
            DoubleVector product = signalVec.mul(kernelVec, mask);
            sum += product.reduceLanes(VectorOperators.ADD, mask);
        }

        return sum;
    }

    double[] reverseKernel(double[] kernel) {
        final double[] flippedKernel = ArrayUtils.clone(kernel);
        ArrayUtils.reverse(flippedKernel);
        return flippedKernel;
    }
}