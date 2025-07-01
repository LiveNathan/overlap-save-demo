package dev.nathanlively.overlap_save_demo;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math4.legacy.exception.NoDataException;

import java.util.Objects;

public class VectorApiAdapter implements Convolution {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    @Override
    public double[] with(double[] signal, double[] kernel) {
        validateInputs(signal, kernel);

        final double[] paddedSignal = padSignal(signal, kernel.length);
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

        int i = 0;

        // Process full vectors
        for (; i + vectorLength <= kernelLength; i += vectorLength) {
            DoubleVector signalVec = DoubleVector.fromArray(SPECIES, paddedSignal, windowStartPos + i);
            DoubleVector kernelVec = DoubleVector.fromArray(SPECIES, preparedKernel, i);

            DoubleVector product = signalVec.mul(kernelVec);
            sum += product.reduceLanes(VectorOperators.ADD);
        }

        // Handle remaining elements with masked operations
        if (i < kernelLength) {
            int remainingLength = kernelLength - i;
            VectorMask<Double> mask = SPECIES.indexInRange(0, remainingLength);

            DoubleVector signalVec = DoubleVector.fromArray(SPECIES, paddedSignal, windowStartPos + i, mask);
            DoubleVector kernelVec = DoubleVector.fromArray(SPECIES, preparedKernel, i, mask);

            DoubleVector product = signalVec.mul(kernelVec, mask);
            sum += product.reduceLanes(VectorOperators.ADD, mask);
        }

        return sum;
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

    private void validateInputs(double[] signal, double[] kernel) {
        Objects.requireNonNull(signal, "signal cannot be null");
        Objects.requireNonNull(kernel, "kernel cannot be null");

        if (signal.length == 0 || kernel.length == 0) {
            throw new NoDataException();
        }
    }
}