package dev.nathanlively.overlap_save_demo;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.exception.NoDataException;
import org.apache.commons.math3.util.MathUtils;

import java.util.Arrays;

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

        // Step 1: Pad the signal with zeros on both sides
        final int padding = kernelLength - 1;
        final int paddedLength = signalLength + 2 * padding;
        final double[] paddedSignal = new double[paddedLength];
        System.arraycopy(signal, 0, paddedSignal, padding, signalLength);

        // Step 2: Flip the kernel for convolution (vs. correlation)
        final double[] flippedKernel = ArrayUtils.clone(kernel);
        ArrayUtils.reverse(flippedKernel);

        // Step 3: Slide the flipped kernel over the padded signal
        for (int outputPos = 0; outputPos < resultLength; outputPos++) {
            int windowStartPos = outputPos + padding - kernelLength + 1;

            // Extract the current window from the padded signal
            double[] signalWindow = Arrays.copyOfRange(paddedSignal, windowStartPos, windowStartPos + kernelLength);

            // Compute dot product of window and flipped kernel
            double sum = 0;
            for (int i = 0; i < kernelLength; i++) {
                sum += signalWindow[i] * flippedKernel[i];
            }
            result[outputPos] = sum;
        }

        return result;
    }
}
