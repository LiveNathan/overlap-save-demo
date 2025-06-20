package dev.nathanlively.overlap_save_demo;

import org.apache.commons.math3.exception.NoDataException;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.MathUtils;

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

        for (int resultIndex = 0; resultIndex < resultLength; resultIndex++) {
            double sum = 0;
            int kernelIndex = FastMath.max(0, resultIndex + 1 - signalLength);
            int signalIndex = resultIndex - kernelIndex;
            while (kernelIndex < kernelLength && signalIndex >= 0) {
                sum += signal[signalIndex--] * kernel[kernelIndex++];
            }
            result[resultIndex] = sum;
        }

        return result;
    }
}
