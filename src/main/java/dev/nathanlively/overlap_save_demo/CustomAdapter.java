package dev.nathanlively.overlap_save_demo;

import org.apache.commons.math3.exception.NoDataException;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.MathUtils;

public class CustomAdapter implements Convolution {
    @Override
    public double[] with(double[] x, double[] h) {
        MathUtils.checkNotNull(x);
        MathUtils.checkNotNull(h);

        final int xLen = x.length;
        final int hLen = h.length;

        if (xLen == 0 || hLen == 0) {
            throw new NoDataException();
        }

        // initialize the output array
        final int totalLength = xLen + hLen - 1;
        final double[] y = new double[totalLength];

        // straightforward implementation of the convolution sum
        for (int n = 0; n < totalLength; n++) {
            double yn = 0;
            int k = FastMath.max(0, n + 1 - xLen);
            int j = n - k;
            while (k < hLen && j >= 0) {
                yn += x[j--] * h[k++];
            }
            y[n] = yn;
        }

        return y;
    }
}
