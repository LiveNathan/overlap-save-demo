package dev.nathanlively.overlap_save_demo;

import org.apache.commons.math4.legacy.exception.NoDataException;
import org.apache.commons.math4.transform.FastFourierTransform;
import org.apache.commons.numbers.complex.Complex;

import java.util.Objects;

public class SignalTransformer {
    public static double[] padArray(double[] array, int targetLength) {
        double[] padded = new double[targetLength];
        System.arraycopy(array, 0, padded, 0, array.length);
        return padded;
    }

    public static Complex[] transform(double[] signal) {
        FastFourierTransform fft = new FastFourierTransform(FastFourierTransform.Norm.STD);
        return fft.apply(signal);
    }

    public static double[] inverseTransformRealOnly(Complex[] transform) {
        FastFourierTransform ifft = new FastFourierTransform(FastFourierTransform.Norm.STD, true);
        Complex[] result = ifft.apply(transform);

        double[] realResult = new double[result.length];
        for (int i = 0; i < result.length; i++) {
            realResult[i] = result[i].getReal();
        }
        return realResult;
    }

    public static Complex[] multiplyTransforms(Complex[] transform1, Complex[] transform2) {
        Complex[] result = new Complex[transform1.length];
        for (int i = 0; i < transform1.length; i++) {
            result[i] = transform1[i].multiply(transform2[i]);
        }
        return result;
    }

    public static void validateInputs(double[] signal, double[] kernel) {
        Objects.requireNonNull(signal, "signal cannot be null");
        Objects.requireNonNull(kernel, "kernel cannot be null");

        if (signal.length == 0 || kernel.length == 0) {
            throw new NoDataException();
        }
    }
}
