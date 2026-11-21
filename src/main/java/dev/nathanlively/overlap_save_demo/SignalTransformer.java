package dev.nathanlively.overlap_save_demo;

import org.apache.commons.math4.legacy.exception.NoDataException;
import org.apache.commons.math4.transform.FastFourierTransform;
import org.apache.commons.numbers.complex.Complex;

import java.util.Objects;

public class SignalTransformer {

    // Cache frequently used FFT instances for better performance
    private static final ThreadLocal<FastFourierTransform> FORWARD_FFT =
            ThreadLocal.withInitial(() -> new FastFourierTransform(FastFourierTransform.Norm.STD));

    private static final ThreadLocal<FastFourierTransform> INVERSE_FFT =
            ThreadLocal.withInitial(() -> new FastFourierTransform(FastFourierTransform.Norm.STD, true));

    public static double[] pad(double[] array, int startPaddingAmount, int endPaddingAmount) {
        if (startPaddingAmount < 0 || endPaddingAmount < 0) {
            throw new IllegalArgumentException("Padding amounts must be non-negative");
        }

        if (startPaddingAmount == 0 && endPaddingAmount == 0) {
            return array;
        }

        double[] padded = new double[array.length + startPaddingAmount + endPaddingAmount];
        System.arraycopy(array, 0, padded, startPaddingAmount, array.length);
        return padded;
    }

    public static double[] padSymmetric(double[] array, int padding) {
        return pad(array, padding, padding);
    }

    public static double[] pad(double[] array, int targetLength) {
        if (array.length >= targetLength) {
            return array;
        }

        double[] padded = new double[targetLength];
        System.arraycopy(array, 0, padded, 0, array.length);
        return padded;
    }

    public static Complex[] fft(double[] signal) {
        return FORWARD_FFT.get().apply(signal);
    }

    public static double[] ifft(Complex[] transform) {
        Complex[] result = INVERSE_FFT.get().apply(transform);

        double[] realResult = new double[result.length];
        for (int i = 0; i < result.length; i++) {
            realResult[i] = result[i].getReal();
        }
        return realResult;
    }

    public static Complex[] multiply(Complex[] transform1, Complex[] transform2) {
        if (transform1.length != transform2.length) {
            throw new IllegalArgumentException("Transform arrays must have same length");
        }

        Complex[] result = new Complex[transform1.length];
        for (int i = 0; i < transform1.length; i++) {
            result[i] = transform1[i].multiply(transform2[i]);
        }
        return result;
    }

    /**
     * Same element-wise product, over value records.
     */
    static ComplexValue[] multiplyValue(ComplexValue[] transform1, ComplexValue[] transform2) {
        if (transform1.length != transform2.length) {
            throw new IllegalArgumentException("Transform arrays must have same length");
        }

        ComplexValue[] result = new ComplexValue[transform1.length];
        for (int i = 0; i < transform1.length; i++) {
            result[i] = transform1[i].multiply(transform2[i]);
        }
        return result;
    }

    /**
     * Same element-wise product, over one interleaved flat array
     * ({@code [re, im, re, im, ...]}). No object is created per element.
     */
    public static double[] multiplyInterleaved(double[] transform1, double[] transform2) {
        if (transform1.length != transform2.length) {
            throw new IllegalArgumentException("Transform arrays must have same length");
        }

        double[] result = new double[transform1.length];
        for (int i = 0; i < transform1.length; i += 2) {
            double aRe = transform1[i];
            double aIm = transform1[i + 1];
            double bRe = transform2[i];
            double bIm = transform2[i + 1];
            result[i] = aRe * bRe - aIm * bIm;
            result[i + 1] = aRe * bIm + aIm * bRe;
        }
        return result;
    }

    // Boundary conversions between the FFT's Complex[] and the multiply
    // stage's representations. Each is O(N), the same order as the multiply
    // itself — whether they pay for themselves is what the end-to-end test
    // measures, so they live here next to the multiply they serve.

     static ComplexValue[] toComplexValue(Complex[] transform) {
        ComplexValue[] result = new ComplexValue[transform.length];
        for (int i = 0; i < transform.length; i++) {
            result[i] = new ComplexValue(transform[i].getReal(), transform[i].getImaginary());
        }
        return result;
    }

     static Complex[] toComplex(ComplexValue[] transform) {
        Complex[] result = new Complex[transform.length];
        for (int i = 0; i < transform.length; i++) {
            result[i] = Complex.ofCartesian(transform[i].real(), transform[i].imaginary());
        }
        return result;
    }

    public static double[] toInterleaved(Complex[] transform) {
        double[] result = new double[2 * transform.length];
        for (int i = 0; i < transform.length; i++) {
            result[2 * i] = transform[i].getReal();
            result[2 * i + 1] = transform[i].getImaginary();
        }
        return result;
    }

    public static Complex[] fromInterleaved(double[] transform) {
        Complex[] result = new Complex[transform.length / 2];
        for (int i = 0; i < result.length; i++) {
            result[i] = Complex.ofCartesian(transform[2 * i], transform[2 * i + 1]);
        }
        return result;
    }

    public static void validate(double[] signal, double[] kernel) {
        Objects.requireNonNull(signal, "signal cannot be null");
        Objects.requireNonNull(kernel, "kernel cannot be null");

        if (signal.length == 0 || kernel.length == 0) {
            throw new NoDataException();
        }
    }
}