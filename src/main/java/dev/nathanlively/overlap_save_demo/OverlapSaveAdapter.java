package dev.nathanlively.overlap_save_demo;

import org.apache.arrow.memory.util.CommonUtil;
import org.apache.commons.math4.legacy.exception.NoDataException;
import org.apache.commons.math4.transform.FastFourierTransform;
import org.apache.commons.numbers.complex.Complex;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class OverlapSaveAdapter implements Convolution {
    @Override
    public double[] with(double[] signal, double[] kernel) {
        validateInputs(signal, kernel);

        return overlapSaveConvolution(signal, kernel);
    }

    private double[] overlapSaveConvolution(double[] signal, double[] kernel) {
        int kernelLength = kernel.length;
        int chunkSize = calculateOptimalChunkSize(kernelLength);
        int overlap = kernelLength - 1;
        int validOutputPerChunk = chunkSize - overlap;

        // Pre-compute kernel FFT (reused for all chunks)
        double[] paddedKernel = padArray(kernel, chunkSize);
        Complex[] kernelTransform = transform(paddedKernel);

        List<double[]> results = new ArrayList<>();
        int signalPosition = 0;

        while (signalPosition < signal.length) {
            // Extract chunk with overlap from previous chunk
            double[] chunk = extractChunk(signal, signalPosition, chunkSize, overlap);

            // Convolve this chunk
            Complex[] chunkTransform = transform(chunk);
            Complex[] productTransform = multiplyTransforms(chunkTransform, kernelTransform);
            double[] chunkResult = inverseTransformRealOnly(productTransform);

            // Keep only the valid portion (discard first 'overlap' samples)
            double[] validPortion = extractValidPortionFromChunk(chunkResult, overlap, validOutputPerChunk);
            results.add(validPortion);

            // Move to next chunk position
            signalPosition += validOutputPerChunk;
        }

        // Concatenate all valid portions
        return concatenateResults(results, signal.length + kernel.length - 1);
    }

    private double[] extractChunk(double[] signal, int startPos, int chunkSize, int overlap) {
        double[] chunk = new double[chunkSize];

        // Calculate where to start reading from signal (accounting for overlap)
        int signalStart = Math.max(0, startPos - overlap);
        int chunkStart = Math.max(0, overlap - startPos);

        // Copy available signal data into chunk
        int copyLength = Math.min(signal.length - signalStart, chunkSize - chunkStart);
        if (copyLength > 0) {
            System.arraycopy(signal, signalStart, chunk, chunkStart, copyLength);
        }

        return chunk;
    }

    private double[] extractValidPortionFromChunk(double[] chunkResult, int overlap, int validLength) {
        int actualValidLength = Math.min(validLength, chunkResult.length - overlap);
        double[] validPortion = new double[actualValidLength];

        if (actualValidLength > 0) {
            System.arraycopy(chunkResult, overlap, validPortion, 0, actualValidLength);
        }

        return validPortion;
    }

    private double[] concatenateResults(List<double[]> results, int totalExpectedLength) {
        double[] finalResult = new double[totalExpectedLength];
        int position = 0;

        for (double[] chunk : results) {
            int copyLength = Math.min(chunk.length, totalExpectedLength - position);
            if (copyLength > 0) {
                System.arraycopy(chunk, 0, finalResult, position, copyLength);
                position += copyLength;
            }
        }

        return finalResult;
    }

    int calculateOptimalChunkSize(int kernelLength) {
        // Use 8x kernel length as heuristic, then round up to next power of 2
        int targetSize = kernelLength * 8;
        return CommonUtil.nextPowerOfTwo(targetSize);
    }

    // Utility methods (same as FrequencyDomainAdapter)
    double[] padArray(double[] array, int targetLength) {
        double[] padded = new double[targetLength];
        System.arraycopy(array, 0, padded, 0, array.length);
        return padded;
    }

    Complex[] transform(double[] signal) {
        FastFourierTransform fft = new FastFourierTransform(FastFourierTransform.Norm.STD);
        return fft.apply(signal);
    }

    private Complex[] multiplyTransforms(Complex[] transform1, Complex[] transform2) {
        Complex[] result = new Complex[transform1.length];
        for (int i = 0; i < transform1.length; i++) {
            result[i] = transform1[i].multiply(transform2[i]);
        }
        return result;
    }

    double[] inverseTransformRealOnly(Complex[] transform) {
        FastFourierTransform fft = new FastFourierTransform(FastFourierTransform.Norm.STD, true);
        Complex[] result = fft.apply(transform);

        double[] realResult = new double[result.length];
        for (int i = 0; i < result.length; i++) {
            realResult[i] = result[i].getReal();
        }
        return realResult;
    }

    private void validateInputs(double[] signal, double[] kernel) {
        Objects.requireNonNull(signal, "signal cannot be null");
        Objects.requireNonNull(kernel, "kernel cannot be null");

        if (signal.length == 0 || kernel.length == 0) {
            throw new NoDataException();
        }
    }
}