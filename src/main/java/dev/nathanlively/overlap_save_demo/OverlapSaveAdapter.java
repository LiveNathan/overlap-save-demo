package dev.nathanlively.overlap_save_demo;

import org.apache.arrow.memory.util.CommonUtil;
import org.apache.commons.numbers.complex.Complex;

import java.util.ArrayList;
import java.util.List;

public class OverlapSaveAdapter implements Convolution {
    @Override
    public double[] with(double[] signal, double[] kernel) {
        SignalTransformer.validate(signal, kernel);

        int kernelLength = kernel.length;
        int chunkSize = calculateOptimalChunkSize(kernelLength);
        int overlap = kernelLength - 1;
        int validOutputPerChunk = chunkSize - overlap;

        // Pre-compute kernel FFT (reused for all chunks)
        double[] paddedKernel = SignalTransformer.pad(kernel, chunkSize);
        Complex[] kernelTransform = SignalTransformer.fft(paddedKernel);

        List<double[]> results = new ArrayList<>();
        int signalPosition = 0;

        while (signalPosition < signal.length) {
            // Extract a chunk with overlap from the previous chunk
            double[] chunk = extractChunk(signal, signalPosition, chunkSize, overlap);

            // Convolve this chunk
            Complex[] chunkTransform = SignalTransformer.fft(chunk);
            Complex[] productTransform = SignalTransformer.multiply(chunkTransform, kernelTransform);
            double[] chunkResult = SignalTransformer.ifft(productTransform);

            // Keep only the valid portion (discard first 'overlap' samples)
            double[] validPortion = extractValidPortionFromChunk(chunkResult, overlap, validOutputPerChunk);
            results.add(validPortion);

            // Move to the next chunk position
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

        // Copy available signal data into a chunk
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
        if (kernelLength <= 32) return 32;
        return CommonUtil.nextPowerOfTwo(kernelLength);
    }

}