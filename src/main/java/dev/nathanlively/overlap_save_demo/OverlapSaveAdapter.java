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
        int blockSize = calculateOptimalBlockSize(kernelLength);
        int overlap = kernelLength - 1;
        int validOutputPerBlock = blockSize - overlap;

        // Pre-compute kernel FFT (reused for all blocks)
        double[] paddedKernel = SignalTransformer.pad(kernel, blockSize);
        Complex[] kernelTransform = SignalTransformer.fft(paddedKernel);

        List<double[]> results = new ArrayList<>();
        int signalIndex = 0;

        while (signalIndex < signal.length) {
            // Extract a block with overlap from the previous block
            double[] block = extractBlock(signal, signalIndex, blockSize, overlap);

            // Convolve this block
            Complex[] blockTransform = SignalTransformer.fft(block);
            Complex[] productTransform = SignalTransformer.multiply(blockTransform, kernelTransform);
            double[] blockResult = SignalTransformer.ifft(productTransform);

            // Keep only the valid portion (discard first 'overlap' samples)
            double[] validPortion = extractValidPortionFromBlock(blockResult, overlap, validOutputPerBlock);
            results.add(validPortion);

            // Move to the next block position
            signalIndex += validOutputPerBlock;
        }

        // Concatenate all valid portions
        return concatenateResults(results, signal.length + kernel.length - 1);
    }

    private double[] extractBlock(double[] signal, int startPosition, int blockSize, int overlap) {
        double[] block = new double[blockSize];

        // Calculate where to start reading from signal (accounting for overlap)
        int signalStart = Math.max(0, startPosition - overlap);
        int blockStart = Math.max(0, overlap - startPosition);

        // Copy available signal data into a block
        int copyLength = Math.min(signal.length - signalStart, blockSize - blockStart);
        if (copyLength > 0) {
            System.arraycopy(signal, signalStart, block, blockStart, copyLength);
        }

        return block;
    }

    private double[] extractValidPortionFromBlock(double[] blockResult, int overlap, int validLength) {
        int actualValidLength = Math.min(validLength, blockResult.length - overlap);
        double[] validPortion = new double[actualValidLength];

        if (actualValidLength > 0) {
            System.arraycopy(blockResult, overlap, validPortion, 0, actualValidLength);
        }

        return validPortion;
    }

    private double[] concatenateResults(List<double[]> results, int totalExpectedLength) {
        double[] finalResult = new double[totalExpectedLength];
        int position = 0;

        for (double[] block : results) {
            int copyLength = Math.min(block.length, totalExpectedLength - position);
            if (copyLength > 0) {
                System.arraycopy(block, 0, finalResult, position, copyLength);
                position += copyLength;
            }
        }

        return finalResult;
    }

    int calculateOptimalBlockSize(int kernelLength) {
        if (kernelLength <= 32) return 32;
        return CommonUtil.nextPowerOfTwo(kernelLength);
    }
}