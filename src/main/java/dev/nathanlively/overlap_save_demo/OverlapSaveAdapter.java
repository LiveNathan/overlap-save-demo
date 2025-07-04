package dev.nathanlively.overlap_save_demo;

import org.apache.arrow.memory.util.CommonUtil;
import org.apache.commons.numbers.complex.Complex;

public class OverlapSaveAdapter implements Convolution {
    @Override
    public double[] with(double[] signal, double[] kernel) {
        SignalTransformer.validate(signal, kernel);

        int kernelLength = kernel.length;
        int fftSize = calculateOptimalFftSize(signal.length, kernelLength);
        int blockSize = fftSize - kernelLength + 1;
        int blockStartIndex = kernelLength - 1;
        int resultLength = signal.length + kernelLength - 1;

        // Pre-compute kernel FFT (zero-padded to FFT size)
        double[] paddedKernel = SignalTransformer.pad(kernel, fftSize);
        Complex[] kernelTransform = SignalTransformer.fft(paddedKernel);

        // Pre-allocate a result array
        double[] result = new double[resultLength];
        int resultPosition = 0;

        // Create padded signal with initial zeros for overlap
        double[] paddedSignal = SignalTransformer.pad(signal, blockStartIndex, resultLength - signal.length - blockStartIndex);

        // Process blocks
        int signalPosition = 0;
        while (signalPosition < signal.length) {
            // Extract block with proper overlap handling
            double[] block = extractBlock(paddedSignal, signalPosition, fftSize);

            // Convolve block in frequency domain
            Complex[] blockTransform = SignalTransformer.fft(block);
            Complex[] convolutionTransform = SignalTransformer.multiply(blockTransform, kernelTransform);
            double[] blockResult = SignalTransformer.ifft(convolutionTransform);

            // Extract valid portion (discard first kernelLength-1 samples due to aliasing)
            int validLength = Math.min(blockSize, resultLength - resultPosition);

            if (validLength > 0) {
                System.arraycopy(blockResult, blockStartIndex, result, resultPosition, validLength);
                resultPosition += validLength;
            }

            signalPosition += blockSize;
        }

        return result;
    }

    private double[] padSignalStart(double[] signal, int resultLength, int blockStartIndex) {
        double[] paddedSignal = new double[resultLength];
        System.arraycopy(signal, 0, paddedSignal, blockStartIndex, signal.length);
        return paddedSignal;
    }

    private double[] extractBlock(double[] paddedSignal, int position, int fftSize) {
        double[] block = new double[fftSize];
        int copyLength = Math.min(fftSize, paddedSignal.length - position);

        if (copyLength > 0) {
            System.arraycopy(paddedSignal, position, block, 0, copyLength);
        }

        return block;
    }

    int calculateOptimalFftSize(int signalLength, int kernelLength) {
        // Minimum size needed for linear convolution without aliasing
        int minSize = 2 * kernelLength - 1;

        // For very small kernels, use a reasonable minimum
        if (minSize < 64) {
            minSize = 64;
        }

        // Find the next power of 2 that's at least minSize
        int optimalSize = CommonUtil.nextPowerOfTwo(minSize);

        // For larger signals, consider efficiency trade-offs
        // Larger FFT sizes reduce the number of blocks but increase per-block cost
        int totalConvolutionLength = signalLength + kernelLength - 1;

        // If the signal is much larger than kernel, try larger FFT sizes
        if (signalLength > 10 * kernelLength) {
            // Calculate efficiency for different FFT sizes
            int bestSize = optimalSize;
            double bestEfficiency = calculateEfficiency(totalConvolutionLength, kernelLength, optimalSize);

            // Try powers of 2 up to a reasonable maximum
            for (int size = optimalSize * 2; size <= Math.min(8192, totalConvolutionLength); size *= 2) {
                double efficiency = calculateEfficiency(totalConvolutionLength, kernelLength, size);
                if (efficiency > bestEfficiency) {
                    bestSize = size;
                    bestEfficiency = efficiency;
                } else {
                    break; // Efficiency is decreasing, stop searching
                }
            }

            return bestSize;
        }

        return optimalSize;
    }

    private double calculateEfficiency(int totalLength, int kernelLength, int fftSize) {
        int blockSize = fftSize - kernelLength + 1;
        int numBlocks = (totalLength + blockSize - 1) / blockSize;
        double operationsPerSample = (numBlocks * fftSize * Math.log(fftSize)) / totalLength;
        return 1.0 / operationsPerSample; // Higher is better
    }
}