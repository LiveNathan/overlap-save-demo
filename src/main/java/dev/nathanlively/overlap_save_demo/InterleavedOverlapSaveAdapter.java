package dev.nathanlively.overlap_save_demo;

import org.apache.commons.numbers.complex.Complex;

/**
 * {@link OverlapSaveAdapter} with the per-block multiply over one interleaved
 * flat array ({@code [re, im, re, im, ...]}). It extends the Apache adapter
 * only to share its FFT-size logic ({@link #calculateOptimalFftSize}); the
 * block loop is its own.
 *
 * <p>The kernel conversion happens once, outside the block loop, so its cost
 * is amortised over every block — that is the point of overlap-save. The
 * per-block conversion of the block transform to and from interleaved form
 * is still paid once per block, and whether the flat multiply's win survives
 * that is what the performance test measures.
 */
public class InterleavedOverlapSaveAdapter extends OverlapSaveAdapter {
    @Override
    public double[] with(double[] signal, double[] kernel) {
        SignalTransformer.validate(signal, kernel);

        int kernelLength = kernel.length;
        int fftSize = calculateOptimalFftSize(signal.length, kernelLength);
        int blockSize = fftSize - kernelLength + 1;
        int blockStartIndex = kernelLength - 1;
        int resultLength = signal.length + kernelLength - 1;

        // Pre-compute kernel FFT once, and convert it once: reused by every block.
        double[] paddedKernel = SignalTransformer.pad(kernel, fftSize);
        Complex[] kernelTransform = SignalTransformer.fft(paddedKernel);
        double[] kernelFlat = SignalTransformer.toInterleaved(kernelTransform);

        double[] result = new double[resultLength];

        double[] paddedSignal = SignalTransformer.pad(
                signal,
                blockStartIndex,
                resultLength - signal.length - blockStartIndex
        );

        // Block count is based on the result length, not the signal length:
        // the result is kernelLength-1 samples longer than the signal, and
        // basing it on the signal left the tail unwritten (zeros) whenever
        // the last block's overshoot past the signal was smaller than
        // kernelLength-1 — e.g. 32768/64 lost its last 54 samples.
        int totalBlocks = (resultLength + blockSize - 1) / blockSize;
        for (int blockIndex = 0; blockIndex < totalBlocks; blockIndex++) {
            int nextBlockStartIndex = blockIndex * blockSize;

            double[] block = extractSignalBlock(paddedSignal, nextBlockStartIndex, fftSize);

            Complex[] blockTransform = SignalTransformer.fft(block);
            double[] blockFlat = SignalTransformer.toInterleaved(blockTransform);
            double[] productFlat = SignalTransformer.multiplyInterleaved(blockFlat, kernelFlat);
            Complex[] convolutionTransform = SignalTransformer.fromInterleaved(productFlat);
            double[] blockResult = SignalTransformer.ifft(convolutionTransform);

            int validLength = Math.min(blockSize, resultLength - nextBlockStartIndex);

            if (validLength > 0) {
                System.arraycopy(
                        blockResult,
                        blockStartIndex,
                        result,
                        nextBlockStartIndex,
                        validLength);
            }
        }

        return result;
    }

    // Duplicated from OverlapSaveAdapter: it is private there, and this
    // subclass's block loop needs its own copy.
    private double[] extractSignalBlock(double[] paddedSignal, int nextBlockStartIndex, int fftSize) {
        double[] block = new double[fftSize];
        int copyLength = Math.min(fftSize, paddedSignal.length - nextBlockStartIndex);

        if (copyLength > 0) {
            System.arraycopy(
                    paddedSignal,
                    nextBlockStartIndex,
                    block,
                    0,
                    copyLength
            );
        }

        return block;
    }
}
