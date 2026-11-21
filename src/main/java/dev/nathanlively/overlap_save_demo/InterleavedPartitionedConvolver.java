package dev.nathanlively.overlap_save_demo;

import org.apache.commons.math4.legacy.exception.NoDataException;

import java.util.Arrays;

/**
 * {@link PartitionedConvolver} with the per-block multiply over one interleaved
 * flat array ({@code [re, im, re, im, ...]}). Same FDL / overlap-add block
 * loop; only the spectrum representation differs: spectra are flat
 * {@code double[]} of length 2 * FFT size, kernel spectra are converted
 * once in the constructor, and each block pays one conversion to and one
 * from interleaved form around the flat multiply.
 */
public class InterleavedPartitionedConvolver implements StreamingConvolver {
    private final int partitionSize;
    private final int fftSize;
    private final int spectrumLength;
    private final int partitions;
    private final double[][] kernelSpectra;
    private final double[][] delayLine;
    private final double[] overlap;
    private int position;

    public InterleavedPartitionedConvolver(double[] kernel, int partitionSize) {
        if (kernel == null) {
            throw new NullPointerException("kernel cannot be null");
        }
        if (kernel.length == 0) {
            throw new NoDataException();
        }
        if (partitionSize < 1 || (partitionSize & (partitionSize - 1)) != 0) {
            throw new IllegalArgumentException("partitionSize must be a power of two");
        }
        this.partitionSize = partitionSize;
        this.fftSize = 2 * partitionSize;
        this.spectrumLength = 2 * fftSize;
        this.partitions = (kernel.length + partitionSize - 1) / partitionSize;

        // Pre-compute kernel FFTs once, converted to interleaved form.
        this.kernelSpectra = new double[partitions][];
        double[] paddedKernel = SignalTransformer.pad(kernel, partitions * partitionSize);
        for (int j = 0; j < partitions; j++) {
            double[] chunk = Arrays.copyOfRange(paddedKernel, j * partitionSize, (j + 1) * partitionSize);
            kernelSpectra[j] = SignalTransformer.toInterleaved(SignalTransformer.fft(SignalTransformer.pad(chunk, fftSize)));
        }

        // Frequency-domain delay line: one zero spectrum per slot, shared
        // (entries are replaced, never mutated, so sharing is safe).
        double[] zeroSpectrum = new double[spectrumLength];
        this.delayLine = new double[partitions][];
        Arrays.fill(delayLine, zeroSpectrum);

        this.overlap = new double[partitionSize];
        this.position = 0;
    }

    @Override
    public double[] processBlock(double[] block) {
        if (block == null) {
            throw new NullPointerException("block cannot be null");
        }
        if (block.length != partitionSize) {
            throw new IllegalArgumentException("block must be partitionSize samples");
        }

        double[] inputSpectrum = SignalTransformer.toInterleaved(SignalTransformer.fft(SignalTransformer.pad(block, fftSize)));
        delayLine[position] = inputSpectrum;

        double[] sum = new double[spectrumLength];
        for (int j = 0; j < partitions; j++) {
            int index = (position - j + partitions) % partitions;
            double[] product = SignalTransformer.multiplyInterleaved(delayLine[index], kernelSpectra[j]);
            for (int i = 0; i < spectrumLength; i++) {
                sum[i] += product[i];
            }
        }
        position = (position + 1) % partitions;

        double[] blockResult = SignalTransformer.ifft(SignalTransformer.fromInterleaved(sum));
        double[] output = new double[partitionSize];
        for (int i = 0; i < partitionSize; i++) {
            output[i] = blockResult[i] + overlap[i];
        }
        System.arraycopy(blockResult, partitionSize, overlap, 0, partitionSize);
        return output;
    }

    @Override
    public double[] flush() {
        double[] tail = new double[partitions * partitionSize];
        double[] zeros = new double[partitionSize];
        for (int b = 0; b < partitions - 1; b++) {
            double[] out = processBlock(zeros);
            System.arraycopy(out, 0, tail, b * partitionSize, partitionSize);
        }
        System.arraycopy(overlap, 0, tail, (partitions - 1) * partitionSize, partitionSize);
        return tail;
    }

    @Override
    public int partitionSize() {
        return partitionSize;
    }

    @Override
    public int partitions() {
        return partitions;
    }
}
