package dev.nathanlively.overlap_save_demo;

import org.apache.commons.math4.legacy.exception.NoDataException;

import java.util.Arrays;

/**
 * {@link PartitionedConvolver} with the per-block multiply over value
 * records ({@link ComplexValue}). Same FDL / overlap-add block loop; only
 * the spectrum representation differs: spectra are {@code ComplexValue[]},
 * kernel spectra are converted once in the constructor, and each block
 * pays one conversion to and one from value form around the value multiply.
 */
public class ValuePartitionedConvolver implements StreamingConvolver {
    private final int partitionSize;
    private final int fftSize;
    private final int partitions;
    private final ComplexValue[][] kernelSpectra;
    private final ComplexValue[][] delayLine;
    private final double[] overlap;
    private int position;

    public ValuePartitionedConvolver(double[] kernel, int partitionSize) {
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
        this.partitions = (kernel.length + partitionSize - 1) / partitionSize;

        // Pre-compute kernel FFTs once, converted to value form.
        this.kernelSpectra = new ComplexValue[partitions][];
        double[] paddedKernel = SignalTransformer.pad(kernel, partitions * partitionSize);
        for (int j = 0; j < partitions; j++) {
            double[] chunk = Arrays.copyOfRange(paddedKernel, j * partitionSize, (j + 1) * partitionSize);
            kernelSpectra[j] = SignalTransformer.toComplexValue(SignalTransformer.fft(SignalTransformer.pad(chunk, fftSize)));
        }

        // Frequency-domain delay line: one zero spectrum per slot, shared
        // (entries are replaced, never mutated, so sharing is safe).
        ComplexValue[] zeroSpectrum = new ComplexValue[fftSize];
        Arrays.fill(zeroSpectrum, new ComplexValue(0.0, 0.0));
        this.delayLine = new ComplexValue[partitions][];
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

        ComplexValue[] inputSpectrum = SignalTransformer.toComplexValue(SignalTransformer.fft(SignalTransformer.pad(block, fftSize)));
        delayLine[position] = inputSpectrum;

        ComplexValue[] sum = new ComplexValue[fftSize];
        Arrays.fill(sum, new ComplexValue(0.0, 0.0));
        for (int j = 0; j < partitions; j++) {
            int index = (position - j + partitions) % partitions;
            ComplexValue[] product = SignalTransformer.multiplyValue(delayLine[index], kernelSpectra[j]);
            for (int i = 0; i < fftSize; i++) {
                sum[i] = sum[i].add(product[i]);
            }
        }
        position = (position + 1) % partitions;

        double[] blockResult = SignalTransformer.ifft(SignalTransformer.toComplex(sum));
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
