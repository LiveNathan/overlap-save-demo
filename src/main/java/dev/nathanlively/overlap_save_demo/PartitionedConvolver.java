package dev.nathanlively.overlap_save_demo;

import org.apache.commons.math4.legacy.exception.NoDataException;
import org.apache.commons.numbers.complex.Complex;

import java.util.Arrays;

/**
 * Stateful uniform partitioned convolution, KFR {@code convolve_filter} shape:
 * the kernel is split into P-sample partitions, each transformed once; each
 * input block is transformed once and kept in a frequency-domain delay line
 * (ring buffer of spectra); per block the products of the delay line against
 * the kernel partitions are summed and inverse-transformed once, and the
 * second half of the result is the overlap-add carry into the next block.
 *
 * <p>Per input block of P samples the work is one FFT of size 2P, M complex
 * multiplies of 2P bins, and one IFFT of size 2P — the cost Sheta derives as
 * C = 4k0 P log2(2P) + 2k1 L. The first partition has size P = B, so the
 * filter adds no signal-alignment latency: the first output sample
 * corresponds to the first input sample.
 *
 * <p>The delay works because kernel partition j starts at offset j*P in the
 * kernel, which cancels the -j*P delay of input block n-j: the t-th sample
 * of the local convolution of input block n-j with partition j lands at
 * absolute time n*P + t. So the summed spectrum's first P samples are the
 * current output block (plus the saved overlap) and its second P samples
 * are the overlap for the next block. This is KFR's
 * y_k[n] = sum_j y_{k-j,j}[n], with y[n] = sum_k y_k[n - kB].
 *
 * <p>After the last input block, {@link #flush()} feeds zero blocks until the
 * delay line drains and returns the remaining output (M*P samples); the
 * caller trims to the true result length. This is KFR's {@code apply_zeros}.
 */
public class PartitionedConvolver implements StreamingConvolver {
    private final int partitionSize;
    private final int fftSize;
    private final int partitions;
    private final Complex[][] kernelSpectra;
    private final Complex[][] delayLine;
    private final double[] overlap;
    private int position;

    public PartitionedConvolver(double[] kernel, int partitionSize) {
        if (kernel.length == 0) {
            throw new NoDataException();
        }
        if (partitionSize < 1 || (partitionSize & (partitionSize - 1)) != 0) {
            throw new IllegalArgumentException("partitionSize must be a power of two");
        }
        this.partitionSize = partitionSize;
        this.fftSize = 2 * partitionSize;
        this.partitions = (kernel.length + partitionSize - 1) / partitionSize;

        // Pre-compute kernel FFTs once: one spectrum per partition.
        this.kernelSpectra = new Complex[partitions][];
        double[] paddedKernel = SignalTransformer.pad(kernel, partitions * partitionSize);
        for (int j = 0; j < partitions; j++) {
            double[] chunk = Arrays.copyOfRange(paddedKernel, j * partitionSize, (j + 1) * partitionSize);
            kernelSpectra[j] = SignalTransformer.fft(SignalTransformer.pad(chunk, fftSize));
        }

        // Frequency-domain delay line: one zero spectrum per slot, shared
        // (entries are replaced, never mutated, so sharing is safe).
        Complex[] zeroSpectrum = new Complex[fftSize];
        Arrays.fill(zeroSpectrum, Complex.ofCartesian(0.0, 0.0));
        this.delayLine = new Complex[partitions][];
        Arrays.fill(delayLine, zeroSpectrum);

        this.overlap = new double[partitionSize];
        this.position = 0;
    }

    /**
     * Processes one block of {@code partitionSize} samples and returns
     * {@code partitionSize} output samples.
 */
    public double[] processBlock(double[] block) {
        if (block.length != partitionSize) {
            throw new IllegalArgumentException("block must be partitionSize samples");
        }

        Complex[] inputSpectrum = SignalTransformer.fft(SignalTransformer.pad(block, fftSize));
        delayLine[position] = inputSpectrum;

        Complex[] sum = new Complex[fftSize];
        Arrays.fill(sum, Complex.ofCartesian(0.0, 0.0));
        for (int j = 0; j < partitions; j++) {
            int index = (position - j + partitions) % partitions;
            Complex[] product = SignalTransformer.multiply(delayLine[index], kernelSpectra[j]);
            for (int i = 0; i < fftSize; i++) {
                sum[i] = sum[i].add(product[i]);
            }
        }
        position = (position + 1) % partitions;

        double[] blockResult = SignalTransformer.ifft(sum);
        double[] output = new double[partitionSize];
        for (int i = 0; i < partitionSize; i++) {
            output[i] = blockResult[i] + overlap[i];
        }
        System.arraycopy(blockResult, partitionSize, overlap, 0, partitionSize);
        return output;
    }

    /**
     * Drains the delay line after the last input block: feeds zero blocks
     * until every partition's contribution has been emitted, and returns the
     * remaining output (partitions * partitionSize samples, including the
     * final overlap). The caller trims to the true result length.
     */
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
