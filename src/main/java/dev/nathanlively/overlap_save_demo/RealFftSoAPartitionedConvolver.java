package dev.nathanlively.overlap_save_demo;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;
import org.apache.commons.math4.legacy.exception.NoDataException;

import java.util.Arrays;

/**
 * {@link SoAPartitionedConvolver} with the hand-rolled real FFT
 * ({@link HandRolledFft}): the forward transform of each real block
 * runs one half-size complex FFT plus O(N) packing, writing the spectrum
 * straight into the delay line, and the inverse runs one half-size
 * complex FFT plus O(N) unpacking, writing real samples straight into
 * the block result. No {@code Complex[]} anywhere, and no full complex
 * transform anywhere — both directions are real in, real out.
 *
 * <p>Same FDL / overlap-add block loop and same Vector API
 * multiply-accumulate as the SoA and InPlace arms, so the comparison
 * against InPlace isolates exactly what the real-FFT halving buys:
 * FftPerformanceTest measured the FFT-512 round trip at 4.54 us hand
 * real vs 6.29 us in place with reused buffers, so the per-block
 * median should drop by roughly 1.75 us from InPlace's ~16.7 us.
 *
 * <p>One note on allocation: {@code forwardReal} / {@code inverseReal}
 * each allocate two half-size scratch arrays per call internally.
 * They are plain {@code double[]}s — the same kind FftPerformanceTest
 * showed to be ~free — and the measured 4.54 us includes them.
 */
public class RealFftSoAPartitionedConvolver implements StreamingConvolver {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private final HandRolledFft fft = new HandRolledFft();

    private final int partitionSize;
    private final int fftSize;
    private final int partitions;
    private final double[][] kernelSpectraRe;
    private final double[][] kernelSpectraIm;
    private final double[][] delayLineRe;
    private final double[][] delayLineIm;
    private final double[] overlap;
    private int position;

    public RealFftSoAPartitionedConvolver(double[] kernel, int partitionSize) {
        if (kernel.length == 0) {
            throw new NoDataException();
        }
        if (partitionSize < 1 || (partitionSize & (partitionSize - 1)) != 0) {
            throw new IllegalArgumentException("partitionSize must be a power of two");
        }
        this.partitionSize = partitionSize;
        this.fftSize = 2 * partitionSize;
        this.partitions = (kernel.length + partitionSize - 1) / partitionSize;

        // Pre-compute kernel FFTs once, real in, split spectrum out.
        this.kernelSpectraRe = new double[partitions][];
        this.kernelSpectraIm = new double[partitions][];
        double[] paddedKernel = SignalTransformer.pad(kernel, partitions * partitionSize);
        for (int j = 0; j < partitions; j++) {
            double[] chunk = new double[fftSize];
            System.arraycopy(paddedKernel, j * partitionSize, chunk, 0, partitionSize);
            double[] re = new double[fftSize];
            double[] im = new double[fftSize];
            fft.forwardReal(chunk, re, im);
            kernelSpectraRe[j] = re;
            kernelSpectraIm[j] = im;
        }

        // Frequency-domain delay line: one zero spectrum per slot, shared
        // (entries are replaced, never mutated, so sharing is safe).
        double[] zeroSpectrum = new double[fftSize];
        this.delayLineRe = new double[partitions][];
        this.delayLineIm = new double[partitions][];
        Arrays.fill(delayLineRe, zeroSpectrum);
        Arrays.fill(delayLineIm, zeroSpectrum);

        this.overlap = new double[partitionSize];
        this.position = 0;
    }

    @Override
    public double[] processBlock(double[] block) {
        if (block.length != partitionSize) {
            throw new IllegalArgumentException("block must be partitionSize samples");
        }

        // Forward real transform straight into the delay line's fresh buffers.
        double[] paddedBlock = new double[fftSize];
        System.arraycopy(block, 0, paddedBlock, 0, partitionSize);
        double[] inputRe = new double[fftSize];
        double[] inputIm = new double[fftSize];
        fft.forwardReal(paddedBlock, inputRe, inputIm);
        delayLineRe[position] = inputRe;
        delayLineIm[position] = inputIm;

        double[] sumRe = new double[fftSize];
        double[] sumIm = new double[fftSize];
        for (int j = 0; j < partitions; j++) {
            int index = (position - j + partitions) % partitions;
            multiplyAccumulate(delayLineRe[index], delayLineIm[index],
                    kernelSpectraRe[j], kernelSpectraIm[j], sumRe, sumIm);
        }
        position = (position + 1) % partitions;

        // Inverse real transform straight into the block result — no
        // merge, no Complex[], no copy.
        double[] blockResult = new double[fftSize];
        fft.inverseReal(sumRe, sumIm, blockResult);
        double[] output = new double[partitionSize];
        for (int i = 0; i < partitionSize; i++) {
            output[i] = blockResult[i] + overlap[i];
        }
        System.arraycopy(blockResult, partitionSize, overlap, 0, partitionSize);
        return output;
    }

    /**
     * Element-wise complex multiply-accumulate: {@code sum += a * b} with
     * a, b and sum in split-complex form. Vectorized over the bins; the
     * scalar tail covers any remainder when the species length does not
     * divide the FFT size. Same as the SoA arm.
     */
    private static void multiplyAccumulate(double[] aRe, double[] aIm, double[] bRe, double[] bIm,
                                           double[] sumRe, double[] sumIm) {
        int length = aRe.length;
        int i = 0;
        int upperBound = SPECIES.loopBound(length);
        for (; i < upperBound; i += SPECIES.length()) {
            DoubleVector aReV = DoubleVector.fromArray(SPECIES, aRe, i);
            DoubleVector aImV = DoubleVector.fromArray(SPECIES, aIm, i);
            DoubleVector bReV = DoubleVector.fromArray(SPECIES, bRe, i);
            DoubleVector bImV = DoubleVector.fromArray(SPECIES, bIm, i);
            DoubleVector prodRe = aReV.mul(bReV).sub(aImV.mul(bImV));
            DoubleVector prodIm = aReV.mul(bImV).add(aImV.mul(bReV));
            DoubleVector.fromArray(SPECIES, sumRe, i).add(prodRe).intoArray(sumRe, i);
            DoubleVector.fromArray(SPECIES, sumIm, i).add(prodIm).intoArray(sumIm, i);
        }
        for (; i < length; i++) {
            sumRe[i] += aRe[i] * bRe[i] - aIm[i] * bIm[i];
            sumIm[i] += aRe[i] * bIm[i] + aIm[i] * bRe[i];
        }
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
