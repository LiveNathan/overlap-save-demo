package dev.nathanlively.overlap_save_demo;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;
import org.apache.commons.math4.legacy.exception.NoDataException;
import org.apache.commons.math4.transform.FastFourierTransform;

import java.util.Arrays;

/**
 * {@link SoAPartitionedConvolver} with Apache's FFT called in place:
 * the forward transform writes straight into the delay line's fresh
 * split-complex buffers, and the inverse runs in place on the summed
 * spectra, whose real parts are the block result. No {@code Complex[]}
 * is ever built — no {@code apply}, no split, no merge, no
 * {@code createComplex} / {@code createRealImaginary} copies.
 *
 * <p>Same FDL / overlap-add block loop and same Vector API
 * multiply-accumulate as the SoA arm, so the comparison against it
 * isolates exactly what the {@code apply} wrapper cost: FftPerformanceTest
 * measured the round trip at 8.96 us through {@code apply} vs 6.29 us
 * in place with reused buffers (FFT-512 medians).
 *
 * <p>The two transform instances are stateless apart from norm and
 * direction, so one shared pair serves every convolver (same as
 * FftPerformanceTest). Normalization is STD on both, matching
 * {@link SignalTransformer}.
 */
public class InPlaceSoAPartitionedConvolver implements StreamingConvolver {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private static final FastFourierTransform FORWARD =
            new FastFourierTransform(FastFourierTransform.Norm.STD);
    private static final FastFourierTransform INVERSE =
            new FastFourierTransform(FastFourierTransform.Norm.STD, true);

    private final int partitionSize;
    private final int fftSize;
    private final int partitions;
    private final double[][] kernelSpectraRe;
    private final double[][] kernelSpectraIm;
    private final double[][] delayLineRe;
    private final double[][] delayLineIm;
    private final double[] overlap;
    private int position;

    public InPlaceSoAPartitionedConvolver(double[] kernel, int partitionSize) {
        if (kernel.length == 0) {
            throw new NoDataException();
        }
        if (partitionSize < 1 || (partitionSize & (partitionSize - 1)) != 0) {
            throw new IllegalArgumentException("partitionSize must be a power of two");
        }
        this.partitionSize = partitionSize;
        this.fftSize = 2 * partitionSize;
        this.partitions = (kernel.length + partitionSize - 1) / partitionSize;

        // Pre-compute kernel FFTs once, in place on split arrays.
        this.kernelSpectraRe = new double[partitions][];
        this.kernelSpectraIm = new double[partitions][];
        double[] paddedKernel = SignalTransformer.pad(kernel, partitions * partitionSize);
        for (int j = 0; j < partitions; j++) {
            double[] re = new double[fftSize];
            double[] im = new double[fftSize];
            System.arraycopy(paddedKernel, j * partitionSize, re, 0, partitionSize);
            FORWARD.transformInPlace(new double[][]{re, im});
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

        // Forward transform straight into the delay line's fresh buffers.
        double[] inputRe = new double[fftSize];
        double[] inputIm = new double[fftSize];
        System.arraycopy(block, 0, inputRe, 0, partitionSize);
        FORWARD.transformInPlace(new double[][]{inputRe, inputIm});
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

        // Inverse transform in place on the sums; the real parts are the
        // block result — no merge, no Complex[], no copy.
        INVERSE.transformInPlace(new double[][]{sumRe, sumIm});
        double[] output = new double[partitionSize];
        for (int i = 0; i < partitionSize; i++) {
            output[i] = sumRe[i] + overlap[i];
        }
        System.arraycopy(sumRe, partitionSize, overlap, 0, partitionSize);
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
