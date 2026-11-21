package dev.nathanlively.overlap_save_demo;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;
import org.apache.commons.math4.legacy.exception.NoDataException;
import org.apache.commons.numbers.complex.Complex;

import java.util.Arrays;

/**
 * {@link InterleavedPartitionedConvolver} with split-complex (SoA) spectra:
 * every spectrum lives as two flat arrays, real and imaginary separately,
 * and the per-block multiply-accumulate runs over them with the Vector API.
 * Same FDL / overlap-add block loop, same FFTs, same conversions in and
 * out — only the multiply differs, so the comparison against the interleaved
 * arm isolates the multiply representation.
 *
 * <p>This is deliberately NOT the earlier {@link VectorApiAdapter} shape:
 * that one reduced per output sample (one vector reduction per output,
 * 0.31x). Here the Vector API does element-wise complex multiply and
 * accumulation across partitions — one vector op per bin, no reduction —
 * which is the shape Hienzsch's split-complex finding points at: separate
 * real/imag arrays make SIMD much easier for the compiler (and here,
 * explicit).
 *
 * <p>Species is {@code SPECIES_PREFERRED}: on Apple Silicon (NEON, 128-bit)
 * that is 2 doubles per vector, on AVX2 4. The tail is a scalar loop, so
 * the code is correct for any species length.
 */
public class SoAPartitionedConvolver implements StreamingConvolver {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private final int partitionSize;
    private final int fftSize;
    private final int partitions;
    private final double[][] kernelSpectraRe;
    private final double[][] kernelSpectraIm;
    private final double[][] delayLineRe;
    private final double[][] delayLineIm;
    private final double[] overlap;
    private int position;

    public SoAPartitionedConvolver(double[] kernel, int partitionSize) {
        if (kernel.length == 0) {
            throw new NoDataException();
        }
        if (partitionSize < 1 || (partitionSize & (partitionSize - 1)) != 0) {
            throw new IllegalArgumentException("partitionSize must be a power of two");
        }
        this.partitionSize = partitionSize;
        this.fftSize = 2 * partitionSize;
        this.partitions = (kernel.length + partitionSize - 1) / partitionSize;

        // Pre-compute kernel FFTs once, split into real/imaginary parts.
        this.kernelSpectraRe = new double[partitions][];
        this.kernelSpectraIm = new double[partitions][];
        double[] paddedKernel = SignalTransformer.pad(kernel, partitions * partitionSize);
        for (int j = 0; j < partitions; j++) {
            double[] chunk = Arrays.copyOfRange(paddedKernel, j * partitionSize, (j + 1) * partitionSize);
            Complex[] spectrum = SignalTransformer.fft(SignalTransformer.pad(chunk, fftSize));
            kernelSpectraRe[j] = new double[fftSize];
            kernelSpectraIm[j] = new double[fftSize];
            split(spectrum, kernelSpectraRe[j], kernelSpectraIm[j]);
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

        Complex[] inputSpectrum = SignalTransformer.fft(SignalTransformer.pad(block, fftSize));
        double[] inputRe = new double[fftSize];
        double[] inputIm = new double[fftSize];
        split(inputSpectrum, inputRe, inputIm);
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

        double[] blockResult = SignalTransformer.ifft(merge(sumRe, sumIm));
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
     * divide the FFT size.
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

    private static void split(Complex[] spectrum, double[] re, double[] im) {
        for (int i = 0; i < spectrum.length; i++) {
            re[i] = spectrum[i].getReal();
            im[i] = spectrum[i].getImaginary();
        }
    }

    private static Complex[] merge(double[] re, double[] im) {
        Complex[] spectrum = new Complex[re.length];
        for (int i = 0; i < re.length; i++) {
            spectrum[i] = Complex.ofCartesian(re[i], im[i]);
        }
        return spectrum;
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
