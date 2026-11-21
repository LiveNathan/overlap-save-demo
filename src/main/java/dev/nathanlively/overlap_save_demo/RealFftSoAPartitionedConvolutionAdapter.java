package dev.nathanlively.overlap_save_demo;

/**
 * {@link PartitionedConvolutionAdapter} over {@link RealFftSoAPartitionedConvolver}:
 * the per-block multiply runs over split-complex spectra with the Vector
 * API, as in the SoA arm, but both FFT directions are the hand-rolled
 * real transforms — one half-size complex FFT plus O(N) packing each
 * way, real in and real out. Only the convolver differs; the block
 * driver lives in the parent.
 */
public class RealFftSoAPartitionedConvolutionAdapter extends PartitionedConvolutionAdapter {
    public RealFftSoAPartitionedConvolutionAdapter() {
        super();
    }

    public RealFftSoAPartitionedConvolutionAdapter(int partitionSize) {
        super(partitionSize);
    }

    @Override
    protected StreamingConvolver newConvolver(double[] kernel) {
        return new RealFftSoAPartitionedConvolver(kernel, partitionSize());
    }
}
