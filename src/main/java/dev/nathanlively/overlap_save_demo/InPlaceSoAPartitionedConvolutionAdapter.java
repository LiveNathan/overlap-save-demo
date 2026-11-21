package dev.nathanlively.overlap_save_demo;

/**
 * {@link PartitionedConvolutionAdapter} over {@link InPlaceSoAPartitionedConvolver}:
 * the per-block multiply runs over split-complex spectra with the Vector
 * API, as in the SoA arm, but Apache's FFT runs in place on the split
 * arrays — no {@code Complex[]} is built anywhere. Only the convolver
 * differs; the block driver lives in the parent.
 */
public class InPlaceSoAPartitionedConvolutionAdapter extends PartitionedConvolutionAdapter {
    public InPlaceSoAPartitionedConvolutionAdapter() {
        super();
    }

    public InPlaceSoAPartitionedConvolutionAdapter(int partitionSize) {
        super(partitionSize);
    }

    @Override
    protected StreamingConvolver newConvolver(double[] kernel) {
        return new InPlaceSoAPartitionedConvolver(kernel, partitionSize());
    }
}
