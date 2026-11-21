package dev.nathanlively.overlap_save_demo;

/**
 * {@link PartitionedConvolutionAdapter} over {@link SoAPartitionedConvolver}:
 * the per-block multiply runs over split-complex spectra with the Vector
 * API. Only the convolver differs; the block driver lives in the parent.
 */
public class SoAPartitionedConvolutionAdapter extends PartitionedConvolutionAdapter {
    public SoAPartitionedConvolutionAdapter() {
        super();
    }

    public SoAPartitionedConvolutionAdapter(int partitionSize) {
        super(partitionSize);
    }

    @Override
    protected StreamingConvolver newConvolver(double[] kernel) {
        return new SoAPartitionedConvolver(kernel, partitionSize());
    }
}
