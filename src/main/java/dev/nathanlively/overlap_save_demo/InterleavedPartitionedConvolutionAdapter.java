package dev.nathanlively.overlap_save_demo;

/**
 * {@link PartitionedConvolutionAdapter} over
 * {@link InterleavedPartitionedConvolver}: the per-block multiply runs over
 * one interleaved flat array. Only the convolver differs; the block driver
 * lives in the parent.
 */
public class InterleavedPartitionedConvolutionAdapter extends PartitionedConvolutionAdapter {
    public InterleavedPartitionedConvolutionAdapter() {
        super();
    }

    public InterleavedPartitionedConvolutionAdapter(int partitionSize) {
        super(partitionSize);
    }

    @Override
    protected StreamingConvolver newConvolver(double[] kernel) {
        return new InterleavedPartitionedConvolver(kernel, partitionSize());
    }
}
