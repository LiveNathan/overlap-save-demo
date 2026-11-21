package dev.nathanlively.overlap_save_demo;

/**
 * {@link PartitionedConvolutionAdapter} over
 * {@link ValuePartitionedConvolver}: the per-block multiply runs over value
 * records. Only the convolver differs; the block driver lives in the parent.
 */
public class ValuePartitionedConvolutionAdapter extends PartitionedConvolutionAdapter {
    public ValuePartitionedConvolutionAdapter() {
        super();
    }

    public ValuePartitionedConvolutionAdapter(int partitionSize) {
        super(partitionSize);
    }

    @Override
    protected StreamingConvolver newConvolver(double[] kernel) {
        return new ValuePartitionedConvolver(kernel, partitionSize());
    }
}
