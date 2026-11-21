package dev.nathanlively.overlap_save_demo;

/**
 * {@link Convolution} over {@link PartitionedConvolver}: uniform partitioned
 * convolution with partition size 256 by default (5.33 ms per block at
 * 48 kHz, 32 partitions for an 8192-tap kernel). Whole-signal driver: feeds
 * P-sample blocks, drains with {@link PartitionedConvolver#flush()}, trims
 * to the true result length.
 */
public class PartitionedConvolutionAdapter implements Convolution {
    private final int partitionSize;

    public PartitionedConvolutionAdapter() {
        this(256);
    }

    public PartitionedConvolutionAdapter(int partitionSize) {
        this.partitionSize = partitionSize;
    }

    /**
     * The streaming convolver this adapter drives. Subclasses override it
     * to swap the multiply representation; the block driver below stays put.
     */
    protected StreamingConvolver newConvolver(double[] kernel) {
        return new PartitionedConvolver(kernel, partitionSize);
    }

    protected int partitionSize() {
        return partitionSize;
    }

    @Override
    public double[] with(double[] signal, double[] kernel) {
        SignalTransformer.validate(signal, kernel);

        int resultLength = signal.length + kernel.length - 1;
        StreamingConvolver convolver = newConvolver(kernel);

        double[] result = new double[resultLength];
        int totalBlocks = (signal.length + partitionSize - 1) / partitionSize;
        int written = 0;
        for (int b = 0; b < totalBlocks; b++) {
            double[] block = new double[partitionSize];
            int copyLength = Math.min(partitionSize, signal.length - b * partitionSize);
            if (copyLength > 0) {
                System.arraycopy(signal, b * partitionSize, block, 0, copyLength);
            }
            double[] output = convolver.processBlock(block);
            int outLength = Math.min(partitionSize, resultLength - written);
            System.arraycopy(output, 0, result, written, outLength);
            written += outLength;
        }
        if (written < resultLength) {
            double[] tail = convolver.flush();
            int tailLength = Math.min(tail.length, resultLength - written);
            System.arraycopy(tail, 0, result, written, tailLength);
        }
        return result;
    }
}
