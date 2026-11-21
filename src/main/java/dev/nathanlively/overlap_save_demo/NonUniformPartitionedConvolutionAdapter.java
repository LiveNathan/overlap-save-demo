package dev.nathanlively.overlap_save_demo;

import java.util.Arrays;

/**
 * {@link Convolution} over {@link NonUniformPartitionedConvolver}: the Wefers
 * layout 128^4 512^15 by default (8192 taps at B=256). Whole-signal driver:
 * feeds blockSize blocks, drains with {@link NonUniformPartitionedConvolver#flush()},
 * trims to the true result length. Short kernels are zero-padded to the
 * layout total inside the convolver, so the layout is fixed even for tiny
 * test cases.
 */
public class NonUniformPartitionedConvolutionAdapter implements Convolution {
    public static final int[][] WEFERS_8192 = {{4, 128}, {15, 512}};

    private final int blockSize;
    private final int[][] groups;

    public NonUniformPartitionedConvolutionAdapter() {
        this(256, WEFERS_8192);
    }

    public NonUniformPartitionedConvolutionAdapter(int blockSize, int[][] groups) {
        this.blockSize = blockSize;
        this.groups = deepCopy(groups);
    }

    /**
     * The streaming convolver this adapter drives. Subclasses override it
     * to swap the layout; the block driver below stays put.
     */
    protected StreamingConvolver newConvolver(double[] kernel) {
        return new NonUniformPartitionedConvolver(kernel, blockSize, groups);
    }

    protected int blockSize() {
        return blockSize;
    }

    protected int[][] groups() {
        return deepCopy(groups);
    }

    @Override
    public double[] with(double[] signal, double[] kernel) {
        SignalTransformer.validate(signal, kernel);

        int resultLength = signal.length + kernel.length - 1;
        StreamingConvolver convolver = newConvolver(kernel);

        double[] result = new double[resultLength];
        int totalBlocks = (signal.length + blockSize - 1) / blockSize;
        int written = 0;
        for (int b = 0; b < totalBlocks; b++) {
            double[] block = new double[blockSize];
            int copyLength = Math.min(blockSize, signal.length - b * blockSize);
            if (copyLength > 0) {
                System.arraycopy(signal, b * blockSize, block, 0, copyLength);
            }
            double[] output = convolver.processBlock(block);
            int outLength = Math.min(blockSize, resultLength - written);
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

    private static int[][] deepCopy(int[][] groups) {
        int[][] copy = new int[groups.length][];
        for (int i = 0; i < groups.length; i++) {
            copy[i] = Arrays.copyOf(groups[i], groups[i].length);
        }
        return copy;
    }
}
