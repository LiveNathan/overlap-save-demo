package dev.nathanlively.overlap_save_demo;

import org.apache.commons.math4.legacy.exception.NoDataException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Non-uniform partitioned convolution (Sheta / Gardner / Wefers): the kernel
 * is split into groups of same-size partitions, each group runs as one uniform
 * partitioned stage ({@link InterleavedPartitionedConvolver}) in parallel, and
 * the stages' outputs are summed at their absolute positions. A stage with
 * partition size P runs once per P/blockSize blocks, so a big stage's cost is
 * spread over many blocks: the multiply work per output sample drops from
 * 2L/P (uniform) toward the Wefers optimum, while the first stage (P at most
 * blockSize) keeps zero added latency.
 *
 * <p>Scheduling: input arrives in {@code blockSize} blocks. A stage with
 * partition size P consumes input in P-sized chunks; a chunk that completes
 * during a block produces P output samples at absolute position
 * chunkIndex * P + offset, where offset is the stage's kernel offset. The
 * driver adds each chunk into a shared output accumulator and emits one
 * block per input block. This is safe only when every stage's output for a
 * block is computed by the time the block is emitted: it holds when each P
 * divides or is a multiple of blockSize and each offset is a multiple of its
 * P (both validated in the constructor) — a stage's chunk then completes at
 * most at the block boundary its first output sample is due (clearance >= 0).
 *
 * <p>The multiply representation is fixed at interleaved flat
 * {@code double[]}: the per-block measurement showed it 6.6x faster than
 * Apache {@code Complex} and 3.4x faster than value records for uniform
 * partitions, so the layout — not the representation — is the variable
 * under test here. Short kernels are zero-padded to the layout total, so
 * the layout is fixed even for tiny test cases.
 */
public class NonUniformPartitionedConvolver implements StreamingConvolver {
    private final int blockSize;
    private final int totalPartitions;
    private final int maxPartitionSize;
    private final Stage[] stages;

    private double[] out = new double[4096];
    private int outStart;
    private int outEnd;
    private int inputFed;

    private static final class Stage {
        final InterleavedPartitionedConvolver convolver;
        final int partitionSize;
        final int offset;
        final double[] inputBuffer;
        int filled;
        int chunkIndex;

        Stage(InterleavedPartitionedConvolver convolver, int partitionSize, int offset) {
            this.convolver = convolver;
            this.partitionSize = partitionSize;
            this.offset = offset;
            this.inputBuffer = new double[partitionSize];
        }
    }

    /**
     * @param kernel the impulse response; zero-padded to the layout total
     *               when shorter, refused when longer
     * @param blockSize the input block size in samples (power of two)
     * @param groups the layout: one {@code {count, size}} pair per stage,
     *               in kernel order, sizes powers of two
     */
    public NonUniformPartitionedConvolver(double[] kernel, int blockSize, int[][] groups) {
        if (kernel.length == 0) {
            throw new NoDataException();
        }
        if (blockSize < 1 || (blockSize & (blockSize - 1)) != 0) {
            throw new IllegalArgumentException("blockSize must be a power of two");
        }
        if (groups.length == 0) {
            throw new IllegalArgumentException("groups must not be empty");
        }
        this.blockSize = blockSize;
        int total = 0;
        int partitions = 0;
        int maxPs = 0;
        for (int[] group : groups) {
            int count = group[0];
            int size = group[1];
            if (count < 1) {
                throw new IllegalArgumentException("group count must be >= 1");
            }
            if (size < 1 || (size & (size - 1)) != 0) {
                throw new IllegalArgumentException("group size must be a power of two");
            }
            if (size % blockSize != 0 && blockSize % size != 0) {
                throw new IllegalArgumentException("group size must divide or be a multiple of blockSize");
            }
            total += count * size;
            partitions += count;
            maxPs = Math.max(maxPs, size);
        }
        if (total < kernel.length) {
            throw new IllegalArgumentException("layout total " + total + " is shorter than the kernel " + kernel.length);
        }
        this.totalPartitions = partitions;
        this.maxPartitionSize = maxPs;

        double[] paddedKernel = SignalTransformer.pad(kernel, total);
        this.stages = new Stage[groups.length];
        int offset = 0;
        for (int s = 0; s < groups.length; s++) {
            int count = groups[s][0];
            int size = groups[s][1];
            if (offset % size != 0) {
                throw new IllegalArgumentException("stage offset " + offset + " is not a multiple of its partition size " + size);
            }
            double[] slice = Arrays.copyOfRange(paddedKernel, offset, offset + count * size);
            stages[s] = new Stage(new InterleavedPartitionedConvolver(slice, size), size, offset);
            offset += count * size;
        }
    }

    @Override
    public double[] processBlock(double[] block) {
        if (block.length != blockSize) {
            throw new IllegalArgumentException("block must be blockSize samples");
        }
        for (Stage stage : stages) {
            feedStage(stage, block);
        }
        inputFed += blockSize;
        return emitBlock();
    }

    private void feedStage(Stage stage, double[] block) {
        int pos = 0;
        while (pos < block.length) {
            int take = Math.min(stage.partitionSize - stage.filled, block.length - pos);
            System.arraycopy(block, pos, stage.inputBuffer, stage.filled, take);
            stage.filled += take;
            pos += take;
            if (stage.filled == stage.partitionSize) {
                double[] output = stage.convolver.processBlock(stage.inputBuffer);
                addAt(stage.chunkIndex * stage.partitionSize + stage.offset, output);
                stage.chunkIndex++;
                stage.filled = 0;
            }
        }
    }

    private void addAt(int pos, double[] chunk) {
        if (pos < outStart) {
            throw new IllegalStateException("stage output landed behind the emit cursor at " + pos);
        }
        ensureRoom(pos + chunk.length);
        int startIndex = pos - outStart;
        int validEnd = outEnd - outStart;
        // Indices beyond the valid pending data hold already-emitted
        // samples: emitBlock shifts the remainder left but never clears
        // what it consumed. Clear the span this chunk will touch before
        // accumulating, or it would add onto stale values — this corrupted
        // exactly block 1 on short kernels, where the second stage
        // contributes nothing to mask it. The span inside the valid data
        // is a genuine second contribution and accumulates as before.
        for (int i = Math.max(validEnd, startIndex); i < startIndex + chunk.length; i++) {
            out[i] = 0.0;
        }
        for (int i = 0; i < chunk.length; i++) {
            out[startIndex + i] += chunk[i];
        }
        outEnd = Math.max(outEnd, pos + chunk.length);
    }

    private void ensureRoom(int end) {
        int needed = end - outStart;
        if (needed > out.length) {
            double[] grown = new double[Math.max(out.length * 2, needed)];
            System.arraycopy(out, 0, grown, 0, outEnd - outStart);
            out = grown;
        }
    }

    private double[] emitBlock() {
        int available = outEnd - outStart;
        if (available < blockSize) {
            throw new IllegalStateException("only " + available + " output samples ready for a " + blockSize + "-sample block");
        }
        double[] block = new double[blockSize];
        System.arraycopy(out, 0, block, 0, blockSize);
        int remaining = available - blockSize;
        if (remaining > 0) {
            System.arraycopy(out, blockSize, out, 0, remaining);
        }
        outStart += blockSize;
        outEnd = outStart + remaining;
        return block;
    }

    /**
     * Drains the stream after the last input block: pads the input to a
     * multiple of the largest partition size so every stage is chunk-aligned,
     * then drains each stage's delay line and returns everything still
     * pending. The caller trims to the true result length.
     */
    @Override
    public double[] flush() {
        List<double[]> parts = new ArrayList<>();
        int total = 0;
        while (inputFed % maxPartitionSize != 0) {
            double[] block = processBlock(new double[blockSize]);
            parts.add(block);
            total += block.length;
        }
        for (Stage stage : stages) {
            double[] tail = stage.convolver.flush();
            addAt(stage.chunkIndex * stage.partitionSize + stage.offset, tail);
        }
        int remaining = outEnd - outStart;
        if (remaining > 0) {
            double[] rest = new double[remaining];
            System.arraycopy(out, 0, rest, 0, remaining);
            parts.add(rest);
            total += remaining;
            outStart += remaining;
            outEnd = outStart;
        }
        double[] result = new double[total];
        int pos = 0;
        for (double[] part : parts) {
            System.arraycopy(part, 0, result, pos, part.length);
            pos += part.length;
        }
        return result;
    }

    @Override
    public int partitionSize() {
        return blockSize;
    }

    @Override
    public int partitions() {
        return totalPartitions;
    }

}
