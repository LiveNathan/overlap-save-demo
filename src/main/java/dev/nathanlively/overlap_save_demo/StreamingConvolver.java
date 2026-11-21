package dev.nathanlively.overlap_save_demo;

/**
 * A streaming convolver: feed fixed-size blocks, drain the tail when the
 * stream ends. Implemented by the partitioned convolvers, one per multiply
 * representation; the adapters drive them whole-signal, and the performance
 * test drives each through this interface for the per-block deadline
 * measurement.
 */
public interface StreamingConvolver {
    double[] processBlock(double[] block);

    double[] flush();

    int partitionSize();

    int partitions();
}
