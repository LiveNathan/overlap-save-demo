package dev.nathanlively.overlap_save_demo;

/**
 * {@link NonUniformPartitionedConvolutionAdapter} with the hand-rolled
 * real FFT in every stage: the Wefers layouts re-run under the settled
 * FFT, so the layout — not the FFT — is the variable against uniform
 * RealFft. Default layout is Wefers 128^4 512^15 at B=256.
 */
public class RealFftNonUniformPartitionedConvolutionAdapter extends NonUniformPartitionedConvolutionAdapter {
    public RealFftNonUniformPartitionedConvolutionAdapter() {
        super();
    }

    public RealFftNonUniformPartitionedConvolutionAdapter(int blockSize, int[][] groups) {
        super(blockSize, groups);
    }

    @Override
    protected StreamingConvolver newConvolver(double[] kernel) {
        return new NonUniformPartitionedConvolver(kernel, blockSize(), groups(), RealFftSoAPartitionedConvolver::new);
    }
}
