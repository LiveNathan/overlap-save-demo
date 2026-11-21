package dev.nathanlively.overlap_save_demo;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Hand-rolled radix-2 FFT on split-complex arrays, written to find out
 * whether Apache's butterfly core can be beaten — and by how much.
 *
 * <p>Two differences from Apache's {@code FastFourierTransform} core:
 * <ul>
 *   <li>Twiddles are a full precomputed table — one cos/sin pair per
 *       butterfly, loaded, never accumulated by repeated complex
 *       multiplication the way Apache advances its per-stage primitive
 *       root (4 mults + 2 adds per butterfly, on a serial chain).</li>
 *   <li>Real variants: the forward input is real and the inverse output
 *       is real, so each direction runs one half-size complex FFT plus
 *       O(N) packing — roughly half the butterflies of the full complex
 *       transform.</li>
 * </ul>
 * Normalization matches Apache STD: forward unscaled, inverse scaled by
 * 1/N. Twiddle plans are cached per size, per instance; a plan holds N/2 roots.
 *
 * <p>Measured head to head against Apache in {@code FftPerformanceTest};
 * now wired to {@code RealFftSoAPartitionedConvolver} — see
 * {@code PERFORMANCE_RESULTS.md} for the numbers.
 *
 * <p>Instances are safe to share across threads as long as each call uses
 * its own buffers: the only shared state is the plan cache, and plans
 * are immutable once built.
 */
public final class HandRolledFft {
    /** Creates an FFT with an empty plan cache; plans are built lazily per size. */
    public HandRolledFft() {
    }

    private record Plan(double[] wRe, double[] wImForward, double[] wImInverse) {
    }

    private final Map<Integer, Plan> plans = new ConcurrentHashMap<>();

    private Plan planFor(int size) {
        return plans.computeIfAbsent(size, this::buildPlan);
    }

    private Plan buildPlan(int size) {
        if (size < 1 || (size & (size - 1)) != 0) {
            throw new IllegalArgumentException("size must be a power of two >= 1");
        }
        int half = size / 2;
        double[] wRe = new double[half];
        double[] wImForward = new double[half];
        double[] wImInverse = new double[half];
        for (int k = 0; k < half; k++) {
            double angle = 2.0 * Math.PI * k / size;
            wRe[k] = Math.cos(angle);
            wImForward[k] = -Math.sin(angle);
            wImInverse[k] = Math.sin(angle);
        }
        return new Plan(wRe, wImForward, wImInverse);
    }

    /** Complex forward transform, in place. Unscaled, like Apache STD. */
    public void forward(double[] re, double[] im) {
        transform(re, im, false);
    }

    /** Complex inverse transform, in place. Scaled by 1/N, like Apache STD. */
    public void inverse(double[] re, double[] im) {
        transform(re, im, true);
    }

    private void transform(double[] re, double[] im, boolean inverse) {
        int n = re.length;
        if (im.length != n) {
            throw new IllegalArgumentException("real and imaginary parts must have the same length");
        }
        Plan plan = planFor(n);
        bitReverse(re, im);
        double[] wIm = inverse ? plan.wImInverse() : plan.wImForward();
        for (int len = 2; len <= n; len <<= 1) {
            int half = len >> 1;
            int step = n / len;
            for (int i = 0; i < n; i += len) {
                for (int j = 0; j < half; j++) {
                    int wIndex = j * step;
                    double wr = plan.wRe()[wIndex];
                    double wi = wIm[wIndex];
                    int u = i + j;
                    int v = u + half;
                    double vr = re[v] * wr - im[v] * wi;
                    double vi = re[v] * wi + im[v] * wr;
                    double ur = re[u];
                    double ui = im[u];
                    re[u] = ur + vr;
                    im[u] = ui + vi;
                    re[v] = ur - vr;
                    im[v] = ui - vi;
                }
            }
        }
        if (inverse) {
            double scale = 1.0 / n;
            for (int i = 0; i < n; i++) {
                re[i] *= scale;
                im[i] *= scale;
            }
        }
    }

    private void bitReverse(double[] re, double[] im) {
        int n = re.length;
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) {
                j ^= bit;
            }
            j ^= bit;
            if (i < j) {
                double temp = re[i];
                re[i] = re[j];
                re[j] = temp;
                temp = im[i];
                im[i] = im[j];
                im[j] = temp;
            }
        }
    }

    /**
     * Forward transform of N real samples into N split-complex spectrum
     * bins. One N/2 complex FFT plus O(N) packing: pack even/odd samples
     * into one half-size complex signal, transform it, then split the
     * even and odd spectra apart (E = (Z + conj(Z[M-k])) / 2,
     * O = -i (Z - conj(Z[M-k])) / 2) and combine with the twiddle,
     * X[k] = E[k] + W_N^k O[k], X[k+M] = E[k] - W_N^k O[k].
     */
    public void forwardReal(double[] in, double[] outRe, double[] outIm) {
        int n = in.length;
        if (outRe.length != n || outIm.length != n) {
            throw new IllegalArgumentException("input and output must have the same length");
        }
        if (n == 1) {
            outRe[0] = in[0];
            outIm[0] = 0.0;
            return;
        }
        int m = n / 2;
        double[] pr = new double[m];
        double[] pi = new double[m];
        for (int k = 0; k < m; k++) {
            pr[k] = in[2 * k];
            pi[k] = in[2 * k + 1];
        }
        transform(pr, pi, false);
        Plan plan = planFor(n);
        for (int k = 0; k < m; k++) {
            double zkR = pr[k];
            double zkI = pi[k];
            int mk = (m - k) % m;
            // conj(Z[M-k])
            double zmR = pr[mk];
            double zmI = -pi[mk];
            double eR = 0.5 * (zkR + zmR);
            double eI = 0.5 * (zkI + zmI);
            double dR = zkR - zmR;
            double dI = zkI - zmI;
            // O = -i * (Z - conj(Z[M-k])) / 2
            double oR = 0.5 * dI;
            double oI = -0.5 * dR;
            // W_N^k = e^{-2 pi i k / N}
            double wr = plan.wRe()[k];
            double wi = plan.wImForward()[k];
            double woR = wr * oR - wi * oI;
            double woI = wr * oI + wi * oR;
            outRe[k] = eR + woR;
            outIm[k] = eI + woI;
            outRe[k + m] = eR - woR;
            outIm[k + m] = eI - woI;
        }
    }

    /**
     * Inverse transform of an N-bin Hermitian spectrum into N real
     * samples. One N/2 complex inverse FFT plus O(N) unpacking: recover
     * E[k] = (X[k] + X[k+M]) / 2 and O[k] = conj(W_N^k) (X[k] - X[k+M]) / 2,
     * recombine into Z[k] = E[k] + i O[k], inverse-transform it (scaled
     * 1/M), and read the even samples from the real parts and the odd
     * samples from the imaginary parts. Scaled by 1/N overall, matching
     * Apache STD.
     */
    public void inverseReal(double[] inRe, double[] inIm, double[] out) {
        int n = inRe.length;
        if (inIm.length != n || out.length != n) {
            throw new IllegalArgumentException("input and output must have the same length");
        }
        if (n == 1) {
            out[0] = inRe[0];
            return;
        }
        int m = n / 2;
        double[] zr = new double[m];
        double[] zi = new double[m];
        Plan plan = planFor(n);
        for (int k = 0; k < m; k++) {
            double aR = inRe[k];
            double aI = inIm[k];
            double bR = inRe[k + m];
            double bI = inIm[k + m];
            double eR = 0.5 * (aR + bR);
            double eI = 0.5 * (aI + bI);
            double dR = 0.5 * (aR - bR);
            double dI = 0.5 * (aI - bI);
            // O = conj(W_N^k) * (X[k] - X[k+M]) / 2
            double wr = plan.wRe()[k];
            double wi = plan.wImInverse()[k];
            double oR = wr * dR - wi * dI;
            double oI = wr * dI + wi * dR;
            // Z[k] = E[k] + i * O[k]
            zr[k] = eR - oI;
            zi[k] = eI + oR;
        }
        transform(zr, zi, true);
        for (int k = 0; k < m; k++) {
            out[2 * k] = zr[k];
            out[2 * k + 1] = zi[k];
        }
    }
}
