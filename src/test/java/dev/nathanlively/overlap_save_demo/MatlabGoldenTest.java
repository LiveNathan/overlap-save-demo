package dev.nathanlively.overlap_save_demo;

import org.junit.jupiter.api.Test;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * {@link HandRolledFft} against MATLAB golden vectors (FFTW-backed
 * {@code fft} / {@code ifft}). The vectors live in
 * {@code src/test/resources/matlab-golden.csv}; generate them by running
 * {@code matlab/generate_fft_golden.m} in MATLAB from the repo root.
 * The test skips cleanly while the file is absent.
 *
 * <p>MATLAB's {@code fft} is the unscaled forward DFT and {@code ifft}
 * scales by 1/N — the same convention as Apache STD — so the same 1e-9
 * tolerance as the Apache comparison applies. This is a third,
 * independent implementation of the same definition: it guards against
 * a shared misunderstanding of the convention, not against anything the
 * naive-DFT comparison has not already settled.
 */
class MatlabGoldenTest {
    private static final double TOLERANCE = 1e-9;

    private record Golden(String kind, int n, double[] inRe, double[] inIm,
                          double[] fftRe, double[] fftIm, double[] ifftRe, double[] ifftIm) {
    }

    @Test
    void handRolledFft_agreesWithMatlab() throws Exception {
        InputStream in = getClass().getResourceAsStream("/matlab-golden.csv");
        assumeTrue(in != null, "matlab-golden.csv absent — run matlab/generate_fft_golden.m in MATLAB first");
        List<Golden> goldens = readGoldens(in);
        assertThat(goldens).isNotEmpty();
        HandRolledFft fft = new HandRolledFft();
        for (Golden g : goldens) {
            int n = g.n();
            double[] re = g.inRe().clone();
            double[] im = g.inIm().clone();
            fft.forward(re, im);
            assertThat(maxErr(re, g.fftRe())).as("%s forward re n=%d", g.kind(), n).isLessThan(TOLERANCE);
            assertThat(maxErr(im, g.fftIm())).as("%s forward im n=%d", g.kind(), n).isLessThan(TOLERANCE);
            fft.inverse(re, im);
            assertThat(maxErr(re, g.ifftRe())).as("%s inverse re n=%d", g.kind(), n).isLessThan(TOLERANCE);
            assertThat(maxErr(im, g.ifftIm())).as("%s inverse im n=%d", g.kind(), n).isLessThan(TOLERANCE);

            if (g.kind().equals("real")) {
                double[] frRe = new double[n];
                double[] frIm = new double[n];
                fft.forwardReal(g.inRe(), frRe, frIm);
                assertThat(maxErr(frRe, g.fftRe())).as("real forwardReal re n=%d", n).isLessThan(TOLERANCE);
                assertThat(maxErr(frIm, g.fftIm())).as("real forwardReal im n=%d", n).isLessThan(TOLERANCE);
                double[] back = new double[n];
                fft.inverseReal(frRe, frIm, back);
                assertThat(maxErr(back, g.ifftRe())).as("real inverseReal n=%d", n).isLessThan(TOLERANCE);
            }
        }
    }

    private static List<Golden> readGoldens(InputStream in) throws Exception {
        Map<String, List<String[]>> rows = new LinkedHashMap<>();
        try (BufferedReader r = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) {
            r.readLine(); // header
            String line;
            while ((line = r.readLine()) != null) {
                String[] c = line.split(",");
                rows.computeIfAbsent(c[0] + ":" + c[1], k -> new ArrayList<>()).add(c);
            }
        }
        List<Golden> out = new ArrayList<>();
        for (List<String[]> rs : rows.values()) {
            int n = Integer.parseInt(rs.get(0)[1]);
            double[] inRe = new double[n], inIm = new double[n];
            double[] fftRe = new double[n], fftIm = new double[n];
            double[] ifftRe = new double[n], ifftIm = new double[n];
            for (String[] c : rs) {
                int k = Integer.parseInt(c[2]);
                inRe[k] = Double.parseDouble(c[3]);
                inIm[k] = Double.parseDouble(c[4]);
                fftRe[k] = Double.parseDouble(c[5]);
                fftIm[k] = Double.parseDouble(c[6]);
                ifftRe[k] = Double.parseDouble(c[7]);
                ifftIm[k] = Double.parseDouble(c[8]);
            }
            out.add(new Golden(rs.get(0)[0], n, inRe, inIm, fftRe, fftIm, ifftRe, ifftIm));
        }
        return out;
    }

    private static double maxErr(double[] a, double[] b) {
        double m = 0.0;
        for (int i = 0; i < a.length; i++) {
            m = Math.max(m, Math.abs(a[i] - b[i]));
        }
        return m;
    }
}
