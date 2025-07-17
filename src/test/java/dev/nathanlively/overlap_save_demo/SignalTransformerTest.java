package dev.nathanlively.overlap_save_demo;

import org.apache.commons.numbers.complex.Complex;
import org.junit.jupiter.api.Test;

import java.util.Comparator;

import static org.assertj.core.api.Assertions.assertThat;

class SignalTransformerTest {
    private static Comparator<Double> doubleComparator() {
        return (a, b) -> Math.abs(a - b) < 1.0E-15 ? 0 : Double.compare(a, b);
    }

    @Test
    void transform_computesFFTForPowerOfTwoSignal() {
        double[] signal = {1, 2};

        Complex[] transform = SignalTransformer.fft(signal);

        assertThat(transform).hasSize(2);
        // FFT of [1, 2] should be [3+0i, -1+0i]
        assertThat(transform[0].getReal()).isEqualTo(3.0);
        assertThat(transform[0].getImaginary()).isEqualTo(0.0);
        assertThat(transform[1].getReal()).isEqualTo(-1.0);
        assertThat(transform[1].getImaginary()).isEqualTo(0.0);
    }

    @Test
    void transformRoundTrip_preservesOriginalSignal() {
        double[] original = {1, 2, 3, 4};

        Complex[] transformed = SignalTransformer.fft(SignalTransformer.pad(original, 8));
        double[] roundTrip = SignalTransformer.ifft(transformed);

        assertThat(roundTrip).usingElementComparator(doubleComparator())
                .containsExactly(1, 2, 3, 4, 0, 0, 0, 0);
    }

    @Test
    void preparePaddedSignal_addsCorrectPadding() {
        double[] signal = {1, 2};
        int kernelLength = 3;

        double[] paddedSignal = SignalTransformer.padSymmetric(signal, kernelLength - 1);

        assertThat(paddedSignal).containsExactly(0, 0, 1, 2, 0, 0);
    }
}