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
        FrequencyDomainAdapter adapter = new FrequencyDomainAdapter();
        double[] signal = {1, 2};

        Complex[] transform = SignalTransformer.transform(signal);

        assertThat(transform).hasSize(2);
        // FFT of [1, 2] should be [3+0i, -1+0i]
        assertThat(transform[0].getReal()).isEqualTo(3.0);
        assertThat(transform[0].getImaginary()).isEqualTo(0.0);
        assertThat(transform[1].getReal()).isEqualTo(-1.0);
        assertThat(transform[1].getImaginary()).isEqualTo(0.0);
    }

    @Test
    void transformRoundTrip_preservesOriginalSignal() {
        FrequencyDomainAdapter adapter = new FrequencyDomainAdapter();
        double[] original = {1, 2, 3, 4};

        Complex[] transformed = SignalTransformer.transform(SignalTransformer.padArray(original, 8));
        double[] roundTrip = SignalTransformer.inverseTransformRealOnly(transformed);

        assertThat(roundTrip).usingElementComparator(doubleComparator())
                .containsExactly(1, 2, 3, 4, 0, 0, 0, 0);
    }
}