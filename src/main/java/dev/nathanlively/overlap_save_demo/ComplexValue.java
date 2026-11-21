package dev.nathanlively.overlap_save_demo;

value record ComplexValue(double real, double imaginary) {
    ComplexValue multiply(ComplexValue complexValue) {
        return new ComplexValue(real * complexValue.real - imaginary * complexValue.imaginary, real * complexValue.imaginary + imaginary * complexValue.real);
    }
}
