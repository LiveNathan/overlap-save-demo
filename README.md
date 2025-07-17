# Overlap Save Method for Frequency Domain Convolution

A Java implementation demonstrating the overlap-save method for real-time audio convolution, progressing from simple time domain approaches through frequency domain optimizations.

## Overview

This project provides working Java implementations of several convolution methods:

- **Apache Commons Math** - Reference implementation
- **Time Domain** - Traditional O(N²) approach
- **Frequency Domain** - O(N log N) using FFT
- **Overlap Save** - Block-based frequency domain for real-time processing
- **Vector API** - SIMD-optimized time domain convolution

All implementations share a common interface and are tested for correctness and performance against each other.

## Why This Exists

When building [GainGuardian](https://www.gainguardian.com/), I needed to switch from time domain to frequency domain convolution for performance reasons. While there are many MATLAB and Python examples of overlap-save, I found few readable Java implementations that explain the progression from basic concepts to real-time processing.

This repository fills that gap with:
- Working code you can run and modify
- Comprehensive test coverage ensuring correctness
- Performance benchmarks showing when each method excels
- Clear progression from simple to advanced techniques

## Quick Start

### Requirements
- Java 24 with preview features enabled
- Maven 3.9+

### Running the Tests

```bash
mvn test
```

This runs all convolution implementations against the same test cases, verifying correctness and measuring performance.

### Key Performance Results

For large signal/small kernel scenarios (1024 samples signal, 3 samples kernel):
- **Overlap Save**: 1.49ms (baseline)
- **Time Domain**: 6.75ms (4.5x slower)
- **Frequency Domain**: 11.98ms (8.0x slower)
- **Apache Commons**: 39.52ms (26.5x slower)

## Implementations

### TimeDomainAdapter
Direct implementation of convolution as sliding dot products. Simple to understand but O(N²) complexity.

### FrequencyDomainAdapter
Uses FFT/IFFT with the convolution theorem. Efficient for offline processing but requires entire signal.

### OverlapSaveAdapter
Block-based frequency domain processing enabling real-time convolution with consistent low latency.

### VectorApiAdapter
SIMD-optimized time domain implementation using Java's Vector API for hardware acceleration.

## Project Structure

```
src/main/java/dev/nathanlively/overlap_save_demo/
├── Convolution.java              # Common interface
├── ApacheAdapter.java            # Reference implementation
├── TimeDomainAdapter.java        # Traditional approach
├── FrequencyDomainAdapter.java   # Basic FFT approach
├── OverlapSaveAdapter.java       # Real-time block processing
├── VectorApiAdapter.java         # SIMD optimization
├── SignalTransformer.java        # FFT utilities
├── WavFileReader.java            # Audio file I/O
└── WavFileWriter.java
```

## Usage Example

```java
// Choose your implementation
Convolution convolution = new OverlapSaveAdapter();

// Load audio (or create test signals)
double[] signal = {1.0, 2.0, 3.0, 4.0, 5.0};
double[] kernel = {0.5, 0.25, 0.125};

// Perform convolution
double[] result = convolution.with(signal, kernel);
```

## When to Use Each Method

**Time Domain**: Small kernels (< 64 samples), educational purposes, sample-by-sample control

**Frequency Domain**: Medium kernels (64-1024 samples), offline processing, one-shot convolutions

**Overlap Save**: Real-time processing, long kernels (> 64 samples), streaming applications, best overall performance

**Vector API**: Time domain with hardware acceleration, when SIMD is available

## Audio Processing Example

The project includes WAV file processing capabilities for real-world testing:

```java
WavFileReader reader = new WavFileReader();
WavFile audio = reader.loadFromClasspath("sample.wav");

// Apply convolution (e.g., reverb impulse response)
Convolution convolution = new OverlapSaveAdapter();
double[] processed = convolution.with(audio.signal(), reverbKernel);

// Save result
WavFileWriter writer = new WavFileWriter();
writer.saveToFile(new WavFile(audio.sampleRate(), processed), outputPath);
```

## Learn More

This code accompanies the blog post: [Overlap Save Method for Frequency Domain Convolution: A Developer's Guide](https://open.substack.com/pub/nathanlively/p/overlap-save-frequency-domain-convolutionhtml)

The post explains the theory behind each implementation and walks through the step-by-step refactoring process.

## Building and Testing

```bash
# Compile with preview features
mvn compile

# Run all tests including performance benchmarks  
mvn test

# Run specific test class
mvn test -Dtest=ConvolutionTest

# Run performance comparison
mvn test -Dtest=VectorApiPerformanceTest
```

## License

MIT License - see LICENSE file for details.

## Contributing

Issues and pull requests welcome! This project aims to be educational, so clarity and understandability are prioritized over micro-optimizations.