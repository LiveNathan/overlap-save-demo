% generate_fft_golden.m
% Golden FFT vectors for HandRolledFft verification (see MatlabGoldenTest).
%
% Run from the repo root in MATLAB:
%   >> addpath('matlab'); generate_fft_golden
%
% Writes src/test/resources/matlab-golden.csv with columns:
%   case, n, k, in_re, in_im, fft_re, fft_im, ifft_re, ifft_im
% where fft_* = fft(input) and ifft_* = ifft(fft(input)).
% MATLAB's fft is the unscaled forward DFT and ifft scales by 1/N -
% the same convention as Apache STD and HandRolledFft.
% Inputs are deterministic (rng(42)); sizes 256, 512, 1024.
% Values are written with %.17g so the CSV round-trips doubles exactly.

rng(42);
sizes = [256, 512, 1024];
outPath = fullfile('src', 'test', 'resources', 'matlab-golden.csv');
fid = fopen(outPath, 'w');
fprintf(fid, 'case,n,k,in_re,in_im,fft_re,fft_im,ifft_re,ifft_im\n');
for n = sizes
    % Real input: exercises forwardReal / inverseReal as well as complex.
    x = randn(n, 1);
    X = fft(x);
    xr = ifft(X);
    for k = 1:n
        fprintf(fid, 'real,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n', ...
            n, k-1, x(k), 0, real(X(k)), imag(X(k)), real(xr(k)), imag(xr(k)));
    end
    % Complex input: exercises the full complex forward / inverse.
    z = randn(n, 1) + 1i * randn(n, 1);
    Z = fft(z);
    zr = ifft(Z);
    for k = 1:n
        fprintf(fid, 'complex,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n', ...
            n, k-1, real(z(k)), imag(z(k)), real(Z(k)), imag(Z(k)), real(zr(k)), imag(zr(k)));
    end
end
fclose(fid);
fprintf('Wrote %s\n', outPath);
