# pyfing - Fingerprint recognition in Python

Simple but effective methods for fingerprint recognition.


## Fingerprint segmentation
The following segmentation methods are available:
- GMFS (Gradient-Magnitude Fingerprint Segmentation): straightforward yet effective approach, achieving superior performance compared to previously reported traditional (non deep-learning-based) methods. Its average error rate on the FVC segmentation benchmark \[1\] is 2.98%.
- SUFS (Simplified U-net Fingerprint Segmentation): method based on a simplified U-net architecture that surpasses all previous methods evaluated on the FVC segmentation benchmark \[1\]. Its average error rate on the benchmark is 1.51%.

See \[2\] for a complete description of the two segmentation methods.


## Fingerprint orientation field estimation
The following orientation field estimation methods are available:
- GBFOE (Gradient-Based Fingerprint Orientation Estimation): a simple fingerprint orientation estimation method based on traditional image processing techniques with minimal computational resource requirements that achieves performance comparable to much more complex methods. It achieves an average error of 5.30° and 14.40° on the "good" and "bad" datasets of the FOE-STD-1.0 benchmark \[3\], respectively.
- SNFOE (Simple Network for Fingerprint Orientation Estimation): learning-based fingerprint orientation estimation method that surpasses all previous methods evaluated on public benchmarks and can deal both with plain fingerprints acquired through online sensors, and with latent fingerprints, without requiring any fine-tuning. It achieves an average error of 4.30° and 6.37° on the "good" and "bad" datasets of the FOE-STD-1.0 benchmark \[3\], respectively.

See \[4\] for a complete description of the two orientation field estimation methods.


## Fingerprint frequency estimation
The following frequency estimation methods are available:
- XSFFE (X-Signature-based Fingerprint Frequency Estimation): a simple fingerprint frequency estimation method based on traditional image processing techniques and on the computation of the x-signature. It achieves an average error of 5.10% and 7.84%	on the "good" and "bad" datasets of the FFE benchmark \[5\], respectively.
- SNFFE (Simple Network for Fingerprint Frequency Estimation): learning-based fingerprint frequency estimation method that surpasses all previous methods evaluated on the FFE benchmark. It achieves an average error of 3.62% and 6.69%	on the "good" and "bad" datasets of the FFE benchmark \[5\], respectively.

See \[6\] for a complete description of the two frequency estimation methods.


## Fingerprint enhancement
The following enhancement methods are available:
- GBFEN (Gabor-Based Fingerprint ENhancement): a simple fingerprint enhancement method based on traditional contextual convolution with Gabor filters. Combined with SNFOE and SNFFE, it surpasses the performance of state-of-the-art methods on the challenging NIST SD27 latent fingerprint database \[7]\.
- SNFEN (Simple Network for Fingerprint ENhancement): learning-based fingerprint enhancement method that achieves even better results while maintaining simplicity,
linearity, and ease of implementation.

See \[8\] for a complete description of the two fingerprint enhancement methods.


## End-to-end minutiae extraction
The following end-to-end minutia extraction method is available:
- LEADER (Lightweight End-to-end Attention-gated Dual autoencodER): a neural network that maps raw fingerprint images to minutiae descriptors, including location, direction, and type. With only 0.9 M parameters, it achieves state-of-the-art accuracy on plain fingerprints and robust cross-domain generalization to latent impressions.

See \[9\] for a complete description of LEADER.


## References
\[1\] D. H. Thai, S. Huckemann and C. Gottschlich, "Filter Design and Performance Evaluation for Fingerprint Image Segmentation," PLOS ONE, vol. 11, pp. 1-31, May 2016.

\[2\] R. Cappelli, "Unveiling the Power of Simplicity: Two Remarkably Effective Methods for Fingerprint Segmentation," in IEEE Access, vol. 11, pp. 144530-144544, 2023, [doi: 10.1109/ACCESS.2023.3345644](https://doi.org/10.1109/ACCESS.2023.3345644).

\[3\] FOE benchmarks on FVC-onGoing, https://biolab.csr.unibo.it/fvcongoing/UI/Form/BenchmarkAreas/BenchmarkAreaFOE.aspx

\[4\] R. Cappelli, "Exploring the Power of Simplicity: A New State-of-the-Art in Fingerprint Orientation Field Estimation," in IEEE Access, vol. 12, pp. 55998-56018, 2024, [doi: 10.1109/ACCESS.2024.3389701](https://doi.org/10.1109/ACCESS.2024.3389701).

\[5\] FFE Benchmark, https://github.com/raffaele-cappelli/FFE

\[6\] R. Cappelli, "No Feature Left Behind: Filling the Gap in Fingerprint Frequency Estimation," in IEEE Access, vol. 12, pp. 153605-153617, 2024, doi: [10.1109/ACCESS.2024.3481507](https://doi.org/10.1109/ACCESS.2024.3481507).

\[7\] M. D. Garris and R. M. McCabe, NIST Special Database 27: Fingerprint Minutiae from Latent and Matching Tenprint Images, 2000.

\[8\] (Paper under review)

\[9\] (Paper under review)

