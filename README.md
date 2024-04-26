# pyfing - Fingerprint recognition in Python

pyfing provides simple but effective methods for fingerprint recognition.

## Fingerprint segmentation methods
The following segmentation methods are available:
- GMFS (Gradient-Magnitude Fingerprint Segmentation): straightforward yet effective approach, achieving superior performance compared to previously reported traditional (non deep-learning-based) methods. Its average error rate on the FVC segmentation benchmark \[1\] is 2.98%.
- SUFS (Simplified U-net Fingerprint Segmentation): method based on a simplified U-net architecture that surpasses all previous methods evaluated on the FVC segmentation benchmark \[1\]. Its average error rate on the benchmark is 1.51%.

See \[2\] for a complete description of the two segmentation methods.


## Fingerprint orientation field estimation methods
The following orientation field estimation methods are available:
- GBFOE (Gradient-Based Fingerprint Orientation Estimation): A simple fingerprint orientation estimation method based on traditional image processing techniques with minimal computational resource requirements that achieves performance comparable to much more complex methods. It achieves an average error of 5.30° and 14.40°	on the "good" and "bad" datasets of the FOE-STD-1.0 benchmark \[3\], respectively.
- SNFOE (Simple Network for Fingerprint Orientation Estimation): learning-based fingerprint orientation estimation method that surpasses all previous methods evaluated on public benchmarks and can deal both with plain fingerprints acquired through online sensors, and with latent fingerprints, without requiring any fine-tuning. It achieves an average error of 4.30° and 6.37°	on the "good" and "bad" datasets of the FOE-STD-1.0 benchmark \[3\], respectively.

See \[4\] for a complete description of the two orientation field estimation methods.


## References
\[1\] D. H. Thai, S. Huckemann and C. Gottschlich, "Filter Design and Performance Evaluation for Fingerprint Image Segmentation," PLOS ONE, vol. 11, pp. 1-31, May 2016.

\[2\] R. Cappelli, "Unveiling the Power of Simplicity: Two Remarkably Effective Methods for Fingerprint Segmentation," in IEEE Access, vol. 11, pp. 144530-144544, 2023, [doi: 10.1109/ACCESS.2023.3345644](https://doi.org/10.1109/ACCESS.2023.3345644).

\[3\] FOE benchmarks on FVC-onGoing, https://biolab.csr.unibo.it/fvcongoing/UI/Form/BenchmarkAreas/BenchmarkAreaFOE.aspx

\[4\] R. Cappelli, "Exploring the Power of Simplicity: A New State-of-the-Art in Fingerprint Orientation Field Estimation," in IEEE Access, vol. 12, pp. 55998-56018, 2024, [doi: 10.1109/ACCESS.2024.3389701](https://doi.org/10.1109/ACCESS.2024.3389701).


