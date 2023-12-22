# pyfing - Fingerprint recognition in Python

pyfing provides simple but effective methods for fingerprint recognition.

## Fingerprint segmentation methods
The following segmentation methods are available:
- GMFS (Gradient-Magnitude Fingerprint Segmentation): straightforward yet effective approach, achieving superior performance compared to previously reported traditional (non deep-learning-based) methods. Its average error rate on the FVC segmentation benchmark [1] is 2.98%.
- SUFS (Simplified U-net Fingerprint Segmentation): method based on a simplified U-net architecture that surpasses all previous methods evaluated on the FVC segmentation benchmark [1]. Its average error rate on the benchmark is 1.51%.

See [2] for a complete description of the two segmentation methods.

## References
[1] D. H. Thai, S. Huckemann and C. Gottschlich, "Filter Design and Performance Evaluation for Fingerprint Image Segmentation," PLOS ONE, vol. 11, pp. 1-31, May 2016.

[2] R. Cappelli, "Unveiling the Power of Simplicity: Two Remarkably Effective Methods for Fingerprint Segmentation," in IEEE Access, [doi: 10.1109/ACCESS.2023.3345644](https://doi.org/10.1109/ACCESS.2023.3345644).
