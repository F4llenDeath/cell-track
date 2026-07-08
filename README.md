# Cell Track

This repository contains a notebook-based pipeline for segmenting and tracking cells in 2D darkfield time-lapse microscopy movies acquired on a patterned hydrogel background.

The central difficulty is that cell signal is mixed with a strong structured background pattern that can drift slightly over time. The pipeline therefore combines illumination correction, pattern alignment, low-rank/sparse decomposition, foundation-model segmentation, and global tracking.

## Pipeline summary

The workflow is split across two notebooks:

- `basicpy.ipynb` — BaSiCPy illumination correction.
- `main.ipynb` — ECC alignment, robust PCA background removal, Cellpose-SAM segmentation, and Ultrack tracking.

Conceptually:

```text
raw time-lapse TIFF
    ↓
BaSiCPy illumination correction
    ↓
ECC alignment of the hydrogel pattern
    ↓
Robust PCA: low-rank pattern + sparse cell signal
    ↓
Cellpose-SAM segmentation
    ↓
Ultrack global tracking
    ↓
tracks_df.csv + tracked_labels.tif
```

## Method overview

### Illumination correction with BaSiCPy

The first preprocessing stage uses BaSiCPy <sup>1</sup> to correct smooth illumination artifacts such as shading, vignetting, lamp drift, and frame-to-frame baseline variation. This is run in a separate `basicpy` environment because BaSiCPy dependencies are easier to isolate from the main `ultrack` environment.

BaSiCPy reduces low-frequency illumination effects, but it is not intended to remove the sharp, spatially structured hydrogel pattern. That remaining structured background is handled by the ECC + robust PCA stages.

### ECC alignment of the hydrogel pattern

Robust PCA works best when the background is spatially consistent across frames. In this data, the hydrogel pattern can translate or rotate slightly over time, so each BaSiCPy-corrected frame is aligned to a temporal-median template using Enhanced Correlation Coefficient (ECC) image registration <sup>2</sup>.

The current notebook uses a Euclidean motion model:

```python
ECC_MOTION = cv2.MOTION_EUCLIDEAN
```

This allows translation and rotation. The estimated transforms align the movie before robust PCA, and the extracted sparse cell signal is then transformed back to the original movie coordinates.

### Robust PCA for pattern/cell separation <sup>3, 4</sup>

After alignment, the movie is reshaped into a matrix:

$$ M ∈ R^{pixels × frames} $$

Robust PCA models the movie as:

$$ M = L + S $$

where:

- `L` is a low-rank component representing the recurring hydrogel/background pattern.
- `S` is a sparse component representing transient foreground signal, primarily cells.

This decomposition is appropriate here because the aligned hydrogel pattern is present in every frame and therefore highly correlated over time, whereas cells occupy a relatively small fraction of pixels and change position/morphology across frames.

The implementation uses an inexact augmented Lagrange multiplier approach. The main tuning parameter is `RPCA_LAMBDA_MULTIPLIER`, which controls the penalty on the sparse component. Larger values can suppress residual pattern more strongly, but may also remove faint cell signal if set too high.

### Cellpose-SAM segmentation

The background-subtracted sparse component is normalized frame-by-frame and segmented with Cellpose-SAM <sup>5</sup> through the Cellpose v4 API. The default model is `cpsam_v2`, with `cpsam` as a fallback.

Cellpose-SAM is a good fit for this data because the cells can undergo large morphology changes, from rounded to spread or star-like shapes. The pipeline does not rely on a star-convex or fixed-shape assumption.

The segmentation output is saved as: `outputs/cellpose_labels.tif`

Large full-stack napari label display is avoided by default because it can freeze the VS Code/Jupyter Qt event loop on macOS.

### Ultrack tracking

Cellpose labels are independent in each frame; label IDs are not temporally consistent. Ultrack <sup>6, 7</sup> converts the per-frame labels into foreground and boundary maps, builds candidate segmentation hypotheses and frame-to-frame links, and solves a global integer linear program to recover consistent tracks.

The main tracking outputs are:

```text
outputs/tracks_df.csv
outputs/tracked_labels.tif
```

- `tracks_df.csv` stores track IDs, time points, and centroid coordinates.
- `tracked_labels.tif` stores label images with temporally consistent track IDs.

If Gurobi is unavailable, Ultrack falls back to the CBC solver. CBC works, but exact optimization can be slow; exploratory runs may benefit from a nonzero solution gap, a time limit, or temporal windowing.

## Generated outputs

Common generated files include:

```text
images/*_basicpy_timelapse.tif
images/*_basicpy_flatfield_only.tif
images/basicpy_diagnostics/
outputs/cellpose_labels.tif
outputs/tracks_df.csv
outputs/tracked_labels.tif
data.db
metadata.toml
```

These files can be large and are usually treated as generated artifacts rather than source files.

## References

1. Peng, T., Thorn, K., Schroeder, T., Wang, L., Theis, F. J., Marr, C., & Navab, N. (2017). A BaSiC tool for background and shading correction of optical microscopy images. *Nature Communications*, 8, 14836. https://doi.org/10.1038/ncomms14836

2. Evangelidis, G. D., & Psarakis, E. Z. (2008). Parametric image alignment using enhanced correlation coefficient maximization. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 30(10), 1858–1865. https://doi.org/10.1109/TPAMI.2008.113

3. Candès, E. J., Li, X., Ma, Y., & Wright, J. (2011). Robust principal component analysis? *Journal of the ACM*, 58(3), 1–37. https://doi.org/10.1145/1970392.1970395

4. Lin, Z., Chen, M., & Ma, Y. (2010). The augmented Lagrange multiplier method for exact recovery of corrupted low-rank matrices. arXiv:1009.5055. https://arxiv.org/abs/1009.5055

5. Pachitariu, M., Rariden, M., & Stringer, C. (2025). Cellpose-SAM: superhuman generalization for cellular segmentation. bioRxiv (Cold Spring Harbor Laboratory). https://doi.org/10.1101/2025.04.28.651001

6. Bragantini, J., Lange, M., & Royer, L. (2024). Large-Scale multi-hypotheses cell tracking using ultrametric contours maps. In Lecture notes in computer science (pp. 36–54). https://doi.org/10.1007/978-3-031-72986-7_3

7. Bragantini, J., Theodoro, I., Zhao, X. et al. Ultrack: pushing the limits of cell tracking across biological scales. Nat Methods 22, 2423–2436 (2025). https://doi.org/10.1038/s41592-025-02778-0