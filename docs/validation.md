# Validation

- Unit tests cover min/max normalization, stationary-frame filtering, rectangular spatial attention, the two-stage model, B-spline smoothing and inactive-arm commands.
- A deterministic reduced-backbone check confirmed identical initialization, losses and sampled actions against the development implementation. See `parity_validation.json`.
- Data preparation was checked on 1,000 candidate episodes: 978 passed single-arm selection and 968 remained after joint-outlier filtering, totaling 128,147 processed frames. See `data_validation.json`.
- A CPU smoke test used real DINO weights and HDF5 data with batch 1: one Stage-I update, one frozen-selector continuation update and one Stage-II update. The 154 selector tensors stayed fixed during continuation; 418 backbone/predictor tensors stayed fixed during Stage II.

These checks do not replace full distributed training or a simulator evaluation. A fresh environment installation and full simulator rollouts have not been rerun for this package. Example videos are recorded successful rollouts. Tested environment versions are listed in `tested_versions.txt`.
