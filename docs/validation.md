# Extraction validation

This is a source extraction of the RoboTwin V4 pipeline, not a new training experiment.

- **Unit tests:** 6 passed; one test requiring a local prepared dataset skips in a clean checkout.
- **Model parity:** with a deterministic reduced test backbone, original and extracted implementations have exactly equal initial state tensors, Stage-I loss, policy loss and sampled action chunks. See `parity_validation.json`.
- **Full checkpoint structure:** all original 20K checkpoint tensor names and shapes match the extracted full model.
- **Data reproduction:** audited all 1,000 original candidate episodes; 978 passed single-arm selection. Prepared manifest SHA256 and state/action min/max match the original experiment exactly. 968 episodes / 128,147 frames remain after joint-outlier and stationary-frame filtering. See `data_validation.json`.
- **Real-data training smoke:** CPU, actual DINO weights and actual HDF5, batch 1. Ran one Stage-I update, resumed for one update with both selectors frozen, then transferred into Stage II and ran one policy update. All 154 selector state tensors remained unchanged across frozen continuation; all 418 backbone/predictor tensors remained exactly equal after Stage-II training.
- **Entry points:** Python compilation, shell syntax and training/evaluation/visualization help commands passed.

The full 20K/100K experiment, multi-GPU training, downloading into a fresh environment and simulator rollouts were not repeated from this extracted package. Included videos come from recorded original successful rollouts. Dependencies used for validation are listed in `tested_versions.txt`; use a PyTorch build suitable for your CUDA installation. Check `validation.json` for machine-readable results.
