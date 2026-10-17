# Third-party notices

- RoboTwin / XPolicyLab are external dependencies, fetched from `RoboTwin-Platform/RoboTwin` at `ea8b21121ebb3cd201ff5b3fe361944ac94eda3f`. Their code, simulator assets and demonstration data are not redistributed in this source package. The selected rollout videos were rendered in RoboTwin. See the upstream repository's MIT license and individual asset/data terms.
- DINOv2-base and all-MiniLM-L6-v2 weights are downloaded separately at the revisions recorded in `docs/*_source.json`; their upstream model cards and licenses apply.
- Stationary-frame filtering, min/max preprocessing and B-spline smoothing follow LiLa-WAM at `339860f65d5cde9e229c037b672889bc4f70497c`. See `utils/clean_dataset_stationary.py`, `models/model_runner.py`, and `robotwin_infer.py` in that repository. No pretrained LiLa-WAM model or dataset is redistributed here.
- PyTorch, Transformers, Sentence Transformers, NumPy, SciPy, OpenCV, Matplotlib and scikit-learn remain independently licensed dependencies.

The MIT license for this repository does not grant rights to third-party assets or data beyond their original licenses.
