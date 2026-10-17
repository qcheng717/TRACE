# RoboTwin V4 protocol

| Component | Setting |
|---|---|
| Dataset | RoboTwin2.0 demo_clean, revision `3dc3b798668feb99ac61cc9086d84cbcc3d79186` |
| Tasks | 20 tasks in `src/trace_robotwin/data/tasks.py` |
| Selection | First 50 recorded episodes per task; unambiguous single-arm trajectories only |
| RGB | Head only, 320×240, no crop/resize/augmentation |
| Backbone | Frozen DINOv2-base, zero-based layers 2/5/8/11 |
| Input normalization | ImageNet mean/std exactly once inside the backbone |
| Patch padding | Right 2, bottom 12; 18×23 patch grid |
| Language | Frozen all-MiniLM-L6-v2, 384 dimensions; recorded episode instructions |
| Proprioception | Active-arm world xyz + wxyz quaternion + gripper (8D) |
| Actions | Active-arm absolute six joint angles in radians + native gripper (7D) |
| Selector | Per-layer projection, language-query attention, learned layer weights, FiLM, two spatial blocks |
| Skill dictionary | 15 × 100, normalized vectors; per-patch soft assignments |
| Feature prediction | Single endpoint t+32 on the cleaned timeline; out-of-range target is the last episode frame |
| Targets | EMA target selector; target-current and target-future difference, stop-gradient |
| Selector freeze | Both online and target selectors fixed at 10K; target EMA updates stop |
| Stage I | 20K updates total; the LR schedule remains parameterized by 100K updates |
| Stage II | Frozen predictor/backbone, train router and action diffusion decoder for 100K |
| Diffusion | 32-action chunk, 100 noise steps, 10 inference steps; continuous gripper |
| Optimizer | AdamW, lr 5e-5, weight decay 0, gradient clip 1 |
| Schedule | Warmup 1000, cosine, minimum ratio 0.1 |
| Batch | 128 per GPU × 7 GPUs, accumulation 1; BF16; seed 42 |
| Action padding | Repeat final valid action, including repeated entries in training loss |
| Postprocessing | Inverse min/max → B-spline on all 7 dimensions, degree 3, 8 control points → fill inactive arm with its current joints/gripper → qpos/TOPP |

Stationary runs are detected on both arms' 14D joint/gripper actions with maximum absolute adjacent difference <1e-6, margin 1. An episode with any non-gripper action joint beyond ±π is excluded. State and action min/max are computed separately over retained training frames; spans <1e-6 become 1, with no percentile clipping.

The source files already use `state[t+1] == action[t]` for recorded joints; do not shift the labels a second time. Cleaned frame indices apply to all fields. Thus a cleaned t+32 interval may span more than 32 original frames.

The historical prepared set contains 968 episodes and 128,147 frames after filtering (978 single-arm episodes before joint-outlier filtering). Dataset revisions or different accepted episodes may change these counts.

## What this release does not claim

The model is trained on all retained demonstrations. Training loss or an attractive t-SNE plot does not establish held-out generalization. Included successful videos are qualitative examples. Evaluation uses official expert feasibility and arm selection, and filters scenes outside single-arm scope. The newly downloaded `open_laptop` task is not one of the 20 training tasks.
