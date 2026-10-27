# DAM-VIS — Occlusion-Aware Video Instance Segmentation

Dual-threshold mask-aware association + flow-guided mask propagation.
PyTorch implementation of the DAM-VIS paper draft.

## File structure
    DAM-VIS/
    ├── README.md
    ├── requirements.txt
    ├── configs/
    │   └── mot17.yaml              # all hyperparameters (tune on validation)
    ├── damvis/
    │   ├── __init__.py             # load_config() -> nested namespace
    │   ├── losses.py               # batch-hard triplet loss (ReID training)
    │   ├── tracker.py              # DAMVISTracker main pipeline
    │   ├── track.py                # track lifecycle + embedding gallery
    │   ├── association.py          # joint cost (box/mask IoU + appearance)
    │   ├── kalman.py               # Kalman filter + Mahalanobis gating
    │   └── models/
    │       ├── detector.py         # Mask R-CNN (boxes + masks)
    │       ├── reid.py             # 128-D appearance embedding CNN
    │       └── flow.py             # RAFT optical flow + mask warping
    └── scripts/
        ├── train.py                # ReID training loop (MOT-format gt)
        ├── track.py                # run on a MOT17 sequence -> results.txt
        ├── demo.py                 # annotated video / webcam demo
        └── analyze_tracks.py       # track statistics + plots

## Quick start
    pip install -r requirements.txt

    # 1. train the ReID head (recommended)
    python scripts/train.py --data /path/to/MOT17/train --out reid.pt

    # 2. demo on a video
    python scripts/demo.py --video input.mp4 --out out.mp4 --reid reid.pt

    # 3. evaluate on a MOT17 sequence
    python scripts/track.py --seq /path/to/MOT17-02-FRCNN --out res.txt --reid reid.pt
    python -m motmetrics.apps.eval_mot /path/to/MOT17-02-FRCNN/gt/gt.txt res.txt

    # 4. analysis plots
    python scripts/analyze_tracks.py --results res.txt

## Notes
- First run downloads pretrained Mask R-CNN + RAFT weights (torchvision).
- Thresholds in configs/mot17.yaml are INITIAL settings from the paper
  draft — tune them on a validation split before reporting numbers.
