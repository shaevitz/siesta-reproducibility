# SLEAP-NN configurations

`16fly_pose_sleap_nn.json` and `16fly_finder_sleap_nn.yaml` preserve the SLEAP-NN model, preprocessing, augmentation, optimizer and seed settings used for the 16-fly experiment. Empty label paths and output directories are intentional: supply local paths for a newly generated `.slp` package and a new output directory rather than editing a frozen experimental artifact.

`16fly_fixed_lr_schedule.jsonl` records the constant learning-rate schedule used by the archived training wrapper. These files are configuration references for SLEAP-NN 0.1-era training code; exact compatibility may require the dependency versions recorded by the accompanying run environment.
