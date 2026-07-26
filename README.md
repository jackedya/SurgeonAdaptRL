# SurgeonAdaptRL

SurgeonAdaptRL models surgeon-specific cataract surgery motion from monocular video with a shared transformer policy, lightweight residual adapters, phase-conditioned motion primitives, and simulation-constrained policy refinement. The implementation separates video learning from AMBF safety validation: normalized two-dimensional trajectories are learned from video, while velocity, contact force, and anatomical clearance are measured only inside simulation.

## Environment

The reference environment uses Ubuntu 20.04, CUDA 11.7, cuDNN 8, Python 3.10.12, PyTorch 2.0.1, and torchvision 0.15.2. Training requires four NVIDIA A100 GPUs with 80 GB of memory each. The dependency versions are intentionally fixed because the reference results were produced with this stack.

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate surgeonadaptrl
pip install --no-deps .
```

Build the container:

```bash
docker build -t surgeonadaptrl:cuda117 .
```

## Data

All verified resource locations and their access conditions are listed in `datasets.txt`. CATARACTS uses the official 25-video training and 25-video test division. The training videos contain three surgeons. CaDIS contributes 4,670 segmentation masks from 25 CATARACTS training videos and is used only for segmentation-guided trajectory extraction. It is not an independent generalization set. The Cataract-1K evaluation uses the 56 fully phase-annotated videos from nine surgeons.

Prepare a CSV manifest with these columns:

```text
image,video,surgeon,frame,phase,x,y
```

Coordinates `x` and `y` must be instrument-tip locations normalized to `[0, 1]`. Frames must remain in temporal order. The sequence loader uses 30 observed frames, stride 5, and a 15-frame prediction horizon. Instrument masks are obtained with Mask R-CNN trained on 3,500 annotated CATARACTS frames using a 70/15/15 division, followed by DeepSORT association.

## Training

The primary run uses batch size 256 across four GPUs, 200 full supervised epochs, AdamW with learning rate `3e-4` and weight decay `0.01`, 50,000 meta-optimization iterations, 200 surgeon adaptation updates with inner learning rate `0.01`, and 100,000 AMBF refinement iterations. Five seeds are specified in `configs/main.yaml`.

```bash
torchrun --standalone --nproc_per_node=4 -m surgeonadaptrl.cli --config configs/main.yaml --manifest data/cataracts_train.csv --seed 13
```

The shared policy contains a ResNet-50 ImageNet visual encoder, six transformer encoder layers at width 512 with eight attention heads, four causal decoder layers, and 64-dimensional surgeon adapters. Each axis is represented by 256 displacement bins spanning `[-0.05, 0.05]`. The phase classifier is an eight-layer temporal convolutional network with kernel size five. Motion primitives use a 32-dimensional variational latent space, regularization weight `0.1`, and phase-wise Gaussian mixtures selected by BIC from four through twelve components.

## Experiment matrix

The main CATARACTS setting is in `configs/main.yaml`. Separate files cover the seven Table 7 removals, the Cataract-1K evaluation, and the CaDIS extraction setting. Each removal file declares exactly which scientific component is disabled and inherits all other values from the primary configuration.

The reference CATARACTS target is normalized trajectory error `0.0289 ± 0.0018`, style fidelity `81.4 ± 1.4%`, phase accuracy `94.7 ± 0.8%`, and surgeon adaptation time `7.5 ± 0.9 minutes`. The Cataract-1K target is normalized trajectory error `0.0304 ± 0.0020`, style fidelity `79.6 ± 1.5%`, and adaptation time `8.1 minutes`. Continuous comparisons use paired t-tests, categorical comparisons use McNemar tests, and all multiple comparisons use Bonferroni correction.

## Safety validation

AMBF supplies three-dimensional positions, contact forces, and clearances. The video policy is not interpreted as metric three-dimensional motion. A static pinhole projection maps predicted image trajectories into the simulator. Primal-dual policy optimization constrains end-effector velocity to `15 mm/s`, contact force to `30 mN`, and anatomical clearance to at least `0.2 mm`. Policy and dual learning rates are `3e-4` and `1e-3`. The reference overall simulated violation rate is `2.8%`. This number is a simulation result and does not establish real-world robotic safety.

## Compute budget

The reference primary run consumes 78.4 GPU-hours on four NVIDIA A100 80 GB GPUs. Each surgeon adaptation takes approximately 7.5 minutes. The required aggregate accelerator memory is 320 GB. Dataset storage is additional and depends on the provider archives. The configuration does not include reduced-compute settings.

## Scope

The learned policy predicts normalized two-dimensional instrument motion. It does not consume joint angles, proprioception, calibrated depth, or three-dimensional kinematics during video learning. Simulator measurements are used only during the separate safety-refinement stage. Clinical use and direct robot control are outside the scope of this repository.
