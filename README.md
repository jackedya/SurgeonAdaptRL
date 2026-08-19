# SurgeonAdaptRL

SurgeonAdaptRL models surgeon-specific motion in cataract surgery video. A shared ResNet-50 and transformer policy predicts normalized two-dimensional instrument trajectories; residual adapters specialize the policy for individual surgeons, temporal phase models condition a phase-wise primitive library, and primal-dual optimization evaluates velocity, force, and clearance constraints in AMBF simulation. The vision-language interface described as future integration is outside the evaluated motion-policy core.

## Installation

The reported implementation uses PyTorch 2.0. The environment files pin a compatible Python 3.10, PyTorch 2.0.1, torchvision 0.15.2, CUDA 11.7 stack.

```bash
conda env create -f environment.yml
conda activate surgeonadaptrl
pip install --no-deps .
```

Container build:

```bash
docker build -t surgeonadaptrl:cuda117 .
```

## Data

Verified official project pages are listed in `dataset_urls.txt`.

- CATARACTS contains 50 videos at approximately 30 fps and 1920 × 1080 resolution. Use its official 25/25 training/test division.
- CaDIS contains 4,670 annotated frames sampled from 25 CATARACTS training videos. It is used only for segmentation-guided trajectory extraction, never as an independent generalization set.
- Cataract-1K contributes the 56-video phase-annotated subset. Its official repository reports 1024 × 768 at 30 fps for this subset; this differs from the 540 × 960 at 25 fps stated in the manuscript and is recorded as a source discrepancy in `claim_to_code.json`.

Create a CSV manifest with the following header:

```text
image,video,surgeon,frame,phase,x,y,instrument
```

Coordinates are instrument-tip locations normalized to `[0, 1]`. Preserve chronological frame order and split at video level. The `instrument` column is optional. Image resize, observation window, stride, and prediction horizon are engineering defaults because the manuscript does not fully specify preprocessing. The prediction horizon is 16, matching Figure 1.

## Training

The reported primary setting uses four NVIDIA A100 80 GB GPUs, 200 epochs, batch size 256, AdamW, learning rate `3e-4`, and weight decay `0.01`. Meta-optimization uses 50,000 iterations, inner learning rate `0.01`, outer learning rate `3e-4`, and 200 adaptation updates. Safety refinement uses 100,000 simulator iterations. The manuscript does not state whether batch size 256 is global or per GPU; the configuration preserves the reported value and records this ambiguity rather than asserting an effective batch size.

```bash
torchrun --standalone --nproc_per_node=4 -m surgeonadaptrl.cli --config configs/main.yaml --manifest DATA_MANIFEST.csv --seed 13
```

Configuration values for precision, scheduler, warmup, gradient clipping, seed identities, image resize, and temporal sampling are labeled engineering defaults in `configs/main.yaml`. They are not reported experimental settings.

## Model and objectives

The visual encoder is ImageNet-pretrained ResNet-50. The transformer has six encoder layers, four decoder layers, hidden width 512, and eight attention heads. Each residual adapter uses a 64-dimensional bottleneck and a learned scale initialized to 0.1. The phase classifier is an eight-layer TCN with kernel size five. Motion primitives use a 32-dimensional VAE latent space, KL weight 0.1, and per-phase Gaussian mixtures selected by BIC from 4 through 12 components.

Training combines trajectory prediction with style loss weighted by 0.3. Surgeon Style Fidelity weights velocity, acceleration, trajectory geometry, and phase timing by 0.25, 0.20, 0.30, and 0.25. Velocity similarity uses DTW scale 0.05; trajectory similarity uses discrete Fréchet scale 0.02.

## Simulation safety evaluation

AMBF provides three-dimensional positions, contact forces, and anatomical clearances. These quantities are not inferred from monocular video trajectories. The simulator uses corneal Young's modulus 0.3 MPa and lens-capsule Young's modulus 0.1 MPa. Constraints limit end-effector velocity to 15 mm/s and contact force to 30 mN while maintaining at least 0.2 mm clearance. Policy and dual learning rates are `3e-4` and `1e-3`.

## Evaluation

Run tests and the independent minimum training closure:

```bash
PYTHONPATH=code pytest -q
PYTHONPATH=code python -c "from surgeonadaptrl.verification import run_verification, write_summary; result=run_verification('verification'); write_summary(result, 'verification_report.json')"
```

The reported metrics are calculated over five random seeds. Continuous metrics use paired t-tests, categorical metrics use McNemar tests, and multiple comparisons use Bonferroni correction. The repository does not embed reported table values as executable outputs and does not claim that full dataset training has been rerun.

## Scope and verification status

The video model predicts normalized two-dimensional motion. Direct robot control, clinical use, calibrated three-dimensional reconstruction, an end-to-end vision-language system, and actual AMBF deployment are outside this release's verified scope. `verification_report.json` records only checks executed in the current environment. Because the full datasets, four-GPU training run, and AMBF experiment are not included, the release status remains `PARTIALLY_VERIFIED`.
