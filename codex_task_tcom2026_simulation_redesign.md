# Codex Task: TCOM-Style Simulation Redesign for AGD-DRL UAV Covert Semantic Communication

## 0. Goal
Use the existing repository as the only codebase. Redesign and implement the paper's simulation pipeline by borrowing the **experimental logic** of Yao et al., IEEE TCOM 2026, "UAV-RHS-Enabled Full-Duplex ISAC Covert System: Robust Beamforming and Trajectory Optimization", without copying its unrelated RHS/ISAC variables.

The target paper is a **full-duplex multi-antenna UAV covert semantic communication system** with AGD-DRL. The final experiments must explain *why* the proposed method works, not merely show one larger final number.

Read `AGENTS.md`, `experiment_brief.md`, and the current source before modifying anything.

## 0.1 Mandatory local-PC execution rule
All implementation, auditing, smoke tests, fixed-policy evaluations, retraining, plotting, and result aggregation for this task must run on the **user's local computer through Codex**.

- Do **not** use SSH, rsync, SCP, a remote Linux server, cloud GPU, GitHub Actions, or any other remote compute backend for the experiments.
- Do not generate server commands as the normal execution path.
- At the beginning, detect and record the local environment: OS, Python version, PyTorch version, CUDA availability, GPU model if present, and free disk space relevant to outputs.
- Use `torch.cuda.is_available()` to decide whether the local CUDA GPU can be used. If CUDA is unavailable, Stage 0 and Stage 1 must still run on CPU where practical.
- For Stage 2/3, use the local GPU if available; otherwise use CPU only when runtime remains reasonable.
- If a full 5-seed training job would be impractically slow on the local PC, do **not** switch to a server. Instead, finish the code and smoke tests, estimate the local runtime, make runs resumable, and report the exact local command that the user can continue running on the same PC.
- All outputs must be written under the repository-local `outputs/tcom2026_suite/` tree unless the existing project has a clearly established local output root that should be reused.
- Long jobs must save checkpoints and intermediate CSV/NPZ results so an interrupted local run can resume without losing completed seeds.

## 1. Non-negotiable paper settings
- Nodes: Alice (fixed), UAV (IBFD multi-antenna), mobile Willie.
- Default task length: T=200 slots.
- Default Alice power: P_A=2 W.
- Default UAV jamming power cap: P_max=15 W.
- Default semantic recovery threshold: R_req=0.55.
- Default normalized KL boundary: 1.
- Default antennas: M=4.
- Main comparison: SCA, TD3, DRL-JPPO, AGD-DRL.
- Five independent seeds for paper-level results.
- AGD-DRL components: MHA state encoder + physics guide + conditional diffusion actor + TD3 twin critics + adaptive semantic granularity alpha + closed-form jamming power.
- Main AGD-DRL action should contain trajectory control, four phase controls, and semantic granularity alpha. Jamming power is closed-form, not a free policy output.
- Do not silently alter the paper model to improve plots.
- Do not manually edit result points, interpolate curves, or use different budgets/seeds for different algorithms.

## 2. First action: audit before implementation
Do not start full training immediately.

Create `docs/tcom2026_experiment_audit.md` containing:
1. Actual current environment equations/implemented formulas for:
   - Alice-UAV channel/SINR
   - UAV-Willie beam gain
   - self-interference
   - KL/covert constraint
   - closed-form P_jam
   - semantic rate/SUDT
   - semantic granularity alpha
2. Actual action/state definitions for AGD-DRL, TD3, DRL-JPPO, SCA.
3. Which parameters are already configurable and which are hard-coded.
4. Whether a real DeepSC/MINE pipeline currently exists or only proxy semantic rate.
5. Which existing checkpoints/results can be reused without retraining.
6. Any discrepancy between code and paper assumptions.
7. Local-PC execution environment and whether the current local GPU/CPU can support each planned stage.

If formulas needed for a requested experiment are missing, expose them as config fields/TODOs instead of inventing physics.

## 3. Build a unified experiment runner
Add a reusable parameter-sweep framework rather than one-off scripts.

Recommended files (adapt to repo style if better):
- `scripts/run_tcom_experiment_suite.py`
- `scripts/evaluate_parameter_sweep.py`
- `scripts/plot_tcom_experiment_suite.py`
- `configs/tcom_experiment_suite.yaml`

Requirements:
- supports `--seed`, `--seeds`, `--device`, `--output-root`, and experiment name;
- can load trained checkpoints for fixed-policy sensitivity evaluation;
- clearly marks whether a result is **retrained** or **offline/fixed-policy re-evaluation**;
- saves raw CSV/NPZ before plotting;
- saves per-seed and mean±std results;
- PNG + PDF figures;
- summary CSV + `summary.txt`;
- metadata JSON recording git commit, config, checkpoint, seed, timestamp, evaluation mode, local device and CUDA status;
- supports resumable execution so already completed seeds/parameter points are not rerun unnecessarily.

Do not retrain when a parameter can validly be evaluated by fixed-policy re-evaluation. But if changing a parameter alters observation/action dimension or the policy's training distribution in a way that invalidates fixed-policy evaluation, document this and retrain or label it explicitly as stress analysis.

## 4. Paper experiment suite
Implement in the following priority order.

### E1. Convergence (retain existing result if provenance is valid)
Compare TD3, DRL-JPPO, AGD-DRL using identical environment, training budget and evaluation protocol.

Outputs:
- evaluation reward vs training episode, mean±std across 5 seeds;
- optional training reward separately;
- report approximate convergence episode using an explicit rule, not manual visual selection.

Do not regenerate only to force AGD-DRL to look better. Audit existing convergence result first.

### E2. Overall task performance
Compare SCA, TD3, DRL-JPPO, AGD-DRL under nominal settings.

Primary metric:
- cumulative effective SUDT over 200 slots.

Also export:
- final cumulative SUDT;
- average semantic rate;
- semantic outage ratio;
- KL violation ratio;
- average/max P_jam;
- average alpha;
- average Alice-UAV distance;
- average Willie-direction beam gain;
- average SI leakage/gain.

### E3. UAV trajectory visualization
Plot the UAV trajectories for SCA, TD3, DRL-JPPO, AGD-DRL on the same physical map, together with Alice, Willie trajectory, and UAV start/end points.

Use the same evaluation episode/Willie realization when making the comparison. Export per-slot coordinates and distances.

### E4. Beamforming behavior / beam pattern
For representative nominal slots, compute and plot `G(theta)=|w^H a(theta)|^2` over a physically valid angular range.

Mark Willie direction `theta_UW` and self-interference direction `theta_SI` if the current model defines one. Compare at least AGD-DRL optimized beam, a random/unoptimized phase beam, and a reasonable heuristic beam if supported.

Do not invent a zero-null claim if the implemented constant-modulus phase-only model does not actually create a null.

### E5. UAV jamming power-budget sensitivity
TCOM analogue: performance versus UAV power budget.

Evaluate at nominally reasonable points around current P_max=15 W, e.g. `[5, 7.5, 10, 12.5, 15, 17.5, 20]` W **only if the code's current physical scale supports these values**. Otherwise derive a range from the implemented `P_req/P_max` distribution and document it.

Plot:
1. final effective SUDT vs P_max;
2. KL violation ratio vs P_max;
3. optional power-clipping ratio or average P_req/P_max vs P_max.

Compare SCA, TD3, DRL-JPPO, AGD-DRL where implementations are comparable. Do not fabricate a threshold/saturation effect.

### E6. Residual self-interference sensitivity
First determine the exact SI parameterization in this repository. Then sweep a meaningful range around nominal.

Plot final effective SUDT, semantic rate/outage, and optionally average required P_jam versus residual SI. Use dB only if conversion is unambiguous and document it.

### E7. Covertness requirement sensitivity
The paper uses a KL-based normalized boundary. Do **not** introduce a new epsilon definition unless it maps exactly to the implemented KL formulation.

Sweep the actual covert strictness parameter used by the code. Plot effective SUDT, KL violation ratio, average alpha, and average P_jam versus covert strictness.

### E8. Willie uncertainty / robustness
Prefer one uncertainty parameter that exists naturally in the current model: Willie position estimation error, AoD estimation error, or CSI error. Do not add all three unless already supported.

Plot effective SUDT, KL violation ratio, and Willie beam alignment/gain versus uncertainty. If the policy was not trained for uncertainty, label this as robustness/stress testing rather than robust optimization.

### E9. Semantic granularity mechanism
This experiment is essential because the paper is semantic communication, not merely covert physical-layer control.

At minimum, export over a representative episode:
- alpha[t];
- normalized KL ratio[t];
- P_jam[t];
- R_sem[t];
- Alice-UAV distance[t];
- Willie-direction beam gain[t].

Create aligned time-series or a compact multi-panel figure. Correlation analysis is descriptive, not causal proof.

If the repository contains a real DeepSC decoder, add a semantic-quality experiment over alpha/SNR using a real dataset and report BLEU and/or BERT-based semantic similarity. If only a proxy semantic rate exists, do not fake BLEU/BERT; document the limitation and implement proxy-mechanism plots only.

### E10. Module contribution / ablation
Retain if current code supports them:
- Full AGD-DRL
- w/o adaptive alpha
- w/o MHA
- w/o diffusion
- w/o closed-form power

Do not tune individual ablations to predetermined target values. Report actual results under common conditions.

## 5. Plotting style
Use clean IEEE/communications-journal style:
- no decorative backgrounds;
- readable 8–10 pt equivalent text;
- consistent line/marker mapping;
- uncertainty bands only when informative;
- vector PDF output;
- no misleading axis truncation;
- cumulative curves based on actual slot-level accumulation.

## 6. Reproducibility and result integrity
Every experiment must have:
- raw per-seed data;
- aggregate mean/std;
- exact config;
- checkpoint provenance;
- command used;
- training/fixed-policy label;
- local device information;
- no hand-edited numerical results.

Validation checks:
- closed-form P_jam respects `[0, P_max]`;
- phase values respect model bounds;
- alpha respects its allowed range/grid;
- KL calculation is finite;
- identical Willie realization for trajectory comparisons;
- seeds do not accidentally reuse the same RNG state.

## 7. Execution stages on the local computer
### Stage 0 — audit only
Run no expensive full training. Inspect the local project, record the local environment, audit formulas/checkpoints, and propose exact parameter grids.

### Stage 1 — local smoke tests
For each new runner/metric:
- 1 seed;
- short episode/evaluation;
- verify CSV/NPZ/PDF/PNG generation;
- verify checkpoint loading;
- verify CPU and, when available, local CUDA device selection.

### Stage 2 — local nominal/fixed-policy evaluations
Reuse valid existing checkpoints for E2–E9 where scientifically valid. Run on the local GPU when available.

### Stage 3 — local retraining only where needed
Run 5 seeds only for experiments requiring retraining or final paper confirmation. Make each seed independently resumable and save intermediate checkpoints.

Before starting any long Stage 3 job, print a concise local runtime estimate based on a short benchmark. Do not migrate the run to a remote machine.

## 8. Final deliverables
Create:
- `docs/tcom2026_experiment_audit.md`
- `docs/tcom2026_experiment_report.md`
- reproducible scripts/configs
- `outputs/tcom2026_suite/raw/`
- `outputs/tcom2026_suite/figures/`
- `outputs/tcom2026_suite/tables/`

The report must contain:
`Experiment | Question answered | Parameter sweep | Algorithms | Training or fixed-policy eval | Main metric | Result path | Passed integrity checks | Caveats`

## 9. Completion rule
Do not claim the full task is complete merely because scripts were created. Completion requires executable local smoke tests and at least Stage 0 + Stage 1 results.

If the local PC is too slow for a full final 5-seed training run, do **not** switch to a server or cloud machine. Finish and validate the code locally, make the run resumable, provide the exact local command, estimated runtime, expected output paths, and list which final runs remain pending on the same local PC.
