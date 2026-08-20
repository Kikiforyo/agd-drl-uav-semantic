# Codex Task: TCOM-Style Simulation Redesign for AGD-DRL UAV Covert Semantic Communication

## 0. Goal
Use the existing repository as the only codebase. Redesign and implement the paper's simulation pipeline by borrowing the **experimental logic** of Yao et al., IEEE TCOM 2026, "UAV-RHS-Enabled Full-Duplex ISAC Covert System: Robust Beamforming and Trajectory Optimization", without copying its unrelated RHS/ISAC variables.

The target paper is a **full-duplex multi-antenna UAV covert semantic communication system** with AGD-DRL. The final experiments must explain *why* the proposed method works, not merely show one larger final number.

Read `AGENTS.md`, `experiment_brief.md`, and the current source before modifying anything.

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
- metadata JSON recording git commit, config, checkpoint, seed, timestamp, evaluation mode.

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
This is new and high priority.

Plot the UAV trajectories for SCA, TD3, DRL-JPPO, AGD-DRL on the same physical map, together with:
- Alice;
- Willie trajectory;
- UAV start/end points.

Use the same evaluation episode/Willie realization when making the comparison.

Also export per-slot coordinates and distances so the paper can explain the trajectory physically.

### E4. Beamforming behavior / beam pattern
This is new and high priority.

For representative nominal slots, compute and plot
`G(theta)=|w^H a(theta)|^2`
over a physically valid angular range.

Mark:
- Willie direction theta_UW;
- self-interference direction theta_SI, if the current model defines one.

Compare at least:
- AGD-DRL optimized beam;
- a random/unoptimized phase beam;
- a reasonable heuristic beam if already supported.

Do not invent a zero-null claim if the implemented constant-modulus phase-only model does not actually create a null.

### E5. UAV jamming power-budget sensitivity
TCOM analogue: performance versus UAV power budget.

Evaluate at nominally reasonable points around the current P_max=15 W, e.g. `[5, 7.5, 10, 12.5, 15, 17.5, 20]` W **only if the code's current physical scale supports these values**. Otherwise derive a range from the implemented `P_req/P_max` distribution and document the chosen range.

Plot:
1. final effective SUDT vs P_max;
2. KL violation ratio vs P_max;
3. optional power-clipping ratio or average P_req/P_max vs P_max.

Compare SCA, TD3, DRL-JPPO, AGD-DRL where implementations are comparable.

This experiment must reveal low-power infeasible, transition, and high-power feasible/saturation regimes if they actually exist; do not fabricate a threshold.

### E6. Residual self-interference sensitivity
TCOM analogue: CTR versus residual SI.

First determine the exact SI parameterization in this repository. Then sweep it over a meaningful range around the nominal value.

Plot:
- final effective SUDT vs residual SI level;
- average semantic rate or outage vs residual SI;
- optional average required P_jam.

Use dB only if the internal parameter is physically represented in dB or can be converted unambiguously. Document conversion.

### E7. Covertness requirement sensitivity
TCOM analogue: performance versus covertness coefficient.

The current paper uses a KL-based normalized boundary. Do **not** introduce a new epsilon definition unless it maps exactly to the implemented KL formulation.

Sweep the actual covert strictness parameter used by the code (e.g. KL budget/boundary).

Plot:
- effective SUDT vs covert strictness;
- KL violation ratio vs covert strictness;
- average alpha vs covert strictness;
- average P_jam vs covert strictness.

This experiment should test the paper's claimed mechanism: stricter covertness should induce a meaningful trajectory/beam/power/semantic-granularity response.

### E8. Willie uncertainty / robustness
TCOM analogue: CSI uncertainty.

Prefer a parameter that exists naturally in the current model:
- Willie position estimation error, or
- AoD estimation error, or
- CSI error.

Do not add all three unless the model already supports them.

Plot:
- effective SUDT vs uncertainty;
- KL violation ratio vs uncertainty;
- Willie beam gain/alignment metric vs uncertainty.

Important: distinguish nominal evaluation from stress/robustness evaluation. If the current AGD-DRL was not trained for uncertainty, do not claim robust optimization; call it robustness/stress testing.

### E9. Semantic granularity mechanism
This experiment is essential because the paper is semantic communication, not merely covert physical-layer control.

At minimum, export over a representative evaluation episode:
- alpha[t];
- normalized KL ratio[t];
- P_jam[t];
- R_sem[t];
- Alice-UAV distance[t];
- Willie-direction beam gain[t].

Create aligned time-series figures or a compact multi-panel figure showing how alpha changes with covert/channel pressure.

Also compute correlations with appropriate caution; correlation is descriptive, not causal proof.

If the repository contains a real DeepSC decoder, add a semantic-quality experiment over alpha/SNR using a real dataset and report BLEU and/or BERT-based semantic similarity. If only a proxy semantic rate exists, do not fake BLEU/BERT; record this limitation in `docs/tcom2026_experiment_audit.md` and implement only the proxy mechanism plots.

### E10. Module contribution / ablation
Retain the four paper configurations if current code supports them:
- Full AGD-DRL
- w/o adaptive alpha
- w/o MHA
- w/o diffusion
- w/o closed-form power

Do not tune individual ablations to predetermined target values. Run with common evaluation conditions and report what the code actually produces.

## 5. Plotting style
Use a clean IEEE/communications-journal style:
- no decorative backgrounds;
- readable 8–10 pt equivalent text;
- consistent line/marker mapping across figures;
- uncertainty bands only where they add information;
- vector PDF output;
- no misleading axis truncation;
- cumulative curves should be based on actual slot-level accumulation.

## 6. Reproducibility and result integrity
Every experiment must have:
- raw per-seed data;
- aggregate mean/std;
- exact config;
- checkpoint provenance;
- command used;
- whether training or fixed-policy evaluation;
- no hand-edited numerical results.

Add validation checks:
- closed-form P_jam respects `[0, P_max]`;
- phase values respect the model bounds;
- alpha respects its allowed range/grid;
- KL calculation is finite;
- identical Willie realization for method-comparison trajectory figures;
- seeds do not accidentally reuse the same RNG state.

## 7. Execution stages
### Stage 0 — audit only
Run no expensive full training. Commit the audit and proposed exact parameter grids based on the existing code.

### Stage 1 — smoke tests
For each new runner/metric:
- 1 seed;
- short episode/evaluation;
- verify CSV/NPZ/PDF/PNG generation;
- verify checkpoint loading.

### Stage 2 — nominal/fixed-policy evaluations
Reuse valid existing checkpoints for E2–E9 where scientifically valid.

### Stage 3 — retraining only where needed
Run 5 seeds only for experiments requiring retraining or for final paper confirmation.

## 8. Final deliverables
Create:
- `docs/tcom2026_experiment_audit.md`
- `docs/tcom2026_experiment_report.md`
- reproducible scripts/configs
- `outputs/tcom2026_suite/raw/`
- `outputs/tcom2026_suite/figures/`
- `outputs/tcom2026_suite/tables/`

The report must contain a table with columns:
`Experiment | Question answered | Parameter sweep | Algorithms | Training or fixed-policy eval | Main metric | Result path | Passed integrity checks | Caveats`

## 9. Completion rule
Do not claim the full task is complete merely because scripts were created. Completion requires executable smoke tests and at least Stage 0 + Stage 1 results. If remote GPU/server execution is unavailable from the coding environment, finish the code and exact server commands, then stop and report what must be run remotely.
