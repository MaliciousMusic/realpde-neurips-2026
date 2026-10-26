# NeurIPS 2026 RealPDE Challenge — Sim2Real turbulent wake prediction

**Phase 1 results: 1st, Track 1 (Sim2Real) · 4th, Track 2 (long-horizon test-time adaptation).**

The task: forecast 20 future frames of a **real**, PIV-measured turbulent wake behind an airfoil
(velocity fields on a 32 × 64 grid, across Reynolds numbers and angles of attack) from 20 past frames,
with **calibrated uncertainty bounds**, in an offline container under a strict time budget. Models
may be pre-trained on simulation; the score is measured on real flow.

## What is in this repository

| Path | Content |
|---|---|
| `track1/submission.py` | Final Track 1 submission: official FNO baseline + our calibrated SPS bounds + latency work |
| `track2/submission.py` | Final Track 2 submission: FNO in test-time-adaptation mode, gradient-free EMA bias correction, same bounds |
| `track2_statistical/submission.py` | Checkpoint-free statistical predictor (*phase-locked replay* with causal accumulation), pure PyTorch |
| `notes/` | Working notes written during the competition (French): state of play, measurements, refuted ideas |

Code comments are in French. The official FNO checkpoint (`sim_real_fno_fp16.pth`) and the organisers'
`load_baseline.py` are **not redistributed**; they come with the competition starting kit.

## Key ideas

- **Reverse-engineered scoring.** The final score was recovered to machine precision from the public
  leaderboard (102 rows, rms 3·10⁻¹¹): a weighted power mean
  `100·(0.5·rl^p + 0.1·tke^p + 0.1·mvpe^p + 0.1·time^p + 0.2·sps^p)^(1/p)`, `p = 0.6854`.
  Its gradients told us where each hour of work was worth spending.
- **Uncertainty bounds are where the points are.** Bounds centred on a *Wiener* estimate (FNO
  fluctuations shrunk frame by frame by their measured correlation with the target), half-widths read
  from per-channel tables fitted by exact argmax on residuals, and an in-sample inflation factor
  (λ = 1.60) solved from a single online observation. The bench then predicted the platform within 0.3 %.
- **Statistics before networks.** `rel_l2` is saturated by trivial predictors (the window mean scores
  94.7); the turbulent kinetic energy is a *statistic*, low-rank in (Re, AoA). The checkpoint-free
  predictor replays the observed fluctuations in phase and rescales them; in Track 2, averaging the
  standard deviation over all windows already seen lifts TKE from 73.1 to 78.6.
- **Latency is a sub-score.** No host synchronisation in the step, no CUDA graphs (zero measured gain,
  real crash risk on the grader's PyTorch), kernel-count-driven rewrites, half-precision outputs with
  bounds widened beforehand so coverage is unchanged.
- **Measure, then trust.** Every idea was checked out-of-sample, by leave-one-regime-out. Nine bound
  families, conformal quantiles and global variance rescaling were all measured and rejected.
- **A lesson from the logs.** A single em dash in a comment scored **zero on both tracks**: the
  grader reads `submission.py` with the container's locale. Submissions are now strict ASCII.

## Author

Hadrien Lonjaret — [github.com/MaliciousMusic](https://github.com/MaliciousMusic)
