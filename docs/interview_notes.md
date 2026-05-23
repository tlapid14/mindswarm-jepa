# MindSwarm JEPA — Interview Notes

A file-by-file walkthrough written so I can explain and defend every design
decision in this repo. For each module: what it does, the decisions worth
defending, and the questions an interviewer is likely to ask.

The mental model to keep loaded:

```
simulate.py ──► generate_dataset.py ──► data.py ──► train_{jepa,baseline}.py ──► evaluate.py
     │                                                                                  
     └────────────────────────► demo.py (live inference) ◄─── checkpoints              
network.py is shared by simulate.py, generate_dataset.py, and demo.py.
config.py is imported by everything.
```

---

## `config.py` — single source of truth

**What it does.** Holds every constant: world geometry, agent counts, kinematics,
disruption/jamming parameters, the connectivity-failure rule, the supervised
problem framing (window length, prediction horizon), model dimensions, and viz
colors. Also derives `STATE_DIM` from the layout constants.

**Decisions to defend.**
- *One config module, no scattered magic numbers.* Anything two modules need
  lives here, so the simulator, dataset, and demo can't silently disagree.
- *`MAX_BLUE` / `MAX_RELAY` kept separate from live counts.* They're equal in
  v1, but separating them means an ablation that changes agent counts doesn't
  force a state-vector reshape.
- *`STATE_DIM` is computed, not hard-coded* (`16·5 + 6·2 + 5 = 97`), so the
  layout and the number can't drift apart.

**Likely questions.**
- *Why is `COMM_RANGE` 450 with a comment about 220?* Honesty artifact: at 220,
  a 3×2 relay grid left a coverage hole in the middle that auto-failed episodes
  regardless of disruption. 450 makes the backbone actually cover the world, so
  failures come from disruption rather than geometry. I left the comment so the
  reasoning is auditable.
- *Why these horizons (past 10, future offset 20, failure horizon 35)?* They're
  chosen so the task is predictable-but-not-trivial; they're exactly the kind of
  thing the roadmap's ablations would sweep.

---

## `network.py` — the communication graph (pure NumPy, stateless)

**What it does.** Given agent/relay/disruptor positions *now*, computes the
graph snapshot: candidate edges within range, per-edge disruption intensity,
which edges are active, connectivity via multi-source BFS from the relays, and
the scalar graph metrics that feed the state vector. Also defines
`is_failure_sustained`, the single failure definition.

**Decisions to defend.**
- *Stateless and pure.* It owns no time and no positions — it just answers
  "given these positions, what's the graph?" That's why the live sim, the
  dataset generator, and the demo can all import it and be guaranteed to agree.
- *Geometry separated from disruption.* `_candidate_edges` computes
  in-range pairs ignoring disruption; disruption is applied separately. This
  lets metrics distinguish "edge never possible" from "edge possible but
  disrupted."
- *Connectivity = reachable from any relay,* computed with a single
  multi-source BFS seeded at all relays at once (O(V+E)), not per-agent search.
- *Lorentzian disruption falloff* `I = strength / (1 + (d/ref)²)`, summed over
  disruptors — smooth, bounded, and overlapping fields add up.

**Likely questions.**
- *Known weakness?* Yes — edge disruption is sampled at the edge *midpoint*. A
  long edge whose midpoint is clean but whose body grazes a disruptor survives.
  It's fine for the current ranges; I'd sample along the edge if ranges changed.
  I call this out in a code comment rather than hiding it.
- *Why share one failure function?* So the label the model is trained against
  and the failure the live demo shows are provably the same rule — they can't
  drift.

---

## `simulate.py` — the environment

**What it does.** A headless `Simulator` (no rendering dependency) and a separate
pygame `Renderer`. The simulator owns positions, steps the agents and
disruptors, maintains a delay buffer for stale observations, tracks true
connectivity history, and flattens snapshots into the `(STATE_DIM,)` state
vector.

**Decisions to defend.**
- *Headless sim, separate renderer.* Dataset generation must run without a
  display and without importing pygame; the renderer is a separate class that
  imports pygame lazily inside `__init__`. Training environments don't need a
  GUI library installed to run.
- *Observation delay via a fixed-length deque.* `_obs_buffer[0]` is always the
  K-step-stale snapshot. Snapshots store **copies**, not references, so later
  in-place position updates don't corrupt buffered history.
- *Normalized state vector.* Positions divided by world size, velocities by max
  speed, hops by a normalizer — everything roughly in `[0, 1]` so the networks
  see well-scaled inputs.
- *`flatten_state` ends with an assert* that the layout offsets sum to
  `STATE_DIM` — a cheap guard against silent layout drift.
- *Scripted, not learned, agents.* Agents steer toward objectives; disruptors
  drift toward the agent centroid with noise. The project is about *predicting*
  failure, not *controlling* the swarm, so scripted dynamics are the right scope.

**Likely questions.**
- *Why `delayed=True` for inputs but true connectivity for labels?* The model
  must see what it would see in deployment (stale data), but the *supervisor*
  shouldn't be fooled by latency — so labels use ground-truth connectivity.
- *Is the renderer needed for results?* No. It's purely for the demo. Everything
  measured runs headless.

---

## `generate_dataset.py` — episodes → supervised dataset

**What it does.** Runs N headless episodes, records the delayed-state history and
true connectivity per step, then slices sliding windows into
`(X_past, X_future, y_failure)` tuples and writes compressed `.npz` train/val
splits.

**Decisions to defend.**
- *Split by episode, not by window.* This is the single most important
  correctness decision. Windows from one episode are temporally correlated; if
  some land in train and others in val, the model effectively memorizes the
  episode and val scores are inflated. I split the *episode indices* and assign
  every window accordingly.
- *Label window starts at `t+1`.* A failure already present at the anchor time
  is an input, not a prediction — including it would leak the answer.
- *Anchor-time bounds* are derived so both the past window fits and the future
  offset / failure horizon stay inside the episode.
- *Reports the realized failure rate* with a hint to tune the disruption
  threshold if it drifts far from the 30–50% target band — keeps the task
  balanced enough to be non-trivial.

**Likely questions.**
- *How do you know there's no leakage?* The split is on episode IDs, computed
  once with a seeded RNG; the assignment mask is derived from those IDs. No
  window can appear on both sides.
- *Class balance?* ~32% positive overall, so the majority-class baseline is
  ~67% accuracy — the bar any real model has to clear.

---

## `data.py` — Dataset / DataLoader

**What it does.** `SwarmDataset` loads one `.npz` split eagerly into memory as
tensors; `load_split` / `make_dataloader` are thin helpers.

**Decisions to defend.**
- *Shape validation on load.* It checks the loaded array shapes against
  `config.py` and raises a clear error telling you to regenerate — catches the
  classic "I edited config and forgot to rebuild the dataset" bug.
- *Eager load + `num_workers=0`.* The dataset fits in RAM, so forking workers
  would just duplicate it and add IPC overhead. Labels are cast to float once at
  load instead of every batch.
- *Helpful `FileNotFoundError`* that tells you which command to run.

**Likely questions.**
- *Would this scale to a dataset that doesn't fit in RAM?* No — and that's a
  deliberate scope choice for a 60k-sample prototype. The fix (memory-mapped
  `.npz` or sharded loading) is obvious if it ever mattered.

---

## `models.py` — the two models

**What it does.** Defines the JEPA-style stack and the LSTM baseline.

JEPA stack:
```
past   ─► PastEncoder  (shared embedder + GRU + proj) ─► z_context ─► Predictor ─► z_pred ─► RiskHead ─► logit
future ─► TargetEncoder (same shared embedder)        ─► z_future ──(detach at loss)──┘
```

**Decisions to defend.**
- *Shared embedder between past and target encoders* (SimSiam-style weight
  sharing). One `_StateEmbedder` instance is used by both, so the past and
  future are embedded into the *same* space — which is what makes the MSE
  between predicted and actual future latent meaningful.
- *Projection back to `LATENT_DIM`.* The GRU hidden dim (128) differs from the
  latent dim (64); a linear projection puts `z_context` and `z_future` in the
  same space so the predictor's output and the target are comparable.
- *Stop-gradient lives at the loss site, not in the model.* `forward` returns
  the raw `z_future`; the trainer detaches it. Keeping the model side-effect-free
  means the same forward pass serves both training and the latent-MSE metric.
- *Risk head reads `z_pred` (the predicted future latent), not `z_context`.*
  That's the whole thesis — risk is judged on where the system is *going*.
- *`predict_risk` inference path* skips the target encoder entirely (no future
  needed at inference) — past → context → predicted latent → sigmoid(risk).
- *Baseline hidden size matches the GRU* so JEPA's advantage can't be explained
  away as "it just had more parameters."
- *Logits, not probabilities, from the heads* — paired with
  `BCEWithLogitsLoss` for numerical stability.

**Likely questions (this is where most interview pressure lands).**
- *Why doesn't the latent collapse?* The stop-gradient. Without it, the shared
  embedder could map everything to a constant and drive the MSE to zero. With
  it, the predictor must hit a target it can't trivialize. (The supervised BCE
  term also pushes against collapse, but the detach is the principled fix.)
- *Is this "really" JEPA?* No, and I say so. It's JEPA-*style*: predict-the-
  future-latent + stop-gradient, plus a supervised head because I have labels
  and pure self-supervision wouldn't target failure. I'm precise about this
  precisely because it's the kind of overclaim that gets caught.
- *GRU in the "JEPA" model but LSTM baseline — unfair?* The GRU is the
  *encoder*; the architectural difference being tested is the predictive
  objective, not GRU-vs-LSTM. Hidden sizes are matched. If pressed, swapping the
  encoder cell is a one-line ablation.

---

## `train_jepa.py` — JEPA trainer

**What it does.** Trains the JEPA model on `MSE + λ·BCE`, evaluates each epoch,
and saves the best-by-val-loss checkpoint with full metadata (epoch, metrics,
history, args).

**Decisions to defend.**
- *`.detach()` on the target latent in `_compute_loss`* — flagged in a comment
  as the load-bearing JEPA detail.
- *Best-by-val-loss checkpointing,* not last-epoch, so a late-epoch overfit
  doesn't get saved.
- *Seeded torch + numpy; gradient clipping; Adam with documented defaults.*
- *Checkpoint stores `history` and `args`* so `evaluate.py` can plot learning
  curves and so any run is self-describing.
- *Auto device selection* (CUDA if available) with a `--device` override.

**Likely questions.**
- *Why is `λ` (BCE weight) = 1.0?* It's the obvious starting point and a
  first-class ablation target — it's a named config constant for exactly that
  reason.
- *Val loss mixes MSE and BCE — does that bias checkpoint selection?* It selects
  on the combined objective the model is actually trained on. For the *fair
  comparison*, `evaluate.py` recomputes classification metrics independently of
  the loss used for selection.

---

## `train_baseline.py` — LSTM baseline trainer

**What it does.** Mirrors the JEPA trainer exactly except the model is the LSTM
and the loss is plain BCE (the future state is loaded but unused).

**Decisions to defend.**
- *Deliberate structural mirror.* Same optimizer, LR, batch size, grad clip,
  epochs, checkpoint format, eval logic. The comparison isn't a tuning-knob
  comparison — only the model and objective differ.

**Likely questions.**
- *Why load the future at all if it's unused?* So both trainers iterate the
  identical `DataLoader` and see identical batches; the unused tensor costs
  almost nothing and keeps the data path uniform.

---

## `evaluate.py` — the comparison and the plots

**What it does.** Loads both checkpoints, runs them on the val set, computes
accuracy/precision/recall/F1 and a calibration curve + ECE, prints a per-metric
winner table, and writes `calibration.png`, `metrics.png`, `learning_curves.png`.

**Decisions to defend.**
- *Metrics computed from scratch* (TP/TN/FP/FN), not pulled from training logs,
  so the reported numbers are independent of the loss used for checkpoint
  selection.
- *Calibration / ECE included,* not just accuracy. For an early-warning system,
  whether the probabilities are *trustworthy* matters as much as whether they're
  right — and it's the one axis where the baseline wins, which I report honestly.
- *Headless matplotlib* (`Agg` backend set before importing pyplot) so it runs
  on a server with no display.
- *Per-metric "winner" is direction-aware* — lower ECE wins, higher everything
  else wins.
- *Latent MSE reported for JEPA only,* with `N/A` for the baseline, since the
  baseline has no latent-prediction objective.

**Likely questions.**
- *JEPA wins 3 metrics, baseline wins 2 — is that significant?* On a single seed,
  no claim of statistical significance — that's why multi-seed runs are the #2
  roadmap item. I present it as preliminary evidence, not a benchmark result.
- *Why plot val_bce for JEPA against val_loss for the baseline?* Because the
  baseline's loss *is* BCE, while JEPA's combined loss isn't comparable — so I
  pull out JEPA's BCE component to compare like with like. It's commented.

---

## `demo.py` — live inference

**What it does.** Runs the simulator in real time with a pygame window, feeding a
rolling buffer of the last `PAST_WINDOW` delayed states into both models and
overlaying their risk estimates (HUD + sparkline). R resets, ESC quits.

**Decisions to defend.**
- *The rolling buffer mirrors training exactly* — same `delayed=True` states,
  same window length — so one row of the demo buffer equals one training
  `X_past` row. Train/inference parity, no skew.
- *Returns `(None, None)` until the buffer fills,* so it never feeds a
  short/padded window the models never saw in training.
- *Both checkpoints loaded before the window opens,* so a missing checkpoint
  fails with a clean message instead of mid-render.
- *Freezes on the failure frame* rather than instantly resetting, so you can
  actually see the moment of fragmentation and the risk traces leading into it.

**Likely questions.**
- *Is the demo measuring anything?* No — it's qualitative/illustrative. All
  quantitative claims come from `evaluate.py` on the held-out val set.

---

## Cross-cutting things I'd bring up unprompted

- **Reproducibility.** Seeded RNG in the sim, the split, and both trainers;
  best-checkpoint selection; self-describing checkpoints carrying their own args
  and history.
- **Fairness of the comparison.** Matched capacity, matched training budget,
  identical data and observation staleness — engineered so the result reflects
  the objective, not the setup.
- **Honesty.** "JEPA-inspired," not V-JEPA. Single-seed, abstract simulator,
  state-vector (not visual) inputs. The README's Limitations and Roadmap state
  exactly what's missing and what I'd do next.
- **Separation of concerns.** Headless sim vs. renderer; stateless graph module
  shared by all consumers; one config module. Each file has one job.
