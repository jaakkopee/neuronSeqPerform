# Network Variance Roadmap and Phased Implementation Plan

## 1. Goal
Increase diversity of network activation patterns so output is not mostly synchronized or vibrating in unison.

Primary target behavior:
- More spatial variation in active neurons.
- More temporal variation in burst timing.
- Stable but evolving motifs instead of collapse to one global rhythm.
- Musical controllability maintained from MPD218 and keyboard.

## 2. Current Baseline (What Exists Today)

## 2.1 Core runtime path
- LIF generates spikes, then spikes are pooled/downmixed into a fixed 8x16 synth trigger grid.
- FM envelopes are triggered from this pooled grid.
- Sequencer and micro-step logic run in main loop.

## 2.2 Current controllable network parameters
- Threshold on CC16 and aftertouch routing.
- Tau on CC17 and aftertouch routing.
- Weight scale on CC18.
- Global drive on CC19 and aftertouch routing.
- Topology direct select on CC23.
- Neuron count direct select on CC24.
- LIF micro-steps per tick on CC25.
- Pad bank C utilities for randomize/reset/topology cycle/neuron count step.

## 2.3 Latest range changes already applied
- Threshold range now 0.1 to 2.0.
- Global drive normalized mapping now 0.25 to 3.0.

These changes increase onset probability, but they do not alone guarantee pattern diversity.

## 2.4 Why unison can still happen
- Neurons remain too homogeneous in threshold/tau/refractory/drive.
- Coupling is mostly immediate and globally coherent.
- Inhibitory balancing and delay structure are limited.
- No explicit anti-synchrony feedback loop exists.

## 3. Brainstormed Solution Space

## 3.1 Upgrades to existing modules

### model/lif_network.py
- Add heterogeneity spreads:
- Threshold spread.
- Tau spread.
- Refractory spread.
- Baseline drive spread.
- Add E/I population split:
- Excitatory ratio.
- Inhibitory gain.
- Separate E and I scaling.
- Add delayed coupling:
- Delay bins for short conduction delays.
- Delay jitter.
- Add adaptation:
- Per-neuron adaptation current (spike-frequency adaptation).
- Add structured noise:
- White and pink-like noise options.
- Spatially correlated noise field option.
- Add topology blending:
- Ring to SmallWorld blend factor rather than only hard switching.

### main.py
- Add strategy/scene manager integration.
- Add slow macro modulation buses that animate parameter targets.
- Add pattern metrics loop:
- Synchrony index.
- Spike entropy.
- Active ratio.
- Add optional anti-lock nudging based on metrics.

### control/midi_handler.py
- Add diversity macro controls page.
- Add scene launch controls via keyboard 0 to 9.
- Add scene save recall via Shift+0 to 9.
- Add anti-lock toggle and intensity control.

### view/matrix_view.py
- Add diagnostics overlays:
- Synchrony meter.
- Entropy meter.
- E versus I activity meter.
- Delay utilization meter.
- Add scene indicator and morph progress display.

## 3.2 Possible new components

### scene_manager.py
- Stores scene definitions and morph behavior.
- Handles launch, save, interpolation, rollback.

### pattern_engine.py
- High-level strategy layer writing dynamic targets for network parameters.
- Supports scene-specific motion and modulation rules.

### diversity_controller.py
- Measures pattern collapse and injects corrective nudges.
- Optional mode, intensity-controlled.

### delay_matrix.py
- Manages delayed recurrent influence channels.
- Keeps delay logic isolated from baseline neuron equations.

### modulation_lfo_bank.py
- Low-frequency modulators with independent phase offsets.
- Targets threshold, drive, heterogeneity, inhibition, and topology blend.

## 3.3 New parameters to expose
- Heterogeneity amount.
- Inhibitory ratio.
- Inhibitory gain.
- Delay spread.
- Adaptation strength.
- Noise amount.
- Noise color selector.
- Topology blend.
- Anti-lock strength.
- Scene morph time.
- Spatial drive gradient angle.
- Spatial drive gradient depth.
- Cluster count.
- Burst probability.

## 3.4 Scene strategy idea for keyboard keys 0 to 9
- 0 Calm lattice.
- 1 Traveling wave.
- 2 Pulsing islands.
- 3 Sparse sparks.
- 4 Burst storm.
- 5 Predator prey.
- 6 Small-world drift.
- 7 Edge-of-chaos.
- 8 Polyrhythm mesh.
- 9 Freeze then shatter.

Each scene should define at minimum:
- Parameter defaults.
- Active modulator set.
- Morph in/out behavior.
- Safety clamps.

## 4. Phased Implementation Plan

## Phase 0: Instrumentation and Baseline Capture
Objective:
- Quantify current synchronization and establish before/after metrics.

Deliverables:
- Add metrics in runtime state: synchrony, entropy, active ratio.
- Add minimal on-screen meters.
- Add optional CSV logging of metrics over time.

Acceptance criteria:
- User can see when network collapses into high synchrony.
- Baseline profiles can be saved for comparison.

## Phase 1: Controlled Heterogeneity
Objective:
- Break perfect symmetry without losing musical stability.

Deliverables:
- Add threshold/tau/refractory/drive spreads.
- Add one macro control: heterogeneity amount.
- Add deterministic random seed support for reproducibility.

Acceptance criteria:
- Same scene with low heterogeneity can lock.
- Same scene with moderate heterogeneity shows richer activity and lower synchrony.

## Phase 2: E/I Balance and Delay Structure
Objective:
- Introduce competition and phase diversity.

Deliverables:
- Add excitatory and inhibitory populations.
- Add inhibitory gain and ratio controls.
- Add short delay spread with small jitter.

Acceptance criteria:
- More local motifs and less global all-on/all-off behavior.
- Pattern travel and phase-offset rhythms become visible/audible.

## Phase 3: Adaptation and Structured Noise
Objective:
- Prevent static attractors and provide evolving behavior.

Deliverables:
- Add adaptation current.
- Add selectable noise color and amount.
- Add spatial noise mode for local texture.

Acceptance criteria:
- Sustained runs produce variation over longer windows.
- User can dial between stable groove and exploratory behavior.

## Phase 4: Scene System and Strategy Engine
Objective:
- Make complexity performable and recallable live.

Deliverables:
- Add scene_manager with keyboard 0 to 9 launch.
- Add scene morph time and smooth interpolation.
- Add strategy definitions for each scene.

Acceptance criteria:
- Keys 0 to 9 reliably recall distinct pattern families.
- Morph transitions are smooth and musically usable.

## Phase 5: Diversity Controller (Anti-lock)
Objective:
- Add optional automatic correction when network collapses.

Deliverables:
- Add diversity_controller using synchrony and entropy thresholds.
- Add anti-lock enable toggle and strength parameter.

Acceptance criteria:
- With anti-lock on, long runs avoid prolonged unison lock.
- With anti-lock off, manual behavior remains unchanged.

## Phase 6: Performance Mapping and UX Polish
Objective:
- Finalize controller mappings and visual feedback for live use.

Deliverables:
- Assign new macros to available controls and shifted layers.
- Add UI labels for scene, anti-lock, and diversity controls.
- Add quick help legend for keyboard scene controls.

Acceptance criteria:
- All new major controls are reachable in performance context.
- User can confidently navigate scenes and diversity shaping live.

## 5. Control Mapping Proposal (High-level)
- Keep existing Bank A and Bank B behavior for compatibility.
- Use an alternate control mode or shift layer for diversity macros.
- Reserve keyboard 0 to 9 for scene launch.
- Reserve Shift+0 to 9 for scene save.
- Reserve minus and plus for previous/next scene.
- Reserve M for morph toggle and A for anti-lock toggle.

## 6. Implementation Order Recommendation
- Start with Phase 0 and Phase 1 together.
- Then Phase 2.
- Then Phase 4.
- Then Phase 3.
- Then Phase 5 and Phase 6.

Reasoning:
- Metrics first prevents tuning blind.
- Heterogeneity and E/I balance typically produce the biggest immediate improvement.
- Scene layer should come before final polish so controls stabilize around real workflows.

## 7. Risks and Mitigations
- Risk: Added complexity harms predictability.
- Mitigation: Scene defaults and clamps; anti-lock optional.

- Risk: CPU load rises with delays and adaptation.
- Mitigation: Feature flags, cheap approximations, adaptive quality levels.

- Risk: Too much randomness destroys groove.
- Mitigation: Correlated noise, bounded modulation depth, scene-wise noise policies.

## 8. Validation Plan
- Run each phase with fixed seed and compare metrics to baseline.
- Verify that at least 3 scenes produce clearly distinct pattern statistics.
- Verify that scene switching and morphing do not introduce crashes or shape mismatch issues.
- Perform live-play test with MPD218 to ensure all new controls are reachable and understandable.

## 9. Deliverable Summary
At completion, the system should provide:
- Stable and controllable non-unison network behavior.
- A scene-based performance workflow on keyboard 0 to 9.
- Real-time diversity diagnostics and optional anti-lock protection.
- Backward-compatible core controls for existing sessions.
