# Network Variance Task Checklist

Implementation tickets derived from the roadmap in [docs/network_variance_implementation_plan.md](docs/network_variance_implementation_plan.md).

## Usage
- Mark tickets complete by changing [ ] to [x].
- Implement in phase order unless dependency notes say otherwise.
- Keep each ticket small enough to test in one session.

## Ticket Template
- ID: unique stable identifier.
- Scope: modules expected to change.
- Deliverable: concrete output.
- Done when: acceptance gate.

## Phase 0: Instrumentation and Baseline Capture

- [ ] NV-0001 Add synchrony metric calculation
- Scope: [main.py](main.py), [model/lif_network.py](model/lif_network.py)
- Deliverable: runtime value for synchrony index exposed in config_state.
- Done when: synchrony value updates continuously and responds to pattern collapse.

- [ ] NV-0002 Add spike entropy metric calculation
- Scope: [main.py](main.py)
- Deliverable: entropy metric from pooled spike grid over rolling window.
- Done when: entropy lowers in unison states and rises in varied states.

- [ ] NV-0003 Add active-neuron ratio metric
- Scope: [main.py](main.py)
- Deliverable: fraction of active cells per step and short moving average.
- Done when: both raw and smoothed values are available for display and logging.

- [ ] NV-0004 Add minimal on-screen metric panel
- Scope: [view/matrix_view.py](view/matrix_view.py)
- Deliverable: compact text or bar overlay for synchrony, entropy, active ratio.
- Done when: metrics are visible without occluding primary matrix readability.

- [ ] NV-0005 Add optional CSV metric logging
- Scope: [main.py](main.py)
- Deliverable: opt-in logger writing timestamped metrics rows.
- Done when: CSV file can be enabled/disabled and contains valid rows for full run.

- [ ] NV-0006 Baseline capture runbook
- Scope: [docs/network_variance_task_checklist.md](docs/network_variance_task_checklist.md)
- Deliverable: short manual procedure for recording baseline profile.
- Done when: same procedure can be run twice with comparable output structure.

## Phase 1: Controlled Heterogeneity

- [ ] NV-0101 Add threshold spread parameter
- Scope: [model/lif_network.py](model/lif_network.py)
- Deliverable: configurable per-neuron threshold variance around base threshold.
- Done when: spread 0 behaves like current system; spread > 0 changes firing diversity.

- [ ] NV-0102 Add tau spread parameter
- Scope: [model/lif_network.py](model/lif_network.py)
- Deliverable: configurable per-neuron tau variance.
- Done when: tau spread affects temporal desynchronization without instability at moderate values.

- [ ] NV-0103 Add refractory spread parameter
- Scope: [model/lif_network.py](model/lif_network.py)
- Deliverable: configurable refractory-time variance.
- Done when: visible reduction in lock-step bursts at moderate settings.

- [ ] NV-0104 Add baseline drive spread parameter
- Scope: [model/lif_network.py](model/lif_network.py)
- Deliverable: configurable variation around global drive.
- Done when: population does not collapse into identical firing under static controls.

- [ ] NV-0105 Add heterogeneity macro control
- Scope: [control/midi_handler.py](control/midi_handler.py), [main.py](main.py)
- Deliverable: one macro that scales all spread parameters together.
- Done when: macro 0 is near-homogeneous and macro 1 is clearly more diverse.

- [ ] NV-0106 Add deterministic seed control
- Scope: [model/lif_network.py](model/lif_network.py), [main.py](main.py)
- Deliverable: optional seed used for stochastic initialization and spread sampling.
- Done when: identical seed reproduces similar pattern statistics.

## Phase 2: E/I Balance and Delay Structure

- [ ] NV-0201 Add excitatory and inhibitory population assignment
- Scope: [model/lif_network.py](model/lif_network.py)
- Deliverable: neuron type mask with configurable inhibitory ratio.
- Done when: model computes both E and I contributions and remains stable at defaults.

- [ ] NV-0202 Add inhibitory gain control
- Scope: [model/lif_network.py](model/lif_network.py), [control/midi_handler.py](control/midi_handler.py)
- Deliverable: parameter controlling I contribution strength.
- Done when: increasing inhibitory gain visibly suppresses global lock-in.

- [ ] NV-0203 Split weight scaling into E-scale and I-scale
- Scope: [model/lif_network.py](model/lif_network.py)
- Deliverable: separate scaling terms for excitatory and inhibitory edges.
- Done when: independent control paths are testable and reflected in activity.

- [ ] NV-0204 Add short delay bins for recurrent influence
- Scope: [model/lif_network.py](model/lif_network.py)
- Deliverable: delay buffer with configurable spread in micro-step units.
- Done when: delayed coupling generates phase-shifted motifs.

- [ ] NV-0205 Add delay jitter
- Scope: [model/lif_network.py](model/lif_network.py)
- Deliverable: small random perturbation of assigned delays.
- Done when: jitter reduces repetitive lock patterns without turning to noise.

- [ ] NV-0206 Expose E/I and delay controls in runtime state
- Scope: [main.py](main.py), [control/midi_handler.py](control/midi_handler.py)
- Deliverable: parameters visible, serializable, and controllable.
- Done when: values appear in UI/debug state and can be changed live.

## Phase 3: Adaptation and Structured Noise

- [ ] NV-0301 Add spike-frequency adaptation current
- Scope: [model/lif_network.py](model/lif_network.py)
- Deliverable: adaptation variable with configurable strength and decay.
- Done when: sustained firing self-limits and pattern evolution increases over time.

- [ ] NV-0302 Add noise amount parameter
- Scope: [model/lif_network.py](model/lif_network.py)
- Deliverable: global noise amplitude injected into membrane update.
- Done when: low noise adds variation; high noise predictably increases randomness.

- [ ] NV-0303 Add noise color selection
- Scope: [model/lif_network.py](model/lif_network.py)
- Deliverable: selector for white and pink-like noise modes.
- Done when: noise modes are switchable and produce audibly/visually distinct behavior.

- [ ] NV-0304 Add spatial noise mode
- Scope: [model/lif_network.py](model/lif_network.py)
- Deliverable: spatially correlated field influencing local neighborhoods.
- Done when: clustered activity appears instead of fully independent flicker.

- [ ] NV-0305 Add controls for adaptation and noise
- Scope: [control/midi_handler.py](control/midi_handler.py)
- Deliverable: mapped controls or mode layer for adaptation/noise parameters.
- Done when: performer can dial these parameters in real time.

## Phase 4: Scene System and Strategy Engine

- [ ] NV-0401 Create scene manager component
- Scope: new file scene manager module, [main.py](main.py)
- Deliverable: scene data model, load/save/recall API, parameter interpolation support.
- Done when: scenes can be recalled programmatically and produce deterministic target states.

- [ ] NV-0402 Add keyboard scene launch 0 to 9
- Scope: [main.py](main.py)
- Deliverable: key mapping from numeric keys to scene slots.
- Done when: each key reliably launches matching scene.

- [ ] NV-0403 Add scene morph time control
- Scope: [main.py](main.py), [control/midi_handler.py](control/midi_handler.py)
- Deliverable: parameterized interpolation duration for scene transitions.
- Done when: transitions can be instant or smoothly ramped by control.

- [ ] NV-0404 Define initial 10 scene presets
- Scope: scene definitions file/module, [docs/network_variance_task_checklist.md](docs/network_variance_task_checklist.md)
- Deliverable: scene set 0..9 matching strategy names from roadmap.
- Done when: each scene has defaults, modulation set, and safety clamps.

- [ ] NV-0405 Add strategy engine hooks
- Scope: [main.py](main.py), optional pattern engine module
- Deliverable: per-scene optional periodic modulation routines.
- Done when: scenes can include static or dynamic behavior profiles.

- [ ] NV-0406 Add Shift+0 to 9 scene save behavior
- Scope: [main.py](main.py)
- Deliverable: overwrite/save current state to chosen scene slot.
- Done when: saved scene reload matches prior state within expected tolerance.

## Phase 5: Diversity Controller (Anti-lock)

- [ ] NV-0501 Create diversity controller component
- Scope: new diversity controller module, [main.py](main.py)
- Deliverable: monitor metrics and issue corrective nudges when collapse is detected.
- Done when: sustained high synchrony triggers bounded corrective behavior.

- [ ] NV-0502 Add anti-lock enable toggle
- Scope: [control/midi_handler.py](control/midi_handler.py), [main.py](main.py)
- Deliverable: runtime toggle for controller enable/disable.
- Done when: off mode has zero interventions; on mode allows interventions.

- [ ] NV-0503 Add anti-lock strength parameter
- Scope: [control/midi_handler.py](control/midi_handler.py), [main.py](main.py)
- Deliverable: scalar controlling intervention magnitude.
- Done when: increasing strength noticeably reduces long lock periods.

- [ ] NV-0504 Add intervention safety clamps
- Scope: diversity controller module
- Deliverable: hard limits on nudges to prevent runaway instability.
- Done when: no out-of-range parameter writes occur during stress tests.

## Phase 6: Performance Mapping and UX Polish

- [ ] NV-0601 Define final control mapping matrix
- Scope: [docs/network_variance_task_checklist.md](docs/network_variance_task_checklist.md), [control/midi_handler.py](control/midi_handler.py)
- Deliverable: final mapping for new macros, toggles, and scene functions.
- Done when: every new major parameter has a reachable live control path.

- [ ] NV-0602 Add scene status and morph progress UI
- Scope: [view/matrix_view.py](view/matrix_view.py)
- Deliverable: visible active scene number/name and morph progress.
- Done when: performer can identify current scene state at a glance.

- [ ] NV-0603 Add anti-lock and diversity UI indicators
- Scope: [view/matrix_view.py](view/matrix_view.py)
- Deliverable: clear status indicators for anti-lock and diversity macro values.
- Done when: toggles and strengths are visually confirmed live.

- [ ] NV-0604 Add compact keyboard help legend
- Scope: [view/matrix_view.py](view/matrix_view.py)
- Deliverable: on-screen legend for 0..9 launch and save/morph controls.
- Done when: user can operate scene workflow without external notes.

- [ ] NV-0605 Backward compatibility pass
- Scope: [main.py](main.py), [control/midi_handler.py](control/midi_handler.py), [model/lif_network.py](model/lif_network.py)
- Deliverable: defaults preserve old behavior when new features are neutral.
- Done when: legacy control flow still works with all new features at zero/disabled.

## Cross-Phase Validation Tickets

- [ ] NV-9001 Fixed-seed regression test script
- Scope: test utility module or script
- Deliverable: repeatable run producing summary metrics for comparison.
- Done when: script outputs stable baseline ranges across repeated runs.

- [ ] NV-9002 Scene distinctiveness verification
- Scope: manual test protocol and optional script
- Deliverable: verify at least 3 scenes show clearly distinct metric signatures.
- Done when: scene metrics differ beyond preset thresholds.

- [ ] NV-9003 Long-run stability test
- Scope: runtime test protocol
- Deliverable: 20 to 30 minute run with scene switching and anti-lock enabled.
- Done when: no crashes, no shape mismatch, no uncontrolled drift.

- [ ] NV-9004 Performance rehearsal checklist
- Scope: docs and runtime sanity checks
- Deliverable: pre-show checklist for controller, scene states, metrics visibility.
- Done when: full workflow can be executed end-to-end from hardware controls.

## Suggested Execution Sequence
- Sprint A: NV-0001 through NV-0006 and NV-0101 through NV-0106.
- Sprint B: NV-0201 through NV-0206.
- Sprint C: NV-0401 through NV-0406.
- Sprint D: NV-0301 through NV-0305.
- Sprint E: NV-0501 through NV-0504 and NV-0601 through NV-0605.
- Continuous: NV-9001 through NV-9004.

## Notes
- Keep micro-step shape safety logic intact while adding delayed and adaptive states.
- Prefer feature flags for expensive additions so performance can be tuned live.
- Validate new controls against existing MPD218 mappings before finalizing scene UX.
