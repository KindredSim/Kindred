# Architecture Rules (Volatile Guardrails)

This file is a permanent but volatile architecture guardrail file. It is intended to persist across sessions, but it must be reassessed as the codebase changes.
Read this file **before** adding features or refactoring. If a rule conflicts with fresh repo evidence, stop and report the conflict instead of silently following stale guidance.
Fresh repo evidence and tests override this file for current-code behavior claims. This file records maintained architecture direction and required guardrails; it is not proof that the current implementation already matches every statement.
These rules apply in the fresh local Kindred repo context under `~/kindred-vDEV` and should remain truthful to the current GUI-stage codebase.

Project root (local shorthand): `~/kindred-vDEV`

Standing architecture rules must use stable anchors such as file paths, class names, function names, test names, and rule IDs. Do not add hard-coded current line numbers unless the line number itself is the behavior under test; gather fresh `nl -ba` evidence in run reports when exact file:line proof is needed.

## Evidence-First Rule

Never guess current repo behavior. `Sure` means fresh evidence has been read in the current task and supports the claim. If fresh evidence is missing, say so and read the relevant code or tests before answering.

For any architecture, ownership, UI, cache, simulation, fitting, solver, validation, or workflow claim, inspect the current repo first unless the answer is explicitly framed as unverified memory or a user-supplied assumption.

Architecture audits, plans, and synthesis artifacts must be transferable to a future reader. Use labels and shorthand where they make the architecture easier to discuss, but define non-obvious labels, issue keys, slice names, acronyms, architecture shorthand, and proposed split names at first use. State whether a split or follow-up is approved, proposed, conditional, or explicitly out of scope. Private scratch notes are allowed only as separate scratch documents; do not leave transferable architecture artifacts dependent on hidden conversation context to explain what their terms mean.

For architecture-sensitive work, the default target is the healthiest truthful boundary supported by fresh evidence. Do not preserve a false seam, fake owner, misleading readiness state, or inaccurate lifecycle claim for diff-size reasons. Do not ask Pedro whether to choose a minimal patch or a healthier architecture when current evidence already shows that the existing boundary is false, duplicated, misleading, or responsible for the failure. The healthy truthful boundary is the default; ask only when that boundary would materially broaden approved product behavior, change compatibility or active-work readability posture, violate an explicit non-goal, or touch an ask-before-touch surface.

The smallest safe patch is acceptable only when it is also the healthiest truthful code shape for the approved scope. Patch size, convenience, test continuity, or preserving transitional surfaces must not outweigh truthful ownership, lifecycle, identity, readiness, invalidation, stale-reply rejection, shutdown/kill cleanup, bounded retained state, or user-visible policy ownership.

Maintainability is the architecture target, not purity. Do not measure architecture quality by the number of owners, adapters, ports, services, or files. A boundary is maintainable only when a future change can be made by reading that boundary and its focused tests without mentally reconstructing the old monolith. A wrapper, port, owner, adapter, or service is not an endpoint if it merely renames access to another object's internals. Prefer stable input/output objects, immutable snapshots, explicit owned state, and narrow Qt/presentation adapters over broad live-object ports.

Composition roots are allowed: a Qt window or controller may compose widgets, signals, ports, and owners. But a composition root must not remain the hidden owner of policy, lifecycle, identity, readiness, invalidation, stale-reply rejection, worker/session ownership, or result publication that another object claims to own. If a new boundary increases the number of places a maintainer must read to understand one behavior, it must own real responsibility or be removed.

## Root vs Private Context Documents

Root `ARCH_RULES.md` is the active local architecture guardrail for this checkout.

Files under `private context cache/`, including private `ARCH_RULES.md`, are derived private-context extracts. They are useful for intent, decisions, freshness, and pending-update awareness, but they are not the same document as this root guardrail and are not proof of current code behavior.

Authority order for architecture work:
1. explicit user instructions for the current task;
2. fresh repo evidence and tests for current-code behavior;
3. root `AGENTS.md` and root `ARCH_RULES.md` for local operating guardrails;
4. validated private context for Pedro's intent, decisions, and known risks;
5. derived private extracts only as convenience summaries.

If root guardrails, private extracts, and fresh repo evidence disagree, stop and report the conflict before planning edits.

## Architecture Rule Maintenance

Whenever architectural work is committed, the same slice must reassess this file and update it if the committed work changes, clarifies, retires, or contradicts any architecture rule, owner map, execution boundary, data-flow boundary, validation rule, or approved exception documented here.

If no update is needed, the final report must say `ARCH_RULES.md update: not needed`.

Do not leave known-stale architecture guidance behind after an architecture commit.

## 1) Validation Protocol (Names)

**Rule V1 (Mandatory):** All name validation must import `validate_name` from `kindred.core.validation`. Do not implement `_validate_name` locally.

- Source of truth: `kindred/core/validation.py`
- Allowed usage:
  - `from kindred.core.validation import validate_name`
  - `from .validation import validate_name` (within `kindred/core/**`)
- Prohibited:
- Defining `def _validate_name(...):` (or any equivalent local helper) in `kindred/core/**`
- Copy/pasting validation logic into `mechanism.py`, `species.py`, or any other module

**Rule V2 (Mandatory):** `kindred.core.simulator.parameter_namespace.MechanismParameterNamespace` owns indexed mechanism parameter identity.

Canonical indexed mechanism parameters are exactly:
- irreversible rates: `k<n>`;
- reversible forward rates: `kf<n>`;
- reversible reverse rates: `kr<n>`;
- equilibrium constants: `Keq<n>`.

Exact protected indexed identifiers are globally reserved: `K<n>`, `k<n>`, `kf<n>`, `kr<n>`, and `Keq<n>`. Direct case-insensitive spelling is normalization only and may resolve only to an existing canonical mechanism identity: `K1` to `k1`, `KF1` to `kf1`, `KR1` to `kr1`, and `KEQ1` / `keq1` to `Keq1` when those canonical identities exist. There are no semantic aliases: `K1` must never mean `kf1`, `kr1`, or `Keq1`; on a reversible-only step where only `kf1` / `kr1` / `Keq1` exist, `K1` is invalid. Exact protected indexed names must resolve through `MechanismParameterNamespace` before any schedule, scalar, observable, fitting, runtime, batch, GUI, or public scan namespace can claim them. Unresolved exact protected indexed names are invalid everywhere. Longer non-exact names such as `K1_test`, `dose_K1`, `my_k1`, and `k1_scale` remain ordinary names.

Schedule request parameters do not have a protected-name exception. For example, intervention `amount_param=K1` is exactly equivalent to `amount_param=k1`: it is valid only if canonical mechanism parameter `k1` exists, and otherwise must be rejected. Independent schedule parameters must use ordinary non-protected names such as `dose`, `pulse_amount`, or `K1_test`. This applies to all schedule `_param` fields.

Bare per-step DSL `K=` remains valid equilibrium-constant input syntax; it is not an indexed parameter identifier. Parameter algebra, symbolic translation/proof, observable symbol tables, fitting, runtime/slider overrides, batch/per-set overrides, GUI scans, and validation surfaces must consume `MechanismParameterNamespace` or an immutable policy snapshot derived from it instead of implementing separate parameter-name rules. Observable/scalar symbol tables must not publish generated `K<n>` aliases.

## 2) Simulation Plumbing (No Cycles)

**Rule S1 (Mandatory):** `dsl.py` and `fast_eq.py` MUST NOT import each other. Shared logic must go to `kindred.core.simulator.common.py`.

- DSL parser: `kindred/core/simulator/dsl.py`
- Fast-equilibrium API wrapper: `kindred/core/simulator/fast_eq.py`
- Shared simulator utilities (cycle-breaker): `kindred/core/simulator/common.py`

**Rule S2:** If a symbol is needed by both DSL parsing and fast-equilibrium policy (types, small helpers, result dataclasses, error translation helpers), it belongs in `kindred/core/simulator/common.py`.

**Rule S3 (Mandatory):** Core reaction semantics must preserve physical sides. `kindred.core.mechanism.Reaction` stores immutable positive irreversible `reactants`, `products`, derived `net_stoich`, and `rate_orders`; `Equilibrium` stores immutable positive reversible forward/back side maps. `rate_orders=None` means default to the reactant side; an explicit empty `rate_orders` mapping means zero-order kinetics. Same-side species such as catalysts remain kinetic participants even when their net stoichiometry is zero.

- DSL mechanism construction must pass physical `reactants` and `products` into `Mechanism.add_reaction`.
- ODE irreversible rate laws must use `rate_orders`, while concentration deltas must use `net_stoich`; reversible rate laws must use the equilibrium forward/back side maps.
- Cache keys, serialization, prepared-runtime payloads, fitting, and batch paths must preserve side-aware reaction identity instead of collapsing to net stoichiometry for semantic identity.
- Do not add net-only reaction construction for parsed DSL reactions. If a future caller truly has no physical side information, it must prove a new explicit contract before adding any net-only API.

## 3) Import Hierarchy (Layering)

**Rule I1 (Mandatory):** Low-level modules must never import high-level modules.

Low-level (foundation) modules include (not exhaustive):
- `kindred/core/validation.py`
- `kindred/core/simulator/common.py`

High-level modules include (not exhaustive):
- `kindred/core/mechanism.py`
- `kindred/core/species.py`
- `kindred/core/simulator/dsl.py`
- GUI modules under `kindred/gui/**`

**Enforcement guidance:**
- If a low-level module needs a type from a higher layer, move the type downward, or define a minimal protocol/type in the low-level layer.
- Do not “fix” layering problems with `try/except ImportError` fallbacks that mask cycles.

## 4) Future Features (Where Shared State/Logic Goes)

**Rule F1 (Mandatory):** Before adding any new feature, read this file. If a new feature requires shared state or shared logic, check `kindred/core/simulator/common.py` first.

Decision checklist:
1. Is this logic shared by `dsl.py` and `fast_eq.py` (or any simulator plumbing)? → put it in `kindred/core/simulator/common.py`.
2. Is this logic shared across core models (species/mechanism/etc.)? → consider `kindred/core/validation.py` or a new small core module under `kindred/core/`.
3. Would adding an import create a cycle? → stop and restructure (extract shared bits downward).

## 5) Regression Prevention (Required Checks)

These checks are minimum guardrails, not a complete blast-radius map. Before each change, assess the specific touched files, ownership boundaries, data flow, and user-visible behavior, then choose targeted tests from that fresh assessment. Do not rely on the lists below as a substitute for test impact analysis.

When multiple review passes are requested or active, wait for every review pass to return before acting on any review output. Synthesize all findings together before classifying them as real, not grounded, or out of scope; do not fix or reject a finding based on a single reviewer while another required review pass is still pending.

Local unit tests are necessary but not sufficient for order-sensitive workflows. When touched behavior crosses controller, runtime, cache, fitting, GUI, and core boundaries, add or update deterministic workflow tests that prove the semantic order of operations, not only final state or isolated method calls.

Prefer bounded test-only event ledgers, semantic trace hooks, state-transition assertions, and fake owners or adapters that record intent-level events. Avoid sleeps, wall-clock ordering, pixel assertions, and broad assertions over noisy private implementation details.

Test-suite size and shape are architecture concerns. Architecture slices must not keep growing local, mock-heavy, or seam-preserving tests while the real workflow contract remains under-proved. When touching a behavior cluster, audit nearby tests for duplicate local proof, fake-seam protection, obsolete transitional vocabulary, and assertions that only prove helper plumbing. Prefer one holistic workflow test through the real boundary over several local tests that collectively approximate the same user-visible contract.

Avoid growing the suite with semantically duplicate tests that only vary tiny numeric values, superficial setup, or old implementation vocabulary. When a touched area already has many nearby tests, audit whether they prove distinct contracts; consolidate toward meaningful behavior classes and workflow guarantees where practical. Consolidation is not optional cleanup when the existing tests protect fake seams or make the architecture harder to change: replace them with workflow-level contract tests, then delete or merge the redundant local tests once the replacement shield is proven.

Do not blindly "prefer updating an existing test" when the existing test is a historical regression guard. If an existing test mostly covers the new contract, first add the clearer replacement or increment and run both old and new tests. Delete or merge the old test only after proving the replacement preserves the regression shield; keep the old test when it catches a distinct failure class. Replacement tests should name the behavior contract more clearly than the removed test.

Do not update tests merely to make the suite pass. A passing suite is evidence, not the objective. If behavior intentionally changes, update tests to state the new behavior contract and preserve a meaningful regression guard; do not shim assertions around the current implementation just to silence failures.

Sequence-sensitive contracts include, but are not limited to: canonical inputs are built before cache identity is computed; stale replies are rejected before display or cache mutation; runtime owners are warmed only from active preparation, run, startup, or explicit runtime-readiness paths, not passive UI refresh; dirty and clean preview promotion and reset happen at the documented boundaries; fitting candidate evaluation and final replay use the same execution semantics; project load applies persisted payload, preferences, and defaults in the intended precedence order.

Before merging any change that touches simulator plumbing or core models:
- Run an import smoke test: `python3 -c "import kindred.core.simulator.dsl, kindred.core.simulator.fast_eq, kindred.core.simulator.common, kindred.core.validation"`
- Run targeted tests relevant to the touched area (and ensure no `ImportError` at import time).

Before merging any change that touches simulation GUI ownership, ports, or controller wiring:
- Run the architecture guard tests that cover the touched seam:
  - `tests/test_arch_simulation_controller_port_usage.py`
  - `tests/test_arch_simulation_ports_contract.py`
  - `tests/test_main_window_app_wiring_contract.py`
  - `tests/test_main_window_preview_session_boundary.py`
  - `tests/test_main_window_variable_runtime_boundary.py`
  - `tests/test_main_window_mechanism_helpers_boundary.py`

Before merging any change that touches import pipeline types or resolver:
- Run tests: `tests/test_import_config.py`, `tests/test_import_config_dialog.py`, `tests/test_import_integration.py`, `tests/test_import_sweep_regressions.py`, `tests/test_excel_import.py`

Before merging any change that touches fitting window, worker, or species table:
- Run tests: `tests/test_fitting.py`, `tests/test_fitting_cancellation_contract.py`, `tests/test_fitting_failure_behavior.py`, `tests/test_fitting_launch_owner_contract.py`, `tests/test_fitting_window_gui_fixes.py`, `tests/test_fitting_mixin_dsl_update_errors.py`, `tests/test_fitting_objective_direct_module.py`, `tests/test_fitting_objective_pipeline_contract.py`

Before merging any change that touches species table or fit-universe:
- Run tests: `tests/test_species_sliders_logic.py`, `tests/test_species_statistics_result_selector.py`, `tests/test_species_statistics_table_layout.py`

## 6) GUI Architecture (Controllers and Layout)

### Simulation Ownership and Wiring

**Rule G1 (Mandatory):** The current simulation architecture is the source of truth. `MainWindow` composes simulation plumbing by creating explicit owners first, then wiring them into `SimulationUiPorts`.

Current owner map:
- `slider` port → `MainWindowPreviewSession`
- `runtime` port → `MainWindowVariableRuntime`
- `mechanism_helpers` port → `MainWindowMechanismHelpers`
- `dialogs` port → `kindred.gui.simulation_dialogs.SimulationDialogs`
- `settings` port → `kindred.gui.simulation_settings_owner.SimulationSettingsOwner`
- `run_ui` port → `kindred.gui.simulation_run_ui_owner.SimulationRunUiOwner`
- `results` port → `kindred.gui.controllers.results_controller.ResultsController`
- `provenance` port → `kindred.gui.simulation_provenance_owner.SimulationProvenanceOwner`
- `solver` port → `kindred.gui.simulation_solver_owner.SimulationSolverOwner`
- `mechanism` port → `kindred.gui.simulation_mechanism_owner.SimulationMechanismOwner`
- `batch` port → `kindred.gui.simulation_batch_owner.SimulationBatchOwner`

Do not revert these seams back to broad `MainWindow` injection for slider/runtime/mechanism-helper work.

**Rule G2 (Mandatory):** Ownership must be truthful. State belongs on the component that actually owns the behavior.

Current truthful owners:
- Preview gesture state, pending slider values, and preview debounce timers belong to `MainWindowPreviewSession`
- Prepared preview runtime, runtime invalidation, and variable metadata belong to `MainWindowVariableRuntime`
- Authoritative parsed-structure snapshot reuse for canonical GUI structure consumers belongs to `kindred.core.mechanism_structure_snapshot.MechanismStructureSnapshotOwner`, adapted through `MainWindowMechanismHelpers`
- Last-mechanism snapshot/context and bounded mechanism-helper coordination belong to `MainWindowMechanismHelpers`
- Authoritative mechanism transition epoch/identity, pending-init transition suppression, transition-owned readiness deferral, and transition outcomes for runtime invalidation, active-work supersede, and stale-result protection belong to `kindred.core.mechanism_runtime_transition.MechanismRuntimeTransitionService`
- Simulation modal/message-box display belongs to `kindred.gui.simulation_dialogs.SimulationDialogs`
- Simulation persistent settings lifecycle and writes belong to `kindred.gui.simulation_settings_owner.SimulationSettingsOwner`
- Simulation run control enabled state, runtime-ready gate, progress, and status text belong to `kindred.gui.simulation_run_ui_owner.SimulationRunUiOwner`
- Simulation plotted-result presentation and batch-selection display belong to `kindred.gui.controllers.results_controller.ResultsController`
- Last-run provenance and CTC state for GUI consumers belong to `kindred.gui.simulation_provenance_owner.SimulationProvenanceOwner`
- Simulation solver-control reads and startup solver defaults belong to `kindred.gui.simulation_solver_owner.SimulationSolverOwner` as a thin Qt adapter over existing solver controls
- Simulation mechanism-session text, mechanism editor controls, preview parameter-store schema/fingerprint reads, and mechanism-port override application belong to `kindred.gui.simulation_mechanism_owner.SimulationMechanismOwner` as a thin Qt adapter over existing mechanism session/editor/runtime owners
- Simulation batch table/store selection reads, batch model validation, active batch display selection state, and batch-port display routing belong to `kindred.gui.simulation_batch_owner.SimulationBatchOwner` as a thin Qt adapter over existing batch store/model/cache/display owners
- Mutable GUI batch-run context storage and queue/session transitions belong to `kindred.gui.controllers.batch_run_context_owner.BatchRunContextOwner`, including batch start-run context construction, completion-policy context normalization/serialization, current queue-position hints, completion summaries, runtime-input staleness comparison against supplied current epochs, cache-key updates, parallel success/failure transitions, serial success cursor advancement, serial stale-prefix consumption, active serial runtime-input supersede cursor advancement, guarded stale callback completion/deactivation, runtime-waiting transitions, and deactivation. `SimulationController` may still orchestrate batch transitions, but it must not reintroduce a raw context dict as controller-owned state.
- Simulation user run intent, task/plan assembly, run-start orchestration, cache administration, and worker lifecycle still belong to `SimulationController`. Callback-captured run/request/owner/cache identity belongs to `kindred.gui.controllers.simulation_callback_identity.SimulationCallbackIdentity` and must be passed unchanged through completion and error dispatch once captured. Parallel batch callback identity must use a slim/shared callback context owned by `BatchRunContextOwner` plus per-set callback identity such as set id, submitted-plan simulation identity, and preview cache token; it must not copy full per-set plan, mechanism text, prepared payload, or simulation identity maps once per submitted parallel set. Completion result materialization belongs to `kindred.gui.controllers.simulation_result_materialization.SimulationResultMaterializationOwner`, including completion-mechanism fallback resolution, energy-mode materialization side effects, primary-result mechanism memory, batch species sync after primary completion, and primary-result control refresh. Completion callback policy must remain composition-only: stale callback rejection/decision, cache-key normalization, cache truth/publication, result display, provenance handoff, pending-init completion, batch success transitions, and final lifecycle effects must live in named owner/effect boundaries such as `BatchRunContextOwner`, `SimulationCacheAdmin`, `ResultsController`, `SimulationProvenanceOwner`, and `SimulationLifecycleEffectOwner`. `SimulationController._on_simulation_complete()` must not return to a monolithic cache/display/provenance/batch policy method, and publication must not rediscover plan, set, or cache identity after callback handling.
- Batch dispatch initials materialization belongs to `kindred.gui.controllers.batch_dispatch_materialization.BatchDispatchMaterializationOwner`, including canonical batch initials reads, pending-init seed overlay, preview-initial overlay for fast-mode dispatch, and run-preparation plan input initials. `SimulationController` may compose the owner but must not duplicate that materialization policy in run preparation, parallel dispatch, or serial dispatch paths.
- Warm contained GUI simulation owner slot ownership for ordinary and preview serial runs currently belongs to `SimulationController`, using `kindred.core.simulation_containment.WarmSimulationOwner`; contained serial worker creation and worker identity stamping belong to `kindred.gui.controllers.serial_worker_launch.ContainedSerialWorkerLaunchOwner`; reusable process lifecycle, READY/ACCEPTED gating, request identity, stale-reply rejection, timeout restart, and idempotent close/kill are delegated through `kindred.core.simulation_runtime_service.SimulationRuntimeOwner` to `kindred.core.containment_kernel.ContainmentKernelOwner`
- Batch run/session identity, queue membership, keep-alive intent, active/superseded/shutdown session state, polling/drain semantics, and runtime lifecycle policy belong to `kindred.core.batch_runtime_session.BatchRuntimeSession`.
- Parent-owned warm batch lane pool lifecycle, request worker tracking, completion metadata, lane/request mechanics, soft-supersede generations, stale-reply rejection, and scheduling belong to the non-GUI batch runtime lane owner in `kindred.core.batch_containment`; `ParallelBatchExecutor` is a temporary narrow controller adapter over `BatchRuntimeSession`, not a durable endpoint. Parallel batch runtime readiness state, nonblocking warm lifecycle, and run-path ready/not-ready decision snapshots belong to `kindred.gui.controllers.parallel_batch_runtime_readiness_owner.ParallelBatchRuntimeReadinessOwner`. Startup should schedule nonblocking batch runtime prewarm, and the run path must reuse an existing non-stale warm batch lane pool without a second blocking warm before submission.
- Reusable containment lifecycle and identity contracts belong in non-GUI containment substrate where reasonable, with Qt/GUI worker classes acting as narrow adapters
- Runtime-initial-only transitions must not clear authoritative parsed-structure snapshots or force derived structural UI refresh. They may invalidate affected cache/display and stale-publication authority while preserving structural reuse.

Do not move these responsibilities back onto `MainWindow` or `SimulationRunState` as convenience mirrors.

**Rule G2a (Direction — Simulation Runtime Layering):** Simulation architecture should move incrementally toward a three-layer split where it is reasonable and supported by fresh evidence:

- scientific core owns preparation, solving, and finalization;
- a non-GUI simulation runtime/application layer owns jobs, owners, lanes, containment lifecycle, request identity, stale-reply rejection, timeout and cancellation cleanup, idempotent close/kill, contained payload identity, and reusable execution/runtime ownership;
- GUI owns user intent and presentation policy, including clicks, preview requests, selection/display state, enabled/disabled controls, inline status, modal versus non-modal display, dirty-preview UX, and other user-facing workflow decisions.

This rule is directional and partial-current. Do not document or assume the full split has landed until fresh repo evidence proves it. Do not force a big-bang migration, and do not move GUI presentation policy or dirty-preview UX into scientific core. When a containment lifecycle or identity responsibility is touched, prefer a reusable non-GUI primitive plus a narrow Qt/GUI adapter over adding GUI-only lifecycle or identity rules.

**Rule G2a-C (Mandatory — Campaign A Expansion/Duplication Cleanup):** Campaign A is not complete while runtime/controller extraction mainly adds owners around still-growing monoliths or repeats the same policy across old and new surfaces. Continue read-only-first cleanup before any completion claim for Campaign A or the simulation runtime/controller architecture.

That session must treat diff size as evidence to classify, not as proof by itself. New code may be legitimate when it owns lifecycle, identity, readiness, invalidation, stale-reply rejection, shutdown/kill, user-visible policy, or typed execution authority. New code is debt when it duplicates policy already owned elsewhere, preserves broad pass-through compatibility, copies large mutable context where scalar identity would do, or exists only so tests can keep calling transitional surfaces.

Cleanup passes must inventory at least these recurring debt families before editing:
- completion identity fallback split across callback and publication owners;
- callback identity/context escape hatches that reintroduce full batch context, per-set plan maps, mechanism text maps, prepared payload maps, or simulation identity maps per submitted set;
- runtime readiness snapshot/status construction repeated between controller helpers, batch readiness owners, and core readiness owners;
- scalar/coercion/default helpers repeated with divergent non-finite or default semantics;
- fitting worker launch defaults duplicated between worker constructors and launch owners;
- dependency-lambda pass-throughs where the controller remains the real policy owner despite an extracted owner name.

Required output for cleanup passes: a surface matrix classifying each item as `keep real owner`, `merge into existing owner`, `move to shared helper`, `delete transitional surface`, or `needs product decision`, with file:line evidence and targeted tests for any behavior-sensitive consolidation.

**Rule G2b (Mandatory — Runtime Readiness Is a Product Contract):** Runtime readiness is user-facing, not an optional optimization. Ordinary expected simulation interactions must not be the place where Kindred pays avoidable import, process-startup, owner-warmup, lane-warmup, or prepared-runtime construction cost.

This applies to Run Selected, slider preview and drag, batch and multi-set simulation, preview/runtime owner reuse, fitting/runtime containment, and any future path where startup or warm ownership could leak into a user action. If an import-heavy backend, process owner, worker lane, prepared runtime, or reusable evaluator is needed for a normal interaction, the runtime/application layer must make it ready before the user action depends on it, or expose an explicit non-blocking readiness state. Do not silently defer that cost into the first click, first drag, first selected run, or repeated unchanged interaction.

Implementation implications:
- if the app presents a simulation/fitting/runtime control as usable, the runtime needed for that control's normal action must already be ready;
- if readiness is not available yet, the UI must truthfully show that state instead of pretending the control is ready and then blocking on first use;
- lazy startup is acceptable only for workflows that are not entered, not visible, not selected, or explicitly optional/heavy, and those workflows must expose a clear readiness state when entered;
- the non-GUI runtime/application layer owns readiness, reusable owner identity, invalidation, lifecycle, stale-reply rejection, shutdown/kill, and reuse policy;
- GUI/controller code may request or schedule readiness, but must not become the durable owner of startup/import/warm lifecycle truth;
- runtime owners may be invalidated and rebuilt when the real mechanism/runtime identity changes, but unchanged interactions must reuse already-ready owners rather than synchronously recreating or warming them;
- this rule strengthens the three-layer architecture; it does not justify moving GUI presentation policy into core or bypassing containment ownership.

Testing implications:
- do not use wall-clock assertions for this contract;
- use deterministic owner/factory/lane/evaluator ledgers to prove first-use and repeated-use interactions reuse runtime-ready owners instead of synchronously recreating, importing, warming, or preparing them;
- any feature, fix, or refactor touching runtime readiness must name the affected user interaction and include targeted tests proving the readiness/reuse contract for that interaction.

**Rule G2c (Mandatory — No Fake Runtime Readiness Or Facades):** "App opens; things work" is the product-level runtime readiness contract for ordinary visible simulation interactions. A runtime path is ready only when the exact owner, runtime, lane, prepared payload, or evaluator that the user action will use is already usable without paying avoidable first-use startup, import, READY, preparation, owner replacement, or lane warmup cost on the action path.

The following are not readiness:
- a warm request was scheduled;
- a background warm task is still running;
- a process or lane was merely started;
- an owner, pool, session, adapter, or controller object exists;
- a generic or empty startup owner exists when the first real mechanism/runtime identity still needs replacement;
- a controller boolean says eager creation happened;
- tests prove only final success after the user action already paid the warmup cost.

For Run Selected, slider preview and drag, batch simulation, fitting/runtime-backed controls, and future runtime-backed workflows, visible usable UI state must correspond to truthful readiness for the exact execution path. If a real mechanism or runtime identity change invalidates readiness, the runtime/application layer must rebuild the needed owner or expose a truthful non-ready state before normal interaction depends on it; changing that user-visible behavior requires explicit product approval.

No owner, session, adapter, service, or facade is an architectural endpoint unless it owns the lifecycle truth it claims: readiness, reusable identity, invalidation, stale-reply rejection, active-work accounting, shutdown/kill behavior, and reuse policy for the workflow it covers. Do not represent partial substrate state as readiness, and do not claim a boundary is fixed while the first visible interaction still pays hidden runtime cost.

**Rule G2d (Mandatory — Maintainable Boundary Test):** Architecture work must improve maintainability, not just seam count. A new owner/port/service boundary is acceptable only when it lets a future maintainer understand or change the behavior by reading the boundary, its owned state, and its focused tests. Do not replace direct monolith access with a broad pass-through port slab and call that architectural closure. Narrow adapters are acceptable at the Qt/presentation edge; non-Qt policy should move toward typed snapshots, explicit command/result objects, or owner-owned state.

For fitting specifically, `FittingWindow` may remain the Qt composition root for widgets, layout, signals, presentation, and dialog lifetime. Fitting launch identity, runtime readiness, accepted-launch construction, worker lifecycle, lane-budget authority, stale-result rejection, and result-application policy must not be hidden behind broad pass-through methods on the window. When those policies are touched, prefer `FittingRuntimeIdentity`, `FittingRuntimeAcceptedLaunch`, typed launch-input snapshots, explicit worker-launch inputs, and owner-owned state over live `FittingWindow` access. A private window port is maintainable only if it is narrow, presentation-oriented, and does not force maintainers to read both the old window internals and the new owner to understand one behavior.

**Rule G2c-F (Direction — Fitting Readiness Config/Launch Simplification):** After checkpoint commit `aaeccda`, the next fitting readiness architecture direction is a separate simplification slice, not a continuation of narrow reviewer cleanup. This is future direction, not a current-code claim.

Target shape:
- one typed fit run/launch object representing the actual run after GUI filtering;
- one collector for passive readiness and explicit Run Fit;
- one validation policy over that typed object;
- passive readiness consumes silent blocked/validation reasons;
- explicit Run Fit renders modal/user-facing errors from the same validation result;
- readiness identity, run stamp, accepted launch, and worker inputs derive from that validated object.

Ownership guardrails:
- `FittingWindow` and fitting tabs own GUI state, mechanism-derived parameter/species table freshness, explicit table-state stamping, and non-reentrant refresh; `ParametersIcsTab` owns fitting parameter table state and parameter-config collection for both explicit Run Fit and passive readiness snapshots; `kindred.gui.fitting.evaluator_state.FittingEvaluatorStateOwner` owns mutable current base-evaluator state, prepared-metadata lookup, and the builder/reuse decision for fitting runtime identity;
- `FittingRuntimeReadinessController` owns runtime lifecycle, desired/active/ready identity, preparation lifecycle, stale identity rejection, ready-session reuse/close policy, and accepted-launch publication;
- `FittingRuntimeReadinessController` must not mutate GUI table/species state;
- `FittingRuntimeIdentity` / `FittingRuntimeAcceptedLaunch` are fitting's run-level launch boundary and are built once per launch. Do not copy the simulation callback-identity pattern into fitting unless a fresh fitting audit proves an equivalent per-callback identity need. `GlobalFitWorker` executes accepted launch data and should not rediscover launch readiness beyond narrow defensive checks for supported direct construction.

Demolition rule for this slice: inventory every fitting config, validation, readiness, and launch path before implementation and classify each as `keep owner`, `move into typed run owner`, `delete`, or `test-only fake seam`. Do not preserve duplicate collectors, duplicate validators, wrappers, facades, private-test-driven compatibility paths, or transitional surfaces merely to keep the current maze working under new names.

**Rule G2d (Mandatory — No Unapproved Hard-Coded Values):** Do not introduce hard-coded values unless Pedro explicitly pre-approved that exact value and tradeoff. This applies before implementation and still applies if the value would be stored as a named default, setting, constant, test expectation, timeout, capacity, threshold, cache size, retry cadence, worker/lane count, validation tolerance, or any other user-visible or maintenance-sensitive policy. After approval, store the value in the appropriate named configuration/default location, make it user-configurable whenever it affects user-visible capacity or workflow behavior, and cover it with tests where relevant.

**Rule G3 (Mandatory):** `SimulationUiPorts` is explicit and partitioned. `SimulationController` must route UI work through the appropriate sub-port such as `self.ui.slider`, `self.ui.runtime`, `self.ui.mechanism_helpers`, `self.ui.batch`, `self.ui.run_ui`, `self.ui.results`, `self.ui.provenance`, `self.ui.dialogs`, `self.ui.settings`, `self.ui.mechanism`, and `self.ui.solver`.

Forbidden:
- adding `SimulationUiPorts.__getattr__` or any generic cross-port fallback
- reintroducing flattened `self.ui.<method>` access for behavior that belongs to an explicit sub-port
- hiding cross-port routing behind broad convenience forwarding that makes ownership ambiguous

**Rule G4 (Mandatory):** No fake seams except explicitly transitional ones. A seam is only real if it owns real state or a real bounded responsibility.

Allowed:
- transitional adapters that are explicitly marked as transitional in the change rationale and are clearly bounded
- helper/owner objects that physically own state or enforce a real boundary

Forbidden:
- wrapper classes that only preserve broad `MainWindow` reachthrough while pretending to be finished architecture
- documentation that describes aspirational ownership as if it already exists in code

Boundary strictness gate:
- Any new owner, service, session, adapter, port, or wrapper must name the responsibility it owns before it is treated as architecture progress.
- Acceptable owned responsibilities include state, lifecycle, identity, readiness, invalidation, stale-reply rejection, active-work accounting, shutdown/kill cleanup, bounded retained state, or user-visible presentation policy.
- Forwarding-only seams are allowed only as explicitly transitional adapters. The change rationale must name the old path they are replacing and the condition for later removal.
- A boundary is not complete until the previous owner no longer controls the moved responsibility, or the remaining forwarding is explicitly documented as transitional debt.
- For architecture slices, success is measured by moved ownership and deleted or narrowed legacy responsibility, not by adding a new object, port, facade, or boolean.
- When a bug or failure is caused by a false ownership/readiness/lifecycle boundary, fixing the boundary is in scope for the slice unless doing so triggers a documented ask-before-touch condition. Do not offer or select a symptom patch that leaves the false boundary in place.
- Tests or audits for new boundaries should prove the owned behavior directly, using deterministic owner/factory/lane/evaluator ledgers where lifecycle, readiness, identity, or retained state is involved.

**Rule G5:** For simulation refactors, introduce the owner before the wiring cutover. Move the owned state/logic into the owner, wire it through `SimulationUiPorts`, verify the architecture tests, and only then remove obsolete direct access or forwarding if that cleanup is in scope.

**Rule G6:** Do not use speculative redesign plans as justification to reopen the generic fallback model or broad `MainWindow` injection.

### Controller UI Pattern

Controllers that own complex threading, caching, or state-machine behavior should prefer a narrow UI port/adapter boundary instead of a raw `MainWindow` reference. `SimulationController` is the primary enforced example and must continue to use explicit `SimulationUiPorts` sub-ports.

A raw `MainWindow` reference remains an allowed exception only where the controller is intentionally retained as a direct UI-routing surface and that ownership is explicit and narrow (for example, `ProjectController`).

### UI Declunking Pattern

Do not add new nested `QGroupBox` elements for layout. Existing top-level or legacy `QGroupBox` sections are not automatically violations; avoid opportunistic churn unless the task is layout cleanup. For new or touched UI, prefer flat layouts using bold `QLabel` headers, `QSpacerItem` for pushing elements, strict `setMaximumWidth()` for input boxes (like spinboxes), dynamic `show()/hide()` for empty states, and utilities from `ui_helpers.py` (like `make_placeholder_label`).

## 7) ODE Integration (SciPy Mandate) and Continuity

**Rule O1 (Mandatory — SciPy Mandate):** ODE integration must use SciPy `scipy.integrate.solve_ivp` exclusively. Current exposed/normalized Kindred solver choices are `Radau` and `BDF`.

- Pure-Python solver implementations and “fallback solvers” (for example Rosenbrock-style pure-Python integrators) are strictly forbidden and must not be reintroduced. They have been removed permanently.
- If a UI/API offers a solver choice, it must restrict choices to `Radau` and `BDF`. Do not silently route users to ad-hoc non-stiff methods.

**Rule O2 (Mandatory — C¹ positivity bridging inside RHS/Jacobians):** Any positivity/feasibility enforcement that happens *inside* an ODE RHS function or Jacobian must be **C¹-continuous** (at least).

- Forbidden inside RHS/Jacobian: hard clamps such as `np.maximum(x, 0)`, `np.clip(x, 0, ...)`, `max(0, x)`, or piecewise boolean masking that introduces kinks/discontinuities.
- Required: use a smooth C¹ bridge (for example a cubic bridge over a small transition band near 0) so stiff/implicit solvers (`Radau`/`BDF`) do not stall or take vanishingly small steps at discontinuities.

**Rule O3 (Mandatory — Windows-safe simulation containment):** Kindred is a Windows-first desktop application. Any process boundary used for simulation solving must remain valid under Python multiprocessing `spawn` and frozen Windows application packaging.

- Simulation containment must pass serializable execution payloads across process boundaries and reconstruct solver requests inside the child process.
- Do not make required behavior depend on `fork` inheritance of Python callables, monkeypatches, prepared RHS closures, Qt objects, or other parent-process state.
- Shared containment kernel modules must stay stdlib-only at import time. They must not import NumPy, SciPy, Qt/PySide, solver modules, batch containment/parallel modules, simulation containment, or fitting containment during module import.
- GUI ordinary and preview containment must use explicit warm owner lifecycle, READY/ACCEPTED protocol gating, request/epoch stale-reply rejection, and kill/restart on active timeout or cancellation. The generic kernel may also gate caller-owned reply dimensions when a higher-level owner provides them, but batch and fitting policy remain owned by their respective higher-level adapters until explicitly migrated.
- Batch containment must use parent-owned warm lanes, not nested per-solve children inside batch workers. The non-GUI batch runtime lane owner must preserve lane-local prepared-runtime reuse, own request worker tracking and completion metadata, treat soft supersede as a generation/staleness transition, prefer available compatible lanes, gate active solve timeout after ACCEPTED, map timeout/protocol failures to the affected set, and reject stale run/request/set/epoch replies.
- Fitting containment must use a readiness owner plus the non-GUI fitting runtime session for exact prepared fitting runtimes. The GUI-adjacent fitting readiness owner owns desired/active/ready fitting identity, preparation worker lifecycle, supersede/cancel/close policy, stale preparation rejection, and readiness state transitions before Run Fit can be accepted. The fitting runtime session owns bounded warm evaluator lanes, reusable scheduler/executor lifecycle, reusable prepared payload identity, cancellation-driven kill/close, and deterministic lifecycle ledgers. GUI fitting workers may drive optimization and presentation callbacks, but they must route configured serial fitting evaluation through the session owner rather than recreating containment or scheduler ownership per candidate or per callback.
- Contained payload identity must not rely on raw Python mapping equality over NumPy arrays or rich runtime objects. Use a containment-owned identity/comparison primitive when deciding owner reuse or request elision.
- Preview containment timeouts must preserve dirty staged state and surface a truthful dirty/no-preview state without modal error dialogs; explicit ordinary run failures remain user-visible command failures.
- Platform-specific optimizations are allowed only when the Windows `spawn` path remains fully functional and covered by tests.

**Rule O4 (Mandatory — Species Intervention Schedule Authority):** Species intervention schedules are core execution inputs, not GUI, batch, preview, or fitting-local reinterpretations.

- `kindred.core.intervention_schedule.InterventionSchedule` owns the typed schedule model, canonical payload, validation, normalization, fingerprint, and request-time parameter resolution for schedule directives.
- Schedule payloads must flow through `SimulationPlan` and its nested `SimulationExecutionRequest`, into `PreparedSimulationRun` and `SimulationRequest`, and then execute in the shared solver path through `kindred.core.simulator.intervention_schedule_execution.InterventionScheduleExecutionOwner`; `kindred.core.simulator.solvers.solve_ode()` remains the public solver composition entry point.
- Ordinary simulation, batch simulation, containment, and fitting candidate/final replay paths must preserve schedule payloads through the shared plan/request/prepare/solve boundary. They must not invent separate schedule semantics or silently drop schedules when prepared payloads are absent or stripped.
- Cache identity, preview identity where applicable, fitting run stamps, and process payloads must include schedule identity/fingerprint when the executable request contains a schedule.
- Schedule execution must be real executor behavior owned by `InterventionScheduleExecutionOwner`. Instant set/add/remove/clear events, repeated/pulsed events, source/sink intervals, reservoir/clamp intervals, state-triggered interventions, and fit-parametrized schedule fields may be segmented or resolved per request/candidate, but they must not be represented as presentation annotations, post-processing, or hidden per-path corrections.
- State-triggered schedule execution must resume from the solver event-time state, not from the last requested output sample. Repeated triggers must re-arm when `min_interval` makes them eligible inside the same solver segment, scheduled trigger events must remain terminal regardless of surplus user `event_terminal` flags, user-level solver options such as `first_step` must be normalized for internal segment bounds before calling SciPy, active-interval Jacobian callables, Jacobian sparsity hints, and executable symbolic Jacobians must be disabled truthfully where interval RHS wrapping makes them invalid, and clamp/reservoir interval values must be applied according to interval activity at the actual trigger time.
- Fittable schedule fields use the core schedule payload and `SimulationExecutionRequest`/fitting candidate parameter values. They must not be reinterpreted locally by fitting, batch, preview, or GUI code.
- Invalid schedule definitions should fail through the shared preparation/solver failure path with schedule-specific stage/context rather than becoming opaque worker or GUI errors.
- GUI schedule editing remains separate future work unless explicitly approved for the current slice. Plot annotations are approved only as optional solved-provenance display aids: they must be off by default, user-toggleable, fully hideable, and must never be treated as schedule authority or as proof that solver/fitting schedule behavior works.

**Rule O5 (Mandatory — Symbolic Core Boundary):** `sympy==1.14.0` is a required runtime scientific dependency. It is not an optional backend, plugin, optional extra, or fallback-capable enhancement. `kindred.core.symbolic` owns SymPy imports, backend/version metadata, exact symbolic proof helpers, symbolic expression translation, generated symbolic RHS/Jacobian structure artifacts, immutable evaluation snapshots, artifact fingerprints, and typed symbolic unsupported-case errors.

- Wegscheider cyclicity proof may consume symbolic proof services, but numeric probes, complex-graph-only detection, or local symbolic-looking checkers are not proof authority.
- Generated symbolic Jacobians may enter solving only through the existing `SimulationRequest.jacobian_func` / `solve_ivp(jac=...)` boundary for solver modes that consume Jacobians, currently `BDF` and `Radau`. Do not create a parallel solver path or replace SciPy integration with SymPy.
- Symbolic Jacobian artifacts must separate reusable symbolic structure from immutable evaluation snapshots. Mutable `RateBinding` objects may provide snapshot values, but generated callables and cache/provenance identity must not depend on later mutation of those bindings.
- Symbolic artifact identity must include backend/version/profile, source fingerprint, expression/artifact fingerprint, structure fingerprint, evaluation-snapshot fingerprint, and enough provenance to prevent stale prepared/cache/provenance reuse.
- Scheduled temperature, active intervention intervals, unsupported kinetics, unsupported dynamic expressions, mutable bindings without an immutable snapshot boundary, or missing identity must disable or reject symbolic Jacobian use truthfully. These unsupported paths must pass no Jacobian data: `jacobian_func is None` and `jac_sparsity is None`.
- `jac_sparsity` is a solver-boundary hint only for explicitly supplied, proven-valid sparsity structures. Simulation preparation must not synthesize sparse-Jacobian callbacks or use sparsity hints as a fallback when symbolic Jacobian generation is unavailable.
- GUI, batch, fitting, and solver modules must not import SymPy directly or duplicate symbolic proof/generation logic; they should consume prepared/core symbolic artifacts or typed failure results.

## 8) Dataset Import Pipeline

**Rule D1 (Mandatory):** Each Excel sheet is a fully independent import unit. No cross-sheet validation, no cross-sheet state sharing during resolution. Each sheet has its own `SheetImportIntent` and resolves independently via `resolve_import_plans` in `kindred/gui/widgets/import_config.py`.

**Rule D2 (Mandatory):** `has_unit_row` is a physical property of the file row, computed from ALL columns. It must never be scoped to selected columns. Unit extraction (which units per column) IS scoped to selected columns via `relevant_column_names`. These are separate concerns in `detect_units_from_row_mapping`: `looks_like_unit_row(full_values)` uses the full row, while scoped column extraction uses `relevant_column_names`.

**Rule D3 (Mandatory):** Factor computation uses intent values (`SheetImportIntent.concentration_units`), not raw detection values. Detection auto-populates intent, but the user can override via the combo box. The resolver reads intent, never detection.

**Rule D4 (Mandatory):** No clone paths or broad `or` chains masking incomplete import pipeline state. Incomplete resolver/dialog state must be constructed upstream or rejected clearly; do not hide it with generic fallbacks.

**Rule D5 (Mandatory):** `rebuild_intent_for_target` in `kindred/gui/widgets/import_config.py` rebuilds `concentration_units` per target column. Target detection wins when `target_detection.detected_conc_unit_by_column[col]` is present; when the target has no detected unit for that column, the function carries `source_intent.concentration_units[col]`. Source detection remains physical facts about the source file, not portable.

**Rule D6:** ImportConfig dataclass composition: `ImportConfig` → `UserImportIntent` (file-level) + `per_sheet_intents: Tuple[Tuple[Optional[str], SheetImportIntent], ...]` (per-sheet) + `plans: Tuple[ResolvedSheetPlan, ...]` (validated) + `remaining_file_template: Optional[SheetImportIntent]`. This structure in `kindred/gui/widgets/import_config.py` is the source of truth.

**Rule D7 (Mandatory):** Frozen dataclasses with dict or list fields must use `__post_init__` with `object.__setattr__` to defensively copy all mutable fields. Surface immutability of `frozen=True` does not prevent callers from mutating dict/list contents after construction. Examples in `kindred/gui/widgets/import_config.py`: `UnitDetection.__post_init__`, `SheetImportIntent.__post_init__`, and `ResolvedSheetPlan.__post_init__`.

## 9) Fitting Window Architecture

**Rule F2 (Mandatory):** IC state lives in explicit `_ic_pending[ds_id][species]` / `_ic_applied[ds_id][species]` dicts in `UnifiedSpeciesTable`. The species table is a view of these dicts, not the state itself. Any `_populate_table()` call must not destroy pending IC edits.

**Rule F3 (Mandatory):** `_fit_targets_available_by_dataset` stores the fit-universe (observed ∩ modeled), not raw observed columns. `UnifiedSpeciesTable._recompute_fit_universe()` is the sole method that reconciles this. No other code path may directly modify the fit-universe dict.

**Rule F4 (Mandatory):** Worker signals must be disconnected centrally on completion/error BEFORE any UI work or cleanup scheduling. `QueuedConnection + deleteLater()` is a race condition if signals are not disconnected first. Current enforcement point: `_disconnect_fit_worker_signals()` runs before `_dispatch_fit_worker_finished()` or `_dispatch_fit_worker_error()` continues with UI work or cleanup scheduling.

**Rule F5 (Mandatory):** `_set_running_state(False)` must happen BEFORE any modal `QMessageBox`. Modal dialogs pump the event loop, so timers and queued callbacks can fire during the dialog with stale state. Enforced through `FittingWindow._handle_global_fit_complete`, `FittingWindow._on_worker_error`, and the centralized failed-run visual cleanup helper in `kindred/gui/fitting/window.py`.

**Rule F6:** Results tab uses visible-subtab-only live updates. Hidden subtabs are tracked via `_stale_plot_view_keys` and refreshed by `RunResultsTab` when needed. Worker emits lightweight params every 0.25 s (`best_update_interval_s`) and heavyweight plot arrays every 2.0 s (`plot_update_interval_s`).

**Rule F7 (Mandatory):** When fitting code claims bounded active lanes, workers, futures, or concurrent tasks, any associated retained warmed/reusable state must be bounded by the same reusable owner identity or by an explicit independent cap. Do not key retained warmed evaluator state by dataset, request, run, or task identity unless that identity is independently capped.

**Rule F8 (Mandatory):** `kindred.gui.fitting.runtime_readiness.FittingRuntimeReadinessController` owns fitting runtime readiness and accepted-launch publication for `FittingWindow`: desired identity, active preparation identity, ready identity, preparation worker/thread lifecycle, supersede/cancel/close policy, stale preparation rejection, readiness state transitions, deterministic readiness ledgers, and the resolved `FittingRuntimeAcceptedLaunch` that visible Run Fit consumes. Deferred evaluator construction must resolve the concrete evaluator before deciding whether a runtime session is required; exact `SerialFittingEvaluator` instances require a ready `FittingRuntimeSession`, while generic callable/evaluate-series evaluators must not falsely require one. `FittingRuntimeIdentity` and `FittingRuntimeAcceptedLaunch` are fitting's current launch identity boundary: they may contain run-level datasets/config/stamp/evaluator inputs because they are built and accepted once per launch, not once per dataset callback. Do not introduce simulation-style per-dataset callback snapshots or full-context copies into fitting without a fresh audit proving that fitting has the same callback identity problem. `kindred.gui.fitting.evaluator_state.FittingEvaluatorStateOwner` owns the mutable current base evaluator, prepared-metadata lookup, and builder/reuse decision that feed fitting runtime identity; do not reintroduce a parallel mutable evaluator slot on `FittingWindow`. `kindred.core.fitting_runtime_session.FittingRuntimeSession` owns reusable fitting evaluator runtime state for global fitting. `kindred.core.analysis.global_fit_execution` owns core global fitting execution policy: candidate dataset input materialization, serial/runtime-batch dataset evaluation dispatch, objective residual construction, final replay, fitting diagnostics, and result/completion assembly. `kindred.core.analysis.global_fitting.fit_global()` is the public API composition function and must not regain those execution-policy internals as private monolithic helpers. Candidate dataset evaluation, final replay, cancellation kill/close, and live replay paths must preserve the session-owned runtime identity. Multi-lane candidate scheduling must reuse the session-owned scheduler/executor for that runtime identity rather than constructing a scheduler per objective/candidate evaluation. `FittingWindow` may compute current fitting identity, render readiness state, route visible Stop Fit to preparation/active-run cancellation, and reject stale worker completions at the active-worker boundary; `GlobalFitWorker` may pass cancellation/progress intent, but neither class should discover, create, warm, or become the durable owner of fitting runtime readiness, contained evaluator lane readiness, scheduler lifecycle, or prepared-runtime reuse. Private launch helpers such as `_start_global_fit()` must not assemble worker launch inputs or bypass accepted launch.
