# Kindred: Agent Instructions (Codex)

These instructions define how automated agents (Codex) must work in this repository.
This is the fresh local Kindred repo context under `~/kindred-vDEV` for `KindredSim/Kindred`, and the current stage is GUI work.

## Working directory and execution model

- You will be run directly from the repo root:
  - Repo root: `<repo-root>`
  - Shorthand: `~/kindred-vDEV`
- Run project commands from the repo root using explicit paths, for example:
  - `bash tools/audit/run_ci.sh`
  - `python3 -m pytest -q`
- Do not rely on an implicit current working directory inside the repo. Be explicit.
- Never run commands from inside `_audit_reports/**` (report dirs are artifacts only).
- Audit runners may internally `cd` to the repo root when needed (for example to run `pytest`); do not document internal `cwd` changes as prohibited.
- For Audit K and packaging/headless smoke checks, prefer `run_all.sh` / `run_all.sh --exhaustive` to preserve isolation expectations.

## Non-negotiable workflow

0) Architecture compliance (must happen before any edits)
- Read `AGENTS.md` first.
- Before implementing any code changes, you MUST read `ARCH_RULES.md` and comply with it.
- Treat `AGENTS.md` and `ARCH_RULES.md` as permanent but volatile guardrails. Fresh repo evidence and tests override them for current-code behavior claims.

Root vs private context documents:
- Root `AGENTS.md` and root `ARCH_RULES.md` are the active local operating guardrails for this checkout.
- Files under `private context cache/`, including private `AGENTS.md` and private `ARCH_RULES.md`, are derived private-context extracts. They are useful for intent, decisions, freshness, and pending-update awareness, but they are not the same documents as the root guardrails and are not proof of current code behavior.
- Root `AGENTS.md` and root `ARCH_RULES.md` are intentionally local guardrails in this checkout. Agents must not remind Pedro that these files are ignored, untracked, gitignored, local-only, force-add-required, or unable to land in a normal commit. This is not optional.
- Mention root guardrail document landing/persistence status only when Pedro directly asks whether one of these files will be staged, committed, or persisted; when a formal Documentation Landing Assessment requires tracked-first classification; or when a formal staging proposal must say whether a documentation change is included.
- Even in those allowed cases, mention the status once, use the shortest possible wording, and do not repeat it later in the final summary if it already appeared in the formal landing or staging section. Do not add explanatory reminders, warnings, or caveats.
- Authority order for code work:
  1. explicit user instructions for the current task;
  2. fresh repo evidence and tests for current-code behavior;
  3. root `AGENTS.md` and root `ARCH_RULES.md` for local operating guardrails;
  4. validated private context for Pedro's intent, decisions, and known risks;
  5. derived private extracts only as convenience summaries.
- If root guardrails, private extracts, and fresh repo evidence disagree, stop and report the conflict before planning edits.

1) Read-only investigation first
- Do not edit files until you have a written diagnosis supported by file:line evidence.
- Prefer ripgrep and minimal reproduction scripts over guesswork.

2) Tests before fixes for behavior changes
- For behavior changes and bug fixes, add a failing regression test first (must fail pre-fix), then fix, then re-run targeted tests.

3) Evidence required
- Every diagnosis and every change must be justified with file:line evidence.
- Never guess current repo behavior. `Sure` means fresh evidence has been read in the current task and supports the claim. If fresh evidence is missing, say so and read the relevant code or tests before answering.
- For any answer about code behavior, UI behavior, architecture, cache identity, solver behavior, fitting behavior, tests, or project workflow, inspect the current repo first unless the answer is explicitly framed as unverified memory or a user-supplied assumption.
- Standing docs should prefer stable anchors such as file paths, class names, function names, test names, and rule IDs instead of hard-coded line numbers. Line numbers drift in active code; use fresh `nl -ba <file> | sed -n 'start,endp'` output in each run report when exact file:line evidence is needed.
- Do not add hard-coded current line numbers to standing docs such as `AGENTS.md` or `ARCH_RULES.md` unless the line number itself is the behavior under test. Use stable anchors instead, and gather fresh `nl -ba` evidence during each run.

3a) Contract-first / no-plaster rule (mandatory)
- The user-visible contract is the objective. Internal cleanup, architecture substrate, adapter creation, owner extraction, session objects, readiness flags, and test churn are not progress unless they directly make the named user-visible contract true.
- For architecture-sensitive work, choose the healthiest truthful boundary by default. Do not preserve a false boundary, fake owner, or misleading lifecycle contract for diff-size reasons; compatibility, active-work readability, and explicit user non-goals still keep their normal ask-before-touch protections.
- Do not ask Pedro whether to choose a minimal patch or a healthier architecture when fresh evidence shows the current boundary is false, misleading, duplicated, or responsible for the user-visible failure. The default is the healthiest truthful code shape supported by the audit.
- A smallest safe patch is acceptable only when it is also the healthiest truthful boundary for the approved scope. Diff size, local convenience, or preserving a transitional seam is not a reason to keep bad architecture.
- Ask for a scope decision only when the healthiest truthful boundary would materially expand the approved product behavior, change compatibility or data-readability posture, violate an explicit non-goal, or touch an ask-before-touch surface. Do not ask just to choose between cleanup quality and a patch that leaves the false boundary intact.
- Before behavior-sensitive edits, state the exact workflow that must work, the exact current failure or risk, and the proof that will show it is fixed. The first implementation target must be that workflow, not an adjacent symptom.
- Do not build facades, wrappers, owners, sessions, adapters, or booleans that only rename responsibility. A boundary is useful only if it owns the lifecycle, identity, readiness, invalidation, stale-reply rejection, shutdown/kill, or user-visible policy it claims to own.
- Maintainability is the measure, not architectural purity or object count. Do not judge a slice by how many owners, adapters, ports, services, or files it creates or deletes. A boundary is maintainable only when a future change can be made by reading that boundary and its focused tests without mentally reconstructing the old monolith.
- A wrapper, port, owner, adapter, or service is not an endpoint if it merely renames access to another object's internals. Prefer stable input/output objects, immutable snapshots, explicit owned state, or narrow Qt/presentation adapters over broad live-object ports. If a new boundary increases the number of places a maintainer must read to understand one behavior, it must own real lifecycle, identity, state, policy, invalidation, or presentation responsibility, or it should be removed.
- Composition roots are allowed. A Qt window or controller may compose widgets, signals, ports, and owners, but it must not remain the hidden owner of policy that another object claims to own.
- Do not treat partial substrate progress as success. Pool exists, owner exists, warm requested, process started, background task scheduled, generic startup owner exists, or a controller flag is set are not proof that the workflow is ready.
- If testing, review, or Pedro's repro shows the central contract is still false, stop broadening the patch. Re-audit the exact failing workflow, report the mismatch, and repair that contract before doing adjacent cleanup.
- No plaster: do not patch visible symptoms when fresh evidence shows the ownership or readiness boundary itself is false. Fix the boundary that makes the contract truthful, or stop with evidence and ask for a scope decision.

4) Standing instruction maintenance
- Whenever architectural work, workflow changes, validation policy changes, hidden-feature policy changes, dependency policy changes, or user-visible product-contract changes are committed, reassess `AGENTS.md`.
- Update `AGENTS.md` in the same commit when the work changes, clarifies, retires, or contradicts any agent workflow rule, architecture summary, product contract, test/CI expectation, dependency rule, hidden-feature rule, or output requirement documented here.
- If no update is needed, the final report must say `AGENTS.md update: not needed`.

5) No placeholders
- Do not add TODOs, placeholders, or stubbed code.

6) Git + GitHub workflow (mandatory)
- Never commit on `main`. Work happens on a per-task branch (prefer `feature/<topic>` or `chore/<topic>`).
- If task-specific instructions require staying on an existing non-`main` branch, treat that as the active branch contract for the slice. Do not switch branches, pull, rebase, merge, or create a new branch unless the task explicitly permits it.
- If Pedro explicitly asks for a read-only audit, salvage audit, or review of an already-dirty worktree, treat the dirty files as the audit subject. Record and classify the dirty state, but do not stop solely because the worktree is dirty. Generic dirty-worktree stop rules apply only when the dirty state is not already authorized as part of the task scope.
- Keep `main` green: create a PR, wait for CI to pass, then squash-merge via GitHub.
- Before every commit and before every push, run the gate from the repo root:
  - `bash tools/audit/run_ci.sh`
- Avoid force-push (`git push --force` / `--force-with-lease`). Prefer PR updates via normal pushes.
- Deliverables must include:
  - file:line evidence (use `nl -ba <file> | sed -n 'start,endp'`)
  - diff-based proof of the slice:
    use `git diff --stat` plus `git diff` for key hunks on clean or slice-isolated branches;
    on already-dirty branches, prefer file-scoped diff proof for the touched files

7) Review-pass synthesis (mandatory)
- When multiple review passes are requested or active, wait for every review pass to return before acting on any review output.
- Synthesize all review findings together before classifying them as real, not grounded, or out of scope.
- Do not fix, reject, stage, commit, or otherwise proceed based on a single reviewer while another required review pass is still pending.

7a) Read-only audit subagents
- This section is Pedro's standing explicit authorization to use read-only audit subagents for repository audits when the conditions below apply. Treat feature-session instructions that require or recommend read-only audit subagents as an explicit request for subagents/delegation/parallel audit work, while preserving controller ownership of synthesis and decisions.
- For broad or architecture-sensitive audits, use read-only subagents when independent audit partitions can run in parallel and their findings could affect the plan.
- Keep delegated audit scopes distinct. Do not dispatch duplicate broad passes over the same files for the same question.
- Require file:line evidence, explicit unknowns, and read-only/no-git-mutation behavior from every audit subagent.
- Treat subagent output as evidence to synthesize, not as authority to act. The controller must synthesize all active audit outputs before planning, classifying scope, or proceeding.

7b) Audit, plan, and handoff transferability
- Treat every audit, execution plan, synthesis, and handoff as transferable context for a future reader.
- Use task labels, slice names, acronyms, shorthand, workstream names, and local terms when they help the work, but define them at first use. Do not leave unexplained labels such as "A/B", issue keys, branch nicknames, or architecture shorthand unless the document itself states what they mean.
- Artifacts must be transferable: include enough objective, scope, current-state evidence, definitions, decision status, and unknowns for another agent or Pedro to continue without reconstructing hidden conversation context.
- If a document uses a split such as "current slice" versus "follow-up", state whether the split is already approved, merely proposed, or conditional on audit/implementation evidence.
- Do not rely on the author remembering why a label exists. If the context is worth recording, record it in the artifact.
- Private scratch notes are allowed when useful, but keep them in a clearly separate scratch document. Do not mix scratch-only shorthand into audits, plans, syntheses, handoffs, or other artifacts meant to transfer context.
- When Pedro asks to record backlog, planned features, future plans, follow-ups, future slices, or deferred architecture recommendations, inspect and use the established private planning destinations before inventing a session-only artifact: `Kindred-Backlog.md`, `Kindred-Extended-Plans.md`, and `_meta/Context-Update-Queue.md` as applicable.

8) Handoff policy
- Handoffs preserve context; they do not transfer authority beyond the explicit decisions and instructions they record. The receiving agent must still follow the active thread's current instructions, root guardrails, and fresh repo evidence.
- Treat old handoffs, side-chat conclusions, and prior session notes as historical context only unless the current task explicitly makes them active instructions or fresh repo evidence verifies them.
- Side conversations may discuss or recommend prompt/template/process changes, but they must not continue parent-thread execution. Do not resume parent-thread instructions, plans, tool calls, approvals, edits, reviews, staging, or commits from inside a side conversation unless Pedro explicitly returns that work to the main thread.
- Normal handoff types:
  - `HANDOFF TYPE: Request-slot`
  - `HANDOFF TYPE: Return-to-parent`
- Request-slot handoffs are written only for the `<PASTE PEDRO'S FEATURE / BUG FIX / REFACTOR REQUEST HERE>` section of the Kindred feature-session prompt template. They are not full prompts.
- A Request-slot handoff may preserve direction, boundaries, non-goals, discussed guardrails, relevant historical observations, and explicit constraints for the next session.
- A Request-slot handoff must not discourage or bypass the receiving agent's normal prompt workflow: interview, Understanding Contract, private-context and git/worktree validation, read-only audit, execution-plan synthesis, and explicit approval with `Implement the plan.` unless Pedro explicitly instructs otherwise.
- A Request-slot handoff must not authorize implementation, tracked-file edits, staging, commits, branch changes, pushes, or direct `private context cache/` updates.
- Return-to-parent handoffs transfer useful context from an ephemeral side conversation back to the main thread it forked from. They are not feature-session requests unless explicitly labeled as Request-slot handoffs.
- A Return-to-parent handoff should separate:
  - Pedro's explicit decisions;
  - side-agent inferences;
  - evidence checked;
  - unresolved questions or items needing main-thread verification.
- A Return-to-parent handoff must not ask the main thread to replay the side conversation wholesale, must not override newer main-thread instructions, and must not authorize edits unless Pedro explicitly gave that authorization in the side conversation and the main thread can verify it from the handoff text.
- Pedro may trigger handoffs with short natural-language commands; no exact magic phrase is required. Infer the intended type from context and destination. Requests about a prompt, next session, new agent, or feature-session template usually mean Request-slot. Requests about the main thread, parent thread, current session, returning from `/side`, or "bring this back" usually mean Return-to-parent. Phrases such as `handoff: prompt` and `handoff: main thread` are optional shorthand, not required syntax. Ask one concise clarification only if the destination is genuinely ambiguous.

## Current simulation architecture summary

- `ARCH_RULES.md` is the maintained architecture guardrail file. Keep the summary here aligned with it.
- Keep docs and rules truthful to the current codebase. Do not describe aspirational seams as if they already landed.
- `MainWindow` currently composes simulation plumbing by creating `MainWindowPreviewSession`, `MainWindowVariableRuntime`, `MainWindowMechanismHelpers`, `SimulationRunUiOwner`, `SimulationProvenanceOwner`, `SimulationSettingsOwner`, `SimulationDialogs`, `SimulationSolverOwner`, `SimulationMechanismOwner`, and `SimulationBatchOwner`, then wiring them into `SimulationUiPorts`. `ResultsController` owns the results port.
- `kindred.core.mechanism_structure_snapshot.MechanismStructureSnapshotOwner` owns authoritative parsed-structure snapshot reuse for canonical GUI structure consumers. It is adapted through `MainWindowMechanismHelpers`; it is separate from execution-local request parsing and from runtime initials.
- `kindred.core.mechanism_runtime_transition.MechanismRuntimeTransitionService` owns authoritative mechanism transition epoch/identity, pending-init transition suppression, transition-owned readiness deferral, and transition outcomes for runtime invalidation, active-work supersede, and stale-result protection.
- Core reaction semantics are side-aware. `kindred.core.mechanism.Reaction` owns immutable positive irreversible physical sides: `reactants`, `products`, derived `net_stoich`, and `rate_orders`; `Equilibrium` owns immutable positive reversible forward/back side maps. `rate_orders=None` means default to the reactant side; an explicit empty `rate_orders` mapping means zero-order kinetics. DSL construction must preserve same-side species such as catalysts as kinetic participants even when their net stoichiometry is zero. ODE, cache, serialization, prepared-runtime, fitting, and batch code must consume the field matching its meaning instead of inferring rate laws from collapsed net stoichiometry.
- `sympy==1.14.0` is a mandatory runtime scientific dependency, not an optional backend, plugin, extra, or graceful-degradation feature. Symbolic capability is part of the app's core scientific substrate. `kindred.core.symbolic` owns SymPy imports, backend/version metadata, exact symbolic proof helpers, symbolic expression translation, generated symbolic RHS/Jacobian structure artifacts, immutable evaluation snapshots, artifact fingerprints, and typed symbolic unsupported-case errors. Wegscheider cyclicity proof and bounded symbolic Jacobian generation must consume that core symbolic boundary; GUI, batch, fitting, and solver modules must not import SymPy directly or duplicate symbolic proof/generation logic. Generated symbolic Jacobians enter solving only through the existing `SimulationRequest.jacobian_func` / `solve_ivp(jac=...)` boundary for `BDF` and `Radau`. Symbolic artifacts must separate reusable structure from immutable parameter snapshots so mutable `RateBinding` objects cannot make generated callables or identity stale after later mutation. Scheduled temperature, active intervention intervals, unsupported kinetics, unsupported dynamic expressions, mutable bindings without an immutable snapshot boundary, or missing artifact identity must disable or reject symbolic Jacobian use truthfully and must not substitute a sparsity-hint or generated sparse-Jacobian fallback.
- `SimulationController` is the execution/orchestration owner for user run intent, task/plan assembly, run-start orchestration, cache administration, worker lifecycle, and warm contained GUI simulation owner slots for ordinary and preview serial runs. Callback-captured run/request/owner/cache identity belongs to `kindred.gui.controllers.simulation_callback_identity.SimulationCallbackIdentity` and must stay unchanged through completion and error dispatch once captured. Parallel batch callbacks must use one slim/shared callback context from `BatchRunContextOwner` plus per-set callback identity such as set id, submitted-plan simulation identity, and preview cache token; callback identity must not deep-copy full mutable batch context or full per-set execution payload maps once per submitted set. Batch dispatch initials materialization belongs to `kindred.gui.controllers.batch_dispatch_materialization.BatchDispatchMaterializationOwner`, including canonical batch initials reads, pending-init seed overlay, preview-initial overlay for fast-mode dispatch, and run-preparation plan input initials. Contained serial worker creation and worker identity stamping belong to `kindred.gui.controllers.serial_worker_launch.ContainedSerialWorkerLaunchOwner`. Completion result materialization belongs to `kindred.gui.controllers.simulation_result_materialization.SimulationResultMaterializationOwner`, including completion-mechanism fallback resolution, energy-mode materialization side effects, primary-result mechanism memory, batch species sync after primary completion, and primary-result control refresh. Completion callback policy must remain composition-only: stale callback rejection/decision, cache-key normalization, cache truth/publication, result display, provenance handoff, pending-init completion, batch success transitions, and final lifecycle effects belong to named owner/effect boundaries such as `BatchRunContextOwner`, `SimulationCacheAdmin`, `ResultsController`, `SimulationProvenanceOwner`, and `SimulationLifecycleEffectOwner`; `_on_simulation_complete()` must not return to a monolithic cache/display/provenance/batch policy method, and publication must not rediscover plan, set, or cache identity after callback handling. Mutable GUI batch-run context storage and queue/session transitions belong to `kindred.gui.controllers.batch_run_context_owner.BatchRunContextOwner`, including batch start-run context construction, slim callback context construction, completion-policy context normalization/serialization, current queue-position hints, completion summaries, runtime-input staleness comparison against supplied current epochs, cache-key updates, parallel success/failure transitions, serial success cursor advancement, serial stale-prefix consumption, active serial runtime-input supersede cursor advancement, guarded stale callback completion/deactivation, runtime-waiting transitions, and deactivation. `SimulationController` may still orchestrate batch transitions, but it must not reintroduce a raw context dict as controller-owned state. Batch runtime session lifecycle lives in `kindred.core.batch_runtime_session.BatchRuntimeSession`, above the non-GUI batch lane owner in `kindred.core.batch_containment`; `ParallelBatchExecutor` is only a temporary narrow controller adapter over that session. Parallel batch runtime readiness state, nonblocking warm lifecycle, and run-path ready/not-ready decision snapshots belong to `kindred.gui.controllers.parallel_batch_runtime_readiness_owner.ParallelBatchRuntimeReadinessOwner`; a Run path with a warm reusable batch lane pool must not synchronously recreate/import/warm that pool before submitting unchanged work. Ordinary and preview reusable process lifecycle, READY/ACCEPTED gating, request identity, stale-reply rejection, timeout restart, and idempotent close/kill are delegated through the non-GUI simulation runtime service to the shared containment kernel.
- Species intervention schedules live in `kindred.core.intervention_schedule.InterventionSchedule` and are core execution inputs. Schedules include fixed time events, repeated pulses, continuous intervals, reservoir/clamp intervals, state-triggered interventions, and request-time parameterized fields. Schedules flow through `SimulationPlan` and nested `SimulationExecutionRequest`, into preparation and `SimulationRequest`, then execute in the shared solver path through `kindred.core.simulator.intervention_schedule_execution.InterventionScheduleExecutionOwner`; `kindred.core.simulator.solvers.solve_ode()` remains the public solver composition entry point. State-triggered execution must resume from solver event-time state, re-arm repeated triggers after `min_interval` inside a segment, keep scheduled trigger events terminal, normalize user-level solver options such as `first_step` for internal segment bounds, and apply clamp/reservoir values according to interval activity at the trigger time. Ordinary, batch, containment, and fitting paths must preserve schedule payloads, fingerprints, and request-time parameter resolution through that shared boundary instead of interpreting schedules independently. Schedule plot annotations are optional solved-provenance display aids only; they must default off, be user-toggleable, and never count as schedule authority or solver/fitting proof.
- Global fitting readiness is coordinated by `kindred.gui.fitting.runtime_readiness.FittingRuntimeReadinessController`, which owns the desired/active/ready fitting identity, preparation worker lifecycle, supersede/cancel/close policy, readiness state transitions, and accepted-launch publication for the fitting window. Deferred evaluator construction resolves whether a session is required after the concrete evaluator exists: exact `SerialFittingEvaluator` runs require a ready `FittingRuntimeSession`, while generic evaluator runs do not. Reusable evaluator/runtime lifecycle lives in `kindred.core.fitting_runtime_session.FittingRuntimeSession`, which owns contained evaluator lane readiness, scheduler/executor reuse, prepared-runtime reuse, bounded lane ownership, cancellation kill/close, and deterministic fitting runtime ledgers. Core global fitting execution policy lives in `kindred.core.analysis.global_fit_execution`, including candidate dataset evaluation, serial/runtime-batch evaluation dispatch, final replay, diagnostics, and result/completion assembly; `kindred.core.analysis.global_fitting.fit_global()` remains the public API composition function. `ParametersIcsTab` owns fitting parameter table state and parameter-config collection for explicit Run Fit and passive readiness snapshots. `kindred.gui.fitting.evaluator_state.FittingEvaluatorStateOwner` owns the mutable current base-evaluator slot, prepared-metadata lookup, and builder/reuse decision for fitting runtime identity. `FittingRuntimeIdentity` / `FittingRuntimeAcceptedLaunch` are fitting's run-level launch boundary and are built once per launch; do not import simulation's per-set callback snapshot model into fitting unless a fresh fitting audit proves an equivalent need. `FittingWindow` computes current fitting identity, renders readiness state, and routes visible Run/Stop Fit through the readiness owner; `GlobalFitWorker` drives optimization/progress/cancel using accepted launch data and must not create or discover fitting runtime sessions.
- `SimulationUiPorts` is explicit and partitioned. For simulation work, use the correct sub-port such as `self.ui.slider`, `self.ui.runtime`, `self.ui.mechanism_helpers`, `self.ui.batch`, `self.ui.run_ui`, `self.ui.results`, `self.ui.provenance`, `self.ui.dialogs`, `self.ui.settings`, `self.ui.mechanism`, and `self.ui.solver`.
- Broad fallback is removed. Do not add `SimulationUiPorts.__getattr__`, do not rely on flattened `self.ui.<method>` access where an explicit sub-port exists, and do not treat broad `main_window` injection as the model for slider/runtime/mechanism-helper work.
- Simulation architecture should move incrementally toward a three-layer split where fresh evidence supports it: scientific core owns preparation/solving/finalization; non-GUI simulation runtime/application containment owns jobs, owners, lanes, lifecycle, request identity, stale-reply rejection, timeout/cancel cleanup, idempotent close/kill, contained payload identity, and reusable execution ownership; GUI owns user intent and presentation policy.
- This split is long-term direction, not proof that every current path has migrated. Do not force a big-bang migration, do not move GUI dirty-preview/presentation policy into scientific core, and prefer reusable non-GUI containment primitives with narrow Qt/GUI adapters when touching containment lifecycle or identity.
- Campaign A is not complete while runtime/controller extraction mainly adds owners around still-growing monoliths or repeats policy across old and new surfaces. A large branch diff is not automatically wrong, because real owners, typed execution boundaries, and non-GUI runtime substrate can require new code; but owner extraction is not complete while old monoliths keep growing, policy is duplicated between old and new surfaces, or callback/readiness/default helpers repeat the same authority in multiple places. Continue classifying added code as `real owner`, `thin adapter`, `duplicated policy`, `temporary compatibility`, or `delete`, and remove or consolidate remaining duplicated completion identity fallback, callback context escape hatches, duplicated readiness snapshot construction, repeated scalar/coercion helpers, repeated fitting worker defaults, and dependency-lambda pass-through surfaces unless fresh evidence proves a distinct owner contract.
- Runtime readiness is a global user-facing product contract, not an internal optimization. Normal interactions such as Run Selected, slider preview/drag, batch simulation, fitting/runtime containment, and any future warm-owner path must not pay avoidable NumPy/SciPy/import/process-startup/owner-warmup/prepared-runtime cost at the first click, first drag, or repeated unchanged interaction. The runtime/application layer must make required owners, lanes, imports, and prepared runtimes ready before user actions depend on them, or expose an explicit non-blocking readiness state.
- If the app presents a simulation, fitting, slider, batch, or runtime-backed control as usable, the runtime needed for that control's normal action must already be ready. If readiness is not available yet, the UI must truthfully show that state instead of pretending the control is ready and then blocking on first use. Lazy startup is acceptable only for workflows that are not entered, not visible, not selected, or explicitly optional/heavy, and those workflows must expose a clear readiness state when entered.
- When touching runtime readiness, use deterministic owner/factory/lane/evaluator ledgers instead of wall-clock assertions. Tests must prove first-use and repeated-use interactions reuse already-ready owners and only invalidate/rebuild them when the real mechanism/runtime identity changes.
- Local owner/service tests with mocked readiness, workers, lanes, or slider paths are not enough to prove the visible runtime-readiness contract. For Run, Run Selected, slider preview, and batch simulation work, include at least one workflow-level GUI regression through the real `MainWindow` readiness/runtime boundary unless the test explicitly documents why a dependency is mocked and what narrower contract it proves.
- Truthful ownership matters:
  - `MainWindowPreviewSession` owns preview gesture state and debounce timers.
- `MainWindowVariableRuntime` owns prepared preview runtime state and variable metadata.
- `MainWindowMechanismHelpers` owns last-mechanism snapshot/context, adapts canonical structure snapshot reuse, and provides bounded helper coordination.
- `SimulationSolverOwner` is a thin Qt adapter that owns simulation solver-control reads and startup solver defaults for the solver port.
- `SimulationMechanismOwner` is a thin Qt adapter that owns mechanism-session text, mechanism editor controls, preview parameter-store schema/fingerprint reads, and mechanism-port override application.
- `SimulationBatchOwner` is a thin Qt adapter that owns batch table/store selection reads, batch model validation, active batch display selection state, and batch-port display routing.
- Runtime-initial-only transitions must advance runtime-input truth and stale-publication epochs without clearing authoritative parsed structure snapshots or forcing structural derived-UI refresh.
- When refactoring simulation seams, introduce the real owner first, move the owned state/logic into it, wire it through `SimulationUiPorts`, and only then remove legacy forwarding if that cleanup is in scope.
- Do not create fake seams. If a wrapper does not own real state or a real bounded responsibility, it is not an architectural endpoint.
- Do not replace monolith reach-through with a broad private port slab and call it done. Narrow ports are acceptable at a real Qt/presentation boundary; non-Qt policy should move toward typed snapshots, explicit command/result objects, or owner-owned state so future changes do not require reading both the old monolith and the new owner.

## Core simulation product contract

Core Product Contract:
- There is one canonical mechanism baseline for the current top-level container/window at any given time.
- A set may additionally own local dirty staged preview state, but focus/selection does not create authority.
- Uncommitted staged state is preview-only.
- Commit is the only promotion boundary from staged preview state into canonical mechanism state.
- Run Selected must use canonical state only, not any uncommitted staged workspace or staged concentration overlay.
- Run Selected is also a local reset operation for the targeted sets, but not a promotion boundary:
  - it discards staged dirty state for those targeted sets
  - it runs those sets from canonical state
  - other dirty sets must remain dirty

Dirty / Clean Model:
- If a set has local staged slider workspace or staged species concentration overlay, that set is dirty.
- When a dirty set is focused or reselected, the UI must restore that dirty set’s controls and dirty preview/plot.
- If a set has no local staged workspace, it is clean and should show canonical state/results.
- Explicit canonical cache must not override the focused display for a set that is still dirty.
- If a set is still dirty but no truthful dirty preview is available, the UI must not fall back to canonical explicit result while leaving dirty controls visible.
- In that case, the UI must show a truthful dirty/no-preview state rather than a split canonical/dirty display.

Commit Behavior:
- Commit promotes staged dirty state into canonical base.
- After commit, preserving the currently displayed preview is good UX, but only when the preserved display is still truthful.
- Do not preserve stale explicit overlays just because they were selected.
- Overlay preservation must be based on dirty preview provenance, not generic selected-set membership.
- Preservation must not depend on preview-cache residency; if a truthful dirty preview is already visible, commit may preserve it from the live plot.

Run Selected Behavior:
- Run Selected must ignore all uncommitted staged state as solver input.
- No explicit full result may be sourced from uncommitted staged state.
- That includes:
  - no slider-prepared runtime/bindings in explicit mode
  - no preview initials in explicit mode
  - no workspace-aware explicit cache identity
- After a successful explicit run, targeted dirty state is cleared only for the targeted sets, and by stable set identity, not row index.
- If an explicit run fails, dirty state must not be destroyed.

Selection / Reselect Rules:
- Selection changes must not trigger full recompute.
- Clean selection changes show cached explicit results only.
- Dirty selection changes restore the dirty preview for that set.
- Do not show a split state where:
  - plot/result is canonical explicit
  - but sliders show dirty staged values
- That split is not the intended UX.

Pending-Init Migration Contract:
- Inline initial migration is a special case:
  - the rewrite is authoritative
  - a successful explicit run that produced the rewrite may keep its result visible
- But if the migrated explicit run later aborts, the previously preserved result must be re-invalidated, because the authoritative mechanism has changed and the old result is stale.
- The next real reactions edit or state-network edit must invalidate normally; the pending-init guard must suppress only the migration rewrite itself.

State-Network Guardrails:
- Mechanism lock lives in shared MainWindow state.
- State Network dialog must still open while locked.
- While locked, it must be read-only and visibly locked.
- Disable all mutation paths:
  - Add State
  - Remove Selected
  - Add Edge
  - Remove Selected
  - direct table editing / edit triggers / delegates as needed
- Keep it usable as a viewer.
- Do not disconnect authoritative programmatic setters/signals.
- Do not redesign Apply/Cancel or add rollback semantics.

Priority Rule:
- Do not silently bury correctness bugs under shell polish or unrelated GUI cleanup.
- When fresh audit exposes a user-visible UX issue or bug in the touched workflow, prefer fixing it in the current slice unless doing so materially broadens scope, conflicts with the approved behavioral target, or requires a separate product decision.
- Ask before reprioritizing user-visible correctness issues.

## Audit and CI gates (strict gate must stay green)

Default go or no-go command (run from repo root):
- `bash tools/audit/run_ci.sh`

Other common commands (run from repo root):
- Exhaustive audits A-L:   `bash tools/audit/run_all.sh`
- Strict audits A-H only:  `bash tools/audit/run_strict.sh`

Audit reporting:
- `run_all.sh`, `run_strict.sh`, and `run_ci.sh` print a line:
  - `AUDIT_REPORT_DIR|<path>`
- All stable artifacts live under:
  - `_audit_reports/<UTC>/`
- When strict fails, open the newest report dir and read:
  - `SUMMARY.txt` first

Audit wrapper conventions (do not break):
- Non-fatal audit wrappers must:
  - Write stable artifacts into the report dir
  - Emit a `COUNTS|...` line (if applicable)
  - Append exactly one `Audit X: PASS|WARN|SKIP ...` line to `SUMMARY.txt`
  - Exit 0
- Gating behavior:
  - Strict/CI gates: `run_strict.sh` and `run_ci.sh` run strict audits A-H only.
  - Exhaustive audit runner: `run_all.sh` defaults to A-L; `run_all.sh --strict` runs A-H; `run_all.sh --exhaustive` runs A-L.
  - `run_strict.sh` fails if `SUMMARY.txt` contains any `WARN|FAIL|TIMEOUT` lines or if `run_all.sh --strict` exited non-zero.

## Testing conventions

Pytest configuration:
- `pytest.ini` uses strict markers and strict config.
- Markers present: `unit`, `integration`, `gui`, `slow`, `experimental`.

Test rules:
- Prefer deterministic assertions over timing thresholds.
- Avoid wall-clock performance assertions. Use invariants such as bounded counts, cache sizes, object lifetimes, and marker counts.
- For GUI tests:
  - Run headless.
  - Avoid pixel-based assertions.
  - Assert internal models, registries, signals, and state transitions.
- Avoid brittle static line-number assertions in tests and docs unless the line number itself is the behavior under test.
- Treat test-suite size and shape as architecture. For behavior that crosses GUI, controller, runtime, cache, fitting, or core boundaries, prefer holistic workflow tests through the real boundary over multiple local/mock-heavy tests that only approximate the same contract.
- When touching a test-heavy area, audit nearby tests for duplicate local proof, fake-seam protection, obsolete transitional vocabulary, and helper-plumbing assertions. Replace redundant local tests with clearer workflow-level contract tests, then delete or merge the redundant tests once the replacement shield is proven.
- Keep local/unit tests only when they protect a distinct pure-logic contract, owner boundary, failure class, or deterministic event ledger that the holistic workflow test cannot reasonably isolate.

Default gates:
- Before changing code, perform a fresh blast-radius assessment from the files, data flow, ownership boundaries, and user-visible behavior touched by that specific change. Name the targeted tests selected for that change and why; do not rely on a static recommended test map as a substitute for this assessment.
- Run targeted tests for changed areas first.
- Then run full suite when feasible:
  - `pytest -q`

## GUI rules (PySide6 / Qt)

- Prefer model and delegate patterns for tables.
- Do not use `QTableWidget.setCellWidget` or `setIndexWidget` in `kindred/gui/**`.
- Avoid repeated signal connections in refresh or run loops.
  - Connections should be established once during widget initialization, or guarded.
- Prefer clear, discoverable UX controls over hidden gestures.
- Avoid adding new tooltips unless explicitly requested.
- Existing tooltips are allowed.
- Tooltips must not be the only way to discover required actions; prefer small-font inline help text for primary guidance.
- Do not add new nested `QGroupBox` layout stacks. Existing top-level or legacy `QGroupBox` sections are not automatically violations; avoid opportunistic churn unless the task is layout cleanup. For new or touched UI, prefer flat layouts using bold `QLabel` headers, spacers, width-bounded inputs, dynamic show/hide empty states, and existing helpers from `ui_helpers.py`.

## Performance and caching rules

Caching invariants:
- Simulation caching uses two bounded in-memory caches with deterministic (LRU) eviction:
  - Result cache: stores explicit full-run results.
  - Preview cache: stores slider-triggered fast preview results.
- Default caps (globally persisted):
  - Result cache cap default: 1000 (`QSettings` key `simulation/result_cache_cap`).
  - Preview cache cap default: 1000 (`QSettings` key `simulation/preview_cache_cap`).
- Simulation Settings includes cache caps, cache status (used/cap and approximate memory), and purge controls for result/preview/both caches.
- Re-running a simulation overwrites per-key cache entries (no unbounded per-key history).
- Eviction UX:
  - If the user selects a batch whose explicit cached result is missing (evicted), do not auto-run and do not clear the current plot.
  - Show a clear inline message: `Result not cached (evicted). Press Run to compute.`
- Plot overlays must not accumulate unbounded curve objects across re-runs.
  - Prefer reusing plot items keyed by stable identifiers and updating via `setData`.

Worker lifecycle:
- Background workers, threads, and QObjects must not accumulate across runs.
- Ensure proper cleanup and no lingering references that prevent GC.

Runtime readiness:
- Opening the app must make ordinary expected simulation interactions usable. If a runtime dependency, process owner, worker lane, prepared runtime, or import-heavy backend is required for a normal interaction, startup or another explicit non-blocking readiness path must handle it before the interaction depends on it.
- Visible simulation controls after startup, visible sliders after startup, the current selected Run path after startup, and current batch runtime capacity up to configured bounds must be ready before the user action depends on them.
- Fitting/runtime-heavy optional workflows may initialize when the workflow is opened or made available, but not by hiding a startup/import tax behind the first apparent usable action.
- Do not defer avoidable runtime startup/import/warmup cost into Run Selected, slider preview/drag, batch/multi-set simulation, fitting evaluation, or repeated unchanged interactions.
- Runtime owners may be invalidated and rebuilt only when the real mechanism/runtime identity changes. Unchanged interactions must reuse existing ready owners rather than synchronously recreating or warming them.
- Readiness tests must use deterministic ledgers over owners, factories, lanes, evaluators, imports, starts, warms, and prepares. Do not use wall-clock timing thresholds to prove this contract.
- Do not introduce hard-coded values unless Pedro explicitly pre-approved that exact value and tradeoff. This applies before implementation and still applies if the value would be stored as a named default, setting, constant, test expectation, timeout, capacity, threshold, cache size, retry cadence, worker/lane count, validation tolerance, or any other user-visible or maintenance-sensitive policy. After approval, store the value in the appropriate named configuration/default location, make it user-configurable whenever it affects user-visible capacity or workflow behavior, and cover it with tests where relevant.

Debug and instrumentation:
- Any performance instrumentation must be opt-in and bounded.
- Prefer counters and phase timings.
- If added, gate it behind an environment variable (for example `KINDRED_DEBUG_PERF=1`) and keep output structured.

## Domain invariants (must preserve unless explicitly requested)

Batch Initial Conditions:
- Batch sets are the canonical home for initial concentrations.
- Reaction DSL initial concentrations are used only on first parse, then migrated into batch sets and removed from DSL for subsequent runs.
- Multi-select overlays curves. No recompute on selection changes.
- Full batch simulations run only via an explicit user Run action (never automatically on selection changes).
- Selection changes update plots from cached explicit results only; selection changes must not trigger a full run.
- Slider adjustments may trigger coalesced fast preview runs (intended) and must not overwrite the explicit result cache.
- Invalid batch cells block running and are highlighted. Paste bounds must be enforced. Zero is valid.

Dataset mapping:
- Each dataset maps to one batch set.
- If unmapped, prompt per dataset. It must not silently reuse another dataset’s initial conditions.
- If a mapped batch set is deleted, that dataset becomes unmapped.

Simulation settings:
- Settings beyond the minimal controls belong in the Simulation Settings dialog (menu).
- Temperature is a DSL concern in energy mode, not a solver setting.

Energy and Computational Mode:
- Internal energy math uses J/mol.
- Conversions must be explicit and use the project conversion utilities.
- In energy mode, sliders control barrier and reaction free energies. Derived rates and equilibria are read-only.
- Do not regress Computational Mode parsing, generation, or the `energy=...` directive validation.

ODE Integration:
- ODE integration must use SciPy `scipy.integrate.solve_ivp` only.
- Exposed and normalized solver choices are `Radau` and `BDF`.
- Do not reintroduce pure-Python ODE solvers or ad-hoc non-stiff solver routes.
- Positivity or feasibility enforcement inside RHS or Jacobian code must be C1-continuous; do not add hard clamps or kinked masking inside those functions.
- Kindred is a Windows-first desktop app. Any simulation process containment must work under Python multiprocessing `spawn` and frozen Windows packaging; do not make required behavior depend on `fork` inheritance of prepared RHS closures or parent-process state.
- Shared containment kernel modules must stay stdlib-only at import time; they must not import NumPy, SciPy, Qt/PySide, solver modules, batch containment/parallel modules, simulation containment, or fitting containment during module import.
- GUI ordinary and preview simulation containment must pass serializable execution payloads across spawn, reconstruct solver requests inside the child process, gate active solve timeouts after READY/ACCEPTED, reject stale request/epoch replies, and kill/restart warm owners on timeout or cancellation.
- Batch simulation containment must use parent-owned warm lanes rather than nested per-solve children inside batch workers. The non-GUI batch runtime lane owner owns lane pool lifecycle, request worker tracking, completion metadata, soft-supersede generations, stale-reply rejection, and scheduling. Batch lanes must keep prepared-runtime reuse lane-local, gate active solve timeout after ACCEPTED, map timeout/protocol failures to the affected set, and reject stale run/request/set/epoch replies.
- Contained payload identity must not use raw mapping equality over NumPy arrays or rich runtime objects; owner reuse and request elision must use a containment-owned identity/comparison primitive.

## Dataset import pipeline

File map:
- `kindred/core/datasets/units.py` — SI prefix conversion, Unicode micro normalization, canonical combo source (`TIME_UNIT_DISPLAY`, `CONCENTRATION_UNIT_DISPLAY`)
- `kindred/core/datasets/excel_import.py` — lazy generator streaming Excel reader (openpyxl-backed)
- `kindred/core/datasets/csv_import.py` — CSV reader
- `kindred/gui/widgets/import_config.py` — dataclass composition (`ImportConfig`, `SheetImportIntent`, `UserImportIntent`, `UnitDetection`, `ResolvedSheetPlan`), resolver (`resolve_import_plans`), detection helper (`detect_units_from_row_mapping`), `rebuild_intent_for_target`
- `kindred/gui/widgets/import_config_dialog.py` — per-sheet configuration dialog with phase-split `_build_result()`
- `kindred/gui/widgets/data_manager.py` — `DataManagerPanel` (wiring point between dialog, resolver, and data loading)
- `kindred/gui/fitting/batch_mapping.py` — shared batch-mapping helpers for dataset-to-batch-set flows

Architecture rules (see `ARCH_RULES.md` section 8 for detail):
- Each Excel sheet is a fully independent import unit — no cross-sheet validation or state sharing.
- Per-column unit conversion (no mixed-unit rejection).
- `has_unit_row` = physical property of the full row, never scoped to selected columns.
- Unit extraction scoped to selected columns via `relevant_column_names`.
- Factor computation from intent (`SheetImportIntent.concentration_units`), not raw detection values.
- No clone/fallback path for file-level apply-to-remaining; incomplete import state should be constructed upstream or rejected clearly.
- `rebuild_intent_for_target` is a module-level function in `import_config.py`. Target-detected concentration units win per column; when the target has no detected unit for a column, the function carries `source_intent.concentration_units[col]`.
- Frozen dataclasses with dict fields get `__post_init__` defensive copies via `object.__setattr__`.
- `_build_result()` uses phase-split architecture (4 logical phases, distinct scopes): (1) early exit, (2) collect states + file intent, (3) per-sheet intent/detection/column construction, (4) resolve plans + assemble `ImportConfig`.

Settled decisions (do not reintroduce):
- No clone/fallback path for file-level apply-to-remaining.
- No scoped `has_unit_row` detection.
- No mixed-unit rejection.
- No backward-compatibility shims for deprecated Excel API.
- `.xls` not supported (explicit `ValueError` in `ImportConfigDialog` constructor).

## Fitting window architecture

File map:
- `kindred/core/analysis/global_fit_execution.py` — core global fitting execution owner for candidate dataset evaluation, runtime/serial dispatch, final replay, diagnostics, and result/completion assembly
- `kindred/core/analysis/global_fitting.py` — public `fit_global()` API composition over the core fitting execution boundary
- `kindred/core/fitting_runtime_session.py` — `FittingRuntimeSession` owner for reusable global fitting evaluator/runtime lanes and ledgers
- `kindred/gui/fitting/window.py` — `FittingWindow` (QDialog), orchestrator
- `kindred/gui/fitting/unified_species_table.py` — `UnifiedSpeciesTable` (targets + ICs)
- `kindred/gui/fitting/run_results_tab.py` — `RunResultsTab` (per-dataset subtabs + tracker)
- `kindred/gui/fitting/worker.py` — `GlobalFitWorker` with lightweight/heavyweight split
- `kindred/gui/fitting/parameters_ics_tab.py` — `ParametersIcsTab`
- `kindred/gui/fitting/data_targets_tab.py` — `DataTargetsTab`
- `kindred/gui/fitting/data_tab.py` — `DataTab`
- `kindred/gui/fitting/unified_dataset_list.py` — `UnifiedDatasetList` (SingleSelection mode)
- `kindred/gui/fitting/launch.py` — fit launch orchestration
- `kindred/gui/fitting/worker_lifecycle.py` — `FitWorkerStopPolicy`
- `kindred/gui/fitting/constants.py` — `FITTING_DEFAULT_SOLVER`, `INITIAL_PREFIX`, etc.
- `kindred/gui/fitting/run_stamp.py` — run stamp builder and hasher

Key invariants:
- 3-tab structure: "Data and Targets", "Parameters", "Results" in `FittingWindow._build_ui()`.
- SingleSelection mode on unified dataset list.
- IC state in explicit `_ic_pending` / `_ic_applied` dicts. Table is view, not state.
- Fit-universe = observed ∩ modeled, reconciled solely by `_recompute_fit_universe()`.
- Worker signals are disconnected centrally on completion/error via `_disconnect_fit_worker_signals()` before completion/error handling continues.
- `_set_running_state(False)` must happen before any completion/error modal `QMessageBox`; failure paths may satisfy this through the centralized failed-run visual cleanup helper.
- Results tab: visible-subtab-only live updates via `_stale_plot_view_keys`, refreshed on tab switch in `RunResultsTab`.
- Worker: lightweight params every 0.25 s (`best_update_interval_s`), heavyweight plot arrays every 2.0 s (`plot_update_interval_s`).
- Fitting ODE solver default: `BDF` via `FITTING_DEFAULT_SOLVER`.
- Bounded fitting parallelism must also bound retained warmed/reusable evaluator state by the same reusable owner identity or by an explicit independent cap. Do not key retained warmed state by dataset, request, run, or task identity unless that identity is independently capped.

## Dependencies

- Do not add new dependencies without explicit instruction.
- Current runtime dependencies are PySide6, shiboken6, numpy, scipy, sympy, bottleneck, pyqtdarktheme-fork, pyqtgraph, and openpyxl. SymPy is a fundamental runtime scientific dependency, not an optional backend dependency.
- **Frozen-build dependency constraints (mandatory):** Nuitka packaging work is deferred, but frozen-build support and deterministic Windows `.exe` prerequisites must not be broken. Runtime scientific/GUI dependencies are not optional and must remain **strictly pinned** (exact `==` pins). Agents must never loosen these specifiers to `>=` ranges, remove them, or move them into optional extras in `requirements.txt`, `pyproject.toml`, or related packaging files without explicit approval.
  - Required runtime exact pins (source of truth): `numpy==2.0.0`, `scipy==1.13.1`, `sympy==1.14.0`, `bottleneck==1.6.0`, `PySide6==6.7.2`, `shiboken6==6.7.2`, `pyqtdarktheme-fork==2.3.4`, `pyqtgraph==0.13.7`, `openpyxl==3.1.5`.
  - `[project.optional-dependencies]` is currently test-only. Do not treat test extras as part of the runtime/frozen-build dependency contract.

## Standing rules

- No literal ampersands in ordinary user-visible labels or messages — use "and". Qt menu/action mnemonic markers such as `&File` are allowed where the menu/action is intentionally mnemonic-bearing and covered by existing UI expectations.
- No backward-compatibility shims, deprecated aliases, or "keep both old and new fields." This app is in active development with no external users. Do not add migration logging, version checks, fallback paths, or any code that exists solely to handle old data formats.
- No new broad fallback paths unless explicitly approved. No "fall back to dialog", generic defaulting, or "graceful degradation" patterns that hide incomplete state. If a code path cannot handle a case, it must reject or error clearly. Existing explicit, tested solver normalization and `Radau`/`BDF` alternative retry paths are part of the solver contract and must not be changed without audit.
- Frozen dataclasses (`@dataclass(frozen=True)`) with mutable fields (dict, list) must use `__post_init__` with `object.__setattr__` to defensively copy all mutable fields.
- Third bugfix round on the same abstraction or data structure triggers a mandatory read-only architectural audit before the next fix. The audit must answer whether the abstraction itself is flawed.
- Every execution prompt must require fresh test impact analysis: assess the specific touched files, ownership boundaries, data flow, and user-visible behavior; identify targeted tests before changes; baseline where useful; re-run after. Do not substitute a static recommended test list for this assessment.
- Every review cycle for code transforming user input into computed output must include a data flow trace.
- Every multi-pass review cycle must wait for all review passes to return before acting on any single reviewer output.
- The `_backup_before_*` workflow is dead — never create backup directories or files.

## Audit rigor and anti-gaming rules

- When auditing numeric validation, the agent must test ALL of these input classes: NaN, inf, -inf, zero, negative, and extreme finite values that could cause overflow or underflow in downstream arithmetic (e.g., `math.exp(-1e9/RT)` overflow, `math.exp(-1e9 * RT)` underflow to zero used as divisor). Missing any class makes the audit incomplete.
- Any audit that involves verifying what the code does with specific inputs must include actual runtime probes (Python one-liners or short scripts), not just code-reading conclusions. "I read the code and it appears to..." is not acceptable when "I ran the code and it does..." is feasible. Include probe commands and their output verbatim in the deliverable.
- Producing a plausible-looking table that matches the requested deliverable format is not completion. Every cell that makes a behavioral claim (e.g., "Rejected", "Accepted", "Crashes") must cite either a runtime probe result or an exact file:line with the relevant guard code. If neither is cited, the cell must say "(unverified)".
- Review passes must include an explicit section: "Input classes I did NOT test and why." If this section is empty or missing, the review is incomplete.
- Never state repo facts, parser behavior, or code behavior as certain without file:line evidence from a fresh audit. If evidence is absent, say "I don't know, needs audit." Presenting assumptions as facts is the single worst failure mode.

## Dismissed / deferred

The following have been explicitly dismissed or deferred. Do not reintroduce without explicit instruction:
- Clone/fallback path for file-level apply-to-remaining
- Scoped `has_unit_row` detection
- Nuitka packaging work (deferred; do not remove frozen-build support or loosen pinned runtime dependencies)
- Two-host shell architecture (`998130f`)
- Slider transaction-semantics redesign

## Hidden features (do not delete, do not expose)

The following features are intentionally hidden from user access. They are NOT dead code. Their implementation files, methods, and internal wiring are intact. Only the user-facing entry points (menu actions, toolbar construction) are commented out.

Do not delete these features. Do not re-expose them without explicit instruction. Each must pass a dedicated integration audit before being restored.

| Feature | Entry point location | Implementation files |
|---|---|---|
| Ribbon | main_window.py — `_init_ribbon_host()` call | widgets/ribbon.py; main_window.py: `_init_ribbon_host()`, `_build_view_ribbon_page()` |
| Species Registry | main_window.py — Edit menu action | main_window.py: _open_species_registry(), _gather_species_registry_entries() |
| State Network Editor | main_window.py — Edit menu action | widgets/state_network_editor.py; main_window.py: `_open_state_network()` |
| Computational Mode | main_window.py — Edit menu action | widgets/computational_mode_dialog.py, core/simulator/computational_mode.py; main_window.py: `_open_computational_mode()` |
| Profiles | main_window.py — Profiles menu | config/profiles.py, gui/mixins/profile_mixin.py |
| Temperature Schedule | main_window.py — Tools menu currently hidden; recreate menu/action only after audit | widgets/temperature_schedule_editor.py, core/temperature_dsl.py; main_window.py: `_open_temperature_schedule_editor()` |

## Output format requirements for agent runs

Run outputs and artifacts meant to brief Pedro, another agent, or a future session must be written for transferability. Use labels and shorthand where helpful, but define non-obvious labels, acronyms, issue keys, local shorthand, and proposed slice names before using them as if they are shared knowledge. Preserve the current objective, scope, evidence status, and unresolved decisions in the artifact itself instead of depending on hidden chat context. Scratch notes are allowed only as separate scratch documents.

Hard ban on greentext and ornamental blockquotes:
- Do not use Markdown blockquotes (`> ...`) for ordinary assistant prose.
- Do not use greentext-style formatting in responses, status updates, plans, audits, reviews, handoffs, artifacts, suggested wording, conclusions, or summaries.
- Do not present instructions for another agent, parent-thread wording, prompt snippets, staging proposals, or "send this back" text as a blockquote.
- A Markdown blockquote is allowed only when Pedro explicitly asks for a quoted block, asks for exact quoted wording in blockquote form, or asks for a short verbatim source excerpt.
- If a quoted phrase or suggested wording is needed without that explicit request, use normal prose, bullets, numbered lists, or a fenced code block instead.
- Before sending any response or writing any artifact, scan it for lines beginning with `>` and remove them unless the narrow explicit-quote exception applies.

For code-changing runs, report:
1) Read-only investigation summary with file:line evidence.
2) Root cause analysis with file:line evidence.
3) Modified or added files list.
4) Tests added and why they fail pre-fix, when applicable.
5) Commands executed and pytest outputs, or explicit skip reasons when a gate is intentionally not run.
6) Concise “what changed” list with file:line anchors.

For strict read-only audits, local-only no-commit slices, and docs-only slices, follow the user-requested report structure and explicitly state when modified files, tests added, commits, or pytest outputs are not applicable.
