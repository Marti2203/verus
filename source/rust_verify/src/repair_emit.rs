//! Bridges a failing verification query to the `repair` crate's fault-localization
//! windowing, gated behind `--repair-emit-facts <path>`. Deliberately kept separate
//! from the `--expand-errors` state machine (`expand_errors_driver.rs`) so this
//! experimental feature can't destabilize that existing one.
//!
//! Also re-invokes Z3 for real counterexample values (`air::model::Model`),
//! attached per hard constraint. `repair` itself never touches the solver
//! (see its `hard_constraints` module doc comment) - it only tells us which
//! already-incarnated variable names (e.g. `x@3`) a given constraint
//! references; evaluating those against a live SAT model has to happen here,
//! where a running `air::context::Context` is actually available. The
//! *same, already-running, persistent* context is reused (not a fresh one),
//! since a fresh context would be missing every global declaration
//! (datatypes, function definitions) accumulated from the rest of the file
//! by this point in the run - this query's assertion may well reference
//! those. `check_valid`'s own incarnation pass (`var_to_const::lower_query`)
//! is a pure function of the query alone, so calling it a second time here
//! (Verus's normal verification already called it once for the original,
//! unfocused query) reliably reproduces the same `x@N` names our own
//! `hard_constraints`-side incarnation just computed.
//!
//! Also flattens a single-level `vir::expand_errors::ExpansionTree` (the
//! last of Phase B's three originally-deferred items) into JSON, if the
//! caller computed one - the logical decomposition of the target
//! assertion's own expression (splitting a conjunction, unfolding one level
//! of a spec function call, etc.), the same primitive `--expand-errors`
//! uses, computed exactly once (not driven iteratively the way that
//! feature's own state machine re-verifies each resulting leaf - this crate
//! only wants the *structural* decomposition, not a second round of Z3
//! checks). Actually computing the tree needs `ctx.fun` toggled around the
//! call, which needs `FunctionOpGenerator` (see
//! `rust_verify::commands::FunctionOpGenerator::compute_expansion_tree`,
//! co-located with the analogous `--expand-errors` logic instead of
//! duplicated here) - this module just flattens whatever tree it's handed.
//! Leaf/introduction text is `Debug`-rendered, an honest placeholder (the
//! same bootstrapping step `hard_constraints.rs` took before `translate.rs`
//! existed) - VIR's `Exp` is a different, larger AST than the AIR `Expr`
//! `translate.rs` already handles, and building a matching translator for
//! it is real, separate follow-up work, not attempted here.

use air::ast::{AssertId, CommandX};
use air::context::{QueryContext, ValidityResult};
use std::collections::HashMap;
use std::sync::Arc;
use vir::def::CommandsWithContext;

#[derive(serde::Serialize)]
struct WindowFact {
    step_index: usize,
    step_kind: String,
}

#[derive(serde::Serialize)]
struct DependencyFact {
    hard: Vec<usize>,
    soft: Vec<usize>,
}

#[derive(serde::Serialize)]
struct HardConstraintFact {
    step_index: usize,
    kind: String,
    text: String,
    /// `(variable, concrete value)` pairs from a real Z3 counterexample -
    /// empty if the re-check didn't produce a model (e.g. a bitvector-only
    /// failure, or the re-check unexpectedly succeeded) rather than that
    /// being conflated with "not applicable".
    counterexample: Vec<(String, String)>,
}

#[derive(serde::Serialize)]
struct ExpansionFact {
    /// The trail of structural decompositions (`Debug`-rendered
    /// `Introduction`s, e.g. `SplitEquality(...)`) leading to this leaf -
    /// see the module doc comment for why these aren't rendered more
    /// readably yet.
    introductions: Vec<String>,
    /// Verus's own classification of whether this leaf could be expanded
    /// further (another level of `do_expansion`) - not acted on here, just
    /// surfaced.
    can_expand_further: bool,
    /// `Debug`-rendered leaf expression.
    leaf: String,
}

#[derive(serde::Serialize)]
struct RepairFacts {
    function: String,
    assert_id: Vec<u64>,
    windows: Vec<WindowFact>,
    /// Present only when `repair::dependencies::compute_dependencies` succeeds
    /// on the same trace - experimental, see the project plan.
    dependencies: Option<DependencyFact>,
    /// The "hard" subset of `dependencies`, rendered for inspection, each
    /// with a real counterexample value per referenced variable when the Z3
    /// re-check produces a model.
    hard_constraints: Vec<HardConstraintFact>,
    /// A single-level structural decomposition of the target assertion's
    /// own expression (`vir::expand_errors::ExpansionTree`, flattened) -
    /// empty if no `FuncCheckSst` was available, the assert_id wasn't found
    /// in it (a real, guarded-against mismatch - not expected to happen
    /// since it's the same assert_id `--expand-errors` itself uses
    /// successfully for this same purpose, but not assumed), or the
    /// decomposition step panicked internally.
    expansion: Vec<ExpansionFact>,
    /// The function's own parameters, in declaration order, each already
    /// suffixed the same way `var_to_const` leaves a never-reassigned
    /// parameter (`vir::def::SUFFIX_PARAM`, e.g. `x` -> `x!`) - so each
    /// entry here matches a `target_counterexample` key directly, with no
    /// further name reconciliation needed by a consumer.
    parameters: Vec<String>,
    /// The target (failing) assertion's own expression, rendered the same
    /// way `hard_constraints[].text` is - e.g. `(r == (x + 2))`. Distinct
    /// from `hard_constraints`, which only ever surfaces *preceding*
    /// hypotheses, never the postcondition actually being checked.
    target_assertion: Option<String>,
    /// `(variable, concrete value)` pairs for every free variable of
    /// `target_assertion` - the same real Z3 re-check `hard_constraints`'
    /// counterexamples come from, just evaluated against this expression's
    /// own variables instead. Together with `parameters`, this is what lets
    /// a consumer derive a concrete test case: concrete *inputs* from here,
    /// and the *specified* result at those inputs from `target_assertion`
    /// itself (not from whatever the current, possibly-still-buggy
    /// implementation happens to return) - see the project plan/memory.
    target_counterexample: Vec<(String, String)>,
    /// When a function has *more than one* simultaneously-failing assertion
    /// (confirmed real, e.g. `is_woodall`'s two independent overflow bugs),
    /// one entry per assertion beyond the first (`assert_id`/`target_assertion`/
    /// `target_counterexample` above already cover that one). Deliberately
    /// lighter-weight than the primary target: just the failing condition
    /// itself and its own counterexample, not a full `windows`/`dependencies`/
    /// `hard_constraints` causal trace - the point is visibility that another,
    /// independent condition is *also* currently failing, not a second full
    /// fault-localization pass. Empty when there's only one failing assertion
    /// (the overwhelmingly common case), not omitted, so a consumer never has
    /// to special-case "field absent" vs "field empty".
    other_targets: Vec<OtherTargetFact>,
}

#[derive(serde::Serialize)]
struct OtherTargetFact {
    assert_id: Vec<u64>,
    target_assertion: Option<String>,
    target_counterexample: Vec<(String, String)>,
}

/// Compute stepwise fault-localization windows (and, best-effort, a
/// dependency slice) for `assert_ids[0]` and write them as JSON to
/// `out_path`, along with a lighter-weight `other_targets` entry for every
/// remaining id in `assert_ids` (see `OtherTargetFact`'s doc comment).
/// Silently skips commands whose trace isn't yet supported by
/// `repair::windows`/`repair::dependencies` (loops) rather than failing the
/// whole verification run over an experimental diagnostic. Does nothing if
/// `assert_ids` is empty (the caller already guards this, but this function
/// doesn't rely on that).
#[allow(clippy::too_many_arguments)]
pub fn emit_repair_facts(
    function_name: &str,
    parameters: &[String],
    commands_with_context_list: &Arc<Vec<CommandsWithContext>>,
    assert_ids: &[AssertId],
    out_path: &str,
    air_context: &mut air::context::Context,
    message_interface: &dyn air::messages::MessageInterface,
    diagnostics: &impl air::messages::Diagnostics,
    expansion_tree: Option<vir::expand_errors::ExpansionTree>,
) {
    let Some(assert_id) = assert_ids.first() else {
        return;
    };
    let expansion = match &expansion_tree {
        Some(tree) => {
            let mut facts = Vec::new();
            let mut path = Vec::new();
            flatten_expansion_tree(tree, &mut path, &mut facts);
            facts
        }
        None => Vec::new(),
    };

    let mut all_windows: Vec<WindowFact> = Vec::new();
    let mut dependencies: Option<DependencyFact> = None;
    let mut hard_constraints: Vec<HardConstraintFact> = Vec::new();
    let mut target_assertion: Option<String> = None;
    let mut target_counterexample: Vec<(String, String)> = Vec::new();
    for commands_with_context in commands_with_context_list.iter() {
        if commands_with_context.prover_choice != vir::def::ProverChoice::DefaultProver {
            continue;
        }
        let focused =
            air::focus::focus_commands_on_assert_id(&commands_with_context.commands, assert_id);
        for command in focused.iter() {
            if let CommandX::CheckValid(query) = &**command {
                let (incarnated_query, _snapshots, _decls) = air::var_to_const::lower_query(query);
                if let Ok(windows) =
                    repair::windows::compute_windows(&incarnated_query.assertion, assert_id)
                {
                    all_windows.extend(
                        windows
                            .into_iter()
                            .map(|w| WindowFact { step_index: w.step_index, step_kind: w.step_kind }),
                    );
                }
                if let Ok(deps) = repair::dependencies::compute_dependencies(
                    &incarnated_query.assertion,
                    assert_id,
                ) {
                    dependencies = Some(DependencyFact { hard: deps.hard, soft: deps.soft });
                }

                let constraints = repair::hard_constraints::compute_hard_constraints(
                    &incarnated_query.assertion,
                    assert_id,
                )
                .unwrap_or_default();
                let target_expr =
                    repair::windows::target_assertion_expr(&incarnated_query.assertion, assert_id).ok();
                let target_vars: Vec<String> = target_expr
                    .as_ref()
                    .map(|e| {
                        let mut v: Vec<String> = repair::dependencies::free_vars(e).into_iter().collect();
                        v.sort();
                        v
                    })
                    .unwrap_or_default();

                // One re-check covers every variable either fact source
                // needs - hard constraints' own referenced vars, plus the
                // target assertion's, deduplicated.
                let mut wanted: Vec<String> =
                    constraints.iter().flat_map(|c| c.referenced_vars.iter().cloned()).collect();
                wanted.extend(target_vars.iter().cloned());
                wanted.sort();
                wanted.dedup();
                let values = reevaluate_counterexample(
                    air_context,
                    message_interface,
                    diagnostics,
                    query,
                    &wanted,
                );

                hard_constraints.extend(constraints.into_iter().map(|c| {
                    let counterexample = c
                        .referenced_vars
                        .iter()
                        .filter_map(|v| values.get(v).map(|val| (v.clone(), val.clone())))
                        .collect();
                    HardConstraintFact { step_index: c.step_index, kind: c.kind, text: c.text, counterexample }
                }));

                if let Some(expr) = &target_expr {
                    target_assertion = Some(repair::translate::translate_expr(expr));
                    target_counterexample = target_vars
                        .iter()
                        .filter_map(|v| values.get(v).map(|val| (v.clone(), val.clone())))
                        .collect();
                }
            }
        }
    }

    let other_targets: Vec<OtherTargetFact> = assert_ids[1..]
        .iter()
        .map(|extra_id| {
            compute_other_target(
                commands_with_context_list,
                extra_id,
                air_context,
                message_interface,
                diagnostics,
            )
        })
        .collect();

    let facts = RepairFacts {
        function: function_name.to_string(),
        assert_id: (**assert_id).clone(),
        windows: all_windows,
        dependencies,
        hard_constraints,
        parameters: parameters.to_vec(),
        target_assertion,
        target_counterexample,
        expansion,
        other_targets,
    };
    if let Ok(json) = serde_json::to_string_pretty(&facts) {
        let _ = std::fs::write(out_path, json);
    }
}

/// The lightweight per-assertion computation behind `other_targets`: just
/// the failing condition's own rendered text and a real counterexample for
/// its free variables - the same `focus` -> `lower_query` ->
/// `target_assertion_expr` -> re-check pipeline the primary target uses
/// above, minus `windows`/`dependencies`/`hard_constraints` (a full causal
/// trace per *additional* simultaneous failure is deliberately out of scope
/// here - see `OtherTargetFact`'s doc comment).
fn compute_other_target(
    commands_with_context_list: &Arc<Vec<CommandsWithContext>>,
    assert_id: &AssertId,
    air_context: &mut air::context::Context,
    message_interface: &dyn air::messages::MessageInterface,
    diagnostics: &impl air::messages::Diagnostics,
) -> OtherTargetFact {
    let mut target_assertion: Option<String> = None;
    let mut target_counterexample: Vec<(String, String)> = Vec::new();
    for commands_with_context in commands_with_context_list.iter() {
        if commands_with_context.prover_choice != vir::def::ProverChoice::DefaultProver {
            continue;
        }
        let focused =
            air::focus::focus_commands_on_assert_id(&commands_with_context.commands, assert_id);
        for command in focused.iter() {
            if let CommandX::CheckValid(query) = &**command {
                let (incarnated_query, _snapshots, _decls) = air::var_to_const::lower_query(query);
                let target_expr =
                    repair::windows::target_assertion_expr(&incarnated_query.assertion, assert_id).ok();
                let target_vars: Vec<String> = target_expr
                    .as_ref()
                    .map(|e| {
                        let mut v: Vec<String> = repair::dependencies::free_vars(e).into_iter().collect();
                        v.sort();
                        v
                    })
                    .unwrap_or_default();
                if let Some(expr) = &target_expr {
                    let values = reevaluate_counterexample(
                        air_context,
                        message_interface,
                        diagnostics,
                        query,
                        &target_vars,
                    );
                    target_assertion = Some(repair::translate::translate_expr(expr));
                    target_counterexample = target_vars
                        .iter()
                        .filter_map(|v| values.get(v).map(|val| (v.clone(), val.clone())))
                        .collect();
                }
            }
        }
    }
    OtherTargetFact {
        assert_id: (**assert_id).clone(),
        target_assertion,
        target_counterexample,
    }
}

fn flatten_expansion_tree(
    tree: &vir::expand_errors::ExpansionTree,
    path: &mut Vec<String>,
    out: &mut Vec<ExpansionFact>,
) {
    match tree {
        vir::expand_errors::ExpansionTree::Branch(children) => {
            for child in children {
                flatten_expansion_tree(child, path, out);
            }
        }
        vir::expand_errors::ExpansionTree::Intro(intro, child) => {
            path.push(format!("{:?}", intro));
            flatten_expansion_tree(child, path, out);
            path.pop();
        }
        vir::expand_errors::ExpansionTree::Leaf(_assert_id, exp, can_expand_further) => {
            out.push(ExpansionFact {
                introductions: path.clone(),
                can_expand_further: matches!(can_expand_further, vir::expand_errors::CanExpandFurther::Yes),
                leaf: format!("{:?}", exp),
            });
        }
    }
}

/// Re-runs `query` against the live, persistent solver session and, if it's
/// still invalid with a model, reads off the concrete value of every
/// variable referenced by `constraints` directly from that model. Always
/// finishes the query afterward (mirroring the normal verification loop's
/// own unconditional `finish_query` after every `CheckValid`), regardless
/// of the outcome, so the shared context is left in the correct state for
/// whatever comes next in the overall run.
///
/// Deliberately reads `Model::raw_value` rather than issuing a separate
/// `Context::eval_expr` call per variable afterward - the latter is unsound
/// here: obtaining a model queues a "disable this label" assert for the
/// *next* batch of commands sent to Z3 (so a later check-sat can find
/// additional errors), and that queued assert reaches Z3 before any
/// subsequent `eval_expr` call's own command does, invalidating the model
/// (confirmed by hitting exactly this failure in practice). Reading
/// `raw_value` off the model that's already been parsed sidesteps the
/// ordering hazard entirely - see `air::model::Model`'s doc comment.
fn reevaluate_counterexample(
    air_context: &mut air::context::Context,
    message_interface: &dyn air::messages::MessageInterface,
    diagnostics: &impl air::messages::Diagnostics,
    query: &air::ast::Query,
    wanted_vars: &[String],
) -> HashMap<String, String> {
    let validity = air_context.check_valid(
        message_interface,
        diagnostics,
        query,
        QueryContext { report_long_running: None },
    );
    let mut values = HashMap::new();
    if let ValidityResult::Invalid(Some(model), _, _) = &validity {
        for var in wanted_vars {
            if let Some(value) = model.raw_value(&Arc::new(var.clone())) {
                values.insert(var.clone(), value.to_string());
            }
        }
    }
    air_context.finish_query();
    values
}
