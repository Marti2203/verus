//! Stepwise WP-style windowing over a straight-line AIR trace.
//!
//! This is the Verus/AIR analog of the Boogie fork's `Sp.cs`/`WlpStepwise.cs`:
//! given a failing target assertion, produce the sequence of incremental
//! "windows" along the trace leading to it, one per preceding statement, so
//! each window can be independently re-checked to localize where the proof
//! state stops explaining the target assertion.
//!
//! Scope: all `StmtX` shapes are supported, including `Switch` (if/else) and
//! `Breakable`/`Break` (loops - Rust/Verus loops lower to a single symbolic
//! iteration check via these two constructs, not literal unrolling). This
//! turns out to require no special version-merge-aware logic at all: reading
//! `air::var_to_const`'s `update_branch_to_versions`/`update_breaks_to_versions`
//! shows that branch/break version merging is already fully materialized as
//! literal `Assume(x@merged == x@from)` statements directly in the tree by
//! the time this module ever sees it - the same structural trick that makes
//! `Switch` support sound (pick the one path that leads to the target,
//! discard the rest) applies identically to `Breakable`: recurse into its
//! body, and if the target isn't found there, the untouched `Breakable`
//! statement is still valid, self-contained AIR to include verbatim as one
//! opaque prior step (a `Break` jumps only to the end of its *own* enclosing
//! `Breakable`, so re-embedding the whole thing elsewhere doesn't break
//! anything). See `repair::dependencies` for a caveat this creates for
//! *dependency* tracking specifically (not windowing).

use air::ast::{AssertId, Expr, Stmt, StmtX, Stmts};
use std::sync::Arc;

/// One incremental window along the trace leading to a target assertion.
#[derive(Debug)]
pub struct Window {
    /// Index of the statement this window ends at (0-based, within the
    /// straight-line trace leading to the target assertion).
    pub step_index: usize,
    /// A human-readable tag for the statement at this step (e.g. "assume",
    /// "assign x", "havoc y") - useful for labeling/debugging output.
    pub step_kind: String,
    /// The prefix of the trace up to and including this step, followed by
    /// a re-assertion of the target assertion's original condition. Checking
    /// this Stmt as a query tells you whether the hypotheses accumulated up
    /// to this step already suffice to explain the (failure of the) target
    /// assertion.
    pub prefix_then_reassert: Stmt,
}

#[derive(Debug, PartialEq, Eq)]
pub enum WindowError {
    /// No statement in `stmt` carries the requested `assert_id`.
    AssertNotFound,
}

/// Compute the stepwise window sequence for `assert_id` within `stmt`.
///
/// `stmt` is expected to already be the *incarnated* form of the query body
/// (i.e. the output of `air::var_to_const::lower_query`, before
/// `air::block_to_assert::lower_query` folds it into a single expression) -
/// see the project plan's Phase B notes on why that ordering matters.
pub fn compute_windows(stmt: &Stmt, assert_id: &AssertId) -> Result<Vec<Window>, WindowError> {
    let trace = straight_line_trace_to_assert(stmt, assert_id)?;
    let target_expr = match trace.last().map(|s| &**s) {
        Some(StmtX::Assert(_, _, _, expr)) => expr.clone(),
        _ => return Err(WindowError::AssertNotFound),
    };

    let mut windows = Vec::new();
    // Windows are prefixes ending at each statement *before* the target
    // assertion itself (the target is appended back on as the re-assert).
    for i in 0..trace.len() - 1 {
        let prefix: Stmts = Arc::new(trace[..=i].to_vec());
        let reassert = Arc::new(StmtX::Assert(
            None,
            Arc::new(String::from("repair-window-reassert")) as air::messages::ArcDynMessage,
            None,
            target_expr.clone(),
        ));
        let mut steps = (*prefix).clone();
        steps.push(reassert);
        windows.push(Window {
            step_index: i,
            step_kind: describe_stmt(&trace[i]),
            prefix_then_reassert: Arc::new(StmtX::Block(Arc::new(steps))),
        });
    }
    Ok(windows)
}

/// Returns just the target assertion's own expression - the tail of the
/// same trace `compute_windows`/`compute_dependencies` operate on. Exposed
/// (unlike the `pub(crate)` trace walk itself) so `rust_verify::repair_emit`
/// can render the postcondition being checked, not just the preceding
/// hypotheses `hard_constraints` surfaces - needed to turn a real Z3
/// counterexample into a concrete test case (the counterexample alone gives
/// concrete *inputs*; the target assertion's own expression, evaluated at
/// those inputs, is what tells you the *specified* (not just currently
/// observed) result).
pub fn target_assertion_expr(stmt: &Stmt, assert_id: &AssertId) -> Result<Expr, WindowError> {
    let trace = straight_line_trace_to_assert(stmt, assert_id)?;
    match trace.last().map(|s| &**s) {
        Some(StmtX::Assert(_, _, _, expr)) => Ok(expr.clone()),
        _ => Err(WindowError::AssertNotFound),
    }
}

/// Walk `stmt` down to `assert_id` through arbitrary `Block`/`Switch`/
/// `Breakable` nesting (picking whichever `Switch` branch or `Breakable`
/// interior leads to the target and discarding the rest). Returns the
/// flattened list of statements from the start of `stmt` through (and
/// including) the target `Assert`.
pub(crate) fn straight_line_trace_to_assert(
    stmt: &Stmt,
    assert_id: &AssertId,
) -> Result<Vec<Stmt>, WindowError> {
    match &**stmt {
        StmtX::Assert(id_opt, ..) => {
            if id_opt.as_ref() == Some(assert_id) {
                Ok(vec![stmt.clone()])
            } else {
                Err(WindowError::AssertNotFound)
            }
        }
        StmtX::Assume(_) | StmtX::Havoc(_) | StmtX::Assign(_, _) | StmtX::Snapshot(_) => {
            Err(WindowError::AssertNotFound)
        }
        StmtX::Block(stmts) => {
            let mut acc = Vec::new();
            for s in stmts.iter() {
                match straight_line_trace_to_assert(s, assert_id) {
                    Ok(mut found_trace) => {
                        acc.append(&mut found_trace);
                        return Ok(acc);
                    }
                    Err(WindowError::AssertNotFound) => {
                        acc.push(s.clone());
                    }
                }
            }
            Err(WindowError::AssertNotFound)
        }
        StmtX::Switch(branches) => {
            // The incarnated tree already has branch-specific versioned variable
            // names baked in by `var_to_const`, so picking the one branch that
            // leads to `assert_id` and discarding the rest (mirroring
            // `air::focus`'s "drop all other cases" semantics) is sound here -
            // we never need the post-switch merged/joined versions, since we
            // stop the trace at the assertion itself.
            for branch in branches.iter() {
                match straight_line_trace_to_assert(branch, assert_id) {
                    Ok(found_trace) => return Ok(found_trace),
                    Err(WindowError::AssertNotFound) => continue,
                }
            }
            Err(WindowError::AssertNotFound)
        }
        // Transparent, like `DeadEnd`: if the target lives inside this loop's
        // body (e.g. an invariant-preservation check on the "keep looping"
        // path), find it there. If not, the whole `Breakable` is still valid,
        // self-contained AIR to fold in as one opaque prior step - see the
        // module doc comment for why this is sound.
        StmtX::Breakable(_, inner) => straight_line_trace_to_assert(inner, assert_id),
        // A bare `Break` never contains the target (it has no nested statements);
        // treat it like any other non-matching leaf.
        StmtX::Break(_) => Err(WindowError::AssertNotFound),
        StmtX::DeadEnd(inner) => straight_line_trace_to_assert(inner, assert_id),
    }
}

pub(crate) fn describe_stmt(stmt: &Stmt) -> String {
    match &**stmt {
        StmtX::Assume(_) => "assume".to_string(),
        StmtX::Assert(..) => "assert".to_string(),
        StmtX::Havoc(x) => format!("havoc {x}"),
        StmtX::Assign(x, _) => format!("assign {x}"),
        StmtX::Snapshot(s) => format!("snapshot {s}"),
        StmtX::DeadEnd(_) => "dead-end".to_string(),
        StmtX::Breakable(_, _) => "breakable".to_string(),
        StmtX::Break(_) => "break".to_string(),
        StmtX::Block(_) => "block".to_string(),
        StmtX::Switch(_) => "switch".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use air::ast::{Constant, ExprX};

    fn bool_const(b: bool) -> air::ast::Expr {
        Arc::new(ExprX::Const(Constant::Bool(b)))
    }

    fn assert_stmt(id: u64, expr: air::ast::Expr) -> Stmt {
        Arc::new(StmtX::Assert(
            Some(Arc::new(vec![id])),
            Arc::new(String::from("test assertion")) as air::messages::ArcDynMessage,
            None,
            expr,
        ))
    }

    #[test]
    fn straight_line_trace_produces_one_window_per_preceding_statement() {
        // block { assume true; assign x := true; assert(id=7) true }
        let stmt = Arc::new(StmtX::Block(Arc::new(vec![
            Arc::new(StmtX::Assume(bool_const(true))),
            Arc::new(StmtX::Assign(Arc::new("x@0".to_string()), bool_const(true))),
            assert_stmt(7, bool_const(true)),
        ])));
        let assert_id: AssertId = Arc::new(vec![7]);

        let windows = compute_windows(&stmt, &assert_id).expect("windows should compute");

        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].step_index, 0);
        assert_eq!(windows[0].step_kind, "assume");
        assert_eq!(windows[1].step_index, 1);
        assert_eq!(windows[1].step_kind, "assign x@0");

        // Each window's prefix_then_reassert should be a Block containing
        // exactly (step_index + 1) original statements plus the re-assert.
        for (i, w) in windows.iter().enumerate() {
            match &*w.prefix_then_reassert {
                StmtX::Block(stmts) => assert_eq!(stmts.len(), i + 2),
                _ => panic!("expected a Block"),
            }
        }
    }

    #[test]
    fn target_assertion_expr_returns_just_the_tail_asserts_expression() {
        let stmt = Arc::new(StmtX::Block(Arc::new(vec![
            Arc::new(StmtX::Assume(bool_const(true))),
            assert_stmt(7, bool_const(false)),
        ])));
        let assert_id: AssertId = Arc::new(vec![7]);

        let expr = target_assertion_expr(&stmt, &assert_id).expect("target expr should be found");
        assert!(matches!(&*expr, ExprX::Const(Constant::Bool(false))));
    }

    #[test]
    fn missing_assert_id_is_reported() {
        let stmt = Arc::new(StmtX::Block(Arc::new(vec![assert_stmt(1, bool_const(true))])));
        let missing: AssertId = Arc::new(vec![999]);
        assert_eq!(compute_windows(&stmt, &missing).unwrap_err(), WindowError::AssertNotFound);
    }

    #[test]
    fn switch_picks_the_branch_containing_the_target_and_drops_the_rest() {
        // block { assume true; switch { branch0: assert(id=1); branch1: assert(id=2); } }
        let stmt = Arc::new(StmtX::Block(Arc::new(vec![
            Arc::new(StmtX::Assume(bool_const(true))),
            Arc::new(StmtX::Switch(Arc::new(vec![
                assert_stmt(1, bool_const(true)),
                assert_stmt(2, bool_const(true)),
            ]))),
        ])));
        let id: AssertId = Arc::new(vec![2]);

        let windows = compute_windows(&stmt, &id).expect("windows should compute");
        // Only the leading `assume` precedes the target inside its branch.
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].step_kind, "assume");
    }

    #[test]
    fn assert_inside_a_loop_body_is_found() {
        // breakable { assume true; assert(id=1) }
        // Models an invariant-preservation check on the "keep looping" path.
        let stmt = Arc::new(StmtX::Breakable(
            Arc::new("loop_end".to_string()),
            Arc::new(StmtX::Block(Arc::new(vec![
                Arc::new(StmtX::Assume(bool_const(true))),
                assert_stmt(1, bool_const(true)),
            ]))),
        ));
        let id: AssertId = Arc::new(vec![1]);

        let windows = compute_windows(&stmt, &id).expect("windows should compute");
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].step_kind, "assume");
    }

    #[test]
    fn irrelevant_sibling_breakable_does_not_veto_windowing() {
        // block { breakable { assert(id=1) } ; assume true; assert(id=2) }
        // The target (id=2) is after the loop (using its merged exit values,
        // in a real incarnated tree) - it must still compute windows, with
        // the untouched loop folded in as one opaque prior step.
        let stmt = Arc::new(StmtX::Block(Arc::new(vec![
            Arc::new(StmtX::Breakable(
                Arc::new("loop_end".to_string()),
                assert_stmt(1, bool_const(true)),
            )),
            Arc::new(StmtX::Assume(bool_const(true))),
            assert_stmt(2, bool_const(true)),
        ])));
        let id: AssertId = Arc::new(vec![2]);

        let windows = compute_windows(&stmt, &id).expect("windows should compute");
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].step_kind, "breakable");
        assert_eq!(windows[1].step_kind, "assume");
    }
}
