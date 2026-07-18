//! Consolidates `windows` + `dependencies` into a single per-step "hard
//! constraint" view: for each statement the target assertion's backward
//! def-use slice ([`crate::dependencies`]) marked "hard", pair it with a
//! human-inspectable rendering of its content.
//!
//! Scope note (deliberately smaller than the original plan's description):
//! the plan describes this module as combining `dependencies.rs` with
//! `vir::expand_errors::ExpansionTree` and the AIR counterexample `Model`.
//! `ExpansionTree` is not wired in yet: it requires the full VIR `Ctx`/`Stm`
//! machinery the existing `--expand-errors` driver uses
//! (`vir::expand_errors::get_expansion_ctx`/`do_expansion`) - a real
//! integration, not attempted here.
//!
//! Real Z3 counterexample values (`air::model::Model`) ARE wired in, but not
//! *by this crate* - `repair` itself never invokes the solver (purely
//! syntactic analysis, no IO), so `HardConstraint` only exposes
//! `referenced_vars` (the already-incarnated variable names, e.g. `x@3`,
//! that a live model would need to be evaluated against). The actual
//! solver re-invocation and evaluation happens one layer up, in
//! `rust_verify::repair_emit` (which already has a live, persistent
//! `air::context::Context` in scope - the same one is reused rather than
//! spinning up a fresh one, since a fresh context would be missing the
//! whole file's accumulated global declarations, see that module).
//!
//! The `text` field is rendered via `crate::translate` - readable
//! Rust/Verus-ish syntax for the common cases it recognizes, falling back to
//! a generic-but-still-inspectable form otherwise (see that module).

use crate::dependencies::{compute_dependencies, free_vars};
use crate::translate::translate_expr;
use crate::windows::{WindowError, describe_stmt, straight_line_trace_to_assert};
use air::ast::{AssertId, Expr, Stmt, StmtX};

#[derive(Debug)]
pub struct HardConstraint {
    pub step_index: usize,
    pub kind: String,
    pub text: String,
    /// Already-incarnated free variable names (e.g. `x@3`) referenced by
    /// this constraint's underlying expression, sorted for deterministic
    /// output - what a caller with a live SAT model needs to evaluate to
    /// attach real counterexample values (see the module doc comment).
    pub referenced_vars: Vec<String>,
}

/// Compute the hard-constraint list for `assert_id` within `stmt`: the subset
/// of the trace that `dependencies::compute_dependencies` marks "hard",
/// rendered for inspection. Soft (irrelevant) steps are omitted entirely -
/// that's the point of this being a *filtered* view.
pub fn compute_hard_constraints(
    stmt: &Stmt,
    assert_id: &AssertId,
) -> Result<Vec<HardConstraint>, WindowError> {
    let trace = straight_line_trace_to_assert(stmt, assert_id)?;
    let deps = compute_dependencies(stmt, assert_id)?;

    Ok(deps
        .hard
        .into_iter()
        .map(|i| HardConstraint {
            step_index: i,
            kind: describe_stmt(&trace[i]),
            text: render_stmt(&trace[i]),
            referenced_vars: referenced_vars(&trace[i]),
        })
        .collect())
}

fn stmt_expr(stmt: &Stmt) -> Option<&Expr> {
    match &**stmt {
        StmtX::Assume(e) => Some(e),
        StmtX::Assert(_, _, _, e) => Some(e),
        _ => None,
    }
}

fn referenced_vars(stmt: &Stmt) -> Vec<String> {
    let Some(expr) = stmt_expr(stmt) else { return Vec::new() };
    let mut vars: Vec<String> = free_vars(expr).into_iter().collect();
    vars.sort();
    vars
}

fn render_stmt(stmt: &Stmt) -> String {
    match &**stmt {
        StmtX::Assume(e) => format!("assume({})", translate_expr(e)),
        StmtX::Assert(_, _, _, e) => format!("assert({})", translate_expr(e)),
        // Not expected in practice (dependencies::compute_dependencies only
        // marks `Assume` steps "hard"), but rendered honestly rather than
        // panicking if that ever changes.
        _ => describe_stmt(stmt),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use air::ast::{BinaryOp, Constant, Expr, ExprX};
    use std::sync::Arc;

    fn var(name: &str) -> Expr {
        Arc::new(ExprX::Var(Arc::new(name.to_string())))
    }

    fn bool_const(b: bool) -> Expr {
        Arc::new(ExprX::Const(Constant::Bool(b)))
    }

    fn eq(lhs: Expr, rhs: Expr) -> Expr {
        Arc::new(ExprX::Binary(BinaryOp::Eq, lhs, rhs))
    }

    fn assert_stmt(id: u64, expr: Expr) -> Stmt {
        Arc::new(StmtX::Assert(
            Some(Arc::new(vec![id])),
            Arc::new(String::from("test assertion")) as air::messages::ArcDynMessage,
            None,
            expr,
        ))
    }

    #[test]
    fn only_hard_steps_are_included_and_rendered() {
        // block {
        //   assume(unrelated == true);   // step 0: soft, excluded
        //   assume(x@1 == y@0);          // step 1: hard, defines x@1
        //   assert(id=1) x@1;
        // }
        let stmt = Arc::new(StmtX::Block(Arc::new(vec![
            Arc::new(StmtX::Assume(eq(var("unrelated"), bool_const(true)))),
            Arc::new(StmtX::Assume(eq(var("x@1"), var("y@0")))),
            assert_stmt(1, var("x@1")),
        ])));
        let id: AssertId = Arc::new(vec![1]);

        let constraints =
            compute_hard_constraints(&stmt, &id).expect("hard constraints should compute");
        assert_eq!(constraints.len(), 1);
        assert_eq!(constraints[0].step_index, 1);
        assert_eq!(constraints[0].kind, "assume");
        assert_eq!(constraints[0].text, "assume((x == y))");
        assert_eq!(constraints[0].referenced_vars, vec!["x@1".to_string(), "y@0".to_string()]);
    }
}
