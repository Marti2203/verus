//! Data-dependency tracking over an incarnated AIR trace - the VIR/AIR analog
//! of the Boogie fork's `GetDependencies`/`GetDataDependencies`.
//!
//! This is marked experimental in the project plan: since the repair loop is
//! agentic rather than a scored search, this doesn't feed a ranking formula -
//! it's exposed as one queryable fact source (which preceding statements does
//! the target assertion actually depend on?) that an agent can consult, and
//! whether it earns its keep over windows + counterexamples alone is meant to
//! be evaluated once it exists, not assumed.
//!
//! How it works: after `air::var_to_const::lower_query`, every `Assign(x, e)`
//! has already become `Assume(x@n == e)` (see that module's `lower_stmt`) -
//! so a backward def-use slice is just: start from the target assertion's
//! free variables, walk the trace backward, and whenever an `Assume` defines
//! (via top-level `==`) a variable currently in the frontier, mark it "hard",
//! remove that variable, and add the free variables of its right-hand side to
//! the frontier. Everything else is "soft" (present, but not shown relevant).
//!
//! Known limitation: `windows::straight_line_trace_to_assert` folds a
//! `Switch`/`Breakable` that doesn't directly contain the target into one
//! opaque trace step (see that module's doc comment) - which is sound for
//! *windowing* (re-verifying the prefix), since the whole statement is
//! included verbatim. But *this* module only pattern-matches top-level
//! `Assume(x == rhs)` per step, so a defining assumption buried inside such
//! an opaque step (e.g. a loop's merge-point equality for a variable the
//! target actually depends on) is currently missed and the step is marked
//! "soft" even when it may be causally relevant. See
//! `opaque_loop_step_dependency_is_missed_soft_for_now` below - this is a
//! deliberate, tested first-cut limitation, not an undiscovered gap.

use crate::windows::{WindowError, straight_line_trace_to_assert};
use air::ast::{AssertId, BinaryOp, BindX, Expr, ExprX, Stmt, StmtX};
use std::collections::HashSet;

/// Which preceding statements (by index into the straight-line trace, 0-based,
/// same numbering as `windows::Window::step_index`) the target assertion's
/// value actually depends on ("hard") versus not ("soft").
#[derive(Debug, PartialEq, Eq)]
pub struct DependencyResult {
    pub hard: Vec<usize>,
    pub soft: Vec<usize>,
}

/// Compute the def-use dependency slice for `assert_id` within `stmt`.
///
/// Same scope restrictions as [`crate::windows::compute_windows`]: straight-line
/// `Block`/`Switch` traces only, `stmt` expected to already be `var_to_const`-incarnated.
pub fn compute_dependencies(
    stmt: &Stmt,
    assert_id: &AssertId,
) -> Result<DependencyResult, WindowError> {
    let trace = straight_line_trace_to_assert(stmt, assert_id)?;
    let target_expr = match trace.last().map(|s| &**s) {
        Some(StmtX::Assert(_, _, _, expr)) => expr.clone(),
        _ => return Err(WindowError::AssertNotFound),
    };

    let mut frontier = free_vars(&target_expr);
    let mut hard = Vec::new();
    let mut soft = Vec::new();

    for (i, s) in trace[..trace.len() - 1].iter().enumerate().rev() {
        let defines = match &**s {
            StmtX::Assume(expr) => defining_assignment(expr),
            _ => None,
        };
        match defines {
            Some((defined_var, rhs)) if frontier.contains(&defined_var) => {
                frontier.remove(&defined_var);
                frontier.extend(free_vars(&rhs));
                hard.push(i);
            }
            _ => soft.push(i),
        }
    }
    hard.reverse();
    soft.reverse();
    Ok(DependencyResult { hard, soft })
}

/// If `expr` is a top-level `x == rhs` (the shape `var_to_const` produces for
/// a lowered assignment), return the defined variable name and its rhs.
pub(crate) fn defining_assignment(expr: &Expr) -> Option<(String, Expr)> {
    if let ExprX::Binary(BinaryOp::Eq, lhs, rhs) = &**expr {
        if let ExprX::Var(x) = &**lhs {
            return Some(((**x).clone(), rhs.clone()));
        }
    }
    None
}

/// `pub` (not just `pub(crate)`) so `rust_verify::repair_emit` can compute
/// the free variables of the target assertion itself (not just a hard
/// constraint step) when deriving a counterexample-based test case.
pub fn free_vars(expr: &Expr) -> HashSet<String> {
    let mut out = HashSet::new();
    collect_free_vars(expr, &mut out);
    out
}

fn collect_free_vars(expr: &Expr, out: &mut HashSet<String>) {
    match &**expr {
        ExprX::Const(_) => {}
        ExprX::Var(x) => {
            out.insert((**x).clone());
        }
        ExprX::Old(_, x) => {
            out.insert((**x).clone());
        }
        ExprX::Apply(_, args) => {
            for a in args.iter() {
                collect_free_vars(a, out);
            }
        }
        ExprX::ApplyFun(_, f, args) => {
            collect_free_vars(f, out);
            for a in args.iter() {
                collect_free_vars(a, out);
            }
        }
        ExprX::Unary(_, e) => collect_free_vars(e, out),
        ExprX::Binary(_, l, r) => {
            collect_free_vars(l, out);
            collect_free_vars(r, out);
        }
        ExprX::Multi(_, es) => {
            for e in es.iter() {
                collect_free_vars(e, out);
            }
        }
        ExprX::IfElse(c, t, f) => {
            collect_free_vars(c, out);
            collect_free_vars(t, out);
            collect_free_vars(f, out);
        }
        ExprX::Array(es) => {
            for e in es.iter() {
                collect_free_vars(e, out);
            }
        }
        ExprX::Bind(bind, body) => {
            let mut bound: HashSet<String> = HashSet::new();
            let mut extra_free: HashSet<String> = HashSet::new();
            match &**bind {
                BindX::Let(binders) => {
                    for b in binders.iter() {
                        // The bound value is evaluated in the *outer* scope.
                        collect_free_vars(&b.a, out);
                        bound.insert((*b.name).clone());
                    }
                }
                BindX::Quant(_, binders, _triggers, _qid) => {
                    for b in binders.iter() {
                        bound.insert((*b.name).clone());
                    }
                }
                BindX::Lambda(binders, _triggers, _qid) => {
                    for b in binders.iter() {
                        bound.insert((*b.name).clone());
                    }
                }
                BindX::Choose(binders, _triggers, _qid, cond) => {
                    for b in binders.iter() {
                        bound.insert((*b.name).clone());
                    }
                    collect_free_vars(cond, &mut extra_free);
                }
            }
            let mut body_free: HashSet<String> = HashSet::new();
            collect_free_vars(body, &mut body_free);
            for v in body_free.into_iter().chain(extra_free) {
                if !bound.contains(&v) {
                    out.insert(v);
                }
            }
        }
        ExprX::LabeledAxiom(_, _, e) => collect_free_vars(e, out),
        ExprX::LabeledAssertion(_, _, _, e) => collect_free_vars(e, out),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use air::ast::Constant;
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
    fn defining_assignment_is_marked_hard_and_unrelated_assume_is_soft() {
        // block {
        //   assume(unrelated == true);       // step 0: irrelevant precondition
        //   assume(x@1 == y@0);              // step 1: defines x@1 from y@0
        //   assert(id=1) x@1;                // target reads x@1
        // }
        let stmt = Arc::new(StmtX::Block(Arc::new(vec![
            Arc::new(StmtX::Assume(eq(var("unrelated"), bool_const(true)))),
            Arc::new(StmtX::Assume(eq(var("x@1"), var("y@0")))),
            assert_stmt(1, var("x@1")),
        ])));
        let id: AssertId = Arc::new(vec![1]);

        let deps = compute_dependencies(&stmt, &id).expect("dependencies should compute");
        assert_eq!(deps.hard, vec![1]);
        assert_eq!(deps.soft, vec![0]);
    }

    #[test]
    fn transitive_dependency_chain_is_followed() {
        // x@1 depends on y@0, which is itself defined earlier from z@0.
        let stmt = Arc::new(StmtX::Block(Arc::new(vec![
            Arc::new(StmtX::Assume(eq(var("y@0"), var("z@0")))),
            Arc::new(StmtX::Assume(eq(var("x@1"), var("y@0")))),
            assert_stmt(1, var("x@1")),
        ])));
        let id: AssertId = Arc::new(vec![1]);

        let deps = compute_dependencies(&stmt, &id).expect("dependencies should compute");
        assert_eq!(deps.hard, vec![0, 1]);
        assert!(deps.soft.is_empty());
    }

    #[test]
    fn opaque_loop_step_dependency_is_missed_soft_for_now() {
        // block {
        //   breakable { assume(x@1 == y@0); }   // step 0: a loop whose merge
        //                                        // point actually defines x@1
        //   assert(id=1) x@1;                    // target depends on x@1
        // }
        // Per the module doc comment: this SHOULD ideally be "hard", but the
        // defining assume is buried inside an opaque `Breakable` step that
        // `compute_dependencies` doesn't recurse into, so it's "soft" for now.
        let stmt = Arc::new(StmtX::Block(Arc::new(vec![
            Arc::new(StmtX::Breakable(
                Arc::new("loop_end".to_string()),
                Arc::new(StmtX::Assume(eq(var("x@1"), var("y@0")))),
            )),
            assert_stmt(1, var("x@1")),
        ])));
        let id: AssertId = Arc::new(vec![1]);

        let deps = compute_dependencies(&stmt, &id).expect("dependencies should compute");
        assert!(deps.hard.is_empty());
        assert_eq!(deps.soft, vec![0]);
    }
}
