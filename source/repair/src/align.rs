//! SSA-aware patch alignment - the Verus/AIR analog of the Boogie fork's
//! incarnation-map instantiation in `PatchConstructor.cs`.
//!
//! An agent reasoning about a fix works from `translate.rs`'s readable,
//! surface-named text (e.g. `assume((z == (x + 1)))`) and proposes a
//! replacement using those same surface names. But the actual AIR tree
//! references incarnated names (`x!`, `z@1`, ...), and a variable's *current*
//! incarnation depends on where in the trace you are (an earlier step sees
//! `x@0`, a later one might see `x@2` after reassignment). This module:
//! 1. parses the small, fully-parenthesized surface grammar `translate.rs`
//!    itself emits (see `parser` below) into a lightweight `SurfaceExpr`,
//! 2. computes the surface-name -> incarnated-name mapping active *at* a
//!    given step by replaying the trace's defining assumes up to that point
//!    (reusing `dependencies::defining_assignment`'s pattern match, so the
//!    two modules can't silently disagree on what "defines a variable"
//!    means), and
//! 3. substitutes accordingly, so the parsed patch can be spliced back into
//!    the incarnated AIR tree at the right point.
//!
//! Scope: parses exactly the operators `translate::translate_expr` can
//! produce (see that module) - this is round-trip alignment for our own
//! rendering, not a general Rust-expression parser.

use air::ast::{AssertId, BinaryOp, Constant, Expr, ExprX, MultiOp, Stmt};
use std::collections::HashMap;

use crate::dependencies::defining_assignment;
use crate::translate::surface_name;
use crate::windows::straight_line_trace_to_assert;

#[derive(Debug, PartialEq, Eq)]
pub enum AlignError {
    Parse(String),
    Trace(crate::windows::WindowError),
}

/// The surface-name -> currently-active-incarnated-name mapping, as of just
/// before `step_index` in `trace` (i.e. reflecting every defining assume at
/// indices `0..step_index`, not `step_index` itself - a patch replacing the
/// statement *at* `step_index` should see the scope as it was on entry to
/// that statement).
pub fn scope_before_step(trace: &[Stmt], step_index: usize) -> HashMap<String, String> {
    let mut scope = HashMap::new();
    for stmt in trace.iter().take(step_index) {
        if let air::ast::StmtX::Assume(expr) = &**stmt {
            if let Some((defined, _rhs)) = defining_assignment(expr) {
                scope.insert(surface_name(&defined), defined);
            }
        }
    }
    scope
}

/// Convenience: compute the scope for the step at `step_index` within the
/// trace leading to `assert_id` in `stmt`.
pub fn scope_for_assert_step(
    stmt: &Stmt,
    assert_id: &AssertId,
    step_index: usize,
) -> Result<HashMap<String, String>, AlignError> {
    let trace = straight_line_trace_to_assert(stmt, assert_id).map_err(AlignError::Trace)?;
    Ok(scope_before_step(&trace, step_index))
}

/// Parse `surface_text` and align every identifier to its currently-active
/// incarnated name per `scope`, falling back to appending
/// `vir::def::SUFFIX_PARAM` (`x` -> `x!`) for names not in `scope` - the
/// convention for a variable that's never been reassigned (a function
/// parameter, referenced in its original incarnation throughout).
pub fn align_patch(
    surface_text: &str,
    scope: &HashMap<String, String>,
) -> Result<Expr, AlignError> {
    let parsed = parser::parse(surface_text).map_err(AlignError::Parse)?;
    Ok(align_surface_expr(&parsed, scope))
}

fn align_surface_expr(expr: &parser::SurfaceExpr, scope: &HashMap<String, String>) -> Expr {
    use parser::SurfaceExpr as S;
    match expr {
        S::Bool(b) => std::sync::Arc::new(ExprX::Const(Constant::Bool(*b))),
        S::Number(n) => {
            std::sync::Arc::new(ExprX::Const(Constant::Nat(std::sync::Arc::new(n.clone()))))
        }
        S::Ident(name) => {
            let incarnated = scope
                .get(name)
                .cloned()
                .unwrap_or_else(|| format!("{}{}", name, vir::def::SUFFIX_PARAM));
            std::sync::Arc::new(ExprX::Var(std::sync::Arc::new(incarnated)))
        }
        S::Old(name) => {
            // Best-effort: `old(x)` always refers to the pre-state incarnation,
            // which the surface->incarnated `scope` map (built from *this*
            // trace's defining assumes) doesn't track - fall back to the bare
            // parameter form, same as an unrecognized identifier.
            std::sync::Arc::new(ExprX::Var(std::sync::Arc::new(format!(
                "{}{}",
                name,
                vir::def::SUFFIX_PARAM
            ))))
        }
        S::Call(name, args) => std::sync::Arc::new(ExprX::Apply(
            std::sync::Arc::new(name.clone()),
            std::sync::Arc::new(args.iter().map(|a| align_surface_expr(a, scope)).collect()),
        )),
        S::Unary(op, inner) => {
            let inner = align_surface_expr(inner, scope);
            let unary_op = match op {
                '!' => air::ast::UnaryOp::Not,
                '-' => air::ast::UnaryOp::BitNeg,
                _ => unreachable!("parser only produces '!'/'-' unary ops"),
            };
            std::sync::Arc::new(ExprX::Unary(unary_op, inner))
        }
        S::Chain(op, operands) => {
            let operands: Vec<Expr> =
                operands.iter().map(|o| align_surface_expr(o, scope)).collect();
            build_chain(op, operands)
        }
    }
}

fn build_chain(op: &str, mut operands: Vec<Expr>) -> Expr {
    if let Some(multi_op) = multi_op_for(op) {
        return std::sync::Arc::new(ExprX::Multi(multi_op, std::sync::Arc::new(operands)));
    }
    // Binary-only operator: fold left-associatively (translate.rs never
    // actually emits chains longer than 2 for these, but folding is a
    // reasonable, still-valid interpretation if it ever does).
    let binary_op = binary_op_for(op).expect("parser only produces recognized operators");
    let mut acc = operands.remove(0);
    for rhs in operands {
        acc = std::sync::Arc::new(ExprX::Binary(binary_op.clone(), acc, rhs));
    }
    acc
}

fn multi_op_for(op: &str) -> Option<MultiOp> {
    Some(match op {
        "&&" => MultiOp::And,
        "||" => MultiOp::Or,
        "^" => MultiOp::Xor,
        "+" => MultiOp::Add,
        "-" => MultiOp::Sub,
        "*" => MultiOp::Mul,
        _ => return None,
    })
}

fn binary_op_for(op: &str) -> Option<BinaryOp> {
    Some(match op {
        "==>" => BinaryOp::Implies,
        "==" => BinaryOp::Eq,
        "<=" => BinaryOp::Le,
        ">=" => BinaryOp::Ge,
        "<" => BinaryOp::Lt,
        ">" => BinaryOp::Gt,
        "/" => BinaryOp::EuclideanDiv,
        "%" => BinaryOp::EuclideanMod,
        _ => return None,
    })
}

mod parser {
    #[derive(Debug, Clone, PartialEq)]
    pub enum SurfaceExpr {
        Ident(String),
        Number(String),
        Bool(bool),
        Old(String),
        Call(String, Vec<SurfaceExpr>),
        Unary(char, Box<SurfaceExpr>),
        /// One operator applied across >=2 operands - covers both the
        /// strictly-binary case (comparisons) and n-ary chains
        /// (`translate_multi`'s "(a + b + c)" rendering).
        Chain(String, Vec<SurfaceExpr>),
    }

    #[derive(Debug, Clone, PartialEq)]
    enum Token {
        LParen,
        RParen,
        Comma,
        Ident(String),
        Number(String),
        Op(String),
        Bang,
    }

    const MULTI_CHAR_OPS: &[&str] = &["==>", "==", "<=", ">=", "<<", ">>", "&&", "||"];
    const SINGLE_CHAR_OPS: &[char] = &['<', '>', '+', '-', '*', '/', '%', '^', '&', '|'];

    fn tokenize(input: &str) -> Result<Vec<Token>, String> {
        let chars: Vec<char> = input.chars().collect();
        let mut i = 0;
        let mut tokens = Vec::new();
        while i < chars.len() {
            let c = chars[i];
            if c.is_whitespace() {
                i += 1;
                continue;
            }
            match c {
                '(' => {
                    tokens.push(Token::LParen);
                    i += 1;
                }
                ')' => {
                    tokens.push(Token::RParen);
                    i += 1;
                }
                ',' => {
                    tokens.push(Token::Comma);
                    i += 1;
                }
                '!' => {
                    tokens.push(Token::Bang);
                    i += 1;
                }
                _ if c.is_ascii_digit() => {
                    let start = i;
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        i += 1;
                    }
                    tokens.push(Token::Number(chars[start..i].iter().collect()));
                }
                _ if c.is_alphabetic() || c == '_' => {
                    let start = i;
                    while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                        i += 1;
                    }
                    tokens.push(Token::Ident(chars[start..i].iter().collect()));
                }
                _ => {
                    let rest: String = chars[i..].iter().collect();
                    if let Some(op) = MULTI_CHAR_OPS.iter().find(|op| rest.starts_with(**op)) {
                        tokens.push(Token::Op(op.to_string()));
                        i += op.len();
                    } else if SINGLE_CHAR_OPS.contains(&c) {
                        tokens.push(Token::Op(c.to_string()));
                        i += 1;
                    } else {
                        return Err(format!("unexpected character '{c}' at byte offset {i}"));
                    }
                }
            }
        }
        Ok(tokens)
    }

    pub fn parse(input: &str) -> Result<SurfaceExpr, String> {
        let tokens = tokenize(input)?;
        let mut pos = 0;
        let expr = parse_expr(&tokens, &mut pos)?;
        if pos != tokens.len() {
            return Err(format!("unexpected trailing tokens at position {pos}"));
        }
        Ok(expr)
    }

    fn peek(tokens: &[Token], pos: usize) -> Option<&Token> {
        tokens.get(pos)
    }

    fn parse_expr(tokens: &[Token], pos: &mut usize) -> Result<SurfaceExpr, String> {
        match peek(tokens, *pos) {
            Some(Token::Number(n)) => {
                let n = n.clone();
                *pos += 1;
                Ok(SurfaceExpr::Number(n))
            }
            Some(Token::Bang) => {
                *pos += 1;
                expect(tokens, pos, &Token::LParen)?;
                let inner = parse_expr(tokens, pos)?;
                expect(tokens, pos, &Token::RParen)?;
                Ok(SurfaceExpr::Unary('!', Box::new(inner)))
            }
            Some(Token::Op(op)) if op == "-" => {
                *pos += 1;
                expect(tokens, pos, &Token::LParen)?;
                let inner = parse_expr(tokens, pos)?;
                expect(tokens, pos, &Token::RParen)?;
                Ok(SurfaceExpr::Unary('-', Box::new(inner)))
            }
            Some(Token::Ident(name)) => {
                let name = name.clone();
                *pos += 1;
                match name.as_str() {
                    "true" => Ok(SurfaceExpr::Bool(true)),
                    "false" => Ok(SurfaceExpr::Bool(false)),
                    "old" if peek(tokens, *pos) == Some(&Token::LParen) => {
                        *pos += 1;
                        let inner = match peek(tokens, *pos) {
                            Some(Token::Ident(n)) => n.clone(),
                            other => {
                                return Err(format!(
                                    "expected identifier inside old(...), got {other:?}"
                                ));
                            }
                        };
                        *pos += 1;
                        expect(tokens, pos, &Token::RParen)?;
                        Ok(SurfaceExpr::Old(inner))
                    }
                    _ if peek(tokens, *pos) == Some(&Token::LParen) => {
                        *pos += 1;
                        let mut args = Vec::new();
                        if peek(tokens, *pos) != Some(&Token::RParen) {
                            args.push(parse_expr(tokens, pos)?);
                            while peek(tokens, *pos) == Some(&Token::Comma) {
                                *pos += 1;
                                args.push(parse_expr(tokens, pos)?);
                            }
                        }
                        expect(tokens, pos, &Token::RParen)?;
                        Ok(SurfaceExpr::Call(name, args))
                    }
                    _ => Ok(SurfaceExpr::Ident(name)),
                }
            }
            Some(Token::LParen) => {
                *pos += 1;
                let mut operands = vec![parse_expr(tokens, pos)?];
                let mut op: Option<String> = None;
                loop {
                    match peek(tokens, *pos) {
                        Some(Token::Op(next_op)) => {
                            if let Some(existing) = &op {
                                if existing != next_op {
                                    return Err(format!(
                                        "mixed operators '{existing}' and '{next_op}' in one chain - not produced by translate::translate_expr"
                                    ));
                                }
                            } else {
                                op = Some(next_op.clone());
                            }
                            *pos += 1;
                            operands.push(parse_expr(tokens, pos)?);
                        }
                        _ => break,
                    }
                }
                expect(tokens, pos, &Token::RParen)?;
                let op =
                    op.ok_or_else(|| "empty parenthesized group has no operator".to_string())?;
                Ok(SurfaceExpr::Chain(op, operands))
            }
            other => Err(format!("unexpected token {other:?} at position {}", *pos)),
        }
    }

    fn expect(tokens: &[Token], pos: &mut usize, expected: &Token) -> Result<(), String> {
        match peek(tokens, *pos) {
            Some(t) if t == expected => {
                *pos += 1;
                Ok(())
            }
            other => Err(format!("expected {expected:?}, got {other:?} at position {}", *pos)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translate::translate_expr;
    use air::ast::{AssertId, ExprX, StmtX};
    use std::sync::Arc;

    fn var(name: &str) -> Expr {
        Arc::new(ExprX::Var(Arc::new(name.to_string())))
    }

    fn nat(n: &str) -> Expr {
        Arc::new(ExprX::Const(Constant::Nat(Arc::new(n.to_string()))))
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
    fn scope_reflects_only_assumes_before_the_given_step() {
        // block { assume(x@1 == y!); assume(z@1 == x@1); assert(id=1) z@1 }
        let stmt = Arc::new(StmtX::Block(Arc::new(vec![
            Arc::new(StmtX::Assume(Arc::new(ExprX::Binary(BinaryOp::Eq, var("x@1"), var("y!"))))),
            Arc::new(StmtX::Assume(Arc::new(ExprX::Binary(BinaryOp::Eq, var("z@1"), var("x@1"))))),
            assert_stmt(1, var("z@1")),
        ])));
        let id: AssertId = Arc::new(vec![1]);
        let trace = straight_line_trace_to_assert(&stmt, &id).unwrap();

        // Before step 0: nothing defined yet.
        assert!(scope_before_step(&trace, 0).is_empty());
        // Before step 1: only x has been defined (as x@1).
        let scope1 = scope_before_step(&trace, 1);
        assert_eq!(scope1.get("x"), Some(&"x@1".to_string()));
        assert_eq!(scope1.get("z"), None);
        // Before step 2 (the assert itself): both x and z are defined.
        let scope2 = scope_before_step(&trace, 2);
        assert_eq!(scope2.get("x"), Some(&"x@1".to_string()));
        assert_eq!(scope2.get("z"), Some(&"z@1".to_string()));
    }

    #[test]
    fn align_patch_maps_surface_names_to_scoped_incarnations() {
        let mut scope = HashMap::new();
        scope.insert("x".to_string(), "x@2".to_string());
        // "y" deliberately absent from scope - should fall back to "y!".
        let expr = align_patch("(x + y)", &scope).expect("should parse and align");
        assert_eq!(translate_expr(&expr), "(x + y)"); // re-translating strips suffixes back
        match &*expr {
            ExprX::Multi(MultiOp::Add, args) => {
                assert_eq!(format!("{:?}", args[0]), format!("{:?}", var("x@2")));
                assert_eq!(format!("{:?}", args[1]), format!("{:?}", var("y!")));
            }
            other => panic!("expected Multi(Add, ...), got {other:?}"),
        }
    }

    #[test]
    fn round_trips_through_translate_and_back_for_representative_shapes() {
        let cases: Vec<Expr> = vec![
            nat("2"),
            var("x!"),
            Arc::new(ExprX::Binary(BinaryOp::Eq, var("r!"), var("z@1"))),
            Arc::new(ExprX::Multi(MultiOp::Add, Arc::new(vec![var("x!"), nat("2")]))),
            Arc::new(ExprX::Unary(air::ast::UnaryOp::Not, var("b!"))),
        ];
        for original in cases {
            let rendered = translate_expr(&original);
            let reparsed = align_patch(&rendered, &HashMap::new())
                .unwrap_or_else(|e| panic!("failed to reparse {rendered:?}: {e:?}"));
            // Re-translating the reparsed (unaligned, so surface names pass
            // through as-is minus suffix-stripping) form should match the
            // original rendering, confirming a stable round trip.
            assert_eq!(translate_expr(&reparsed), rendered, "round trip mismatch for {original:?}");
        }
    }

    #[test]
    fn rejects_mixed_operators_in_one_chain() {
        assert!(align_patch("(x + y == z)", &HashMap::new()).is_err());
    }
}
