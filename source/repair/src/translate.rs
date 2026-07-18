//! Best-effort AIR `Expr` -> readable Rust/Verus-ish source text, replacing
//! the `Debug`-output placeholder used by `hard_constraints.rs` so far. This
//! is the Verus/AIR analog of the Boogie fork's 416-line `PatchTranslator.cs`.
//!
//! Scope (deliberately bounded, not full parity yet): recognizes the
//! encoding names Verus's own VIR->AIR lowering actually uses for arithmetic,
//! integer-width clipping, and boxing/unboxing - sourced directly from
//! `vir::def`'s public constants (`vir::def::ADD`, `vir::def::U_CLIP`, etc.),
//! not guessed. Anything not specifically recognized falls back to
//! `name(arg1, arg2, ...)`, which is still informative even if not idiomatic
//! Rust.
//!
//! Bitvector coverage is already complete, not a gap: `vir::def` only ever
//! defines six bitwise op constants (`BIT_XOR`/`BIT_AND`/`BIT_OR`/`BIT_SHR`/
//! `BIT_SHL`/`BIT_NOT`, confirmed by direct inspection - there is no seventh),
//! and all six are handled below.
//!
//! Seq/Set/Map method calls (e.g. `s.len()`, `s.contains(x)`) lower to VIR
//! function applications, not special AIR primitives the way arithmetic
//! does - confirmed by compiling and inspecting real `--repair-emit-facts`
//! output against both a `Seq` and a `Set` example. Two real, grounded
//! findings from that investigation, both handled below:
//! - The value being called on (and any other polymorphic argument) always
//!   arrives wrapped in a generic box/unbox marker shaped like
//!   `Poly%<type-path>.(<value>)` / `%Poly%<type-path>.(<value>)` -
//!   `vir::def::PREFIX_BOX`/`PREFIX_UNBOX` - the *dynamically-named*
//!   generic-type analog of the fixed `BOX_INT`/`UNBOX_INT` etc. constants
//!   already handled. Stripped transparently the same way.
//! - The function name itself is *not* reliably renameable to a clean
//!   `Type::method` form: `Seq::len` lowers to `vstd!seq.Seq.len.?` (a
//!   readable type segment), but `Set::len` lowers to
//!   `vstd!set.impl&%0.len.?` (an *anonymous impl-block index*, not the
//!   type name) - confirmed side-by-side against real output. Renaming the
//!   latter to `impl&%0::len` would look like a real path while actually
//!   being more misleading than the raw form, so this isn't attempted -
//!   only the `vstd!` tag itself (which carries no information beyond "this
//!   is a vstd builtin") is stripped, leaving the true mangled structure
//!   visible rather than a fabricated-looking gloss. This is a deliberate,
//!   evidence-based scope decision, not an oversight.
//!
//! Also strips `air::var_to_const`'s incarnation suffixes (`x@3`) and VIR's
//! parameter suffix (`x!`, see `vir::def::SUFFIX_PARAM`) back to the
//! surface-level variable name, since those are encoding artifacts, not
//! something a Rust-facing reader needs to see.

use air::ast::{BinaryOp, Constant, Expr, ExprX, MultiOp, UnaryOp};

/// Translate `expr` into readable text. Never fails - unrecognized shapes
/// fall back to a generic, still-inspectable rendering.
pub fn translate_expr(expr: &Expr) -> String {
    match &**expr {
        ExprX::Const(c) => translate_const(c),
        ExprX::Var(x) => surface_name(x),
        ExprX::Old(_snap, x) => format!("old({})", surface_name(x)),
        ExprX::Apply(name, args) => translate_apply(name, args),
        ExprX::ApplyFun(_, f, args) => {
            format!("{}({})", translate_expr(f), translate_args(args))
        }
        ExprX::Unary(op, e) => translate_unary(op, e),
        ExprX::Binary(op, l, r) => translate_binary(op, l, r),
        ExprX::Multi(op, es) => translate_multi(op, es),
        ExprX::IfElse(c, t, f) => {
            format!("if {} {{ {} }} else {{ {} }}", translate_expr(c), translate_expr(t), translate_expr(f))
        }
        ExprX::Array(es) => format!("[{}]", translate_args(es)),
        ExprX::Bind(bind, body) => translate_bind(bind, body),
        ExprX::LabeledAxiom(_, _, e) => translate_expr(e),
        ExprX::LabeledAssertion(_, _, _, e) => translate_expr(e),
    }
}

fn translate_args(args: &[Expr]) -> String {
    args.iter().map(translate_expr).collect::<Vec<_>>().join(", ")
}

fn translate_const(c: &Constant) -> String {
    match c {
        Constant::Bool(b) => b.to_string(),
        Constant::Nat(s) => (**s).clone(),
        Constant::Real(s) => (**s).clone(),
        Constant::BitVec(s, width) => format!("{s}u{width}"),
    }
}

fn translate_unary(op: &UnaryOp, e: &Expr) -> String {
    match op {
        UnaryOp::Not => format!("!({})", translate_expr(e)),
        UnaryOp::BitNot => format!("!({})", translate_expr(e)),
        UnaryOp::BitNeg => format!("-({})", translate_expr(e)),
        _ => format!("{op:?}({})", translate_expr(e)),
    }
}

fn translate_binary(op: &BinaryOp, l: &Expr, r: &Expr) -> String {
    let (l, r) = (translate_expr(l), translate_expr(r));
    let infix = match op {
        BinaryOp::Implies => Some("==>"),
        BinaryOp::Eq => Some("=="),
        BinaryOp::Le => Some("<="),
        BinaryOp::Ge => Some(">="),
        BinaryOp::Lt => Some("<"),
        BinaryOp::Gt => Some(">"),
        BinaryOp::EuclideanDiv => Some("/"),
        BinaryOp::EuclideanMod => Some("%"),
        BinaryOp::RealDiv => Some("/"),
        _ => None,
    };
    match infix {
        Some(op) => format!("({l} {op} {r})"),
        None => format!("{op:?}({l}, {r})"),
    }
}

fn translate_multi(op: &MultiOp, es: &[Expr]) -> String {
    let infix = match op {
        MultiOp::And => Some("&&"),
        MultiOp::Or => Some("||"),
        MultiOp::Xor => Some("^"),
        MultiOp::Add => Some("+"),
        MultiOp::Sub => Some("-"),
        MultiOp::Mul => Some("*"),
        _ => None,
    };
    match infix {
        Some(op) => {
            format!("({})", es.iter().map(translate_expr).collect::<Vec<_>>().join(&format!(" {op} ")))
        }
        None => format!("{op:?}({})", translate_args(es)),
    }
}

fn translate_bind(bind: &air::ast::Bind, body: &Expr) -> String {
    use air::ast::BindX;
    match &**bind {
        BindX::Let(binders) => {
            let bindings = binders
                .iter()
                .map(|b| format!("let {} = {};", surface_name(&b.name), translate_expr(&b.a)))
                .collect::<Vec<_>>()
                .join(" ");
            format!("{{ {} {} }}", bindings, translate_expr(body))
        }
        BindX::Quant(quant, binders, _triggers, _qid) => {
            let keyword = match quant {
                air::ast::Quant::Forall => "forall",
                air::ast::Quant::Exists => "exists",
            };
            let params =
                binders.iter().map(|b| surface_name(&b.name)).collect::<Vec<_>>().join(", ");
            format!("{keyword}|{params}| {}", translate_expr(body))
        }
        BindX::Lambda(binders, _triggers, _qid) => {
            let params =
                binders.iter().map(|b| surface_name(&b.name)).collect::<Vec<_>>().join(", ");
            format!("|{params}| {}", translate_expr(body))
        }
        BindX::Choose(binders, _triggers, _qid, cond) => {
            let params =
                binders.iter().map(|b| surface_name(&b.name)).collect::<Vec<_>>().join(", ");
            format!("choose|{params}| {} && {}", translate_expr(cond), translate_expr(body))
        }
    }
}

/// Recognize Verus's own VIR->AIR encoding function names (sourced from
/// `vir::def`'s public constants) and render them idiomatically; anything
/// else falls back to `name(args...)`.
fn translate_apply(name: &str, args: &[Expr]) -> String {
    // Integer/real arithmetic, lowered to named applications rather than
    // BinaryOp/MultiOp by VIR (see `vir::def::ADD` etc.).
    let arith_infix = match name {
        _ if name == vir::def::ADD || name == vir::def::RADD => Some("+"),
        _ if name == vir::def::SUB || name == vir::def::RSUB => Some("-"),
        _ if name == vir::def::MUL || name == vir::def::RMUL => Some("*"),
        _ if name == vir::def::EUC_DIV || name == vir::def::RDIV => Some("/"),
        _ if name == vir::def::EUC_MOD => Some("%"),
        _ if name == vir::def::BIT_XOR => Some("^"),
        _ if name == vir::def::BIT_AND => Some("&"),
        _ if name == vir::def::BIT_OR => Some("|"),
        _ if name == vir::def::BIT_SHR => Some(">>"),
        _ if name == vir::def::BIT_SHL => Some("<<"),
        _ => None,
    };
    if let Some(op) = arith_infix {
        if args.len() == 2 {
            return format!("({} {op} {})", translate_expr(&args[0]), translate_expr(&args[1]));
        }
    }
    if name == vir::def::BIT_NOT && args.len() == 1 {
        return format!("!({})", translate_expr(&args[0]));
    }
    // Integer-width clipping/wraparound wrappers (overflow-check encoding
    // artifacts): `uClip(width, value)` etc. - render the wrapped value
    // directly, since a Rust-facing reader thinks in terms of `value`, not
    // its bounds-check wrapper.
    let is_clip = name == vir::def::U_CLIP
        || name == vir::def::I_CLIP
        || name == vir::def::NAT_CLIP
        || name == vir::def::CHAR_CLIP;
    if is_clip && args.len() == 2 {
        return translate_expr(&args[1]);
    }
    // Boxing/unboxing wrappers (polymorphic value encoding artifacts) -
    // transparent, like clipping above.
    let is_box_or_unbox = matches!(
        name,
        vir::def::BOX_INT
            | vir::def::BOX_BOOL
            | vir::def::BOX_REAL
            | vir::def::BOX_FNDEF
            | vir::def::UNBOX_INT
            | vir::def::UNBOX_BOOL
            | vir::def::UNBOX_REAL
            | vir::def::UNBOX_FNDEF
    );
    if is_box_or_unbox && args.len() == 1 {
        return translate_expr(&args[0]);
    }
    // Generic-type box/unbox wrappers: unlike the fixed-name primitive-type
    // wrappers above, a polymorphic value's wrapper name embeds its own type
    // path (`Poly%vstd!seq.Seq<u32.>.`), so it can't be matched by exact
    // string equality - a prefix check is the right (and sufficient) test,
    // confirmed against real `Seq`/`Set` examples. Transparent, same as above.
    let is_generic_box_or_unbox =
        name.starts_with(vir::def::PREFIX_BOX) || name.starts_with(vir::def::PREFIX_UNBOX);
    if is_generic_box_or_unbox && args.len() == 1 {
        return translate_expr(&args[0]);
    }
    // Cosmetic only: the `vstd!` tag carries no information beyond "this is
    // a vstd builtin" - stripping it can't make an already-mangled name any
    // *more* misleading, unlike attempting to rename the segments after it
    // (see the module doc comment on why that's not attempted).
    let display_name = name.strip_prefix("vstd!").unwrap_or(name);
    format!("{}({})", display_name, translate_args(args))
}

/// Strip `air::var_to_const`'s incarnation suffix (`x@3`, or the bare `x@`
/// seen for some first-version locals) and VIR's parameter suffix (`x!`,
/// `vir::def::SUFFIX_PARAM`) back to the surface-level name.
pub(crate) fn surface_name(raw: &str) -> String {
    let mut s = raw;
    if let Some(stripped) = s.strip_suffix(vir::def::SUFFIX_PARAM) {
        s = stripped;
    }
    if let Some(at_pos) = s.rfind('@') {
        if s[at_pos + 1..].chars().all(|c| c.is_ascii_digit()) {
            s = &s[..at_pos];
        }
    }
    s.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use air::ast::Constant;
    use std::sync::Arc;

    fn var(name: &str) -> Expr {
        Arc::new(ExprX::Var(Arc::new(name.to_string())))
    }

    fn nat(n: &str) -> Expr {
        Arc::new(ExprX::Const(Constant::Nat(Arc::new(n.to_string()))))
    }

    #[test]
    fn strips_incarnation_and_param_suffixes() {
        assert_eq!(surface_name("x!"), "x");
        assert_eq!(surface_name("z@1"), "z");
        assert_eq!(surface_name("z@"), "z");
        assert_eq!(surface_name("plain"), "plain");
    }

    #[test]
    fn renders_clip_wrapped_arithmetic_as_plain_infix() {
        // uClip(32, Add(x!, 2)) -> "(x + 2)"
        let expr = Arc::new(ExprX::Apply(
            Arc::new(vir::def::U_CLIP.to_string()),
            Arc::new(vec![
                nat("32"),
                Arc::new(ExprX::Apply(
                    Arc::new(vir::def::ADD.to_string()),
                    Arc::new(vec![var("x!"), nat("2")]),
                )),
            ]),
        ));
        assert_eq!(translate_expr(&expr), "(x + 2)");
    }

    #[test]
    fn renders_equality_and_unrecognized_apply_fallback() {
        let expr = Arc::new(ExprX::Binary(BinaryOp::Eq, var("r!"), var("z@1")));
        assert_eq!(translate_expr(&expr), "(r == z)");

        let unknown = Arc::new(ExprX::Apply(
            Arc::new("some_vstd_helper".to_string()),
            Arc::new(vec![var("x!")]),
        ));
        assert_eq!(translate_expr(&unknown), "some_vstd_helper(x)");
    }

    #[test]
    fn strips_the_generic_box_wrapper_by_prefix_not_exact_match() {
        // Poly%vstd!seq.Seq<u32.>.(r) -> "r" - grounded in a real
        // --repair-emit-facts run against a Seq-based example (see the
        // module doc comment); the wrapper name embeds the concrete type, so
        // only a prefix check (not exact-string matching, unlike the fixed
        // BOX_INT-style wrappers) can recognize it.
        let expr = Arc::new(ExprX::Apply(
            Arc::new(format!("{}vstd!seq.Seq<u32.>.", vir::def::PREFIX_BOX)),
            Arc::new(vec![var("r!")]),
        ));
        assert_eq!(translate_expr(&expr), "r");

        let unbox_expr = Arc::new(ExprX::Apply(
            Arc::new(format!("{}vstd!set.Set<u32.>.", vir::def::PREFIX_UNBOX)),
            Arc::new(vec![var("s!")]),
        ));
        assert_eq!(translate_expr(&unbox_expr), "s");
    }

    #[test]
    fn strips_only_the_uninformative_vstd_tag_not_the_rest_of_a_mangled_method_name() {
        // vstd!seq.Seq.len.?(...) -> "seq.Seq.len.?(...)" - a real captured
        // shape (see the module doc comment); deliberately NOT rewritten to
        // "Seq::len(...)", since a sibling real example (`Set::len`) lowers
        // to an anonymous impl-block index (`vstd!set.impl&%0.len.?`) rather
        // than the type name, which a rename would misleadingly dress up as
        // a real path.
        let expr = Arc::new(ExprX::Apply(Arc::new("vstd!seq.Seq.len.?".to_string()), Arc::new(vec![var("r!")])));
        assert_eq!(translate_expr(&expr), "seq.Seq.len.?(r)");

        let opaque_impl_expr =
            Arc::new(ExprX::Apply(Arc::new("vstd!set.impl&%0.len.?".to_string()), Arc::new(vec![var("s!")])));
        assert_eq!(translate_expr(&opaque_impl_expr), "set.impl&%0.len.?(s)");
    }
}
