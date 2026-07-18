//! Fault-localization engine for the Verus proof-repair tool.
//!
//! This crate hosts the Verus/AIR analog of the Boogie fork's
//! stepwise WP/SP windowing, dependency tracking, hard-constraint
//! extraction, VC-to-source translation, and SSA-aware patch
//! alignment (see the project plan for details). It starts small
//! and grows incrementally.

pub mod align;
pub mod dependencies;
pub mod hard_constraints;
pub mod translate;
pub mod windows;
