//! Per-swap work counters.
//!
//! Off unless the `cu-counters` feature is on, in which case the swap path increments a
//! thread-local as it runs and [`compute_swap`](crate::compute_swap) reports the diff over
//! its own execution in [`SwapResult::counters`](crate::SwapResult).
//!
//! **These count the work that happened; they never re-derive it.** Every increment sits
//! inside the function doing the work, next to the branch that decides whether the work
//! runs, so a change to the math moves the counter with it. That is the whole point: a
//! caller modelling this swap's on-chain compute cost can read what the loop did instead
//! of replaying it from geometry, and a replay is a second implementation that drifts.
//!
//! Nothing here can move a quote. The counters are write-only from the math's point of
//! view and are never read back into a calculation, so `SwapResult`'s amounts are
//! identical with the feature on or off.
//!
//! ## Why a thread-local rather than a threaded `&mut`
//!
//! `tick_index_to_sqrt_price` and `sqrt_price_to_tick_index` are `pub` and `wasm_expose`d,
//! so their signatures are not ours to change; and a counting *copy* of either ladder
//! would be the second implementation this exists to avoid.
//!
//! ## Why the granularity is per call, not per inner iteration
//!
//! The bumps are one thread-local access per ladder call, with the per-branch counting
//! done in a local `u32` first. Bumping inside `mul_shift_96` would be ~19x the
//! thread-local traffic for the same number, on a path a caller's optimizer runs 20-40
//! times per hop.

/// Work performed by one `compute_swap` call.
///
/// All fields are counts of a specific operation, so a model can price each one against a
/// measured coefficient rather than fitting a single opaque "walk length".
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SwapCounters {
    // ---- walk shape -------------------------------------------------------------------
    /// Outer loop iterations: one per initialized tick the walk targets.
    pub tick_steps: u32,
    /// Inner loop iterations. Equals `tick_steps` on a static-fee pool; on an adaptive-fee
    /// pool it is the tick-group sub-step count, which is the dominant cost term.
    pub substeps: u32,
    /// `compute_swap_step` calls. Equal to `substeps` today; counted separately so a future
    /// early-out cannot silently decouple the two.
    pub swap_steps: u32,
    /// Liquidity crossings — steps that landed exactly on the next initialized tick.
    pub liquidity_crossings: u32,
    /// Sub-steps where the adaptive fee bounded nothing, so `advance_tick_group_after_skip`
    /// ran in place of `advance_tick_group`.
    pub adaptive_skips: u32,

    // ---- tick -> sqrt price ------------------------------------------------------------
    /// `tick_index_to_sqrt_price` calls that took the **positive** ladder.
    pub ladder_pos_calls: u32,
    /// `mul_shift_96` operations across those calls: one **U256** multiply and shift each.
    pub ladder_pos_ops: u32,
    /// `tick_index_to_sqrt_price` calls that took the **negative** ladder.
    pub ladder_neg_calls: u32,
    /// Multiply-and-shift operations across those calls. These are bare `u128` multiplies,
    /// a different and much cheaper primitive than the positive ladder's `mul_shift_96`, so
    /// the two op counts must not share a coefficient.
    pub ladder_neg_ops: u32,

    // ---- sqrt price -> tick ------------------------------------------------------------
    /// `sqrt_price_to_tick_index` calls.
    pub sqrt_to_tick_calls: u32,
    /// Iterations of its log2 approximation loop, summed over those calls.
    pub sqrt_to_tick_log2_iters: u32,
    /// Calls that could not decide between `tick_low` and `tick_high` and so ran a further
    /// `tick_index_to_sqrt_price` to break the tie. That nested call is *also* counted in
    /// the ladder fields above, as it should be — it is work that ran.
    pub sqrt_to_tick_refines: u32,

    // ---- U256 divisions ----------------------------------------------------------------
    //
    // `ethnum::U256` division is a software long division whose iteration count is a
    // function of the two operands' limb widths, not of the type. Every price and amount
    // step in a swap performs one, so on a long walk the *width* of the numbers being
    // divided is a real per-sub-step cost axis that a flat per-sub-step rate cannot carry.
    // Recorded as raw widths rather than as a fitted "work" figure so a caller can price
    // whichever combination its own measurements support.
    /// `numerator / denominator` (with its paired `%`) operations on U256 operands.
    pub u256_divs: u32,
    /// Of those, the ones the *on-chain* `U256Muldiv::div` returns from before doing any
    /// arithmetic: a zero dividend, or a dividend narrower than the divisor.
    pub u256_div_trivial: u32,
    /// Divisions the chain answers with a single native `u128` divide, because the dividend
    /// fits in two 64-bit limbs.
    pub u256_div_u128: u32,
    /// Divisions with a single-limb divisor, which the chain special-cases into a loop of
    /// one `u128` divide per dividend limb — and the summed trip count of those loops.
    pub u256_div_short_calls: u32,
    pub u256_div_short_iters: u32,
    /// Divisions that reach the general normalized long division, and the summed
    /// `num_dividend_words - num_divisor_words + 1` outer iterations they perform.
    pub u256_div_knuth_calls: u32,
    pub u256_div_knuth_iters: u32,
    pub u128_divs: u32,
}

impl SwapCounters {
    pub const ZERO: Self = Self {
        tick_steps: 0,
        substeps: 0,
        swap_steps: 0,
        liquidity_crossings: 0,
        adaptive_skips: 0,
        ladder_pos_calls: 0,
        ladder_pos_ops: 0,
        ladder_neg_calls: 0,
        ladder_neg_ops: 0,
        sqrt_to_tick_calls: 0,
        sqrt_to_tick_log2_iters: 0,
        sqrt_to_tick_refines: 0,
        u256_divs: 0,
        u256_div_trivial: 0,
        u256_div_u128: 0,
        u256_div_short_calls: 0,
        u256_div_short_iters: 0,
        u256_div_knuth_calls: 0,
        u256_div_knuth_iters: 0,
        u128_divs: 0,
    };

    /// Field-wise `self - base`, for turning two snapshots into the work done between them.
    ///
    /// Saturating rather than wrapping so a counter reset between the two snapshots reads
    /// as zero work instead of ~4 billion.
    pub fn since(&self, base: &Self) -> Self {
        Self {
            tick_steps: self.tick_steps.saturating_sub(base.tick_steps),
            substeps: self.substeps.saturating_sub(base.substeps),
            swap_steps: self.swap_steps.saturating_sub(base.swap_steps),
            liquidity_crossings: self
                .liquidity_crossings
                .saturating_sub(base.liquidity_crossings),
            adaptive_skips: self.adaptive_skips.saturating_sub(base.adaptive_skips),
            ladder_pos_calls: self.ladder_pos_calls.saturating_sub(base.ladder_pos_calls),
            ladder_pos_ops: self.ladder_pos_ops.saturating_sub(base.ladder_pos_ops),
            ladder_neg_calls: self.ladder_neg_calls.saturating_sub(base.ladder_neg_calls),
            ladder_neg_ops: self.ladder_neg_ops.saturating_sub(base.ladder_neg_ops),
            sqrt_to_tick_calls: self
                .sqrt_to_tick_calls
                .saturating_sub(base.sqrt_to_tick_calls),
            sqrt_to_tick_log2_iters: self
                .sqrt_to_tick_log2_iters
                .saturating_sub(base.sqrt_to_tick_log2_iters),
            sqrt_to_tick_refines: self
                .sqrt_to_tick_refines
                .saturating_sub(base.sqrt_to_tick_refines),
            u256_divs: self.u256_divs.saturating_sub(base.u256_divs),
            u256_div_trivial: self.u256_div_trivial.saturating_sub(base.u256_div_trivial),
            u256_div_u128: self.u256_div_u128.saturating_sub(base.u256_div_u128),
            u256_div_short_calls: self.u256_div_short_calls.saturating_sub(base.u256_div_short_calls),
            u256_div_short_iters: self.u256_div_short_iters.saturating_sub(base.u256_div_short_iters),
            u256_div_knuth_calls: self.u256_div_knuth_calls.saturating_sub(base.u256_div_knuth_calls),
            u256_div_knuth_iters: self.u256_div_knuth_iters.saturating_sub(base.u256_div_knuth_iters),
            u128_divs: self.u128_divs.saturating_sub(base.u128_divs),
        }
    }
}

#[cfg(feature = "cu-counters")]
mod imp {
    use super::SwapCounters;
    use core::cell::RefCell;

    thread_local! {
        static COUNTERS: RefCell<SwapCounters> = const { RefCell::new(SwapCounters::ZERO) };
    }

    #[inline]
    pub fn bump(f: impl FnOnce(&mut SwapCounters)) {
        // A failed borrow would mean a counter bump reentered from inside another bump,
        // which cannot happen: the closures only touch integers.
        let _ = COUNTERS.try_with(|c| {
            if let Ok(mut c) = c.try_borrow_mut() {
                f(&mut c)
            }
        });
    }

    #[inline]
    pub fn snapshot() -> SwapCounters {
        COUNTERS
            .try_with(|c| c.try_borrow().map(|c| *c).unwrap_or(SwapCounters::ZERO))
            .unwrap_or(SwapCounters::ZERO)
    }
}

/// Record one U256 division, classified into the cost class the **on-chain** program's
/// `U256Muldiv::div` will take for these operand widths.
///
/// Deliberately not a width sum. That division is a four-way branch on the operands'
/// 64-bit limb counts — return early, one native `u128` divide, a per-limb loop, or the
/// general normalized long division — and those classes differ by more than an order of
/// magnitude, so a linear term over pooled widths cannot express them. The classification
/// mirrors `programs/whirlpool/src/math/u256_math.rs` case for case; `num_words()` there is
/// the index of the highest non-zero limb plus one, i.e. exactly `bits.div_ceil(64)`.
///
/// The SDK divides with `ethnum` and the chain with its own `U256Muldiv`, which is why this
/// takes widths rather than counting inside either: the two implementations differ, but they
/// are handed the same operands, and it is the operands that select the chain's branch.
#[inline(always)]
#[allow(unused_variables)]
pub(crate) fn record_u256_div(num_bits: u32, den_bits: u32) {
    #[cfg(feature = "cu-counters")]
    {
        let nd = num_bits.div_ceil(64);
        let dv = den_bits.div_ceil(64);
        bump(|c| {
            c.u256_divs += 1;
            if nd == 0 || nd < dv {
                c.u256_div_trivial += 1;
            } else if nd < 3 {
                c.u256_div_u128 += 1;
            } else if dv == 1 {
                c.u256_div_short_calls += 1;
                c.u256_div_short_iters += nd;
            } else {
                c.u256_div_knuth_calls += 1;
                c.u256_div_knuth_iters += nd - dv + 1;
            }
        });
    }
}

/// Record work against the calling thread's counters. Compiles to nothing without the
/// `cu-counters` feature.
#[inline(always)]
#[allow(unused_variables)]
pub(crate) fn bump(f: impl FnOnce(&mut SwapCounters)) {
    #[cfg(feature = "cu-counters")]
    imp::bump(f);
}

/// The calling thread's counters as they stand. All-zero without the `cu-counters` feature.
#[inline(always)]
pub(crate) fn snapshot() -> SwapCounters {
    #[cfg(feature = "cu-counters")]
    {
        imp::snapshot()
    }
    #[cfg(not(feature = "cu-counters"))]
    {
        SwapCounters::ZERO
    }
}

#[cfg(all(test, feature = "cu-counters", not(feature = "wasm")))]
mod tests {
    use super::*;
    use crate::{sqrt_price_to_tick_index, tick_index_to_sqrt_price, MAX_TICK_INDEX, MIN_TICK_INDEX};

    fn measure(f: impl FnOnce()) -> SwapCounters {
        let before = snapshot();
        f();
        snapshot().since(&before)
    }

    /// The ladders branch on bits 2,4,8,... of `|tick|`, with bit 1 selecting the seed
    /// constant and costing no multiply. So the op count must be `popcount(|tick| >> 1)`.
    ///
    /// This is not how the counter is computed — it counts each taken branch — which is
    /// exactly what makes the test worth having: it pins the closed form a *caller's* cost
    /// model would otherwise have to assume, so a change to either ladder fails here
    /// instead of silently invalidating that assumption somewhere downstream.
    #[test]
    fn ladder_ops_equal_the_set_bits_above_the_seed_bit() {
        for tick in [
            0, 1, 2, 3, 7, 64, 100, 255, 1000, 4096, 65535, 131071, MAX_TICK_INDEX,
        ] {
            let c = measure(|| {
                tick_index_to_sqrt_price(tick);
            });
            assert_eq!(c.ladder_pos_calls, 1, "tick {tick}");
            assert_eq!(c.ladder_neg_calls, 0, "tick {tick}");
            assert_eq!(
                c.ladder_pos_ops,
                (tick.unsigned_abs() >> 1).count_ones(),
                "positive ladder ops for tick {tick}"
            );
        }

        for tick in [-1, -2, -3, -7, -64, -100, -255, -1000, -4096, MIN_TICK_INDEX] {
            let c = measure(|| {
                tick_index_to_sqrt_price(tick);
            });
            assert_eq!(c.ladder_neg_calls, 1, "tick {tick}");
            assert_eq!(c.ladder_pos_calls, 0, "tick {tick}");
            assert_eq!(
                c.ladder_neg_ops,
                (tick.unsigned_abs() >> 1).count_ones(),
                "negative ladder ops for tick {tick}"
            );
        }
    }

    /// Tick 0 takes the positive ladder and does no multiplying at all — the boundary a
    /// sign-keyed cost model has to get right.
    #[test]
    fn tick_zero_is_a_positive_ladder_call_with_no_ops() {
        let c = measure(|| {
            tick_index_to_sqrt_price(0);
        });
        assert_eq!((c.ladder_pos_calls, c.ladder_pos_ops), (1, 0));
        assert_eq!((c.ladder_neg_calls, c.ladder_neg_ops), (0, 0));
    }

    /// The reverse ladder's refine branch runs a nested `tick_index_to_sqrt_price`, and
    /// that nested call must show up in the ladder counters — it is work that ran.
    #[test]
    fn a_refine_also_counts_its_nested_ladder_call() {
        let mut refines = 0u32;
        for tick in (MIN_TICK_INDEX..MAX_TICK_INDEX).step_by(7919) {
            let sqrt_price = tick_index_to_sqrt_price(tick);
            let c = measure(|| {
                sqrt_price_to_tick_index(sqrt_price);
            });
            assert_eq!(c.sqrt_to_tick_calls, 1, "tick {tick}");
            assert!(c.sqrt_to_tick_log2_iters > 0, "tick {tick}");
            if c.sqrt_to_tick_refines == 1 {
                refines += 1;
                assert_eq!(
                    c.ladder_pos_calls + c.ladder_neg_calls,
                    1,
                    "a refine must run exactly one nested ladder call, tick {tick}"
                );
            } else {
                assert_eq!(c.sqrt_to_tick_refines, 0);
                assert_eq!(c.ladder_pos_calls + c.ladder_neg_calls, 0, "tick {tick}");
            }
        }
        assert!(refines > 0, "no refine branch was exercised; the sweep is not covering it");
    }

    /// Counters must never leak between measurements, or every reading after the first is
    /// wrong by everything that came before it.
    #[test]
    fn a_measurement_reports_only_its_own_work() {
        let _ = measure(|| {
            tick_index_to_sqrt_price(65535);
        });
        let c = measure(|| {});
        assert_eq!(c, SwapCounters::ZERO);
    }
}
