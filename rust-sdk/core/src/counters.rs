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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    /// Log2 iterations whose fraction bit came out **clear** (`r >>= 63` rather than `>>= 64`):
    /// the variable shift takes `__lshrti3`'s under-64 path, 8 instructions dearer than the
    /// by-64 one, so a refinement costs 8 CU more per clear bit of the price's log2 fraction.
    pub sqrt_to_tick_log2_clear_bits: u32,
    /// Refinements whose price was narrower / wider than 64 bits, so the initial normalisation
    /// of `r` shifted by a non-zero amount: `__ashlti3` costs 18 against 8 for a zero shift
    /// (`msb < 63`), and `__lshrti3` 12 (`msb > 63`).
    pub sqrt_to_tick_shift_left: u32,
    pub sqrt_to_tick_shift_right: u32,
    /// Of `sqrt_to_tick_refines`, the ones whose `tick_high` is negative, so the tie-break's
    /// nested ladder call takes the negative body (a flat ~1,000) rather than the positive
    /// one (~248 per set bit, those bits being in `ladder_pos_ops`).
    pub sqrt_to_tick_refines_neg: u32,
    /// Conversions of a price under 2^64: `u128::leading_zeros` has no SBF instruction and
    /// its compiled form takes a longer path when the high limb is zero.
    pub sqrt_to_tick_narrow: u32,
    /// `advance_tick_group_after_skip` divides the landing tick by `tick_group_size` twice —
    /// a `%` and a `floor_division`, both signed 64-bit, so both are shift-subtract loops whose
    /// cost is `base + 7 per quotient bit + 2 per set bit of the quotient`. Two calls per skip,
    /// summed here over both, on top of the six every crossing runs. See the bot's
    /// `ORCA_CUS_PER_TICK_DIV_BASE`.
    pub skip_tick_div_qbits: u32,
    pub skip_tick_div_setbits: u32,

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
    /// **The multiply work *inside* the division, in limb products.**
    ///
    /// Knuth D's inner loop multiplies the divisor by the trial quotient digit once per
    /// iteration, so the multiplies scale with the **divisor's** limb count as well as the
    /// iteration count — `iterations * dv` for the general arm, one per iteration for the
    /// short arm where the divisor is a single limb. `u256_mul_word_products` cannot see any
    /// of this: it is recorded at explicit `U256Muldiv::mul` call sites only.
    ///
    /// Separate from `u256_div_knuth_iters` because the two are *not* proportional — `dv`
    /// varies with the operands — and a flat per-iteration cost therefore cannot express a
    /// division whose divisor is wide. Measured downstream, one such division's multiplies are
    /// worth ~750 CU that a per-iteration term charges at zero.
    pub u256_div_mul_words: u32,
    /// Case-2 divisions that are a `U256Muldiv::div` **frame** on chain — the `delta_a` and
    /// `from_a` divisions, whose `mul_div` never reaches `div_loop`'s set-up when the dividend
    /// fits two limbs — as opposed to the b-side price solve, which this SDK divides as a U256
    /// but the chain as an inline `u128`. A cost model prices the frame's own work by this.
    pub u256_div_case2_frames: u32,
    /// Every `__udivti3` call the chain's `U256Muldiv::div` makes — the quotient steps of cases 3
    /// and 4 (`d0 / d1` in `div_loop`, `d1 / d2` in the single-limb loop) and case 2's one
    /// `u128 / u128` — as a histogram over the **branch path** compiler-builtins' `trifecta`
    /// division takes on those operands, indexed by [`udiv_path`]. Each path is a fixed
    /// instruction count on chain, which no width bucket expresses: the same 128-by-64 divide
    /// costs 188, 345, 431, 451, 540, 560, 636 or 765 CU depending on which of the algorithm's
    /// branches the operand *values* select. Counted by running the chain's own long division
    /// on the same operands, limb for limb, so the operands at each step are the chain's — see
    /// `chain_div_arms`.
    pub udiv_paths: [u32; UDIV_PATHS],
    /// Case-4 divisions whose divisor's top limb had leading zeros, so both operands were shifted
    /// left before the loop (Knuth D's normalisation) — a per-limb shift the other cases skip.
    pub u256_div_normalized: u32,
    /// Of those, the divisions whose caller keeps the remainder (a round-up), which the chain
    /// then shifts back right — a 21-instruction block the round-down callers never run.
    pub u256_div_normalized_remainder: u32,
    /// Iterations of `div_loop`'s `qhat` correction loop (`qhat` over-estimated by one or two).
    pub u256_div_qhat_corrections: u32,
    /// `u128` divisions in the fee helpers, a different and much cheaper primitive.
    pub u128_divs: u32,

    // ---- U256 multiplies ---------------------------------------------------------------
    //
    // The on-chain `U256Muldiv::mul` is a schoolbook product whose inner loop runs `m * n`
    // times for operands of `m` and `n` 64-bit limbs — so a 128-bit operand costs twice a
    // 64-bit one, and two of them four times. Every price and amount step performs two or
    // three of these on the pool's liquidity and sqrt prices, which is why two pools with
    // identical walks can have different per-sub-step costs.
    /// U256 multiplies performed.
    pub u256_muls: u32,
    /// Summed `m * n` over those multiplies — the inner-loop trip count, 1..=4 here since
    /// both operands are `u128`.
    pub u256_mul_word_products: u32,

    // ---- which token math ran ----------------------------------------------------------
    //
    // A sub-step calls a different mix of these depending on branches inside
    // `compute_swap_step` — whether the step reached its target, whether the initial fixed
    // delta overflowed — and the four are not equal work: `amount_delta_a` performs a U256
    // division where `amount_delta_b` is a shift. Counting the calls subsumes those branches
    // without a model of them.
    pub amount_delta_a_calls: u32,
    pub amount_delta_b_calls: u32,
    pub next_sqrt_from_a_calls: u32,
    pub next_sqrt_from_b_calls: u32,
}

impl Default for SwapCounters {
    fn default() -> Self {
        Self::ZERO
    }
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
        sqrt_to_tick_refines_neg: 0,
        sqrt_to_tick_narrow: 0,
        skip_tick_div_qbits: 0,
        skip_tick_div_setbits: 0,
        sqrt_to_tick_log2_clear_bits: 0,
        sqrt_to_tick_shift_left: 0,
        sqrt_to_tick_shift_right: 0,
        u256_divs: 0,
        u256_div_trivial: 0,
        u256_div_u128: 0,
        u256_div_short_calls: 0,
        u256_div_short_iters: 0,
        u256_div_knuth_calls: 0,
        u256_div_knuth_iters: 0,
        u256_div_mul_words: 0,
        u256_div_case2_frames: 0,
        udiv_paths: [0; UDIV_PATHS],
        u256_div_normalized: 0,
        u256_div_normalized_remainder: 0,
        u256_div_qhat_corrections: 0,
        u128_divs: 0,
        u256_muls: 0,
        u256_mul_word_products: 0,
        amount_delta_a_calls: 0,
        amount_delta_b_calls: 0,
        next_sqrt_from_a_calls: 0,
        next_sqrt_from_b_calls: 0,
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
            sqrt_to_tick_refines_neg: self
                .sqrt_to_tick_refines_neg
                .saturating_sub(base.sqrt_to_tick_refines_neg),
            sqrt_to_tick_narrow: self
                .sqrt_to_tick_narrow
                .saturating_sub(base.sqrt_to_tick_narrow),
            skip_tick_div_qbits: self
                .skip_tick_div_qbits
                .saturating_sub(base.skip_tick_div_qbits),
            skip_tick_div_setbits: self
                .skip_tick_div_setbits
                .saturating_sub(base.skip_tick_div_setbits),
            sqrt_to_tick_log2_clear_bits: self
                .sqrt_to_tick_log2_clear_bits
                .saturating_sub(base.sqrt_to_tick_log2_clear_bits),
            sqrt_to_tick_shift_left: self
                .sqrt_to_tick_shift_left
                .saturating_sub(base.sqrt_to_tick_shift_left),
            sqrt_to_tick_shift_right: self
                .sqrt_to_tick_shift_right
                .saturating_sub(base.sqrt_to_tick_shift_right),
            u256_divs: self.u256_divs.saturating_sub(base.u256_divs),
            u256_div_trivial: self.u256_div_trivial.saturating_sub(base.u256_div_trivial),
            u256_div_u128: self.u256_div_u128.saturating_sub(base.u256_div_u128),
            u256_div_short_calls: self.u256_div_short_calls.saturating_sub(base.u256_div_short_calls),
            u256_div_short_iters: self.u256_div_short_iters.saturating_sub(base.u256_div_short_iters),
            u256_div_knuth_calls: self.u256_div_knuth_calls.saturating_sub(base.u256_div_knuth_calls),
            u256_div_knuth_iters: self.u256_div_knuth_iters.saturating_sub(base.u256_div_knuth_iters),
            u256_div_mul_words: self.u256_div_mul_words.saturating_sub(base.u256_div_mul_words),
            u256_div_case2_frames: self
                .u256_div_case2_frames
                .saturating_sub(base.u256_div_case2_frames),
            udiv_paths: {
                let mut p = [0u32; UDIV_PATHS];
                for (i, slot) in p.iter_mut().enumerate() {
                    *slot = self.udiv_paths[i].saturating_sub(base.udiv_paths[i]);
                }
                p
            },
            u256_div_normalized: self.u256_div_normalized.saturating_sub(base.u256_div_normalized),
            u256_div_normalized_remainder: self
                .u256_div_normalized_remainder
                .saturating_sub(base.u256_div_normalized_remainder),
            u256_div_qhat_corrections: self
                .u256_div_qhat_corrections
                .saturating_sub(base.u256_div_qhat_corrections),
            u128_divs: self.u128_divs.saturating_sub(base.u128_divs),
            u256_muls: self.u256_muls.saturating_sub(base.u256_muls),
            u256_mul_word_products: self
                .u256_mul_word_products
                .saturating_sub(base.u256_mul_word_products),
            amount_delta_a_calls: self
                .amount_delta_a_calls
                .saturating_sub(base.amount_delta_a_calls),
            amount_delta_b_calls: self
                .amount_delta_b_calls
                .saturating_sub(base.amount_delta_b_calls),
            next_sqrt_from_a_calls: self
                .next_sqrt_from_a_calls
                .saturating_sub(base.next_sqrt_from_a_calls),
            next_sqrt_from_b_calls: self
                .next_sqrt_from_b_calls
                .saturating_sub(base.next_sqrt_from_b_calls),
        }
    }
}

#[cfg(feature = "cu-counters")]
mod imp {
    use super::SwapCounters;
    use core::cell::{Cell, RefCell};

    thread_local! {
        static COUNTERS: RefCell<SwapCounters> = const { RefCell::new(SwapCounters::ZERO) };
        /// Whether this thread is counting. **Off by default** — see [`super::set_enabled`].
        static ENABLED: Cell<bool> = const { Cell::new(false) };
    }

    #[inline]
    pub fn set_enabled(on: bool) -> bool {
        ENABLED.try_with(|e| e.replace(on)).unwrap_or(false)
    }

    #[inline]
    pub fn enabled() -> bool {
        ENABLED.try_with(|e| e.get()).unwrap_or(false)
    }

    #[inline]
    pub fn bump(f: impl FnOnce(&mut SwapCounters)) {
        // One thread-local read and a predictable branch is the whole cost when a caller
        // has not asked to be counted; the `RefCell` behind it is never touched.
        if !enabled() {
            return;
        }
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
        // Nothing has been counted on this thread since counting was switched off, so a
        // `since()` over two of these is all-zero either way — and `compute_swap` takes two
        // per call, which with counting off was two `RefCell` borrows and two 264-byte copies
        // on every quote for a difference that is zero by construction.
        if !enabled() {
            return SwapCounters::ZERO;
        }
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
pub(crate) fn record_u256_div(
    numerator: ethnum::U256,
    denominator: ethnum::U256,
    frame: bool,
    remainder: bool,
) {
    #[cfg(feature = "cu-counters")]
    {
        // Classified inside the closure, so a caller that is not counting pays one
        // thread-local read and nothing else -- not the width arithmetic as well.
        bump(|c| {
            let num_bits = 256 - numerator.leading_zeros();
            let den_bits = 256 - denominator.leading_zeros();
            let nd = num_bits.div_ceil(64);
            let dv = den_bits.div_ceil(64);
            c.u256_divs += 1;
            if nd == 0 || nd < dv {
                c.u256_div_trivial += 1;
            } else if nd < 3 {
                c.u256_div_u128 += 1;
                if frame {
                    c.u256_div_case2_frames += 1;
                }
                // Case 2: one native `u128 / u128`, which on SBF is one `__udivti3` call.
                let (num, den) = (limbs(numerator), limbs(denominator));
                let num = num[0] as u128 | (num[1] as u128) << 64;
                let den = den[0] as u128 | (den[1] as u128) << 64;
                c.udiv_paths[udiv_path(num, den)] += 1;
            } else if dv == 1 {
                c.u256_div_short_calls += 1;
                c.u256_div_short_iters += nd;
                // One limb product per iteration: the divisor is a single limb here.
                c.u256_div_mul_words += nd;
            } else {
                c.u256_div_knuth_calls += 1;
                c.u256_div_knuth_iters += nd - dv + 1;
                // Knuth D multiplies the whole divisor by the trial digit once per iteration.
                c.u256_div_mul_words += (nd - dv + 1) * dv;
            }
            if nd >= 3 {
                let (_, arms) = chain_div_arms(limbs(numerator), limbs(denominator));
                for (slot, n) in c.udiv_paths.iter_mut().zip(arms.paths.iter()) {
                    *slot += n;
                }
                c.u256_div_normalized += arms.normalized;
                if remainder {
                    c.u256_div_normalized_remainder += arms.normalized;
                }
                c.u256_div_qhat_corrections += arms.corrections;
            }
        });
    }
}

/// The four little-endian 64-bit limbs of a U256, as `U256Muldiv::items` holds them.
#[cfg(feature = "cu-counters")]
#[inline(always)]
fn limbs(x: ethnum::U256) -> [u64; 4] {
    let lo = *x.low();
    let hi = *x.high();
    [lo as u64, (lo >> 64) as u64, hi as u64, (hi >> 64) as u64]
}

/// The number of distinct `__udivti3` branch paths [`udiv_path`] distinguishes.
pub const UDIV_PATHS: usize = 28;

/// Which branch path Rust's `compiler_builtins` `u128` division takes on these operands.
///
/// The chain's `__udivti3` is `compiler_builtins::int::specialized_div_rem::u128_div_rem`, the
/// **trifecta** algorithm (`impl_trifecta!` with `n = 64`, `n_h = 32`): a quotient-0-or-1 arm, a
/// native 64-bit half division, a three-half-division short arm for divisors under 2^32, a
/// "two possibility" arm when the operands' leading bits are within 32 of each other, and
/// otherwise an under-subtracting long division loop that runs one or more steps and then
/// leaves through one of the first four arms. Traced at every call site in the program, each
/// path is a constant instruction count to within 3 CU, and the paths differ from each other
/// by up to 580 — so this, not the operand widths, is the quantity a cost model wants.
///
/// The index layout is fixed and the cost table lives with the consumer:
///
/// | index | path |
/// |---|---|
/// | 0 / 1 | quotient 0 / quotient 1 (`div_lz <= duo_lz`) |
/// | 2 | half division (`duo < 2^64`) |
/// | 3 | short division (`div < 2^32`) |
/// | 4 / 5 | two-possibility, uncorrected / corrected (`quo - 1`) |
/// | 6 / 7 | the same with `duo` a full 128 bits, where the sig-bit shifts are by 64 |
/// | 8 + 10·(steps − 1) + 2·exit + zero_shl | long division: `steps` ∈ 1..=2 (a step clears at least 31 bits, so a 128-bit dividend never needs a third), `exit` 0 = quotient 0, 1 = quotient 1, 2 = half, 3 = two-possibility, 4 = corrected two-possibility, and `zero_shl` 1 when any step's `extra_shl` was 0 |
///
/// A faithful port, kept in the algorithm's own variable names so it can be read against
/// `trifecta.rs`; only the quotient bookkeeping is dropped, because the path is the answer.
#[cfg(feature = "cu-counters")]
pub fn udiv_path(duo: u128, div: u128) -> usize {
    const N: u32 = 64;
    const N_H: u32 = 32;
    #[inline(always)]
    fn twopos_corrected(duo: u128, div: u128, duo_lz: u32) -> usize {
        let shift = N - duo_lz;
        let duo_sig_n = (duo >> shift) as u64;
        let div_sig_n = (div >> shift) as u64;
        let quo = duo_sig_n / div_sig_n;
        let div_lo = div as u64;
        let div_hi = (div >> N) as u64;
        let tmp_a = (quo as u128).wrapping_mul(div_lo as u128);
        let (tmp_lo, carry) = (tmp_a as u64, (tmp_a >> N) as u64);
        let tmp_b = (quo as u128)
            .wrapping_mul(div_hi as u128)
            .wrapping_add(carry as u128);
        let (tmp_hi, overflow) = (tmp_b as u64, (tmp_b >> N) as u64);
        let tmp = (tmp_lo as u128) | ((tmp_hi as u128) << N);
        usize::from(overflow != 0 || duo < tmp)
    }
    #[inline(always)]
    fn long(steps: u32, exit: usize, zero_shl: bool) -> usize {
        8 + 10 * (steps.clamp(1, 2) as usize - 1) + 2 * exit + usize::from(zero_shl)
    }
    if div == 0 {
        return 0;
    }
    let div_lz = div.leading_zeros();
    let mut duo_lz = duo.leading_zeros();
    if div_lz <= duo_lz {
        return usize::from(duo >= div);
    }
    if duo_lz >= N {
        return 2;
    }
    if div_lz >= N + N_H {
        return 3;
    }
    let lz_diff = div_lz - duo_lz;
    if lz_diff < N_H {
        return 4 + twopos_corrected(duo, div, duo_lz) + if duo_lz == 0 { 2 } else { 0 };
    }
    let mut duo = duo;
    let div_extra = (N + N_H) - div_lz;
    let div_sig_n_h = (div >> div_extra) as u32;
    let div_sig_n_h_add1 = (div_sig_n_h as u64) + 1;
    let mut steps = 0u32;
    let mut zero_shl = false;
    loop {
        let duo_extra = N - duo_lz;
        let duo_sig_n = (duo >> duo_extra) as u64;
        if div_extra <= duo_extra {
            let quo_part = (duo_sig_n / div_sig_n_h_add1) as u128;
            let extra_shl = duo_extra - div_extra;
            zero_shl |= extra_shl == 0;
            duo = duo.wrapping_sub(div.wrapping_mul(quo_part) << extra_shl);
            steps += 1;
        } else {
            return long(steps, 3 + twopos_corrected(duo, div, duo_lz), zero_shl);
        }
        duo_lz = duo.leading_zeros();
        if div_lz <= duo_lz {
            return long(steps, usize::from(div <= duo), zero_shl);
        }
        if N <= duo_lz {
            return long(steps, 2, zero_shl);
        }
    }
}

/// What `chain_div_arms` counts: one [`udiv_path`] entry per `__udivti3` call the division
/// makes, plus the two data-dependent extras of the Knuth loop.
#[cfg(feature = "cu-counters")]
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct DivArms {
    pub paths: [u32; UDIV_PATHS],
    pub normalized: u32,
    pub corrections: u32,
}

#[cfg(feature = "cu-counters")]
impl Default for DivArms {
    fn default() -> Self {
        Self { paths: [0; UDIV_PATHS], normalized: 0, corrections: 0 }
    }
}

/// The on-chain `U256Muldiv::div` (`programs/whirlpool/src/math/u256_math.rs`), copied limb for
/// limb and instrumented — returning the quotient it computes as well, so a test can hold the
/// port against `ethnum`.
///
/// Every quotient step the chain performs is one `u128 / u64` (case 4: `d0 / d1` with `d1` a
/// normalised limb; case 3: `d1 / d2` with `d2` the raw single limb), and which `__udivti3`
/// branch path that takes is decided by the step's operand *values* — the high half is the
/// remainder carried from the previous step (or the normalisation overflow on the first), which
/// no width of the original operands predicts. It has to be *computed*, and the only faithful
/// way is the chain's own arithmetic on the chain's own operands. This is that, including the
/// `qhat` correction loop and the add-back, because the remainder they leave behind is the next
/// step's high half. Cases 0–2 make no such step and return early exactly as the chain does.
#[cfg(feature = "cu-counters")]
pub(crate) fn chain_div_arms(dividend: [u64; 4], divisor: [u64; 4]) -> ([u64; 4], DivArms) {
    const NUM_WORDS: usize = 4;
    const U64_RESOLUTION: u32 = 64;
    fn num_words(w: &[u64; 4]) -> usize {
        for i in (0..4).rev() {
            if w[i] != 0 {
                return i + 1;
            }
        }
        0
    }
    fn shift_left(w: [u64; 4], mut s: u32) -> [u64; 4] {
        let mut r = w;
        while s >= U64_RESOLUTION {
            r = [0, r[0], r[1], r[2]];
            s -= U64_RESOLUTION;
        }
        if s > 0 {
            for i in (1..NUM_WORDS).rev() {
                r[i] = (r[i] << s) | (r[i - 1] >> (U64_RESOLUTION - s));
            }
            r[0] <<= s;
        }
        r
    }
    let mut arms = DivArms::default();
    let mut quotient = [0u64; 4];
    let mut dividend = dividend;
    let mut divisor = divisor;
    let nd = num_words(&dividend);
    let nv = num_words(&divisor);
    if nv == 0 || nd == 0 || nd < nv || nd < 3 {
        return (quotient, arms);
    }
    // Case 3: single-limb divisor, one `u128 / u64` per dividend limb with the remainder carried.
    if nv == 1 {
        let mut k: u128 = 0;
        let d2 = divisor[0] as u128;
        for j in (0..nd).rev() {
            let d1 = ((k as u64 as u128) << 64) | dividend[j] as u128;
            arms.paths[udiv_path(d1, d2)] += 1;
            let q = d1 / d2;
            k = d1 - d2 * q;
            quotient[j] = q as u64;
        }
        return (quotient, arms);
    }
    // Case 4: Knuth D, normalised.
    let s = divisor[nv - 1].leading_zeros();
    let b = dividend[nd - 1].leading_zeros();
    if s > 0 {
        arms.normalized += 1;
    }
    let mut carry: u64 = 0;
    if nd == NUM_WORDS && b < s {
        carry = dividend[nd - 1] >> (U64_RESOLUTION - s);
    }
    dividend = shift_left(dividend, s);
    divisor = shift_left(divisor, s);
    for j in (0..nd - nv + 1).rev() {
        let use_carry = (j + nv) == NUM_WORDS;
        let div_hi = if use_carry { carry } else { dividend[j + nv] };
        let d0: u128 = ((div_hi as u128) << 64) | dividend[j + nv - 1] as u128;
        let d1: u128 = divisor[nv - 1] as u128;
        arms.paths[udiv_path(d0, d1)] += 1;
        let mut qhat = d0 / d1;
        let mut rhat = d0 - d1 * qhat;
        let d0_2 = dividend[j + nv - 2];
        let d1_2 = divisor[nv - 2] as u128;
        let mut cmp1: u128 = ((rhat as u64 as u128) << 64) | d0_2 as u128;
        let mut cmp2: u128 = qhat.wrapping_mul(d1_2);
        while (qhat >> 64) != 0 || cmp2 > cmp1 {
            arms.corrections += 1;
            qhat -= 1;
            rhat += d1;
            if (rhat >> 64) != 0 {
                break;
            }
            cmp1 = ((rhat as u64 as u128) << 64) | (cmp1 as u64 as u128);
            cmp2 -= d1_2;
        }
        let mut k: u128 = 0;
        let mut t: u128;
        for i in 0..nv {
            let p = qhat * divisor[i] as u128;
            t = (dividend[j + i] as u128).wrapping_sub(k).wrapping_sub(p as u64 as u128);
            dividend[j + i] = t as u64;
            k = ((p >> 64) as u64).wrapping_sub((t >> 64) as u64) as u128;
        }
        let d_head: u128 = if use_carry { carry as u128 } else { dividend[j + nv] as u128 };
        t = d_head.wrapping_sub(k);
        if use_carry {
            carry = t as u64;
        } else {
            dividend[j + nv] = t as u64;
        }
        if k > d_head {
            // The add-back. The chain reads the dividend limb above the window here even when the
            // carry space is in use (an index past the array on a four-limb dividend); mirror the
            // value it would read, which is the carry.
            qhat -= 1;
            k = 0;
            for i in 0..nv {
                t = (dividend[j + i] as u128).wrapping_add(divisor[i] as u128).wrapping_add(k);
                dividend[j + i] = t as u64;
                k = t >> 64;
            }
            let head = if use_carry { carry as u128 } else { dividend[j + nv] as u128 };
            let new_carry = head.wrapping_add(k) as u64;
            if use_carry {
                carry = new_carry;
            } else {
                dividend[j + nv] = new_carry;
            }
        }
        quotient[j] = qhat as u64;
    }
    (quotient, arms)
}

/// One `u128 / u128` the chain performs, classified by the branch path [`udiv_path`] says it
/// takes — with the operands **as the chain computes them**, which is not always how this SDK
/// does: the step fee on a max-swap step is `amount_in * fee_rate / (1e6 - fee_rate)` on chain and
/// `amount_in * 1e6 / (1e6 - fee_rate)` here, and the two dividends can land on different paths.
#[inline(always)]
#[allow(unused_variables)]
pub(crate) fn record_udiv(numerator: u128, denominator: u128) {
    #[cfg(feature = "cu-counters")]
    {
        if denominator != 0 {
            bump(|c| c.udiv_paths[udiv_path(numerator, denominator)] += 1);
        }
    }
}

/// The two divisions the chain's `swap_manager::calculate_fees` performs after every step and
/// this SDK never needs — the protocol fee `fee * protocol_fee_rate / 10_000` (when the rate is
/// set) and the fee-growth update `(fee - protocol_fee) << 64 / liquidity` (when there is
/// liquidity), the latter with the pre-crossing liquidity, as the chain does it. Computed here
/// only to be classified, and only while counting.
#[inline(always)]
#[allow(unused_variables)]
pub(crate) fn record_step_fees(fee_amount: u64, protocol_fee_rate: u16, liquidity: u128) {
    #[cfg(feature = "cu-counters")]
    {
        bump(|c| {
            let mut global_fee = fee_amount as u128;
            if protocol_fee_rate > 0 {
                let numerator = global_fee * protocol_fee_rate as u128;
                c.udiv_paths[udiv_path(numerator, 10_000)] += 1;
                global_fee -= numerator / 10_000;
            }
            if liquidity > 0 {
                c.udiv_paths[udiv_path(global_fee << 64, liquidity)] += 1;
            }
        });
    }
}

/// Record one U256 multiply with its operands' limb widths.
///
/// The chain's `U256Muldiv::mul` runs its inner loop `m * n` times for `m`- and `n`-limb
/// operands (`programs/whirlpool/src/math/u256_math.rs`), so the trip count — not the call
/// count — is the quantity a cost model wants. Both operands here are `u128`, so each is
/// one or two limbs and the product is 1, 2 or 4.
#[inline(always)]
#[allow(unused_variables)]
pub(crate) fn record_u256_mul(a: u128, b: u128) {
    #[cfg(feature = "cu-counters")]
    {
        bump(|c| {
            let m = (128 - a.leading_zeros()).div_ceil(64).max(1);
            let n = (128 - b.leading_zeros()).div_ceil(64).max(1);
            c.u256_muls += 1;
            c.u256_mul_word_products += m * n;
        });
    }
}

/// Turn counting on or off **for the calling thread**, returning the previous setting.
///
/// Off by default, and that default is the point. The bumps sit inside `compute_swap`, which
/// a caller's optimizer runs 20-40 times per hop while searching for a trade size — but a cost
/// model only needs the counts once, for the size it finally picks. Compiling the feature in
/// and leaving this off costs one thread-local read and a predictable branch per bump site
/// (measured at ~0 against a 190 ns swap); leaving it *on* costs ~16% of that swap.
///
/// So the intended shape is a scope, not a global:
///
/// ```ignore
/// let prev = set_enabled(true);
/// let res = compute_swap(..)?;      // the one call whose work we want to price
/// set_enabled(prev);
/// let counts = res.counters;        // exact, not reconstructed
/// ```
///
/// Restore the previous value rather than unconditionally disabling, so a nested scope cannot
/// switch counting off underneath the caller that turned it on.
///
/// A no-op returning `false` without the `cu-counters` feature.
#[inline(always)]
#[allow(unused_variables)]
pub fn set_enabled(on: bool) -> bool {
    #[cfg(feature = "cu-counters")]
    {
        imp::set_enabled(on)
    }
    #[cfg(not(feature = "cu-counters"))]
    {
        false
    }
}

/// Whether the calling thread is counting. Always `false` without the `cu-counters` feature,
/// which is what lets a caller assert it is actually measuring rather than reading zeros.
#[inline(always)]
pub fn enabled() -> bool {
    #[cfg(feature = "cu-counters")]
    {
        imp::enabled()
    }
    #[cfg(not(feature = "cu-counters"))]
    {
        false
    }
}

/// Record work against the calling thread's counters. Compiles to nothing without the
/// `cu-counters` feature, and to one thread-local read plus a branch when counting is off.
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
        let prev = set_enabled(true);
        let before = snapshot();
        f();
        let out = snapshot().since(&before);
        set_enabled(prev);
        out
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

    /// Every branch of the trifecta port, on operands built to select it. The indices are the
    /// layout documented on [`udiv_path`]; a slip here would mis-price a class silently.
    #[test]
    fn udiv_path_selects_each_trifecta_branch() {
        let norm = 1u128 << 63 | 12345; // a normalised 64-bit divisor, as Knuth D hands it over
        assert_eq!(udiv_path(norm - 1, norm), 0, "quotient 0");
        assert_eq!(udiv_path(norm + 1, norm), 1, "quotient 1");
        assert_eq!(udiv_path(1u128 << 40, 1u128 << 20), 2, "half: duo < 2^64, div_lz > duo_lz");
        assert_eq!(udiv_path(1u128 << 100, 1u128 << 20), 3, "short: div < 2^32");
        // two possibility: duo of 65..95 bits against a 64-bit divisor
        assert_eq!(udiv_path(1u128 << 80, norm) & !1, 4);
        // the same with a full 128-bit duo against a divisor within 32 bits of it
        assert_eq!(udiv_path(u128::MAX - 7, 1u128 << 100) & !1, 6);
        // long division: duo of 96+ bits against the 64-bit divisor; one step then an exit
        let p = udiv_path(1u128 << 100, norm);
        assert!((8..18).contains(&p), "one long step, got {p}");
        let p = udiv_path(u128::MAX, norm);
        assert!((18..28).contains(&p), "two long steps, got {p}");
    }

    /// The chain's long division, ported for its branch counts, must still divide correctly —
    /// otherwise the operands it hands the classifier at each step are not the chain's.
    #[test]
    fn chain_div_arms_reproduces_the_quotient() {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut checked = 0;
        for _ in 0..4000 {
            let mut n = [0u64; 4];
            let mut d = [0u64; 4];
            let nw = 3 + (next() % 2) as usize;
            let dw = 1 + (next() % nw as u64) as usize;
            for i in 0..nw {
                n[i] = next() >> (next() % 64);
            }
            for i in 0..dw {
                d[i] = next() >> (next() % 64);
            }
            n[nw - 1] |= 1;
            d[dw - 1] |= 1;
            let to_u256 = |w: [u64; 4]| {
                ethnum::U256::from_words(
                    (w[3] as u128) << 64 | w[2] as u128,
                    (w[1] as u128) << 64 | w[0] as u128,
                )
            };
            let (num, den) = (to_u256(n), to_u256(d));
            let expect = num / den;
            let (q, arms) = chain_div_arms(n, d);
            assert_eq!(to_u256(q), expect, "n={n:?} d={d:?}");
            let steps: u32 = arms.paths.iter().sum();
            if steps > 0 {
                checked += 1;
            }
        }
        assert!(checked > 3000, "too few case-3/4 draws: {checked}");
    }
}
