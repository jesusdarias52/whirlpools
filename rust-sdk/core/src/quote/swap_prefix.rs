//! A prefix table over [`compute_swap`](super::compute_swap)'s walk: the running totals after
//! each sub-step, so an exact-in quote is a binary search plus one partial fill instead of a walk.
//!
//! **Why a table can exist at all.** A sub-step that *reaches* its bounded target consumes an
//! input that does not depend on the trade — `a_t + fee(a_t)` where `a_t` is the fixed delta to
//! the target and `fee(a_t) = ceil(a_t·D/(D−f)) − a_t` (`try_reverse_apply_swap_fee`) — for an
//! output that does not either, and it leaves behind a state (price, tick, liquidity, the
//! adaptive-fee manager) that is the same for every amount that completes it. The step reaches
//! its target iff `floor(rem·(D−f)/D) ≥ a_t` (`try_apply_swap_fee`), which is exactly
//! `rem ≥ ceil(a_t·D/(D−f))`, so the running total of consumed inputs is also the sequence of
//! thresholds at which each further sub-step is entered. The fee at each sub-step is a pure
//! function of the manager's state, and the manager's only dependence on the clock is which
//! branch `update_reference` took at construction — the caller keys its table on that
//! ([`AdaptiveFeeVariablesFacade::reference_branch`]).
//!
//! **The walk ends three ways, and an amount past the last row must answer as the walk would.**
//! [`SwapPrefixTail::Stop`] is an error that does not depend on the amount (the tick sequence ran
//! out, liquidity overflowed): every such amount quotes `Err`, here `0`. [`SwapPrefixTail::Capped`]
//! is the loop exiting normally at the price limit with input left: `compute_swap` returns
//! `amount_calculated` with no leftover check, so it is the last row's output. [`SwapPrefixTail::Open`]
//! is a sub-step no `u64` amount completes — its `u64::MAX` fill errored, fell short of its target
//! (`AMOUNT_EXCEEDS_MAX_U64` at the target, say) or pushed the running total past `u64::MAX` — and
//! it is replayed with the remainder through the same `compute_swap_step` and the same checked
//! arithmetic, so a partial is the walk's partial and an error is the walk's error. A sub-step
//! whose *crossing* fails (`LIQUIDITY_OVERFLOW`) is `Open` too — the walk applies the sub-step
//! before it crosses, so an amount that completes it errors while a smaller one partial-fills it.
//! The walk can never continue past such a unit, so no resume state is kept for it.
//!
//! Built incrementally, whole outer steps at a time: the first quote walks just past its own
//! amount and records where it stopped; a smaller amount is a lookup, a larger one resumes.
//! Nothing here feeds the counters, `trade_fee` or `applied_fee_rate_*` — none of them feed the
//! output, and the counted `compute_swap` is what prices compute units.

use crate::{
    sqrt_price_to_tick_index, tick_index_to_sqrt_price, AdaptiveFeeInfo, FeeRateManager,
    TickArraySequence, WhirlpoolFacade, MAX_SQRT_PRICE, MIN_SQRT_PRICE,
};

use super::swap::{compute_swap_step, get_next_liquidity};

/// One sub-step of the walk that consumed input, with the operands a partial fill of it is
/// replayed from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SwapPrefixUnit {
    pub start: u128,
    pub target: u128,
    pub liquidity: u128,
    pub fee_rate: u32,
}

/// The running totals **after** one consuming sub-step.
#[derive(Clone, Copy, Debug)]
pub struct SwapPrefixRow {
    pub cum_in: u128,
    /// `u64` because `compute_swap`'s `amount_calculated` is, and its `checked_add` overflowing is
    /// an error the table reproduces.
    pub cum_out: u64,
    pub unit: SwapPrefixUnit,
}

/// The walk's state at the top of its outer loop, after the last completed step.
#[derive(Clone, Debug)]
pub struct SwapPrefixResume {
    sqrt_price: u128,
    tick_index: i32,
    liquidity: u128,
    fee_rate_manager: FeeRateManager,
}

/// How the table ends — what an amount past its last row quotes. See the module doc.
#[derive(Clone, Debug)]
pub enum SwapPrefixTail {
    Stop,
    Capped,
    Open(SwapPrefixUnit),
    More(Box<SwapPrefixResume>),
}

/// The walk's running totals for one pool state, direction and reference branch.
#[derive(Clone, Debug)]
pub struct SwapPrefixTable {
    rows: Vec<SwapPrefixRow>,
    tail: SwapPrefixTail,
    /// The clock the table was built against. Any timestamp in the same reference branch walks
    /// identically, so a caller re-running the counted walk for this table's quotes can use it.
    timestamp: u64,
    sqrt_price_limit: u128,
}

impl SwapPrefixTable {
    /// An empty table positioned where the walk starts. Runs `compute_swap`'s gates in its order
    /// — limit bounds, direction, adaptive-info consistency, `FeeRateManager::new` (whose
    /// `update_reference` can refuse the timestamp) — each a `Stop`. The zero-amount gate is the
    /// caller's. With `sqrt_price_limit == 0` the direction's extreme is used, as `compute_swap`
    /// does.
    pub fn start(
        sqrt_price_limit: u128,
        whirlpool: WhirlpoolFacade,
        a_to_b: bool,
        timestamp: u64,
        adaptive_fee_info: Option<AdaptiveFeeInfo>,
    ) -> Self {
        let sqrt_price_limit = if sqrt_price_limit == 0 {
            if a_to_b {
                MIN_SQRT_PRICE
            } else {
                MAX_SQRT_PRICE
            }
        } else {
            sqrt_price_limit
        };
        let stopped = || Self {
            rows: Vec::new(),
            tail: SwapPrefixTail::Stop,
            timestamp,
            sqrt_price_limit,
        };
        if !(MIN_SQRT_PRICE..=MAX_SQRT_PRICE).contains(&sqrt_price_limit) {
            return stopped();
        }
        if a_to_b && sqrt_price_limit >= whirlpool.sqrt_price
            || !a_to_b && sqrt_price_limit <= whirlpool.sqrt_price
        {
            return stopped();
        }
        if whirlpool.is_initialized_with_adaptive_fee() != adaptive_fee_info.is_some() {
            return stopped();
        }
        let Ok(fee_rate_manager) = FeeRateManager::new(
            a_to_b,
            whirlpool.tick_current_index,
            timestamp,
            whirlpool.fee_rate,
            &adaptive_fee_info,
        ) else {
            return stopped();
        };
        Self {
            rows: Vec::new(),
            tail: SwapPrefixTail::More(Box::new(SwapPrefixResume {
                sqrt_price: whirlpool.sqrt_price,
                tick_index: whirlpool.tick_current_index,
                liquidity: whirlpool.liquidity,
                fee_rate_manager,
            })),
            timestamp,
            sqrt_price_limit,
        }
    }

    /// The clock this table was built against.
    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// The input at which each row's sub-step is exactly exhausted — a test instrument, for
    /// quoting at every boundary and one unit either side.
    pub fn row_inputs(&self) -> impl Iterator<Item = u128> + '_ {
        self.rows.iter().map(|r| r.cum_in)
    }

    /// The output of an exact-in swap of `amount_in`, as `compute_swap(amount_in, …)` would
    /// return it in the output token (`token_b` for `a_to_b`, `token_a` otherwise) — and `0`
    /// wherever `compute_swap` would return `Err`. `amount_in` must be non-zero.
    pub fn quote(&mut self, amount_in: u64, tick_sequence: &TickArraySequence, a_to_b: bool) -> u64 {
        let x = u128::from(amount_in);
        // Past everything built, and the walk would go on: extend until a row is past `x` or the
        // walk ends. Each extension runs at least one whole step, so this terminates.
        while matches!(self.tail, SwapPrefixTail::More(_))
            && self.rows.last().is_none_or(|r| r.cum_in <= x)
        {
            self.extend(x, tick_sequence, a_to_b);
        }

        // `rows[..k]` are the sub-steps `x` completes outright.
        let k = self.rows.partition_point(|r| r.cum_in <= x);
        let (out, r) = if k == 0 {
            (0u64, amount_in)
        } else {
            (self.rows[k - 1].cum_out, (x - self.rows[k - 1].cum_in) as u64)
        };
        if r == 0 {
            // Exhausted exactly on a boundary: the walk breaks before the next sub-step.
            return out;
        }
        if let Some(row) = self.rows.get(k) {
            return replay(row.unit, out, r, a_to_b);
        }
        match &self.tail {
            SwapPrefixTail::Stop => 0,
            SwapPrefixTail::Capped => out,
            SwapPrefixTail::Open(unit) => replay(*unit, out, r, a_to_b),
            SwapPrefixTail::More(_) => {
                // Unreachable: the extension loop above leaves the last row past `x`.
                debug_assert!(false, "prefix lookup ran past a table that was built past it");
                0
            }
        }
    }

    /// Resume the walk from `More` and run whole outer steps of `compute_swap`'s loop, every
    /// sub-step filled with `u64::MAX`, until a row is past `up_to` or the walk ends. This is
    /// `compute_swap`'s control flow line for line — the tick lookup, the limit clamp, the
    /// accumulator update, `get_bounded_sqrt_price_target`, the crossing, the tick-group advance
    /// — minus everything that does not feed the output.
    fn extend(&mut self, up_to: u128, tick_sequence: &TickArraySequence, a_to_b: bool) {
        let SwapPrefixTail::More(resume) = &self.tail else { return };
        let mut rs = (**resume).clone();
        let (mut cum_in, mut cum_out) = self.rows.last().map_or((0u128, 0u64), |r| (r.cum_in, r.cum_out));

        // One outer step per iteration. `rs` is the walk's state at the loop top; the tail is
        // rewritten from it after each completed step, and a step that ends the walk returns with
        // the tail set and `rs` discarded.
        loop {
            let next = if a_to_b {
                tick_sequence.prev_initialized_tick(rs.tick_index)
            } else {
                tick_sequence.next_initialized_tick(rs.tick_index)
            };
            let Ok((next_tick, next_tick_index)) = next else {
                self.tail = SwapPrefixTail::Stop;
                return;
            };
            let next_tick_sqrt_price: u128 = tick_index_to_sqrt_price(next_tick_index).into();
            let target_sqrt_price = if a_to_b {
                next_tick_sqrt_price.max(self.sqrt_price_limit)
            } else {
                next_tick_sqrt_price.min(self.sqrt_price_limit)
            };

            loop {
                rs.fee_rate_manager.update_volatility_accumulator();
                let total_fee_rate = rs.fee_rate_manager.get_total_fee_rate();
                let (bounded_sqrt_price_target, adaptive_fee_update_skipped) = rs
                    .fee_rate_manager
                    .get_bounded_sqrt_price_target(target_sqrt_price, rs.liquidity);

                let unit = SwapPrefixUnit {
                    start: rs.sqrt_price,
                    target: bounded_sqrt_price_target,
                    liquidity: rs.liquidity,
                    fee_rate: total_fee_rate,
                };
                let step = match compute_swap_step(
                    u64::MAX,
                    total_fee_rate,
                    rs.liquidity,
                    rs.sqrt_price,
                    bounded_sqrt_price_target,
                    a_to_b,
                    true,
                ) {
                    Ok(step) if step.next_sqrt_price == bounded_sqrt_price_target => step,
                    _ => {
                        self.tail = SwapPrefixTail::Open(unit);
                        return;
                    }
                };
                // `compute_swap` subtracts `amount_in` and `fee_amount` separately; their sum is
                // the amount at which the reach test above first passes.
                let (Some(consumed), Some(total_out)) =
                    (step.amount_in.checked_add(step.fee_amount), cum_out.checked_add(step.amount_out))
                else {
                    self.tail = SwapPrefixTail::Open(unit);
                    return;
                };
                cum_out = total_out;
                cum_in += u128::from(consumed);
                debug_assert!(consumed > 0 || step.amount_out == 0, "a sub-step produced output for no input");
                if consumed > 0 {
                    self.rows.push(SwapPrefixRow { cum_in, cum_out, unit });
                }

                if step.next_sqrt_price == next_tick_sqrt_price {
                    let Ok(liquidity) = get_next_liquidity(rs.liquidity, next_tick, a_to_b) else {
                        // The crossing fails *after* the sub-step is applied: an amount that
                        // completes it errors, one that does not still partial-fills it. So the
                        // row comes back off and the unit becomes the `Open` tail, whose replay
                        // answers 0 on a reach.
                        if consumed > 0 {
                            self.rows.pop();
                        }
                        self.tail = if consumed > 0 { SwapPrefixTail::Open(unit) } else { SwapPrefixTail::Stop };
                        return;
                    };
                    rs.liquidity = liquidity;
                    rs.tick_index = if a_to_b { next_tick_index - 1 } else { next_tick_index };
                } else if step.next_sqrt_price != rs.sqrt_price {
                    rs.tick_index = sqrt_price_to_tick_index(step.next_sqrt_price.into()).into();
                }
                rs.sqrt_price = step.next_sqrt_price;

                if !adaptive_fee_update_skipped {
                    rs.fee_rate_manager.advance_tick_group();
                } else {
                    rs.fee_rate_manager.advance_tick_group_after_skip(
                        rs.sqrt_price,
                        next_tick_sqrt_price,
                        next_tick_index,
                    );
                }

                // The walk's other break, `amount_remaining == 0`, is the lookup's.
                if rs.sqrt_price == target_sqrt_price {
                    break;
                }
            }

            if rs.sqrt_price == self.sqrt_price_limit {
                self.tail = SwapPrefixTail::Capped;
                return;
            }
            if cum_in > up_to {
                self.tail = SwapPrefixTail::More(Box::new(rs));
                return;
            }
        }
    }
}

/// One sub-step filled with the remainder `r`, through the walk's own `compute_swap_step` on the
/// walk's own operands, with the walk's own checked arithmetic — so an error here is the walk's
/// error (`0`).
///
/// A replay that **reaches** its target answers 0. On a row that is only ever partial-filled here
/// (`r` is below its consumption by construction) that never fires; on an `Open` unit it is the
/// definition — no amount completes an `Open` unit *and carries on*: either the fill itself fails
/// at any amount that reaches (an amount-independent error, or the running total overflowing),
/// or the crossing after it does.
fn replay(unit: SwapPrefixUnit, out: u64, r: u64, a_to_b: bool) -> u64 {
    match compute_swap_step(r, unit.fee_rate, unit.liquidity, unit.start, unit.target, a_to_b, true) {
        Ok(step) if step.next_sqrt_price == unit.target => 0,
        Ok(step) => out.checked_add(step.amount_out).unwrap_or(0),
        Err(_) => 0,
    }
}

#[cfg(all(test, not(feature = "wasm")))]
mod tests {
    use super::*;
    use crate::{
        compute_swap, AdaptiveFeeConstantsFacade, AdaptiveFeeVariablesFacade, OracleFacade,
        TickArrayFacade, TickFacade, TICK_ARRAY_SIZE,
    };

    type Arrays = Vec<Option<TickArrayFacade>>;

    fn empty_array(start_tick_index: i32) -> TickArrayFacade {
        TickArrayFacade { start_tick_index, ticks: [TickFacade::default(); TICK_ARRAY_SIZE] }
    }

    fn now() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    fn whirlpool(sqrt_price: u128, liquidity: u128, tick_spacing: u16, adaptive: bool) -> WhirlpoolFacade {
        WhirlpoolFacade {
            tick_current_index: sqrt_price_to_tick_index(sqrt_price),
            fee_rate: if adaptive { 1000 } else { 3000 },
            liquidity,
            sqrt_price,
            fee_tier_index_seed: if adaptive { [128 + 64, 0] } else { [tick_spacing as u8, 0] },
            tick_spacing,
            ..WhirlpoolFacade::default()
        }
    }

    fn tick(liquidity_net: i128) -> TickFacade {
        TickFacade {
            initialized: liquidity_net != 0,
            liquidity_net,
            ..TickFacade::default()
        }
    }

    /// Five arrays around 0 with an initialized tick at every slot — the dense fixture of
    /// `swap.rs`'s tests, with the ticks' liquidity deltas made irregular so consecutive
    /// crossings differ.
    fn dense_arrays(tick_spacing: u16) -> Arrays {
        let span = TICK_ARRAY_SIZE as i32 * tick_spacing as i32;
        let mut arrays: Arrays = Vec::new();
        for start in [0, span, 2 * span, -span, -2 * span] {
            let mut ticks = [TickFacade::default(); TICK_ARRAY_SIZE];
            for (j, t) in ticks.iter_mut().enumerate() {
                let net = if start < 0 { 1000 + j as i128 * 7 } else { -(1000 + j as i128 * 7) };
                *t = tick(net);
            }
            arrays.push(Some(TickArrayFacade { start_tick_index: start, ticks }));
        }
        arrays
    }

    /// Sparse: one initialized tick per array, at differing offsets, so the walk alternates long
    /// and short steps and lands on an array sentinel at the end.
    fn sparse_arrays_with(tick_spacing: u16, net: i128) -> Arrays {
        let span = TICK_ARRAY_SIZE as i32 * tick_spacing as i32;
        let mut arrays: Arrays = Vec::new();
        for (i, start) in [0, span, 2 * span, -span, -2 * span].into_iter().enumerate() {
            let mut ticks = [TickFacade::default(); TICK_ARRAY_SIZE];
            let slot = (i * 17 + 3) % TICK_ARRAY_SIZE;
            ticks[slot] = tick(if start < 0 { net } else { -net });
            arrays.push(Some(TickArrayFacade { start_tick_index: start, ticks }));
        }
        arrays
    }

    fn sparse_arrays(tick_spacing: u16) -> Arrays {
        sparse_arrays_with(tick_spacing, 100_000)
    }

    fn empty_arrays(tick_spacing: u16) -> Arrays {
        let span = TICK_ARRAY_SIZE as i32 * tick_spacing as i32;
        let mut arrays: Arrays = Vec::new();
        for start in [0, span, 2 * span, -span, -2 * span] {
            arrays.push(Some(empty_array(start)));
        }
        arrays
    }

    fn oracle(now: u64, last_ref: u64, last_major: u64, vol_ref: u32, idx_ref: i32, vol_acc: u32) -> OracleFacade {
        OracleFacade {
            trade_enable_timestamp: now,
            adaptive_fee_constants: AdaptiveFeeConstantsFacade {
                filter_period: 30,
                decay_period: 600,
                adaptive_fee_control_factor: 5_000,
                reduction_factor: 500,
                max_volatility_accumulator: 88 * 3 * 10_000,
                tick_group_size: 64,
                major_swap_threshold_ticks: 64,
            },
            adaptive_fee_variables: AdaptiveFeeVariablesFacade {
                last_reference_update_timestamp: last_ref,
                last_major_swap_timestamp: last_major,
                volatility_reference: vol_ref,
                tick_group_index_reference: idx_ref,
                volatility_accumulator: vol_acc,
            },
        }
    }

    /// The counted walk's answer for an exact-in swap, in the output token, `0` on `Err`.
    fn walk(amount: u64, wp: WhirlpoolFacade, seq: &TickArraySequence, a_to_b: bool, ts: u64, afi: Option<AdaptiveFeeInfo>) -> u64 {
        match compute_swap(amount, 0, wp, seq, a_to_b, true, ts, afi) {
            Ok(r) => if a_to_b { r.token_b } else { r.token_a },
            Err(_) => 0,
        }
    }

    /// Every amount in `amounts`, in three build orders, on a fresh table each: the table's
    /// answer must equal the walk's to the unit. Returns how many rows the fullest table held.
    fn check(wp: WhirlpoolFacade, arrays: Arrays, a_to_b: bool, ts: u64, afi: Option<AdaptiveFeeInfo>, ladder: &[u64]) -> (usize, usize) {
        let seq = TickArraySequence::new(arrays, wp.tick_spacing).unwrap();
        // The boundaries, off a throwaway table built to the top of the ladder.
        let mut probe = SwapPrefixTable::start(0, wp, a_to_b, ts, afi);
        let _ = probe.quote(*ladder.last().unwrap(), &seq, a_to_b);
        let mut amounts: Vec<u64> = ladder.to_vec();
        for b in probe.row_inputs() {
            for a in [b.saturating_sub(1), b, b + 1] {
                if a > 0 && a <= u64::MAX as u128 {
                    amounts.push(a as u64);
                }
            }
        }
        amounts.sort_unstable();
        amounts.dedup();
        let n = amounts.len();
        let mut interleaved = Vec::with_capacity(n);
        for j in 0..n.div_ceil(2) {
            interleaved.push(amounts[j]);
            if n - 1 - j > j {
                interleaved.push(amounts[n - 1 - j]);
            }
        }
        let mut nonzero = 0;
        for order in [amounts.clone(), amounts.iter().rev().copied().collect(), interleaved] {
            let mut table = SwapPrefixTable::start(0, wp, a_to_b, ts, afi);
            for &a in &order {
                let expect = walk(a, wp, &seq, a_to_b, ts, afi);
                let got = table.quote(a, &seq, a_to_b);
                assert_eq!(got, expect, "a_to_b={a_to_b} amount={a}: table {got}, walk {expect}");
                nonzero += (expect > 0) as usize;
            }
        }
        (probe.rows.len(), nonzero)
    }

    fn ladder(max: u64) -> Vec<u64> {
        let mut v = vec![1u64, 2, 3, 10, 100];
        let mut a = 1_000u64;
        while a < max {
            v.push(a);
            a = a.saturating_mul(3) / 2 + 1;
        }
        v.push(max);
        v
    }

    #[test]
    fn static_dense_crossings_match_the_walk_in_both_directions() {
        for a_to_b in [true, false] {
            let (rows, nonzero) = check(whirlpool(1 << 64, 100_000_000, 2, false), dense_arrays(2), a_to_b, now(), None, &ladder(1 << 40));
            assert!(rows > 20, "the table walked {rows} rows");
            assert!(nonzero > 0);
        }
    }

    #[test]
    fn static_sparse_reaches_the_sequence_end_and_stops() {
        for a_to_b in [true, false] {
            let wp = whirlpool(1 << 64, 265_000, 2, false);
            let (rows, _) = check(wp, sparse_arrays(2), a_to_b, now(), None, &ladder(1 << 30));
            assert!(rows >= 1);
            let seq = TickArraySequence::new(sparse_arrays(2), 2).unwrap();
            let mut table = SwapPrefixTable::start(0, wp, a_to_b, now(), None);
            let _ = table.quote(u64::MAX, &seq, a_to_b);
            assert!(matches!(table.tail, SwapPrefixTail::Stop | SwapPrefixTail::Open(_)), "{:?}", table.tail);
        }
    }

    #[test]
    fn a_failing_crossing_is_an_open_unit() {
        // Liquidity 265_000 against a tick whose net removes 5_000_000 on the way down: the
        // crossing overflows, so an amount that completes the first sub-step errors while a
        // smaller one partial-fills it.
        let wp = whirlpool(1 << 64, 265_000, 2, false);
        check(wp, sparse_arrays_with(2, 5_000_000), true, now(), None, &ladder(1 << 30));
        let seq = TickArraySequence::new(sparse_arrays_with(2, 5_000_000), 2).unwrap();
        let mut table = SwapPrefixTable::start(0, wp, true, now(), None);
        assert_eq!(table.quote(u64::MAX, &seq, true), 0);
        assert!(table.rows.is_empty());
        assert!(matches!(table.tail, SwapPrefixTail::Open(_)), "{:?}", table.tail);
    }

    #[test]
    fn empty_sequence_open_step_is_replayed() {
        // No initialized tick and a tiny liquidity: one sub-step to the array sentinel that no
        // `u64` completes at the bound, so every amount is a partial of it.
        for a_to_b in [true, false] {
            let wp = whirlpool(1 << 64, 1_000, 64, false);
            check(wp, empty_arrays(64), a_to_b, now(), None, &ladder(1 << 50));
        }
    }

    #[test]
    fn adaptive_substeps_match_the_walk_across_reference_branches() {
        let now = now();
        let afi = |o: OracleFacade| Some(AdaptiveFeeInfo::from(o));
        // High-frequency (keep), reduction, reset, and a hot accumulator with references set.
        let oracles = [
            oracle(now, now, now, 0, 0, 0),
            oracle(now, now - 100, now - 100, 0, 0, 100_000),
            oracle(now, now - 1000, now - 1000, 0, 0, 100_000),
            oracle(now, now - 5, now - 5, 50_000, 3, 200_000),
        ];
        for o in oracles {
            for a_to_b in [true, false] {
                let (rows, nonzero) = check(
                    whirlpool(tick_index_to_sqrt_price(0).into(), 1_000_000, 64, true),
                    empty_arrays(64),
                    a_to_b,
                    now,
                    afi(o),
                    &ladder(1 << 34),
                );
                assert!(rows > 3, "sub-stepped {rows} rows");
                assert!(nonzero > 0);
                check(
                    whirlpool(tick_index_to_sqrt_price(0).into(), 100_000_000, 64, true),
                    dense_arrays(64),
                    a_to_b,
                    now,
                    afi(o),
                    &ladder(1 << 44),
                );
            }
        }
    }

    #[test]
    fn a_stale_timestamp_is_a_stop() {
        let now = now();
        let o = oracle(now, now + 10, now, 0, 0, 0);
        let wp = whirlpool(tick_index_to_sqrt_price(0).into(), 1_000_000, 64, true);
        let seq = TickArraySequence::new(empty_arrays(64), 64).unwrap();
        let mut table = SwapPrefixTable::start(0, wp, true, now, Some(o.into()));
        assert!(matches!(table.tail, SwapPrefixTail::Stop));
        assert_eq!(table.quote(1_000, &seq, true), 0);
        assert!(compute_swap(1_000, 0, wp, &seq, true, true, now, Some(o.into())).is_err());
    }

    #[test]
    fn reference_branch_names_update_references_outcome() {
        let c = oracle(0, 0, 0, 0, 0, 0).adaptive_fee_constants;
        let base = AdaptiveFeeVariablesFacade {
            last_reference_update_timestamp: 10_000,
            last_major_swap_timestamp: 10_020,
            volatility_reference: 7,
            tick_group_index_reference: 3,
            volatility_accumulator: 40_000,
        };
        for ts in (9_990..=14_000).step_by(1) {
            let branch = base.reference_branch(ts, &c);
            let mut v = base;
            let res = v.update_reference(9, ts, &c);
            let expect = match res {
                Err(_) => 0,
                Ok(()) if v.last_reference_update_timestamp == 10_000 => 2,
                Ok(()) if v.volatility_reference == 0 && ts - 10_000 > crate::MAX_REFERENCE_AGE => 1,
                Ok(()) if v.volatility_reference == 0 => 4,
                Ok(()) => 3,
            };
            assert_eq!(branch, expect, "ts={ts}");
        }
    }
}
