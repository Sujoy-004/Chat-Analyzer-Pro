# Incremental sliding-window design for `calculate_rolling_health_score`

**Status:** research-only design (no production files modified).
**Target:** replace the per-window full recompute (~35 s for 1418 dates) with an
incremental sliding window that is **bit-identical** on the rolling output columns
(`date`, `health_score`, `grade`, `message_count`) for both the sequential and
the parallel/verification paths.

All claims below were verified by throwaway probes in a scratch directory
(`np.random` fixtures, 2–4 senders, dense calendars, calendars with gaps > 7 days,
and thin windows below `min_messages`):

- starter-flag invariance across a slide: **0 differences** over 1421 surviving rows;
- concatenated per-day starter flags == full-window recompute: **0 mismatching windows**;
- a cheap scalar score pass vs the four reference metric functions: **0/234 windows differ**
  (incl. a 3-sender fix: `groupby('sender')['message_length'].sum()` is indexed by
  sender *name ascending*, not count-descending — see Pitfall #2);
- end-to-end incremental passthrough: **2.24 s for 1418 dates (1.58 ms/window)** vs the
  present ~25 ms/window; outputs bit-identical to the reference on 30 sampled windows.

---

## 1. What is expensive today (measured, ~430-row window)

| component | cost | note |
|---|---|---|
| `identify_conversation_starters` per window | ~2.5–3.5 ms | size-*independent* floor: `to_datetime` 0.69 + `sort_values` 0.22 + normalize-compare 0.39 + `diff()` 0.47 ms |
| `calculate_initiator_ratio` | ~3.1 ms | value_counts + dict |
| `analyze_response_patterns` | ~22.6 ms | pair-groupby 2.7 + pair round/to_dict 4.1 + 6-agg groupby 1.35 + stats .round(2) 1.1 + stats to_dict 1.4 + mask/rename 1.7 ms |
| `calculate_dominance_scores` | ~17.5 ms | burst run-length `cumcount`/`transform('size')` 2.7 ms + groupby aggs + 3× `to_dict` |
| per-window `pd.concat` of date frames | ~0.6 ms | already cheap; dropping it buys little |

⇒ The four metric functions (esp. `to_dict()` on MultiIndex groupbys and the burst
block) are the bottleneck, not the window construction.

## 2. Starter-flag analysis (what changes when the oldest day leaves)

`is_conversation_starter[i] = (time_diff_minutes[i] > 60) OR (day[i] != day[i-1])`, row 0 forced True.

When the oldest day's rows are removed from a window:

- **New row 0** → forced True. It was *already* True (its predecessor was the removed
  day's last row, a different calendar day → day-change). Flag unchanged.
- **Every other surviving row** keeps the exact same in-window predecessor (the removed
  day was a strict prefix of the window), so `day-change` and `gap` recompute identically.
  Flag unchanged.

Empirically confirmed: **zero flag changes** across 1414 slides (1421 surviving rows).

**Key corollary for response extraction:** a *non-starter* row can never have a
day-boundary predecessor (non-starter ⇔ same calendar day, ≤ 60 min). So for every
score-relevant row, `prev_sender` and `time_diff_minutes` are unchanged by a slide —
per-day-local values are exactly correct for all non-starter rows.

Consequences:

- **Precompute starter flags once per date** (day-local; first row of every day is a
  starter). Concatenating per-day flags reproduces the full-window `identify_…` refresh
  exactly (verified, 0 mismatches). This eliminates the ~3 ms/window `identify` call.
- The only window-local values that change across a slide (row 0 and each day-boundary
  first row's `prev_sender`/`time_diff_minutes`) are **inert**: those rows are starters,
  excluded from response analysis, and dominance/initiator read only sender + flags +
  `message_length`. (Set `st[0] = True` after concat anyway as a documented no-op guard.)

## 3. Minimal incremental state

Global (one-time, O(total rows)):

- `dates` (sorted unique dates), `by_date` frames (same grouping as today).
- global `sender → code` mapping (`pd.factorize`) + `rank_by_code` (lexicographic name
  rank of each code — needed to reproduce `groupby(..., sort=True)` row order).
- per-date segments (each a tuple of numpy arrays):
  - `st`   bool starter flags (day-local),
  - `enc`  int64 sender codes,
  - `td`   float64 `time_diff_minutes` (day-local),
  - `len`  int64 `message_length`.

Per-step (amortized O(1) day pops + O(window rows) concat):

- `active = deque[(date, segment)]`; on `current_date`:
  1. `while active and active[0].date < current_date - window_days: active.popleft()`
  2. `active.append((current_date, segment))`
  3. if total rows < `min_messages`: **skip** (identical filtering to today).
  4. build window column buffers `st, enc, td, lens = np.concatenate([...])`; `st[0] = True`.
  5. run the cheap scalar pass (sec. 4) → emit `{'date', 'health_score', 'grade', 'message_count'}`.

Window row order is preserved exactly: days concatenate in ascending date order, each
segment keeps the day's datetime-sorted order = today's concat order. No `sort_values`.

## 4. Exact per-window score (the cheap pass)

Reimplement only the four *scalars* the health score consumes; replicate the reference
operators verbatim (same call → same bits).

**Initiator balance** — integer-exact:
`starters = bincount(enc[st]); total = st.sum()`;
`b_init = 0.0 if total == 0 or <2 distinct starter-senders else 1 - |c0 - c1| / total`
where `c0, c1` are the two largest counts (value_counts is count-desc; ties are
irrelevant — equal values give zero `abs` diff either way).

**Response** (`responsiveness_score`, `response_balance_score`):
`vmask = ~st & (prev_enc != enc) & ~isnan(td)`, with `prev_enc[i] = enc[i-1]` (window-local).
`rtimes = td[vmask]`. Then the **only two float aggregates that must be recomputed per
window**:

- `overall = rtimes.mean()` (replicates `response_analysis_df['response_time_minutes'].mean()`).
- `per = pd.Series(rtimes, index=enc[vmask]).groupby(level=0).mean().round(2)`,
  reordered by `rank_by_code`; `r1, r2 = per.values[0], per.values[1]`
  (the two *alphabetically-first* responders = `groupby('responder', sort=True)`).

Scores: `resp_s = max(0, 1 - overall/120)`; `bal_s = max(0, 1 - |r1 - r2|/60)`
if ≥ 2 responders else `1.0`. Empty-response window → `(0, 1.0)` (mirrors the
`{'error': …}` branch → `.get(..., 0)/.get(..., 1)`).
Grouping by int codes instead of strings changes nothing: same membership, same value
order per group, same groupby mean arithmetic.

**Dominance** (`composite_dominance_score = (msg_bal + len_bal + control)/3`):

- `msg_bal`: `0.0 if <2 senders else 1 - |c0 - c1| / n` (two largest `bincount(enc)`).
- `len_bal`: `ld = pd.Series(lens, index=enc).groupby(level=0).sum()`, reorder by
  `rank_by_code`, take **indices [0] and [1]** (two alphabetically-first senders — this
  is *not* the two largest sums; see Pitfall #2). `1 - |l0 - l1| / total_chars`
  if ≥ 2 senders and `total_chars > 0`, else `1.0`. Guard `'message_length' in columns`
  as today.
- `control`: endings = senders at `st[i+1]` True or last row →
  `bincount(endings)`; `0.0 if <2 distinct enders else 1 - |e0 - e1| / len(endings)`
  (two largest counts).

**Health score**: `overall = 0.25*b_init + 0.35*resp_s + 0.20*bal_s + 0.20*comp`
(identical expression-tree/order as `calculate_relationship_health_score`), then the
same `grade` thresholds. Because every consumed float is unchanged bit-for-bit, the
weighted sum is bit-identical and a grade-boundary flip is impossible.

## 5. What is exact-incremental vs. what needs a local recompute

**Exact running counts (integers only):** per-sender starter counts, total
conversations, per-sender message counts, per-sender length sums, `total_messages`,
`total_chars`, per-sender ending counts. A day's contributions to all of these are
themselves invariant (flags are invariant), so pop/subtract + push/add are exact.

**Cannot be exactly incremental as running aggregates:** the two response means.
`mean()`/`groupby.mean().round(2)` in pandas do not equal `sum/count` running formulas
(Welford / pairwise summation), so a running counter drifts in the last ulp. Cheapest
EXACT option: keep the response rows window-ordered and call the *same* pandas
`groupby(...).mean().round(2)` over only the response rows each window.
Cost: measured ~0.2 ms at 10 responses, ~0.5 ms at 100+ → **~0.3–0.7 s total** for
1418 windows. This is the accepted local recompute.

**Not consumed by any score (drop entirely for rolling parity):** `response_stats`
median/std/min/max, `response_pairs`, `response_time_difference`, initiator ratios
dict, `message_distribution`/`length_distribution`/`avg_message_lengths`/
`conversation_enders`/`burst_stats`. The **whole burst block** (2.7 ms RLE + groupby
aggs) and the two MultiIndex `to_dict()` conversions are dead weight for the rolling
output — they only matter if you later demand dict-level parity too.

## 6. Recommended architecture (1418-window / ~300-row-window case)

**Recommendation: one sliding deque of per-day numpy segment arrays + per-window cheap
scalar pass (sec. 3–4).** Do *not* build running counters yet.

Variant comparison (measured):

| variant | per-window | 1418 dates |
|---|---|---|
| current (concat + 4 metric functions) | ~25 ms (user-reported) | ~35 s |
| DataFrame slide + per-window `identify` + cheap pass | 9.9 ms | ~14 s |
| **deque of per-day arrays + maintained flags + cheap pass** | **1.6 ms** | **~2.2 s** |
| running counters for int-aggregates + deque response arrays | est. < 1 ms | est. < 1.5 s |

The running-counter refinement (pop/subtract a popped day's precomputed per-sender
starter/message/length/ending counts, push/add the new day's) is real but adds
bookkeeping surface for at most ~2× over the 1.6 ms variant. Skip it for now; add only
if a profile on 300-row windows shows the per-window concat/groupby dominating.

**Parallel/sequential parity.** Keep `_rolling_window_score(window_df, current_date,
min_messages)` unchanged as the independent per-window reference worker. Use the
incremental loop as the main path for every date count; raise the default
`_ROLLING_PARALLEL_MIN_DATES` so production never spawns workers (the incremental loop
makes them pointless and the workers' per-window recompute would restore the 35 s).
The parallel branch remains reachable by the ME-04 parity test
(`monkeypatch … = 1`), which then genuinely cross-checks incremental vs. the full
recompute worker — both are bit-exact, so `sequential.equals(parallel)` holds.

## 7. File-level structure

Everything under the DAY-9 section of `relationship_health.py`:

- `_rolling_window_score` — **unchanged** (reference worker, parallel harness).
- new `_prepare_date_segments(df, window_days)` → `(dates, segments, rank_by_code)`:
  group by date, global `pd.factorize`, one per-day `identify_conversation_starters`
  call, extract the four numpy columns.
- new `_score_window_columns(st, enc, td, lens, rank_by_code, min_messages)` → optional
  `None` (skip) or `(health_score, grade, message_count)` dict — the cheap pass.
- restructured `calculate_rolling_health_score`: same signature, same output
  DataFrame construction (`pd.DataFrame(records)` from emitted dicts, date objects =
  `dates`), per-window `try/except → skip + warn` to preserve the reference's
  skip-on-error semantics, calling the incremental loop. Keep the `_ROLLING_PARALLEL_*`
  constants (retuned) and the worker branch for ME-04.

## 8. Float / rounding pitfalls checklist (parity-critical)

1. **`.round(2)` happens on response_stats**, and the two balance means are the
   *rounded* values. Reproduce by the same `groupby(...).mean().round(2)` call; never
   `np.round` a DIY mean.
2. **`.values[0]/[1]` ordering is aggregate-specific** — the recurring trap:
   - `value_counts()` → sorted by **count desc** → take the two largest counts;
   - `df.groupby('sender')['message_length'].sum()` → indexed by **sender name asc** →
     take the two *alphabetically-first* senders, NOT the largest sums
     (this alone broke 3-sender parity: 35/59 windows divergent before the fix);
   - `response_stats['mean'].values` → **responder name asc** → two alphabetically-first
     responders.
3. Keep counts as ints through `/` (`np.int64 / np.int64 == int/int` here); never float
   the numerator first.
4. Preserve the exact expression tree of the weighted sum and of `max(0, 1 - x)`
   (Python `max` returns the *int* `0` on negatives — replicate, don't `np.clip`).
5. Because every consumed float comes from the identical pandas call on identically
   ordered data, last-ulp or `0.8999… vs 0.9000…` grade-flip risk is eliminated rather
   than "made improbable". A running-sum `overall` mean is off the table (pairwise
   summation mismatch).
6. `message_length` is int64 (guaranteed by `messages_to_dataframe`), so length sums are
   int-exact; keep the column-presence guard anyway.
7. Output dtypes: `date` = `datetime.date` objects (from `dates`), `health_score` =
   float64, `grade` = str, `message_count` = int; build the frame exactly as today.
8. If dict-level parity is ever needed later, `value_counts` tie order is stable
   first-appearance — a dict-reconstruction must preserve insertion order; not needed
   for the four rolling columns.