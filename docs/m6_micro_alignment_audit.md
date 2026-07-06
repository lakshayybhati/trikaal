# M6 micro-alignment lag-sharpness audit — PASS

**What this closes:** the FP-3 residual — could the aggTrades-derived microstructure features
sit one minute off the kline bar they claim to describe? Two independent probes; the windowing
under test is `[open, open+60s)` (`bar_id = transact_time // 60000`, taker BUY =
`is_buyer_maker == false`).

## (A) in-lake lag sharpness — 59 (symbol, month) samples, ALL PASS

corr(kline log-volume z (x_5), aggTrades volume reconstruction z(x_9)+z(x_10)) at lag 0 vs ±1;
adjacent usable minutes only; decisiveness floor: r0 must beat both by ≥ 0.02.

| symbol | month | n_pairs | r_lag0 | r_lag_plus1 | r_lag_minus1 | margin | pass |
|---|---|---|---|---|---|---|---|
| BTCUSDT | 2021-02 | 40129 | 0.911 | 0.5422 | 0.5691 | 0.3419 | True |
| BTCUSDT | 2021-09 | 42866 | 0.9687 | 0.6275 | 0.6411 | 0.3276 | True |
| BTCUSDT | 2022-06 | 42807 | 0.9228 | 0.5926 | 0.6227 | 0.3001 | True |
| BTCUSDT | 2022-11 | 39814 | 0.8366 | 0.6493 | 0.6485 | 0.1873 | True |
| BTCUSDT | 2023-05 | 42674 | 0.9043 | 0.6215 | 0.6291 | 0.2752 | True |
| BTCUSDT | 2023-10 | 43805 | 0.951 | 0.6707 | 0.6844 | 0.2666 | True |
| BTCUSDT | 2024-03 | 43520 | 0.9334 | 0.6534 | 0.6686 | 0.2648 | True |
| BTCUSDT | 2024-08 | 44057 | 0.9512 | 0.6792 | 0.687 | 0.2642 | True |
| BTCUSDT | 2024-12 | 44119 | 0.9468 | 0.6473 | 0.6588 | 0.2881 | True |
| ETHUSDT | 2021-02 | 40023 | 0.9219 | 0.5782 | 0.5982 | 0.3237 | True |
| ETHUSDT | 2021-09 | 42980 | 0.9642 | 0.6227 | 0.6335 | 0.3306 | True |
| ETHUSDT | 2022-06 | 42845 | 0.925 | 0.5991 | 0.6146 | 0.3104 | True |
| ETHUSDT | 2022-11 | 39952 | 0.8272 | 0.6256 | 0.6259 | 0.2013 | True |
| ETHUSDT | 2023-05 | 33337 | 0.929 | 0.6152 | 0.616 | 0.313 | True |
| ETHUSDT | 2023-10 | 44032 | 0.9643 | 0.6199 | 0.6275 | 0.3368 | True |
| ETHUSDT | 2024-03 | 43753 | 0.9485 | 0.6518 | 0.6664 | 0.2821 | True |
| ETHUSDT | 2024-08 | 44164 | 0.9511 | 0.6587 | 0.668 | 0.2831 | True |
| ETHUSDT | 2024-12 | 44216 | 0.9644 | 0.6624 | 0.6698 | 0.2946 | True |
| SOLUSDT | 2021-02 | 40236 | 0.8681 | 0.4265 | 0.4355 | 0.4326 | True |
| SOLUSDT | 2021-09 | 43076 | 0.9091 | 0.603 | 0.6252 | 0.2839 | True |
| SOLUSDT | 2022-06 | 42624 | 0.9565 | 0.5869 | 0.599 | 0.3575 | True |
| SOLUSDT | 2022-11 | 40043 | 0.8294 | 0.6294 | 0.6279 | 0.2 | True |
| SOLUSDT | 2023-05 | 43000 | 0.8978 | 0.511 | 0.5138 | 0.384 | True |
| SOLUSDT | 2023-10 | 44427 | 0.9107 | 0.5653 | 0.5802 | 0.3305 | True |
| SOLUSDT | 2024-03 | 44327 | 0.8473 | 0.5155 | 0.5433 | 0.304 | True |
| SOLUSDT | 2024-08 | 44246 | 0.9143 | 0.5962 | 0.6095 | 0.3049 | True |
| SOLUSDT | 2024-12 | 44347 | 0.9617 | 0.649 | 0.653 | 0.3087 | True |
| DOGEUSDT | 2021-02 | 40144 | 0.8382 | 0.5387 | 0.5516 | 0.2867 | True |
| DOGEUSDT | 2021-09 | 42891 | 0.9289 | 0.5543 | 0.5771 | 0.3518 | True |
| DOGEUSDT | 2022-06 | 43010 | 0.9508 | 0.5585 | 0.567 | 0.3838 | True |
| DOGEUSDT | 2022-11 | 40118 | 0.7494 | 0.5572 | 0.5563 | 0.1922 | True |
| DOGEUSDT | 2023-05 | 43026 | 0.8959 | 0.4752 | 0.4752 | 0.4207 | True |
| DOGEUSDT | 2023-10 | 44428 | 0.9505 | 0.5052 | 0.5059 | 0.4445 | True |
| DOGEUSDT | 2024-03 | 44361 | 0.9088 | 0.598 | 0.6111 | 0.2978 | True |
| DOGEUSDT | 2024-08 | 44073 | 0.9716 | 0.6511 | 0.6507 | 0.3206 | True |
| DOGEUSDT | 2024-12 | 44240 | 0.9529 | 0.6578 | 0.6641 | 0.2889 | True |
| ADAUSDT | 2021-02 | 40065 | 0.9228 | 0.6122 | 0.6192 | 0.3035 | True |
| ADAUSDT | 2021-09 | 42923 | 0.964 | 0.6397 | 0.6482 | 0.3157 | True |
| ADAUSDT | 2022-06 | 42900 | 0.9483 | 0.5496 | 0.5598 | 0.3885 | True |
| ADAUSDT | 2022-11 | 40055 | 0.8139 | 0.526 | 0.5244 | 0.2878 | True |
| ADAUSDT | 2023-05 | 43013 | 0.9225 | 0.4639 | 0.4639 | 0.4586 | True |
| ADAUSDT | 2023-10 | 44454 | 0.9389 | 0.4694 | 0.469 | 0.4695 | True |
| ADAUSDT | 2024-03 | 44189 | 0.9556 | 0.6311 | 0.6337 | 0.3219 | True |
| ADAUSDT | 2024-08 | 44502 | 0.9744 | 0.5902 | 0.5907 | 0.3838 | True |
| ADAUSDT | 2024-12 | 44333 | 0.9687 | 0.6734 | 0.6755 | 0.2932 | True |
| GALAUSDT | 2021-09 | 17684 | 0.9244 | 0.5581 | 0.5799 | 0.3446 | True |
| GALAUSDT | 2022-06 | 42999 | 0.9476 | 0.5431 | 0.5568 | 0.3908 | True |
| GALAUSDT | 2022-11 | 40141 | 0.7712 | 0.4723 | 0.4672 | 0.2989 | True |
| GALAUSDT | 2023-05 | 43035 | 0.9251 | 0.5028 | 0.5023 | 0.4223 | True |
| GALAUSDT | 2023-10 | 44371 | 0.942 | 0.5144 | 0.5158 | 0.4261 | True |
| GALAUSDT | 2024-03 | 44345 | 0.922 | 0.5989 | 0.6053 | 0.3167 | True |
| GALAUSDT | 2024-08 | 44526 | 0.9768 | 0.5903 | 0.5921 | 0.3847 | True |
| GALAUSDT | 2024-12 | 44340 | 0.9748 | 0.6477 | 0.6491 | 0.3258 | True |
| FRONTUSDT | 2023-10 | 44452 | 0.9231 | 0.4718 | 0.4747 | 0.4483 | True |
| FRONTUSDT | 2024-03 | 44559 | 0.8592 | 0.5612 | 0.566 | 0.2932 | True |
| FRONTUSDT | 2024-08 | 32136 | 0.9532 | 0.5769 | 0.5773 | 0.3759 | True |
| 1000BONKUSDT | 2024-03 | 44507 | 0.8098 | 0.5253 | 0.5354 | 0.2743 | True |
| 1000BONKUSDT | 2024-08 | 44504 | 0.9525 | 0.5761 | 0.5811 | 0.3714 | True |
| 1000BONKUSDT | 2024-12 | 44509 | 0.9486 | 0.5814 | 0.5866 | 0.362 | True |

Skipped/absent combos (late listings / thin months): GALAUSDT 2021-02 (absent), FRONTUSDT 2021-02 (absent), FRONTUSDT 2021-09 (absent), FRONTUSDT 2022-06 (absent), FRONTUSDT 2022-11 (absent), FRONTUSDT 2023-05 (absent), FRONTUSDT 2024-12 (absent), 1000BONKUSDT 2021-02 (absent), 1000BONKUSDT 2021-09 (absent), 1000BONKUSDT 2022-06 (absent), 1000BONKUSDT 2022-11 (absent), 1000BONKUSDT 2023-05 (absent), 1000BONKUSDT 2023-10 (absent).

## (B) definitive raw-day recompute — 4 days, ALL PASS

Fresh daily aggTrades archives re-downloaded from data.binance.vision, independently
re-aggregated per minute with the explicit [open, open+60s) window, compared to the lake's
stored TFI (x_7) / signed-count-imbalance (x_8) — BOUNDED features, so the lag-0 comparison is
float32-exact (atol 2e-06), and a ±1-minute shift must collapse the correlation.
Archives are cached in `raw/binance/_audit_cache/` (gitignored — never redistributed); their
digests below are the audit trail.

| symbol | day | n_minutes_compared | max_abs_diff_tfi | max_abs_diff_count_imb | r_lag0 | r_lag_plus1 | r_lag_minus1 | pass |
|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 2021-06-15 | 1439 | 2.78e-08 | 1.49e-08 | 1.0 | 0.0493 | 0.0493 | True |
| ETHUSDT | 2022-11-09 | 1439 | 1.95e-08 | 1.44e-08 | 1.0 | 0.1151 | 0.1151 | True |
| SOLUSDT | 2023-12-01 | 1438 | 2.97e-08 | 2.91e-08 | 1.0 | 0.0618 | 0.0618 | True |
| DOGEUSDT | 2024-06-03 | 1424 | 2.96e-08 | 2.96e-08 | 1.0 | 0.0465 | 0.0465 | True |

| archive | sha256 |
|---|---|
| BTCUSDT-aggTrades-2021-06-15.zip | `1bf901e1cc8dd5653f14c124fe9d17ba2a99459a241b1c164a224e2e359c528b` |
| ETHUSDT-aggTrades-2022-11-09.zip | `0bc1bac799c04ad623c1c25aed0375d93ae6a3fa642ca85f1a9fe494e9047672` |
| SOLUSDT-aggTrades-2023-12-01.zip | `c61302961200e5957bba0b60b4bc264c5759d72ab8c2150da055f4494ca498c1` |
| DOGEUSDT-aggTrades-2024-06-03.zip | `30f3ebe5e038ba3dd1198c4bc5731dfa5047c59ddb050571badbba389c3fe0c0` |

**Reading:** part A shows the cross-source correlation peaking at lag 0 on every sample across
years and archive eras; part B shows the stored aggTrades features are bit-consistent (float32)
with an independent [open, open+60s) recompute on raw archives and that a one-minute shift
destroys the match — the microstructure block is aligned to the bar it claims to describe.
Machine-readable copy: `runs_manifest/m6_micro_alignment.json`.
