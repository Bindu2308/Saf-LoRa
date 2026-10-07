# missing_reports_asymmetry_summary.csv

Validates the informative-report-loss bias (paper Section VII) across the
full gateway-asymmetry range already used in Section VI-C (Table "Asymmetry
sweep", dp = 0.0-0.4), rather than the single dp=0.2 point tested in
`missing_reports_summary.csv`.

## Generation
    NSEEDS=500 python3 sim_missing_reports_asymmetry.py

54 rows = 9 asymmetry levels (dp) x 2 coupling levels (rho = 1.0 uplink
predicts downlink, rho = 0.0 independent) x 3 report-loss settings.

## Columns
| column                     | meaning |
|-----------------------------|---------|
| dp                          | gateway-2 stationary GOOD-probability deficit (asymmetry), same grid as Table XI |
| rho                         | uplink-downlink coupling (1.0 = uplink predicts downlink exactly, 0.0 = independent) |
| setting                     | complete / random-missing / informative-missing report-loss regime |
| n_seeds                     | paired seeds this row is averaged over |
| report_pct                  | % of ACKs with an accepted receipt report under `random` policy |
| true_delivery_pct           | actual delivery rate (ground truth, all ACKs) |
| reported_estimate_pct       | delivery estimated from reported ACKs only (what a real deployment would measure) |
| bias_pp                     | reported_estimate_pct - true_delivery_pct, in percentage points |
| snr_greedy_loss_pct ... oracle_pct | mean delivery (%) per policy under this setting |
| adaptive_minus_ewma_pp      | paired difference, SAF-LoRa adaptive fusion minus EWMA |
| adaptive_minus_ewma_ci95    | 95% CI half-width on that paired difference |

## Headline finding
The informative-loss bias grows with asymmetry (~+4.9pp at dp=0 to ~+6.8pp
at dp=0.4) but the paired policy comparison (adaptive_minus_ewma_pp) shifts
by at most ~0.5-0.9pp across the same range -- roughly an order of
magnitude smaller than the absolute bias, at every asymmetry level tested.
This confirms the bias affects absolute delivery levels, not the
cross-policy comparisons the paper draws.
