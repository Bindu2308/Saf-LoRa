"""EWMA smoothing-factor sensitivity (Layer B setup). Reports G_norm per alpha/severity."""
import statistics as st
import run_benchmark_layerb as B

ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.5]
SEVS = [0.0, 1.0, 2.0, 3.0, 5.0]

print(f"{'alpha':>6} " + " ".join(f"{s:>6.1f}dB" for s in SEVS) + "   mean")
for a in ALPHAS:
    B.EWMA_ALPHA = a
    row = []
    for sev in SEVS:
        rnd, orc, ew = [], [], []
        for seed in range(B.NUM_SEEDS):
            rounds = B.simulate_rounds(sev, seed=seed)
            rnd.append(B.run_policy("random", rounds, seed + 100000))
            orc.append(B.run_policy("oracle", rounds, seed + 100000))
            ew.append(B.run_policy("ewma", rounds, seed + 100000))
        r, o, e = st.mean(rnd), st.mean(orc), st.mean(ew)
        row.append((e - r) / (o - r))
    print(f"{a:6.2f} " + " ".join(f"{g:8.3f}" for g in row) + f"   {st.mean(row):.3f}")
