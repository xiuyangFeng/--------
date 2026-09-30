# T2 five-seed IND ensemble (deployment-grade) vs X5Dcap five-seed ensemble, 91 recovered units

| | X5Dcap 5-seed | T2 5-seed | Δ |
|---|---:|---:|---:|
| R²_cb (Pa, 91) | 0.7644 | 0.7758 | +0.0115 |
| R²_cb AG (n=10) | 0.7804 | 0.7789 | -0.0015 |
| R²_cb AAA (n=6) | 0.8649 | 0.8539 | -0.0110 |
| R²_cb ILO (n=75) | 0.7568 | 0.7707 | +0.0139 |
| jet subset R²_cb (p99>40 Pa, n=5) | 0.6951 | 0.7233 | +0.0282 |
| top10 SSE share | 0.6543 | 0.5947 | -0.0596 |
| top10 under-estimation fraction | 0.7263 | 0.6579 | -0.0684 |
| top10 pred/true mean ratio | 0.8283 | 0.8841 | +0.0559 |

per-unit ensemble R² delta: mean -0.0012, median -0.0008, improved 43/91; worst 3: ILO/WANG_LI_MIN-0/before -0.075, AG/fast/PENG_JI_MING -0.056, ILO/DONG_KE_QIN-0/before -0.056; best 3: ILO/LIU_BAO_JUN-0/after +0.048, ILO/SUN_XU_XIA-1/after +0.047, ILO/SUN_CHUN_PU-0/after +0.038

single-seed R²_cb: s1234 0.7466→0.7535 (+0.0069), s7 0.7330→0.7533 (+0.0203), s2025 0.7417→0.7500 (+0.0082), s11 0.7527→0.7605 (+0.0078), s2026 0.7380→0.7466 (+0.0086)

reference: v5.2 readout X5Dcap five-seed ensemble 0.764; frozen X5D_v51 five-seed on the same 91 units 0.749 (before end-radius rebuild).
