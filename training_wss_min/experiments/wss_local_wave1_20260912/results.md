# wss_local_wave1_20260912 结果（自动生成 2026-09-12 14:00）

单 seed 1234、已暴露 test34；参照 = 同期对照 X0（同配置重跑）与历史 C1；单 seed 对单 seed 的 95% 带约 ±0.034 物理 R²_cb / ±0.009 归一化。只作筛选，不作显著性或泛化结论。

| 臂 | 变化 | ckpt | Pa R²_cb | Δ vs X0 | Δ vs C1 | norm R²_cb | Δ vs X0 | MAE Pa | case P10 | high-WSS R² | top10 比 | p99 比 | IoU | 负R² |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 历史 | 参照 | best | 0.6575 | — | — | 0.8410 | — | 1.845 | 0.5050 | 0.300 | 0.639 | 0.624 | 0.4698 | 0 |
| X0 | C1 contemporaneous control (identical config, fresh run) | best | 0.6173 | +0.0000 | -0.0402 | 0.8347 | +0.0000 | 1.909 | 0.4825 | 0.223 | 0.611 | 0.590 | 0.4462 | 0 |
| X0 | C1 contemporaneous control (identical config, fresh run) | last | 0.6201 | +0.0028 | -0.0374 | 0.8339 | -0.0009 | 1.913 | 0.4782 | 0.239 | 0.625 | 0.603 | 0.4470 | 0 |
| X1 | P4 weight EMA 0.999 (extra ckpt_ema evaluation) | best | 0.6297 | +0.0124 | -0.0278 | 0.8364 | +0.0017 | 1.908 | 0.4587 | 0.209 | 0.604 | 0.581 | 0.4544 | 0 |
| X1 | P4 weight EMA 0.999 (extra ckpt_ema evaluation) | last | 0.6328 | +0.0155 | -0.0247 | 0.8357 | +0.0010 | 1.907 | 0.4626 | 0.218 | 0.607 | 0.583 | 0.4556 | 0 |
| X1 | P4 weight EMA 0.999 (extra ckpt_ema evaluation) | ema | 0.6298 | +0.0125 | -0.0277 | 0.8360 | +0.0012 | 1.908 | 0.4622 | 0.206 | 0.600 | 0.576 | 0.4549 | 0 |
| X2 | F1 bend-referenced circumferential angle (+ upstream 2D/5D lags, torsion) | best | 0.6510 | +0.0337 | -0.0065 | 0.8433 | +0.0086 | 1.815 | 0.5046 | 0.307 | 0.643 | 0.636 | 0.4636 | 0 |
| X2 | F1 bend-referenced circumferential angle (+ upstream 2D/5D lags, torsion) | last | 0.6522 | +0.0349 | -0.0053 | 0.8436 | +0.0089 | 1.808 | 0.5057 | 0.317 | 0.651 | 0.646 | 0.4654 | 0 |
| X3 | F2 bifurcation-referenced angles (carina side, plane, angle, sibling radius) | best | 0.6459 | +0.0286 | -0.0116 | 0.8378 | +0.0030 | 1.866 | 0.5154 | 0.291 | 0.649 | 0.632 | 0.4531 | 0 |
| X3 | F2 bifurcation-referenced angles (carina side, plane, angle, sibling radius) | last | 0.6436 | +0.0263 | -0.0139 | 0.8366 | +0.0019 | 1.876 | 0.5099 | 0.285 | 0.647 | 0.628 | 0.4515 | 0 |
| X4 | F3 upstream history (s/D, upstream radius extrema, upstream max kappa*R) | best | 0.6419 | +0.0246 | -0.0156 | 0.8383 | +0.0035 | 1.895 | 0.4750 | 0.257 | 0.630 | 0.612 | 0.4563 | 1 |
| X4 | F3 upstream history (s/D, upstream radius extrema, upstream max kappa*R) | last | 0.6392 | +0.0219 | -0.0183 | 0.8377 | +0.0029 | 1.898 | 0.4746 | 0.251 | 0.627 | 0.608 | 0.4561 | 1 |
| X5 | F6 Murray flow-share prior (log Q share, log tau0) | best | 0.7178 | +0.1005 | +0.0603 | 0.8559 | +0.0212 | 1.733 | 0.6430 | 0.422 | 0.701 | 0.691 | 0.5068 | 0 |
| X5 | F6 Murray flow-share prior (log Q share, log tau0) | last | 0.7134 | +0.0961 | +0.0559 | 0.8560 | +0.0213 | 1.740 | 0.6331 | 0.403 | 0.683 | 0.672 | 0.5085 | 0 |
| X6 | P5 population prior ln WSS (train138, leave-one-out) | best | 0.6327 | +0.0154 | -0.0248 | 0.8372 | +0.0025 | 1.880 | 0.4941 | 0.248 | 0.631 | 0.628 | 0.4617 | 0 |
| X6 | P5 population prior ln WSS (train138, leave-one-out) | last | 0.6341 | +0.0168 | -0.0234 | 0.8367 | +0.0019 | 1.884 | 0.5010 | 0.256 | 0.639 | 0.636 | 0.4609 | 0 |
| X7 | S1a query patch offsets in the query tangent frame | best | 0.6548 | +0.0375 | -0.0027 | 0.8389 | +0.0042 | 1.873 | 0.4506 | 0.272 | 0.633 | 0.623 | 0.4655 | 0 |
| X7 | S1a query patch offsets in the query tangent frame | last | 0.6546 | +0.0373 | -0.0029 | 0.8391 | +0.0043 | 1.873 | 0.4516 | 0.274 | 0.635 | 0.623 | 0.4655 | 0 |
| X8 | S1b query-conditioned attention pooling over the patch | best | 0.6292 | +0.0119 | -0.0283 | 0.8380 | +0.0032 | 1.890 | 0.4727 | 0.230 | 0.603 | 0.579 | 0.4628 | 0 |
| X8 | S1b query-conditioned attention pooling over the patch | last | 0.6289 | +0.0116 | -0.0286 | 0.8378 | +0.0031 | 1.892 | 0.4712 | 0.228 | 0.603 | 0.580 | 0.4621 | 0 |
| X9 | S1c tangent frame + attention + K=32 patch | best | 0.6419 | +0.0246 | -0.0156 | 0.8411 | +0.0064 | 1.878 | 0.4839 | 0.266 | 0.635 | 0.626 | 0.4721 | 0 |
| X9 | S1c tangent frame + attention + K=32 patch | last | 0.6389 | +0.0216 | -0.0186 | 0.8407 | +0.0060 | 1.882 | 0.4920 | 0.259 | 0.634 | 0.624 | 0.4702 | 0 |
| X10 | T1 tangent-plane WSS direction auxiliary head (cosine loss 0.1) | best | 0.6363 | +0.0190 | -0.0212 | 0.8411 | +0.0064 | 1.863 | 0.4543 | 0.246 | 0.637 | 0.620 | 0.4589 | 0 |
| X10 | T1 tangent-plane WSS direction auxiliary head (cosine loss 0.1) | last | 0.6371 | +0.0199 | -0.0204 | 0.8413 | +0.0065 | 1.862 | 0.4478 | 0.247 | 0.637 | 0.621 | 0.4614 | 0 |
| X11 | S3 centreline section-token context replacing the per-case FiLM vector | best | 0.6703 | +0.0530 | +0.0128 | 0.8457 | +0.0110 | 1.833 | 0.5778 | 0.338 | 0.678 | 0.689 | 0.4571 | 0 |
| X11 | S3 centreline section-token context replacing the per-case FiLM vector | last | 0.6711 | +0.0538 | +0.0136 | 0.8456 | +0.0108 | 1.831 | 0.5778 | 0.342 | 0.680 | 0.692 | 0.4572 | 0 |
| X12 | S5 mixture-of-experts head gated by query input features (4 experts) | best | 0.6517 | +0.0344 | -0.0058 | 0.8397 | +0.0050 | 1.829 | 0.4812 | 0.306 | 0.646 | 0.627 | 0.4711 | 0 |
| X12 | S5 mixture-of-experts head gated by query input features (4 experts) | last | 0.6513 | +0.0340 | -0.0062 | 0.8392 | +0.0045 | 1.832 | 0.4709 | 0.304 | 0.648 | 0.628 | 0.4679 | 0 |
| X13a | P1 stage 1: all-frame pretraining with phase features (peak prob 1/9) | best | 0.6256 | +0.0083 | -0.0319 | 0.8290 | -0.0057 | 1.929 | 0.5008 | 0.229 | 0.597 | 0.565 | 0.4319 | 0 |
| X13a | P1 stage 1: all-frame pretraining with phase features (peak prob 1/9) | last | 0.6276 | +0.0103 | -0.0299 | 0.8287 | -0.0060 | 1.938 | 0.4891 | 0.228 | 0.603 | 0.577 | 0.4319 | 0 |
| X13b | P1 stage 2: peak-frame fine-tuning from X13a (150 epochs, lr 5e-4) | best | 0.6429 | +0.0256 | -0.0146 | 0.8409 | +0.0062 | 1.855 | 0.4997 | 0.291 | 0.652 | 0.632 | 0.4558 | 0 |
| X13b | P1 stage 2: peak-frame fine-tuning from X13a (150 epochs, lr 5e-4) | last | 0.6451 | +0.0278 | -0.0124 | 0.8408 | +0.0060 | 1.855 | 0.4799 | 0.287 | 0.651 | 0.634 | 0.4529 | 0 |
| X15 | F-all: F1 + F2 + F3 + F6 appended together | best | 0.7118 | +0.0945 | +0.0543 | 0.8627 | +0.0280 | 1.698 | 0.6332 | 0.397 | 0.693 | 0.673 | 0.5334 | 0 |
| X15 | F-all: F1 + F2 + F3 + F6 appended together | last | 0.7122 | +0.0949 | +0.0547 | 0.8622 | +0.0275 | 1.702 | 0.6322 | 0.396 | 0.695 | 0.673 | 0.5339 | 0 |
| X16 | Combo: S1c + F-all + T1 direction head | best | 0.7140 | +0.0968 | +0.0566 | 0.8642 | +0.0295 | 1.720 | 0.5998 | 0.430 | 0.703 | 0.697 | 0.5294 | 0 |
| X16 | Combo: S1c + F-all + T1 direction head | last | 0.7145 | +0.0972 | +0.0570 | 0.8649 | +0.0301 | 1.714 | 0.6071 | 0.430 | 0.702 | 0.695 | 0.5324 | 0 |

## 分域物理 R²_cb（best）

| 臂 | AG | AAA | ILO |
|---|---:|---:|---:|
| C1 历史 | 0.6937 | 0.6342 | 0.6315 |
| X0 | 0.6482 | 0.5866 | 0.6010 |
| X1 | 0.6894 | 0.5838 | 0.5949 |
| X2 | 0.6703 | 0.6294 | 0.6406 |
| X3 | 0.6646 | 0.6248 | 0.6357 |
| X4 | 0.6997 | 0.6145 | 0.5967 |
| X5 | 0.7723 | 0.6957 | 0.6734 |
| X6 | 0.6619 | 0.5972 | 0.6217 |
| X7 | 0.7143 | 0.5897 | 0.6334 |
| X8 | 0.6731 | 0.6052 | 0.5954 |
| X9 | 0.6966 | 0.6152 | 0.5993 |
| X10 | 0.6729 | 0.5874 | 0.6267 |
| X11 | 0.7250 | 0.6398 | 0.6305 |
| X12 | 0.6715 | 0.6166 | 0.6502 |
| X13a | 0.6761 | 0.6190 | 0.5731 |
| X13b | 0.6675 | 0.6025 | 0.6398 |
| X15 | 0.7652 | 0.7004 | 0.6611 |
| X16 | 0.7586 | 0.7270 | 0.6559 |

## 逐例配对（best，相对 X0 的 Pa R² 变化）

| 臂 | 变好例数/34 | 中位 Δ | 最差 8 例（按 X0）均值 Δ |
|---|---:|---:|---:|
| X1 | 17/34 | +0.0012 | -0.0183 |
| X2 | 22/34 | +0.0259 | +0.0249 |
| X3 | 18/34 | +0.0090 | +0.0379 |
| X4 | 15/34 | -0.0021 | -0.0264 |
| X5 | 28/34 | +0.0757 | +0.1551 |
| X6 | 19/34 | +0.0070 | +0.0340 |
| X7 | 21/34 | +0.0214 | -0.0401 |
| X8 | 20/34 | +0.0031 | +0.0172 |
| X9 | 21/34 | +0.0156 | +0.0110 |
| X10 | 23/34 | +0.0241 | +0.0204 |
| X11 | 25/34 | +0.0509 | +0.0945 |
| X12 | 23/34 | +0.0192 | +0.0171 |
| X13a | 13/34 | -0.0160 | +0.0093 |
| X13b | 22/34 | +0.0102 | +0.0470 |
| X15 | 31/34 | +0.0884 | +0.1731 |
| X16 | 30/34 | +0.0945 | +0.1504 |

队列状态：complete；作业 14160；跳过的候选：T4 anisotropic local-difference loss (E6 interaction unresolved); F4 geodesic/HKS descriptors; F5 section-shape features (geometry candidate still training_allowed=false); F7 predicted wall-pressure gradient cascade; S2 windowed attention; S4 (s,theta) chart U-Net; S6 DiffusionNet branch; P2/P3 pretraining data; T3 hotspot cascade
