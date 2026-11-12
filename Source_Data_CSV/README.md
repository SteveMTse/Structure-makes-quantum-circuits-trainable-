Source data for "Structure makes quantum circuits trainable" (W. Xie, A. Grama, W. Szpankowski)
=====================================================================================================

Data behind the plotted figures of the latest version of the paper (revision of 23 October 2026).
Figures 1 and 2 are schematic illustrations and have no underlying data. The Supplementary Information has no figures.

Files
-----

- `Fig3.csv`: Figure 3, panels (a) and (b), $n = 2, \dots, 20$.
- `Fig4.csv`: Figure 4, $n = 4, \dots, 20$.
- `make_source_data.py`: Writes `Fig3.csv`, `Fig4.csv` and `Source_Data.xlsx` (one sheet for each figure, the closed forms as spreadsheet formulas):
  ```
  python3 make_source_data.py fig4_exact_rule_data.csv OUTPUT_FOLDER
  ```
  `fig4_exact_rule_data.csv` is in `Figure4_code.zip`. Requires Python 3 and openpyxl.

Figure 3 (Fig3.csv)
-------------------

Exact variance (solid line), Case-3 lower bound of Equation (12) (squares), Case-3 upper bound of Equation (11) (circles)
and trivial bound $4 (\lVert H \rVert_\infty)^2 (\lVert M \rVert_\infty)^2$ (dashed line), on a logarithmic scale. Both bounds are evaluated after
compression to the irreducible sub-representation that carries $\rho$. The squares are drawn only where the lower bound is
positive: $n \geq 5$ in panel (a) and $n \geq 3$ in panel (b). Columns `a_*` belong to panel (a), columns `b_*` to panel (b).

(a) Free fermions (Supplementary Section S7.4.1): $\rho = \lvert 0^n \rangle\langle 0^n \rvert$, $M = Z_1 = -i \gamma_1 \gamma_2$, $H = \gamma_1 \gamma_2 / 2$.

- `a_exact_variance`: $4(n-1) / \left(n (2n-1)^2\right)$
- `a_lower_bound`: equal to the exact variance for $n \geq 8$, and $(n-1)(n-4) / \left(n (2n-1)^2\right)$ for $n < 8$
- `a_upper_bound`: $\left[1 + (\sqrt{2}/n) \sum_{j=1}^{\lfloor n/2 \rfloor} \lvert 8 j^2 - n \rvert \sqrt{C(2n, n-2j) / C(2n, n)}\right] / (2n-1)$, evaluated on the even-parity sector
- `a_trivial_bound`: $1$, since $\lVert H \rVert_\infty = 1/2$ and $\lVert M \rVert_\infty = 1$

(b) $\mathrm{SU}(2)$ collective spin (Supplementary Section S7.3): $\rho$ on the spin-$n/2$ sector, $M = S_X + S_Y + S_Z$, $H = -i S_Z$.

- `b_exact_variance`: $n^2 / 6$
- `b_lower_bound`: equal to the exact variance for $n \geq 4$, $3/4$ at $n = 3$, and $0$ at $n = 2$
- `b_upper_bound`: $n^2 / \left(2 \sqrt{2n+1}\right) \sum_{\lambda=0}^{n} \sqrt{2 \lambda + 1} \, \lvert \lambda (\lambda + 1) - n (n+2)/2 \rvert$, evaluated on the spin-$n/2$ sector
- `b_trivial_bound`: $3 n^4 / 4$, since $\lVert H \rVert_\infty = n/2$ and $\lVert M \rVert_\infty = \sqrt{3}\, n/2$

Figure 4 (Fig4.csv)
-------------------

Setting (Supplementary Section S7.5): $G = \mathrm{SU}(n)$ acting on $n$ qubits through the compound (number-conserving fermionic)
representation; $\rho = \lvert + \rangle\langle + \rvert^{\otimes n}$, $M = X_1 X_2$, $H = -(i/2)(X_1 X_2 + Y_1 Y_2)$; $U_-$ and $U_+$ Haar-distributed on $\Pi(\mathrm{SU}(n))$.

- `exact_variance`: $(2n^3 - 3n - 2) / \left(n^2 (n-1) (n+1)^2\right)$ (solid line)
- `in_algebra_part`: $n / \left((n-1) (n+1)^2\right)$ (dashed line). The figure draws both lines for real $n$ between 4 and 20; the file gives their values at integer $n$.
- `case3_lower_bound`: Case-3 lower bound of Equation (12), as drawn in the figure (squares)
- `algorithm_estimate`: mean of $N$ runs of Algorithm 1 in its exact form (four-term shift rule, $\gamma = 2$), every run simulated by `fig4_exact_rule_data.py` with seed 2026 (circles)
- `error_bar`: certified accuracy $0.999/n^2$, guaranteed with probability at least $1 - \delta$, $\delta = 10^{-3}$
- `runs_exact_rule`: $N = \left\lceil 32 \ln(2/\delta) / (0.999/n^2)^2 \right\rceil$
- `runs_finite_difference`: number of runs that the finite-difference form would need for the same guarantee
