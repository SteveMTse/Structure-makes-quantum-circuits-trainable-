# Structure makes quantum circuits trainable [![DOI](https://zenodo.org/badge/1455684352.svg)](https://doi.org/10.5281/zenodo.24178165)

Code and data for the paper *Structure makes quantum circuits trainable* by Wenbo Xie, Ananth Grama and Wojciech Szpankowski (Purdue University).

| Folder | Contents |
|---|---|
| [`sun_large`](sun_large) | CUDA C++ implementation of the sampling-based variance estimator (Algorithm 1 of the paper) for the $\mathrm{SU}(n)$ compound ansatz, simulated on NVIDIA GPUs |
| [`Source_Data_CSV`](Source_Data_CSV) | Numerical values plotted in Figures 3 and 4 of the paper |
| `sun_large.zip`, `Source_Data_CSV.zip` | The same two folders as zip archives |

## `sun_large`: the sampling-based variance estimator on GPUs

### What the program computes

`sun_experiment` runs Algorithm 1 of the paper on simulated circuits. It estimates the variance of the partial derivative of the cost function with respect to the parameter of one gate.

- **Circuit.** The circuit acts on $n$ qubits and consists of $L = 2n$ layers. Each layer applies, one after another, the $K = n^2 - 1$ generators of $\mathfrak{su}(n)$ in the compound (number-conserving fermionic) representation, each normalized to spectral norm 1:
  - for every pair of qubits $i < j$, the generators $-\tfrac12\left(X_i Z\cdots Z X_j + Y_i Z\cdots Z Y_j\right)$ and $-\tfrac12\left(Y_i Z\cdots Z X_j - X_i Z\cdots Z Y_j\right)$, where $Z\cdots Z$ acts on the qubits between $i$ and $j$;
  - for every $r = 1, \dots, n-1$, the generator $-\tfrac12\left(Z_r - Z_{r+1}\right)$.

  Every gate is $e^{i\theta \tilde H}$ for its generator $\tilde H$, with the angle $\theta$ drawn uniformly from $[-\pi, \pi)$. The gate whose parameter is differentiated sits in the middle of the circuit.
- **Input state and observable.** The input state is $\lvert + \rangle^{\otimes n}$. The observable is $M = X^{\otimes k}$ on the first $k$ qubits, so that $k = 2$ gives $M = X^{(1)} X^{(2)}$. The differentiated generator is chosen with `--target-j`; the default, `0`, is $-\tfrac12\left(X^{(1)} X^{(2)} + Y^{(1)} Y^{(2)}\right)$.
- **Estimator.** The program uses the finite-difference form of Algorithm 1. Each shot returns $\varepsilon^{-2}(-1)^{b_1 + b_2} O_A O_B$, where $\varepsilon = \sqrt{3a/32}$ is the finite-difference step and $a$ is the target accuracy (`--accuracy`). The mean of the shots is $\left[\left(f(+\varepsilon) - f(-\varepsilon)\right)/(2\varepsilon)\right]^2$, averaged over the random circuits, where $f(\pm\varepsilon)$ is the expectation value of $M$ with the differentiated gate shifted by $\pm\varepsilon$. This mean approximates the variance of the partial derivative.
- **Result.** The program reports the estimate, its standard error, and whether the standard error is at most $a/2$.

The two copies of the state in Algorithm 1 are never entangled, because every operator in the algorithm is a tensor product. The program therefore simulates two state vectors of $2^n$ amplitudes instead of one of $4^n$.

### Requirements

- An NVIDIA GPU and the CUDA Toolkit (`nvcc`).
- A C++17 host compiler: GCC or Clang on Linux, MSVC on Windows.
- GNU make.

For $n$ up to about 12 in FP32 (11 in FP64), depending on the shared memory of the GPU, each circuit is simulated in the shared memory of one thread block. For larger $n$, up to 26, the state lives in global memory: $2^n$ amplitudes for each circuit, for example 8 MB at $n = 20$ in FP32. The number of circuits run in parallel is then chosen from the free GPU memory.

### Building

```sh
cd sun_large
make                                             # FP32 (default)
make PREC=fp64                                   # FP64
make HOST=msvc                                   # MSVC host compiler (the default on Windows)
make ARCH="-gencode arch=compute_89,code=sm_89"  # name the GPU architecture explicitly
make config                                      # show the detected settings
```

The build produces `sun_experiment.exe` on Windows and `sun_experiment` on Linux. FP32 is the default because it is about 30 times faster on GeForce GPUs, whose FP64 throughput is 1/64 of their FP32 throughput.

### Running

On Windows:

```sh
.\sun_experiment.exe --n 15 --k 2 --accuracy 0.001 --shots 5e14
```

On Linux:

```sh
./sun_experiment --n 15 --k 2 --accuracy 0.001 --shots 5e14
```

This command:

- simulates $n = 15$ qubits: $K = 224$ generators in each of $L = 30$ layers, so 6,720 gates in each circuit;
- measures $M = X^{(1)} X^{(2)}$ ($k = 2$);
- sets the target accuracy to $a = 10^{-3}$, so that the finite-difference step is $\varepsilon = \sqrt{3a/32} \approx 9.68 \times 10^{-3}$ and the run reports whether the standard error is at most $a/2 = 5 \times 10^{-4}$;
- uses $5 \times 10^{14}$ shots in total, drawn as $5 \times 10^{7}$ shots on each of $10^{7}$ random circuits;
- uses FP32, the default seed 20240601 and every available GPU. Since $n = 15 > 12$, the circuits are simulated in global memory.

The run can be stopped early with Ctrl+C or with `--stop-stderr`; the summary then covers the circuits completed so far.

### Options

| Option | Meaning | Default |
|---|---|---|
| `--n INT` | Number of qubits, from 2 to 26 | 10 |
| `--k INT` | The observable is $X^{\otimes k}$ on the first $k$ qubits | 2 |
| `--target-j INT` | Index of the differentiated generator, from 0 to $n^2 - 2$ | 0 |
| `--accuracy FLOAT` | Target accuracy $a$; sets the finite-difference step $\varepsilon = \sqrt{3a/32}$ | $1/n^2$ |
| `--shots FLOAT` | Total number of shots; `0` uses the analytic bound $8 \ln(2/\delta) / \left((a/2)^2 \varepsilon^4\right)$ | 0 |
| `--delta FLOAT` | Failure probability $\delta$ in the analytic bound | 0.999 |
| `--circuits LONG` | Number of simulated random circuits; `0` chooses it automatically (at most $10^7$ unless `--shots-per-circuit` is set) | 0 |
| `--shots-per-circuit LONG` | Shots drawn on each circuit; `0` divides the total number of shots evenly over the circuits, `1` gives fully independent shots | 0 |
| `--estimator STR` | `sampled` (measurement outcomes) or `exact` (exact expectation values of $M$, available only in simulation) | `sampled` |
| `--ansatz STR` | `full`, or `legacy` (the gate schedule of an earlier Python version, kept for comparison) | `full` |
| `--edges STR` | `all` pairs $i < j$, or `chain` (nearest-neighbour pairs only, which still generate $\mathfrak{su}(n)$) | `all` |
| `--path STR` | `auto`, `shared` (only for $n \le 12$) or `global` | `auto` |
| `--batch INT` | Circuits per kernel launch on the global-memory path | chosen from free memory |
| `--stop-stderr FLOAT` | Stop once the standard error falls below this value | off |
| `--gpus INT` | Use at most this many GPUs | all |
| `--seed ULONG` | Random seed | 20240601 |
| `--out PATH` | Output CSV file | `sun_experiment_YYYYMMDD_HHMMSS.csv` |
| `--selftest` | Print $f(+\varepsilon)$ and $f(-\varepsilon)$ for one circuit | off |
| `--log STR` | `legacy` (the console format of the earlier Python version) or `modern` (adds a progress line) | `legacy` |
| `--sign STR` | `positive` (report the variance), or `legacy` (keep the leading minus sign of the earlier Python version) | `positive` |

### Output

The program prints a summary to the console and writes a CSV file. Lines of the CSV file that start with `#` record the settings of the run: $n$, $k$, the differentiated generator, $K$, $L$, the numbers of gates on each side of the differentiated gate, the target accuracy, $\delta$, $\varepsilon$, the analytic shot bound, the estimator, the numbers of circuits and of shots per circuit, the precision and the seed. They are followed by a table with the columns `metric,value`:

| Metric | Meaning |
|---|---|
| `circuits` | Number of random circuits simulated |
| `effective_shots` | Total number of shots |
| `n_plus` | Number of shots with outcome $+\varepsilon^{-2}$ |
| `mean` | Estimate of the variance of the partial derivative |
| `stderr` | Standard error of the estimate |
| `var_circuit` | Variance of the per-circuit values across circuits |
| `wall_seconds` | Running time in seconds |

## `Source_Data_CSV`: data of Figures 3 and 4

| File | Contents |
|---|---|
| `Fig3.csv` | Figure 3, panels (a) and (b), $n = 2, \dots, 20$: exact variance, lower bound of Equation (12), upper bound of Equation (11) and trivial bound |
| `Fig4.csv` | Figure 4, $n = 4, \dots, 20$: exact variance, its in-algebra part, lower bound of Equation (12), estimates of Algorithm 1 with their error bars, and numbers of runs |
| `README.txt` | Definition of every column and the closed-form expressions behind it |
