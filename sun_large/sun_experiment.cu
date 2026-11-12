/* =============================================================================
 *  sun_experiment.cu  --  SU(n) randomized-tester experiment, CUDA C++
 *
 *  Build:  make                 (FP32, default)
 *          make PREC=fp64
 *  Run:    ./sun_experiment --n 10 --k 2 --shots 1e12
 *
 * =============================================================================
 *  WHAT CHANGED RELATIVE TO THE ORIGINAL PYTORCH SCRIPT
 * =============================================================================
 *
 *  1. STATE VECTORS, NOT DENSITY MATRICES; THE TWO COPIES NEVER ENTANGLE.
 *     rho = |psi><psi| is pure and the algorithm only ever needs
 *     diag(V^dag W rho W^dag V) = |V^dag W|psi>|^2, so the N x N matrices never
 *     need to exist. Moreover every operator acting on rho^{otimes 2} is a
 *     tensor product -- (U+)^{otimes 2}, (U-^dag)^{otimes 2}, U_eps = E (x) I
 *     and V_eps = I (x) E -- so rho_out stays a product state and the 4^n
 *     two-copy object is exactly two N-vectors (2 x 1024 rather than 2^20 at
 *     n = 10). Since (U+)^{otimes 2} rho^{otimes 2} (U+^dag)^{otimes 2} =
 *     (U+ rho U+^dag)^{otimes 2}, U+ is run once and the state then duplicated.
 *
 *  2. NO EIGENDECOMPOSITION -- CLOSED-FORM PAULI ROTATIONS.
 *     Every generator is a sum of two COMMUTING Pauli strings that each square
 *     to the identity:
 *         Hx = -1/4 (X_i Z..Z X_j + Y_i Z..Z Y_j)      [P1 P2 = -Z_i Z_j]
 *         Hy = -1/4 (Y_i Z..Z X_j - X_i Z..Z Y_j)      [Q1 Q2 = +Z_i Z_j]
 *         Hz = -1/4 (Z_r - Z_{r+1})
 *     Since P1P2 is a non-identity Pauli its spectrum is {+1,-1}, so
 *     H^2 = 1/8 (I +- P1P2) has spectrum {0, 1/4} and ||H||_2 = 1/2 EXACTLY.
 *     The normalised generator is therefore -(P1 -+ P2)/2 and
 *         exp(i theta H~) = [cos(t) I + i a1 sin(t) P1][cos(t) I + i a2 sin(t) P2]
 *     with t = theta/2 and a = -+1. A Pauli rotation on a state vector mixes the
 *     amplitude pairs (c, c^flipmask): O(N) per gate, not O(N^2). Together with
 *     (1) this is ~1000x fewer flops per gate and it removes cuSOLVER and the
 *     eigenvector tables (335 MB/GPU at n=9) entirely.
 *     The commutation and non-degeneracy facts above are re-verified
 *     symbolically at startup for every generator (verify_algebra()).
 *
 *  3. THE WHOLE STATE LIVES IN SHARED MEMORY.
 *     One block per circuit; all L*K gates run without touching global memory.
 *     Amplitudes are stored SoA with a (c >> 5) skew so the strided pair access
 *     c, c^flip is bank-conflict free.
 *
 *  4. NO MEASUREMENT EIGENBASIS.
 *     M is a Pauli string, so its eigenvalues are exactly +-1 and only <M> is
 *     needed, an O(N) reduction rather than an N x N GEMM plus a multinomial.
 *
 *  5. CLUSTER SAMPLING OVER SHOTS  (--shots-per-circuit, the big one).
 *     Every outcome equals -(1/eps^2) (-1)^{b1+b2} O_A O_B with all three
 *     factors in {+1,-1}, so a shot is a single +-1/eps^2 coin. Averaging the
 *     coin over b1,b2 and over the two measurement draws gives, exactly,
 *         P(outcome = +1/eps^2) = 1/2 + s Delta^2 / 8,  Delta = f(+eps) - f(-eps)
 *     with the sign s fixed by note 10 below.
 *     where f(x) = <psi| U+^dag E(x)^dag U- M U-^dag E(x) U+ |psi>. So one
 *     circuit determines p, and M shots on that circuit is one Binomial(M, p)
 *     draw. The estimator stays unbiased and
 *         Var(mean) = Var_circ * M / N  +  sigma_shot^2 / N
 *     against (Var_circ + sigma_shot^2)/N for fully independent shots. Since
 *     sigma_shot^2 = 1/eps^4 ~ 1.1e6 and Var_circ ~ 1e-4, M up to ~1e5-1e6 costs
 *     nothing statistically. Var_circ is measured empirically and the inflation
 *     factor is reported, so this is checked rather than assumed.
 *     --shots-per-circuit 1 recovers the fully independent estimator exactly.
 *
 *  6. THE TWO BRANCHES ARE EVALUATED DETERMINISTICALLY.
 *     The original drew b1,b2 at random, so half of all shots evaluated f(+eps)
 *     twice and learned nothing about f(-eps). Here each circuit computes both
 *     signs once. Same cost, same expectation, strictly less noise. (5) then
 *     folds the b1,b2 average in analytically.
 *
 *  7. EXACT ESTIMATOR (--estimator exact).
 *     A simulator can evaluate E[O|circuit] = <M> directly, which hardware
 *     cannot. The per-circuit value is then -(Delta/(2 eps))^2 with no
 *     measurement noise at all, dropping the variance from ~1/eps^4 to
 *     ~E[<M>^4]/eps^4. Typically a few times 1e4 circuits match what 1e12
 *     sampled shots buy. Use this unless the point is to study the
 *     shot-noise-limited estimator itself.
 *
 *  8. ANSATZ CORRECTION (default; --ansatz legacy restores the old behaviour).
 *     The Python's complete-layer loops iterate over range(j) where range(K)
 *     was meant, so at target_j = 0 they are empty, U_minus is the identity and
 *     both L and ell are inert. With range(K) the circuit is ell*K + j gates
 *     before the probed gate and (K-j) + (L-ell-1)*K from it onward, summing to
 *     L*K. --edges chain restricts each layer to nearest-neighbour (i,i+1)
 *     blocks, which still generate su(n) and cut the gate count by ~n/3.
 *
 * 10. SIGN CONVENTION (--sign; default 'positive').
 *     Algorithm "Sampling-Based Variance Estimator" returns
 *         + (1/eps^2) * (-1)^{b1+b2} * O,
 *     with no leading minus. The Python's -(1.0/(epsilon**2)) is a transcription
 *     error rather than a convention: O_A and O_B are independent draws, so
 *     E[O_A O_B | b1,b2] = f(eps_b1) f(eps_b2) and the four-term signed average
 *     collapses to a perfect square,
 *         E[(-1)^{b1+b2} O_A O_B] = [ (f(+eps) - f(-eps)) / 2 ]^2 ,
 *     making the Python's expectation <= 0 for EVERY circuit -- not merely
 *     negative on average. The estimand is a variance,
 *         Var = [ (f(+eps) - f(-eps)) / (2 eps) ]^2  ~  (d<M>/d theta_j)^2 >= 0.
 *     --sign legacy restores the minus for comparison against old runs.
 *
 *  9. THREADS, NOT PROCESSES; ON-DEVICE ACCUMULATION.
 *     One worker thread per GPU. Per-shot CSV rows are gone: in sampled mode
 *     the entire experiment is (total_shots, n_plus), two integers, so the
 *     original 4M-row file was 80 MB encoding one number. The CSV now holds
 *     metadata plus a progress log with running mean and standard error.
 *
 *  NOTE: the shared-memory design bounds n. FP32 reaches n = 12, FP64 n = 11 on
 *  a 100 KB/SM part; the limit is checked at startup.
 * =============================================================================
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

/* ------------------------------------------------------------------ config */

#ifndef TPB
#define TPB 256
#endif

#if defined(SUN_FP32)
typedef float real;
#define PREC_NAME "FP32 (complex64)"
#else
typedef double real;
#define PREC_NAME "FP64 (complex128)"
#endif

#define MAX_GEN 1024

/* Above this many shots per circuit the exact Bernoulli sum is replaced by a
 * rounded normal; M p (1-p) then exceeds 1e6 and the difference is invisible in
 * the mean and variance the estimator consumes. */
#define EXACT_BINOMIAL_MAX (1LL << 22)

/* Above this many shots per circuit the Bernoulli sum is replaced by a rounded
 * normal; at that size M*p*(1-p) exceeds 1e6 and the difference is invisible in
 * the mean and variance the estimator uses. */
#define EXACT_BINOMIAL_MAX (1LL << 22)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Skew the shared-memory index by one slot every 32 amplitudes. The pair
 * access c, c^flip is otherwise a 2-way bank conflict for every flip bit
 * below 5; with the skew, indices 32 apart land one bank apart. */
#define SKEW(c)  ((c) + ((c) >> 5))
#define SSIZE(N) ((N) + ((N) >> 5))

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t e__ = (call);                                              \
        if (e__ != cudaSuccess) {                                              \
            std::fprintf(stderr, "[CUDA] %s:%d  %s\n", __FILE__, __LINE__,     \
                         cudaGetErrorString(e__));                             \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

static std::atomic<bool> g_stop{false};
static void sigint_handler(int) { g_stop.store(true); }

/* ==========================================================================
 * Generator descriptors
 *
 * A Pauli string is stored symplectically: `flip` marks the X and Y sites
 * (the XOR mask on basis indices), `yz` marks the Y and Z sites (the sign
 * mask). Its matrix element is
 *     P[c ^ flip, c] = i^{nY} * (-1)^{popcount(c & yz)} .
 * Qubit q occupies bit (nq - 1 - q), matching torch.kron(result, op).
 *
 * Both terms of every generator share the same `flip`, so a gate loads each
 * amplitude pair once and applies both rotations in registers.
 * ========================================================================== */

struct GenDesc {
    unsigned int flip;
    unsigned int yz0, yz1;
    int          g0, g1;    /* phase i^g, g = (nY + 1) mod 4                  */
    float        a0, a1;    /* sign of the rotation coefficient, |a| = 1/2    */
};

__constant__ GenDesc c_gen[MAX_GEN];

/* --------------------------------------------------------- device helpers */

__device__ __forceinline__ void mul_ig(int g, real xr, real xi, real& yr, real& yi)
{
    switch (g & 3) {
        case 0:  yr =  xr; yi =  xi; break;
        case 1:  yr = -xi; yi =  xr; break;
        case 2:  yr = -xr; yi = -xi; break;
        default: yr =  xi; yi = -xr; break;
    }
}

__device__ __forceinline__ real sigma(unsigned int c, unsigned int yz)
{
    return (__popc(c & yz) & 1) ? (real)(-1) : (real)(1);
}

/* psi'[c] = cs psi[c] + sn sigma(d) i^g psi[d],  d = c ^ flip
 * psi'[d] = cs psi[d] + sn sigma(c) i^g psi[c]                              */
__device__ __forceinline__ void term_pair(real& vcr, real& vci, real& vdr, real& vdi,
                                          real cs, real sn, int g, real sc, real sd)
{
    real icr, ici, idr, idi;
    mul_ig(g, vcr, vci, icr, ici);
    mul_ig(g, vdr, vdi, idr, idi);
    real ncr = cs * vcr + sn * sd * idr;
    real nci = cs * vci + sn * sd * idi;
    real ndr = cs * vdr + sn * sc * icr;
    real ndi = cs * vdi + sn * sc * ici;
    vcr = ncr; vci = nci; vdr = ndr; vdi = ndi;
}

__device__ __forceinline__ void term_diag(real& vr, real& vi, real cs, real sn,
                                          int g, real s)
{
    real ir, ii;
    mul_ig(g, vr, vi, ir, ii);
    real nr = cs * vr + sn * s * ir;
    real ni = cs * vi + sn * s * ii;
    vr = nr; vi = ni;
}

/* exp(i theta H~_g) with cs = cos(theta/2), sn = sin(theta/2) */
__device__ __forceinline__ void apply_gen(real* re, real* im, unsigned int N,
                                          const GenDesc& g, real cs, real sn)
{
    if (g.flip == 0u) {                       /* Hz: both terms are phases */
        for (unsigned int c = threadIdx.x; c < N; c += TPB) {
            unsigned int i = SKEW(c);
            real vr = re[i], vi = im[i];
            term_diag(vr, vi, cs, sn * (real)g.a0, g.g0, sigma(c, g.yz0));
            term_diag(vr, vi, cs, sn * (real)g.a1, g.g1, sigma(c, g.yz1));
            re[i] = vr; im[i] = vi;
        }
    } else {
        const unsigned int q    = (unsigned int)(__ffs((int)g.flip) - 1);
        const unsigned int lm   = (1u << q) - 1u;
        const unsigned int half = N >> 1;
        for (unsigned int p = threadIdx.x; p < half; p += TPB) {
            unsigned int c  = ((p & ~lm) << 1) | (p & lm);  /* bit q of c is 0 */
            unsigned int d  = c ^ g.flip;
            unsigned int ic = SKEW(c), id = SKEW(d);
            real vcr = re[ic], vci = im[ic], vdr = re[id], vdi = im[id];
            term_pair(vcr, vci, vdr, vdi, cs, sn * (real)g.a0, g.g0,
                      sigma(c, g.yz0), sigma(d, g.yz0));
            term_pair(vcr, vci, vdr, vdi, cs, sn * (real)g.a1, g.g1,
                      sigma(c, g.yz1), sigma(d, g.yz1));
            re[ic] = vcr; im[ic] = vci; re[id] = vdr; im[id] = vdi;
        }
    }
}

__device__ __forceinline__ double rng_u(unsigned long long seed, long long circ,
                                        unsigned int ctr)
{
    curandStatePhilox4_32_10_t st;
    curand_init(seed, (unsigned long long)circ, (unsigned long long)ctr * 4ull, &st);
    return curand_uniform_double(&st);
}

/* <M> = Re sum_c conj(psi[c]) coeff(c ^ flipM) psi[c ^ flipM] ; caller reduces */
__device__ __forceinline__ double expval(const real* re, const real* im,
                                         unsigned int N, unsigned int flipM,
                                         unsigned int yzM, int gM)
{
    double acc = 0.0;
    for (unsigned int c = threadIdx.x; c < N; c += TPB) {
        unsigned int d = c ^ flipM;
        unsigned int ic = SKEW(c), id = SKEW(d);
        real wr, wi;
        mul_ig(gM, re[id], im[id], wr, wi);
        double s = (__popc(d & yzM) & 1) ? -1.0 : 1.0;
        acc += s * ((double)re[ic] * (double)wr + (double)im[ic] * (double)wi);
    }
    return acc;
}

__device__ __forceinline__ double block_sum(double v, double* scratch)
{
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = v;
    __syncthreads();
    double t = 0.0;
    if (threadIdx.x == 0)
        for (int i = 0; i < TPB / 32; ++i) t += scratch[i];
    return t;                                   /* valid on thread 0 only */
}

__device__ __forceinline__ unsigned long long block_sum_ull(unsigned long long v,
                                                            unsigned long long* scratch)
{
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = v;
    __syncthreads();
    unsigned long long t = 0ull;
    if (threadIdx.x == 0)
        for (int i = 0; i < TPB / 32; ++i) t += scratch[i];
    return t;
}

/* Run one phase of the circuit. Angles are drawn from a counter-based stream
 * keyed by (seed, circuit, gate) so they do not depend on the launch geometry.
 * `reverse` handles U_plus = S_1..S_m acting on a ket (S_m first); `sinScale`
 * is -1 for the daggered U_minus. */
__device__ void run_phase(real* reA, real* imA, real* reB, real* imB,
                          unsigned int N,
                          const unsigned short* __restrict__ sched, int ngates,
                          bool reverse, real sinScale, real* trig,
                          unsigned long long seed, long long circ,
                          unsigned int ctrBase)
{
    for (int base = 0; base < ngates; base += TPB) {
        __syncthreads();                    /* previous chunk fully consumed */
        {
            int u = base + (int)threadIdx.x;
            if (u < ngates) {
                int t = reverse ? (ngates - 1 - u) : u;
                double th = rng_u(seed, circ, ctrBase + (unsigned int)t) * 2.0 * M_PI - M_PI;
                double sh, ch;
                sincos(0.5 * th, &sh, &ch);
                trig[2 * threadIdx.x]     = (real)ch;
                trig[2 * threadIdx.x + 1] = (real)sh * sinScale;
            }
        }
        __syncthreads();

        int lim = min(base + TPB, ngates);
        for (int u = base; u < lim; ++u) {
            int t = reverse ? (ngates - 1 - u) : u;
            const GenDesc g = c_gen[sched[t]];
            real cs = trig[2 * (u - base)];
            real sn = trig[2 * (u - base) + 1];
            apply_gen(reA, imA, N, g, cs, sn);
            if (reB) apply_gen(reB, imB, N, g, cs, sn);
            __syncthreads();
        }
    }
}


/* ==========================================================================
 * Small-n kernel: one block per circuit, state resident in shared memory.
 * ========================================================================== */

template <bool EXACT>
__global__ __launch_bounds__(TPB)
void k_circuits(long long circ0, long long ncirc, int nq,
                const unsigned short* __restrict__ schedP, int nP,
                const unsigned short* __restrict__ schedM, int nM,
                int targetJ, unsigned int flipM, unsigned int yzM, int gM,
                real epsCos, real epsSin, double invEps2, double signConv,
                long long shotsPerCirc, unsigned long long seed,
                unsigned long long* nplusOut, unsigned long long* shotsOut,
                unsigned long long* circOut,
                double* sumOut, double* sumsqOut, double* dbg)
{
    extern __shared__ char smem[];
    const unsigned int N  = 1u << nq;
    const unsigned int SS = SSIZE(N);

    real* reA  = (real*)smem;
    real* imA  = reA + SS;
    real* reB  = imA + SS;
    real* imB  = reB + SS;
    real* trig = imB + SS;

    __shared__ double             dscratch[TPB / 32];
    __shared__ unsigned long long uscratch[TPB / 32];

    double sum = 0.0, comp = 0.0, sumsq = 0.0;
    unsigned long long nplus = 0ull, nshots = 0ull, ncircDone = 0ull;

    const real amp = (real)(1.0 / sqrt((double)N));

    for (long long c = circ0 + blockIdx.x; c < circ0 + ncirc; c += gridDim.x) {

        for (unsigned int i = threadIdx.x; i < N; i += TPB) {
            unsigned int k = SKEW(i);
            reA[k] = amp; imA[k] = (real)0;
        }
        __syncthreads();

        /* U_plus |psi> */
        run_phase(reA, imA, nullptr, nullptr, N, schedP, nP, true, (real)1,
                  trig, seed, c, 8u);

        /* branch A = E(+eps), branch B = E(-eps), both from the same U_plus */
        {
            const GenDesc g = c_gen[targetJ];
            for (unsigned int i = threadIdx.x; i < N; i += TPB) {
                unsigned int k = SKEW(i);
                reB[k] = reA[k]; imB[k] = imA[k];
            }
            __syncthreads();
            apply_gen(reA, imA, N, g, epsCos,  epsSin);
            apply_gen(reB, imB, N, g, epsCos, -epsSin);
            __syncthreads();
        }

        /* U_minus^dag on both branches */
        run_phase(reA, imA, reB, imB, N, schedM, nM, false, (real)(-1),
                  trig, seed, c, 8u + (unsigned int)nP);

        double fA = block_sum(expval(reA, imA, N, flipM, yzM, gM), dscratch);
        __syncthreads();
        double fB = block_sum(expval(reB, imB, N, flipM, yzM, gM), dscratch);
        __syncthreads();

        /* broadcast p to the block for the Binomial draw */
        __shared__ double s_p;
        __shared__ double s_v;
        if (threadIdx.x == 0) {
            double delta = fA - fB;
            /* signConv = +1 gives the variance (delta/(2 eps))^2 >= 0;
             * signConv = -1 reproduces the Python's leading minus. */
            double v     = signConv * invEps2 * 0.25 * delta * delta;
            double p     = 0.5 + signConv * 0.125 * delta * delta;
            s_p = (p < 0.0) ? 0.0 : (p > 1.0 ? 1.0 : p);
            s_v = v;
            if (dbg && blockIdx.x == 0 && c == circ0) {
                dbg[0] = fA; dbg[1] = fB; dbg[2] = v; dbg[3] = p;
            }
        }
        __syncthreads();

        /* per-circuit statistics, used for the variance-inflation report */
        if (threadIdx.x == 0) {
            double y = s_v - comp, t = sum + y;
            comp = (t - sum) - y;
            sum  = t;
            sumsq += s_v * s_v;
            ++ncircDone;
        }

        if (!EXACT) {
            curandStatePhilox4_32_10_t st;
            curand_init(seed ^ 0x9E3779B97F4A7C15ull,
                        (unsigned long long)c * TPB + threadIdx.x, 0ull, &st);
            const double p = s_p;
            unsigned long long local = 0ull;
            for (long long t = threadIdx.x; t < shotsPerCirc; t += TPB)
                if (curand_uniform_double(&st) < p) ++local;
            unsigned long long tot = block_sum_ull(local, uscratch);
            if (threadIdx.x == 0) { nplus += tot; nshots += (unsigned long long)shotsPerCirc; }
            __syncthreads();
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(circOut,  ncircDone);
        atomicAdd(sumOut,   sum);
        atomicAdd(sumsqOut, sumsq);
        if (!EXACT) { atomicAdd(nplusOut, nplus); atomicAdd(shotsOut, nshots); }
    }
}

/* ==========================================================================
 * Host: generator table
 * ========================================================================== */

struct HostGen {
    unsigned int flip, yz0, yz1;
    int          g0, g1;
    float        a0, a1;
    int          i, j;      /* the mode pair, for --edges filtering */
    char         kind;      /* 'x', 'y' or 'z' */
};

static inline unsigned int BITQ(int q, int nq) { return 1u << (nq - 1 - q); }

static void build_generators(int nq, std::vector<HostGen>& G)
{
    G.clear();
    for (int i = 0; i < nq; ++i) {
        for (int j = i + 1; j < nq; ++j) {
            unsigned int zs = 0u;
            for (int r = i + 1; r < j; ++r) zs |= BITQ(r, nq);
            const unsigned int fij = BITQ(i, nq) | BITQ(j, nq);

            HostGen hx;                       /* -(P1 + P2)/2 */
            hx.flip = fij;
            hx.yz0  = zs;                     /* X_i Z..Z X_j : nY = 0 -> g = 1 */
            hx.g0   = 1;  hx.a0 = -1.0f;
            hx.yz1  = zs | fij;               /* Y_i Z..Z Y_j : nY = 2 -> g = 3 */
            hx.g1   = 3;  hx.a1 = -1.0f;
            hx.i = i; hx.j = j; hx.kind = 'x';
            G.push_back(hx);

            HostGen hy;                       /* -(Q1 - Q2)/2 */
            hy.flip = fij;
            hy.yz0  = zs | BITQ(i, nq);       /* Y_i Z..Z X_j : nY = 1 -> g = 2 */
            hy.g0   = 2;  hy.a0 = -1.0f;
            hy.yz1  = zs | BITQ(j, nq);       /* X_i Z..Z Y_j : nY = 1 -> g = 2 */
            hy.g1   = 2;  hy.a1 = +1.0f;
            hy.i = i; hy.j = j; hy.kind = 'y';
            G.push_back(hy);
        }
    }
    for (int r = 0; r + 1 < nq; ++r) {        /* -(Z_r - Z_{r+1})/2 */
        HostGen hz;
        hz.flip = 0u;
        hz.yz0  = BITQ(r, nq);      hz.g0 = 1; hz.a0 = -1.0f;
        hz.yz1  = BITQ(r + 1, nq);  hz.g1 = 1; hz.a1 = +1.0f;
        hz.i = r; hz.j = r + 1; hz.kind = 'z';
        G.push_back(hz);
    }
}

static int popcnt(unsigned int v) { int c = 0; while (v) { c += v & 1u; v >>= 1; } return c; }

/* The factorisation exp(i t H~) = exp(i t a0 P0) exp(i t a1 P1) and the value
 * ||H||_2 = 1/2 both rest on: P0 and P1 commute, and P0 P1 is not +-I. Both are
 * pure symplectic statements, so they are checked exactly rather than assumed. */
static void verify_algebra(const std::vector<HostGen>& G)
{
    for (size_t t = 0; t < G.size(); ++t) {
        const HostGen& g = G[t];
        const unsigned int a0 = g.flip, b0 = g.yz0;
        const unsigned int a1 = g.flip, b1 = g.yz1;
        if ((popcnt(a0 & b1) + popcnt(a1 & b0)) & 1) {
            std::fprintf(stderr, "generator %zu: terms anticommute\n", t);
            std::exit(EXIT_FAILURE);
        }
        if (((a0 ^ a1) | (b0 ^ b1)) == 0u) {
            std::fprintf(stderr, "generator %zu: P0 P1 = +-I, norm is not 1/2\n", t);
            std::exit(EXIT_FAILURE);
        }
    }
}

/* ==========================================================================
 * Host: configuration and schedules
 * ========================================================================== */

struct Config {
    int    n = 10, k_meas = 2, target_j = 0;
    double delta = 0.999;
    double shots = 0.0;              /* 0 -> use the analytic formula */
    long long circuits = 0;          /* 0 -> auto */
    long long shots_per_circuit = 0; /* 0 -> auto */
    unsigned long long seed = 20240601ull;
    bool   exact = false;
    bool   full_ansatz = true;
    bool   chain_edges = false;
    bool   selftest = false;
    double stop_stderr = 0.0;
    int    gpu_limit = 0;
    bool   legacy_log = true;   /* match the original Python console format */
    bool   legacy_sign = false; /* true = keep the Python's leading minus     */
    double accuracy = 0.0;      /* 0 -> 1/n^2                                 */
    int    path = 0;            /* 0 auto, 1 shared, 2 global                 */
    int    batch = 0;           /* circuits per launch, global path (0 auto)  */
    std::string out;
};

struct Schedule { std::vector<unsigned short> plus, minus; };

/* One circuit phase in the order the gates are actually applied to a ket,
 * together with its RNG counters and the fused gate groups (runs of equal flip
 * mask). Used by the large-n path in sun_large.cuh. */
struct Phase {
    std::vector<unsigned short> gen;
    std::vector<unsigned int>   ctr;
    std::vector<int>            gstart, gcnt;
};

static Phase build_phase(const std::vector<unsigned short>& sched, bool reverse,
                         unsigned int ctrBase, const std::vector<HostGen>& G)
{
    Phase ph;
    const int n = (int)sched.size();
    for (int u = 0; u < n; ++u) {
        const int t = reverse ? (n - 1 - u) : u;
        ph.gen.push_back(sched[(size_t)t]);
        ph.ctr.push_back(ctrBase + (unsigned int)t);
    }
    for (int i = 0; i < n; ) {                 /* fuse runs of equal flip mask */
        const unsigned int f = G[ph.gen[(size_t)i]].flip;
        int j = i + 1;
        while (j < n && G[ph.gen[(size_t)j]].flip == f && (j - i) < 32) ++j;
        ph.gstart.push_back(i);
        ph.gcnt.push_back(j - i);
        i = j;
    }
    return ph;
}

static Schedule build_schedule(const std::vector<int>& layerGens, int posOfTarget,
                               int L, int ell, bool full)
{
    Schedule s;
    const int Kl = (int)layerGens.size();
    const int j  = posOfTarget;
    const int inner = full ? Kl : j;

    for (int t = j; t < Kl; ++t) s.plus.push_back((unsigned short)layerGens[t]);
    for (int l = ell + 1; l < L; ++l)
        for (int t = 0; t < inner; ++t) s.plus.push_back((unsigned short)layerGens[t]);

    for (int l = 0; l < ell; ++l)
        for (int t = 0; t < inner; ++t) s.minus.push_back((unsigned short)layerGens[t]);
    if (ell < L)
        for (int t = 0; t < j; ++t) s.minus.push_back((unsigned short)layerGens[t]);
    return s;
}

/* ==========================================================================
 * Host: per-GPU worker
 * ========================================================================== */

struct Slot {
    unsigned long long nplus = 0, shots = 0, circuits = 0;
    double sum = 0.0, sumsq = 0.0;
};

/* Each worker republishes its own cumulative counters after every launch, so
 * the monitor can evaluate --stop-stderr on live data. Slots are per GPU, so
 * republishing overwrites rather than accumulates. */
struct Totals {
    std::mutex        mtx;
    std::vector<Slot> live;
    Slot sum_all() const {
        Slot a;
        for (const Slot& s : live) {
            a.nplus += s.nplus; a.shots += s.shots; a.circuits += s.circuits;
            a.sum   += s.sum;   a.sumsq += s.sumsq;
        }
        return a;
    }
};

struct Stats { double mean, err, totalN, varC, shotTerm, inflation; };

/* Single definition of the estimator statistics, used by both the live monitor
 * and the final report so the two can never disagree. */
static Stats compute_stats(const Slot& a, bool exact, double invEps2, long long spc)
{
    Stats st{0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    if (a.circuits == 0) return st;
    const double C = (double)a.circuits;
    const double meanC = a.sum / C;
    st.varC = a.sumsq / C - meanC * meanC;
    if (st.varC < 0.0) st.varC = 0.0;

    if (exact) {
        st.totalN = C;
        st.mean   = meanC;
        st.err    = std::sqrt(st.varC / C);
    } else {
        st.totalN = (double)a.shots;
        if (st.totalN <= 0.0) return st;
        const double phat = (double)a.nplus / st.totalN;
        st.mean = invEps2 * (2.0 * phat - 1.0);
        /* Var(mean) = varC * M / N + sigma_shot^2 / N, the shot term measured
         * directly from the binomial spread rather than assumed. */
        st.shotTerm = invEps2 * invEps2 * 4.0 * phat * (1.0 - phat);
        st.err = std::sqrt((st.varC * (double)spc + st.shotTerm) / st.totalN);
        const double indep = std::sqrt((st.varC + st.shotTerm) / st.totalN);
        st.inflation = st.err / std::max(1e-300, indep);
    }
    return st;
}

struct DeviceRes {
    unsigned short *schedP = nullptr, *schedM = nullptr;
    unsigned long long *d_nplus = nullptr, *d_shots = nullptr, *d_circ = nullptr;
    double *d_sum = nullptr, *d_sumsq = nullptr, *d_dbg = nullptr;
};

static std::atomic<long long> g_circ_next{0};
static long long              g_circ_total = 0;

struct Ctx {
    Config      cfg;
    Schedule    sched;
    size_t      smem = 0;
    int         nq = 0;
    unsigned int flipM = 0, yzM = 0;
    int         gM = 0;
    real        epsCos = 0, epsSin = 0;
    double      invEps2 = 0;
    double      signConv = 1.0;
    long long   spc = 1;
    std::vector<int> blocks;    /* resident blocks per GPU, sized in main() */
    Phase pp, pm;               /* applied-order phases for the global path  */
};

static void worker(int gpu, const Ctx* ctx, Totals* tot, const std::vector<HostGen>* G)
{
    CUDA_CHECK(cudaSetDevice(gpu));
    const Config& cfg = ctx->cfg;

    std::vector<GenDesc> gd(G->size());
    for (size_t i = 0; i < G->size(); ++i) {
        gd[i].flip = (*G)[i].flip;
        gd[i].yz0  = (*G)[i].yz0;  gd[i].yz1 = (*G)[i].yz1;
        gd[i].g0   = (*G)[i].g0;   gd[i].g1  = (*G)[i].g1;
        gd[i].a0   = (*G)[i].a0;   gd[i].a1  = (*G)[i].a1;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_gen, gd.data(), sizeof(GenDesc) * gd.size()));

    DeviceRes r;
    const int nP = (int)ctx->sched.plus.size();
    const int nM = (int)ctx->sched.minus.size();
    if (nP) {
        CUDA_CHECK(cudaMalloc(&r.schedP, sizeof(unsigned short) * nP));
        CUDA_CHECK(cudaMemcpy(r.schedP, ctx->sched.plus.data(),
                              sizeof(unsigned short) * nP, cudaMemcpyHostToDevice));
    }
    if (nM) {
        CUDA_CHECK(cudaMalloc(&r.schedM, sizeof(unsigned short) * nM));
        CUDA_CHECK(cudaMemcpy(r.schedM, ctx->sched.minus.data(),
                              sizeof(unsigned short) * nM, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&r.d_nplus, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&r.d_shots, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&r.d_circ,  sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&r.d_sum,   sizeof(double)));
    CUDA_CHECK(cudaMalloc(&r.d_sumsq, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&r.d_dbg,   sizeof(double) * 4));
    CUDA_CHECK(cudaMemset(r.d_nplus, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(r.d_shots, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(r.d_circ,  0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(r.d_sum,   0, sizeof(double)));
    CUDA_CHECK(cudaMemset(r.d_sumsq, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(r.d_dbg,   0, sizeof(double) * 4));

    if (ctx->smem > 48u * 1024u) {
        CUDA_CHECK(cudaFuncSetAttribute(
            cfg.exact ? (const void*)k_circuits<true> : (const void*)k_circuits<false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ctx->smem));
    }

    const int blocks = ctx->blocks[(size_t)gpu];
    if (cfg.legacy_log)
        std::printf("  [+] Worker %d bound to cuda:%d | Assigned Batch Size: %lld\n",
                    gpu, gpu, (long long)blocks * 4 * ctx->spc);
    else
        std::fprintf(stderr, "  [+] cuda:%d  %d blocks  %.1f KB shared/block\n",
                     gpu, blocks, ctx->smem / 1024.0);
    std::fflush(stdout);

    long long perLaunch = (long long)blocks * 4;
    double    lastSecs  = 0.0;
    bool      selftest_done = false;

    while (!g_stop.load()) {
        long long take = 0, start = 0;
        {
            long long cur = g_circ_next.load();
            while (cur < g_circ_total) {
                long long want = std::min<long long>(perLaunch, g_circ_total - cur);
                if (g_circ_next.compare_exchange_weak(cur, cur + want)) {
                    start = cur; take = want; break;
                }
            }
        }
        if (take <= 0) break;

        auto t0 = std::chrono::steady_clock::now();
        if (cfg.exact)
            k_circuits<true><<<blocks, TPB, ctx->smem>>>(
                start, take, ctx->nq, r.schedP, nP, r.schedM, nM, cfg.target_j,
                ctx->flipM, ctx->yzM, ctx->gM, ctx->epsCos, ctx->epsSin,
                ctx->invEps2, ctx->signConv, ctx->spc, cfg.seed,
                r.d_nplus, r.d_shots, r.d_circ,
                r.d_sum, r.d_sumsq, cfg.selftest ? r.d_dbg : nullptr);
        else
            k_circuits<false><<<blocks, TPB, ctx->smem>>>(
                start, take, ctx->nq, r.schedP, nP, r.schedM, nM, cfg.target_j,
                ctx->flipM, ctx->yzM, ctx->gM, ctx->epsCos, ctx->epsSin,
                ctx->invEps2, ctx->signConv, ctx->spc, cfg.seed,
                r.d_nplus, r.d_shots, r.d_circ,
                r.d_sum, r.d_sumsq, cfg.selftest ? r.d_dbg : nullptr);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        lastSecs = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - t0).count();

        if (cfg.legacy_log) {
            std::printf("[Worker %d | cuda:%d] Evaluated batch of %lld shots in %.4f seconds.\n",
                        gpu, gpu, take * ctx->spc, lastSecs);
            std::fflush(stdout);
        }

        /* aim for ~0.4 s launches: responsive to Ctrl+C, negligible overhead */
        if (lastSecs > 1e-4) {
            double scale = 0.4 / lastSecs;
            scale = std::min(4.0, std::max(0.25, scale));
            perLaunch = std::max<long long>(blocks, (long long)(perLaunch * scale));
        }

        {   /* republish this worker's cumulative counters */
            Slot sl;
            CUDA_CHECK(cudaMemcpy(&sl.nplus,    r.d_nplus, sizeof(sl.nplus),    cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&sl.shots,    r.d_shots, sizeof(sl.shots),    cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&sl.circuits, r.d_circ,  sizeof(sl.circuits), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&sl.sum,      r.d_sum,   sizeof(sl.sum),      cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&sl.sumsq,    r.d_sumsq, sizeof(sl.sumsq),    cudaMemcpyDeviceToHost));
            std::lock_guard<std::mutex> lk(tot->mtx);
            tot->live[(size_t)gpu] = sl;
        }

        if (cfg.selftest && !selftest_done) {
            selftest_done = true;
            double dbg[4];
            CUDA_CHECK(cudaMemcpy(dbg, r.d_dbg, sizeof(dbg), cudaMemcpyDeviceToHost));
            std::fprintf(stderr,
                         "  [selftest cuda:%d] f(+eps)=%.12g  f(-eps)=%.12g  "
                         "value=%.12g  p=%.12g\n", gpu, dbg[0], dbg[1], dbg[2], dbg[3]);
        }
    }

    {   /* final publish; the device counters are cumulative for this worker */
        Slot sl;
        CUDA_CHECK(cudaMemcpy(&sl.nplus,    r.d_nplus, sizeof(sl.nplus),    cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&sl.shots,    r.d_shots, sizeof(sl.shots),    cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&sl.circuits, r.d_circ,  sizeof(sl.circuits), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&sl.sum,      r.d_sum,   sizeof(sl.sum),      cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&sl.sumsq,    r.d_sumsq, sizeof(sl.sumsq),    cudaMemcpyDeviceToHost));
        std::lock_guard<std::mutex> lk(tot->mtx);
        tot->live[(size_t)gpu] = sl;
    }

    if (r.schedP) CUDA_CHECK(cudaFree(r.schedP));
    if (r.schedM) CUDA_CHECK(cudaFree(r.schedM));
    CUDA_CHECK(cudaFree(r.d_nplus)); CUDA_CHECK(cudaFree(r.d_shots));
    CUDA_CHECK(cudaFree(r.d_circ));  CUDA_CHECK(cudaFree(r.d_sum));
    CUDA_CHECK(cudaFree(r.d_sumsq)); CUDA_CHECK(cudaFree(r.d_dbg));
}

#include "sun_large.cuh"

/* ========================================================================== */

static void usage(const char* p)
{
    std::printf(
        "Usage: %s [options]\n"
        "  --n INT                  qubits, up to 26                  (10)\n"
        "  --accuracy FLOAT         target accuracy (0 = 1/n^2)       (0)\n"
        "  --path STR               'auto', 'shared' (n<=12) or 'global'\n"
        "  --batch INT              circuits per launch, global path  (auto)\n"
        "  --k INT                  qubits measured by M              (2)\n"
        "  --target-j INT           index of the probed generator     (0)\n"
        "  --delta FLOAT            failure probability               (0.999)\n"
        "  --shots FLOAT            total shots, e.g. 1e12 (0 = analytic bound)\n"
        "  --circuits LONG          circuit simulations (0 = auto)\n"
        "  --shots-per-circuit LONG shots drawn per circuit (0 = auto, 1 = independent)\n"
        "  --estimator STR          'sampled' or 'exact'              (sampled)\n"
        "  --ansatz STR             'full' or 'legacy'                (full)\n"
        "  --edges STR              'all' or 'chain'                  (all)\n"
        "  --stop-stderr FLOAT      halt once the standard error drops below this\n"
        "  --gpus INT               use at most this many GPUs\n"
        "  --seed ULONG             RNG seed                          (20240601)\n"
        "  --out PATH               CSV path\n"
        "  --selftest               print f(+eps), f(-eps) for one circuit\n"
        "  --log STR                'legacy' (original console format) or 'modern'\n"
        "  --sign STR               'positive' (report the variance) or 'legacy'\n"
        "                           (keep the Python's leading minus)  (positive)\n",
        p);
}

int main(int argc, char** argv)
{
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { usage(argv[0]); std::exit(1); }
            return argv[++i];
        };
        if      (a == "--n")                 cfg.n = std::atoi(next());
        else if (a == "--k")                 cfg.k_meas = std::atoi(next());
        else if (a == "--target-j")          cfg.target_j = std::atoi(next());
        else if (a == "--delta")             cfg.delta = std::atof(next());
        else if (a == "--shots")             cfg.shots = std::atof(next());
        else if (a == "--circuits")          cfg.circuits = std::atoll(next());
        else if (a == "--shots-per-circuit") cfg.shots_per_circuit = std::atoll(next());
        else if (a == "--stop-stderr")       cfg.stop_stderr = std::atof(next());
        else if (a == "--gpus")              cfg.gpu_limit = std::atoi(next());
        else if (a == "--seed")              cfg.seed = std::strtoull(next(), nullptr, 10);
        else if (a == "--out")               cfg.out = next();
        else if (a == "--selftest")          cfg.selftest = true;
        else if (a == "--accuracy")          cfg.accuracy = std::atof(next());
        else if (a == "--batch")             cfg.batch = std::atoi(next());
        else if (a == "--path") {
            std::string v = next();
            if      (v == "auto")   cfg.path = 0;
            else if (v == "shared") cfg.path = 1;
            else if (v == "global") cfg.path = 2;
            else { std::fprintf(stderr, "--path must be auto, shared or global\n"); return 1; }
        }
        else if (a == "--sign") {
            std::string v = next();
            if      (v == "positive") cfg.legacy_sign = false;
            else if (v == "legacy")   cfg.legacy_sign = true;
            else { std::fprintf(stderr, "--sign must be 'positive' or 'legacy'\n"); return 1; }
        }
        else if (a == "--log") {
            std::string v = next();
            if      (v == "legacy") cfg.legacy_log = true;
            else if (v == "modern") cfg.legacy_log = false;
            else { std::fprintf(stderr, "--log must be 'legacy' or 'modern'\n"); return 1; }
        }
        else if (a == "--estimator") {
            std::string v = next();
            if      (v == "exact")   cfg.exact = true;
            else if (v == "sampled") cfg.exact = false;
            else { std::fprintf(stderr, "--estimator must be 'exact' or 'sampled'\n"); return 1; }
        } else if (a == "--ansatz") {
            std::string v = next();
            if      (v == "full")   cfg.full_ansatz = true;
            else if (v == "legacy") cfg.full_ansatz = false;
            else { std::fprintf(stderr, "--ansatz must be 'full' or 'legacy'\n"); return 1; }
        } else if (a == "--edges") {
            std::string v = next();
            if      (v == "all")   cfg.chain_edges = false;
            else if (v == "chain") cfg.chain_edges = true;
            else { std::fprintf(stderr, "--edges must be 'all' or 'chain'\n"); return 1; }
        } else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "unknown option: %s\n", a.c_str()); usage(argv[0]); return 1; }
    }

    if (cfg.n < 2 || cfg.n > 26) { std::fprintf(stderr, "n must be in [2, 26]\n"); return 1; }
    if (cfg.k_meas < 1 || cfg.k_meas > cfg.n) { std::fprintf(stderr, "k out of range\n"); return 1; }

    std::signal(SIGINT, sigint_handler);

    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount == 0) { std::fprintf(stderr, "CUDA is strictly required.\n"); return 1; }
    if (cfg.gpu_limit > 0) devCount = std::min(devCount, cfg.gpu_limit);

    const int nq = cfg.n;
    const unsigned int N = 1u << nq;
    const int K = nq * nq - 1;
    const int L = 2 * nq;
    const int ell = L / 2;
    const double target_accuracy = (cfg.accuracy > 0.0) ? cfg.accuracy
                                                        : 1.0 / ((double)nq * nq);
    const double teps = target_accuracy / 2.0;

    if (cfg.target_j < 0 || cfg.target_j >= K) {
        std::fprintf(stderr, "target-j must be in [0, %d)\n", K); return 1;
    }

    std::vector<HostGen> G;
    build_generators(nq, G);
    if ((int)G.size() != K) { std::fprintf(stderr, "generator count mismatch\n"); return 1; }
    verify_algebra(G);

    /* which generators appear in a layer */
    std::vector<int> layerGens;
    for (int t = 0; t < K; ++t)
        if (!cfg.chain_edges || G[t].j == G[t].i + 1) layerGens.push_back(t);
    int posOfTarget = -1;
    for (size_t t = 0; t < layerGens.size(); ++t)
        if (layerGens[t] == cfg.target_j) posOfTarget = (int)t;
    if (posOfTarget < 0) {
        std::fprintf(stderr, "target-j %d is not in the chain edge set; use --edges all\n",
                     cfg.target_j);
        return 1;
    }

    Schedule sched = build_schedule(layerGens, posOfTarget, L, ell, cfg.full_ansatz);
    const int nP = (int)sched.plus.size();
    const int nM = (int)sched.minus.size();

    /* All normalised generators and M have unit spectral norm, and the nuclear
     * norm of a pure-state rho is its trace, so every norm in the Python's
     * epsilon formula is 1 (as it was there too, since build_su_n_operators
     * normalised before measuring). */
    const double num = 3.0 * (target_accuracy - teps);
    const double den = 16.0;
    const double epsilon = std::sqrt(num / den);
    const double invEps2 = 1.0 / (epsilon * epsilon);

    const double N_analytic = (8.0 / (teps * teps * std::pow(epsilon, 4)))
                              * std::log(2.0 / cfg.delta);
    double totalShots = (cfg.shots > 0.0) ? cfg.shots : N_analytic;

    long long spc = cfg.shots_per_circuit;
    long long circuits = cfg.circuits;
    if (cfg.exact) {
        spc = 1;
        if (circuits <= 0) circuits = (long long)std::min(totalShots, 1e7);
    } else {
        if (circuits <= 0 && spc <= 0) {
            /* cap circuits at 1e7; the shot-noise term dominates long before
               the cluster term becomes visible (see note 5) */
            circuits = (long long)std::min(totalShots, 1e7);
            spc = std::max<long long>(1, (long long)std::llround(totalShots / (double)circuits));
        } else if (circuits <= 0) {
            circuits = std::max<long long>(1, (long long)std::llround(totalShots / (double)spc));
        } else if (spc <= 0) {
            spc = std::max<long long>(1, (long long)std::llround(totalShots / (double)circuits));
        }
    }
    g_circ_total = circuits;

    const size_t SS = SSIZE(N);
    const size_t smem = 4u * SS * sizeof(real) + 2u * TPB * sizeof(real);

    /* Path selection. The shared-memory kernel needs 4*N amplitudes resident
     * per block; above ~n = 12 (FP32) or 11 (FP64) that no longer fits and the
     * state has to live in global memory. */
    size_t maxSmem = 48u * 1024u;
    {
        CUDA_CHECK(cudaSetDevice(0));
        cudaDeviceProp pr0; CUDA_CHECK(cudaGetDeviceProperties(&pr0, 0));
        maxSmem = pr0.sharedMemPerBlock;
        int optin = 0;
        if (cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0)
            == cudaSuccess && optin > (int)maxSmem) maxSmem = (size_t)optin;
    }
    bool useGlobal;
    if      (cfg.path == 1) useGlobal = false;
    else if (cfg.path == 2) useGlobal = true;
    else                    useGlobal = (smem > maxSmem);
    if (!useGlobal && smem > maxSmem) {
        std::fprintf(stderr,
                     "--path shared needs %.1f KB/block but the device offers %.1f KB; "
                     "use --path global or build PREC=fp32.\n",
                     smem / 1024.0, maxSmem / 1024.0);
        return 1;
    }

    Ctx ctx;
    ctx.cfg = cfg; ctx.sched = sched; ctx.smem = smem; ctx.nq = nq;
    ctx.flipM = ((1u << cfg.k_meas) - 1u) << (nq - cfg.k_meas);
    ctx.yzM = 0u; ctx.gM = 0;                 /* M = X^{otimes k}: no Y, no Z */
    ctx.epsCos = (real)std::cos(0.5 * epsilon);
    ctx.epsSin = (real)std::sin(0.5 * epsilon);
    ctx.invEps2 = invEps2;
    ctx.signConv = cfg.legacy_sign ? -1.0 : 1.0;
    ctx.spc = spc;
    ctx.pp = build_phase(sched.plus,  true,  8u,                      G);
    ctx.pm = build_phase(sched.minus, false, 8u + (unsigned int)nP,   G);

    /* Resident blocks per GPU, resolved here so the launch sizes can be
     * reported before the workers start (the original printed them too). */
    ctx.blocks.assign((size_t)devCount, 0);
    std::string batchStr = "{";
    for (int g = 0; g < devCount && !useGlobal; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        if (smem > 48u * 1024u) {
            CUDA_CHECK(cudaFuncSetAttribute(
                cfg.exact ? (const void*)k_circuits<true> : (const void*)k_circuits<false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        }
        cudaDeviceProp pr; CUDA_CHECK(cudaGetDeviceProperties(&pr, g));
        int perSM = 0;
        if (cfg.exact)
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM, k_circuits<true>,  TPB, smem));
        else
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM, k_circuits<false>, TPB, smem));
        if (perSM < 1) {
            std::fprintf(stderr,
                         "[cuda:%d] no blocks fit (%zu B shared/block). Lower n or build PREC=fp32.\n",
                         g, smem);
            return 1;
        }
        ctx.blocks[(size_t)g] = pr.multiProcessorCount * perSM;
        char t[80];
        std::snprintf(t, sizeof t, "%s'cuda:%d': %lld", g ? ", " : "", g,
                      (long long)ctx.blocks[(size_t)g] * 4 * spc);
        batchStr += t;
    }
    if (useGlobal) batchStr = "{auto, sized from free VRAM}";
    else           batchStr += "}";
    CUDA_CHECK(cudaSetDevice(0));

    if (cfg.legacy_log) {
        std::printf("\n--- CUDA Diagnostics ---\n");
        std::printf("Total GPUs detected: %d\n", devCount);
        for (int g = 0; g < devCount; ++g) {
            cudaDeviceProp pr; CUDA_CHECK(cudaGetDeviceProperties(&pr, g));
            std::printf("  [cuda:%d] %s (VRAM: %.2f GB)\n", g, pr.name,
                        pr.totalGlobalMem / 1073741824.0);
        }
        std::printf("------------------------\n\n");
        std::printf("Setting up SU(n) Experiment...\n");
        std::printf("\n--- EXPERIMENT SETUP ---\n");
        std::printf("Target Accuracy:                %.17g\n", target_accuracy);
        std::printf("Calculated Epsilon:             %.8f\n", epsilon);
        std::printf("Calculated N_shots (Raw Float): %.17g\n", N_analytic);
        std::printf("Calculated N_shots (Ceiled):    %.0f\n",
                    (double)circuits * (double)spc);
        std::printf("Calculated Batch Sizes:         %s\n", batchStr.c_str());
        std::printf("Precision / Estimator:          %s / %s\n", PREC_NAME,
                    cfg.exact ? "exact" : "sampled");
        std::printf("Execution path:                 %s\n", useGlobal
                    ? "global memory (large n, fused gate groups)"
                    : "shared memory (state resident per block)");
        if (useGlobal)
            std::printf("Fused passes U+ / U-:           %zu / %zu  (of %d / %d gates)\n",
                        ctx.pp.gstart.size(), ctx.pm.gstart.size(), nP, nM);
        std::printf("Sign convention:                %s\n", cfg.legacy_sign
                    ? "legacy (Python's leading minus; always <= 0)"
                    : "positive (variance)");
        std::printf("Circuits x shots/circuit:       %lld x %lld\n", circuits, spc);
        std::printf("Gates U+ / U- / total:          %d / %d / %d\n", nP, nM, nP + nM);
    } else {
        std::printf("\n--- SU(n) TESTER ---\n");
        for (int g = 0; g < devCount; ++g) {
            cudaDeviceProp pr; CUDA_CHECK(cudaGetDeviceProperties(&pr, g));
            std::printf("  [cuda:%d] %s  (%d SMs, %.1f GB)\n", g, pr.name,
                        pr.multiProcessorCount, pr.totalGlobalMem / 1073741824.0);
        }
        std::printf("Precision:              %s\n", PREC_NAME);
        std::printf("n / k / target_j:       %d / %d / %d\n", nq, cfg.k_meas, cfg.target_j);
        std::printf("K / L / ell:            %d / %d / %d\n", K, L, ell);
        std::printf("Edges / ansatz:         %s / %s\n",
                    cfg.chain_edges ? "chain" : "all", cfg.full_ansatz ? "full" : "legacy");
        std::printf("Gates U+ / U- / total:  %d / %d / %d\n", nP, nM, nP + nM);
        std::printf("Target accuracy:        %.17g   (teps %.17g)\n", target_accuracy, teps);
        std::printf("Epsilon / 1/eps^2:      %.10f / %.4f\n", epsilon, invEps2);
        std::printf("Analytic shot bound:    %.6g\n", N_analytic);
        std::printf("Estimator:              %s\n", cfg.exact ? "exact" : "sampled");
        std::printf("Sign convention:        %s\n",
                    cfg.legacy_sign ? "legacy (always <= 0)" : "positive (variance)");
        std::printf("Circuits x shots/circ:  %lld x %lld  = %.6g shots\n",
                    circuits, spc, (double)circuits * (double)spc);
        std::printf("Path:                   %s\n", useGlobal ? "global" : "shared");
        if (!useGlobal) std::printf("Shared per block:       %.1f KB\n", smem / 1024.0);
        else std::printf("Fused passes U+ / U-:   %zu / %zu\n",
                         ctx.pp.gstart.size(), ctx.pm.gstart.size());
        std::printf("--------------------\n");
    }

    std::string csv = cfg.out;
    if (csv.empty()) {
        char stamp[64]; std::time_t tt = std::time(nullptr); std::tm tmv{};
#if defined(_WIN32)
        localtime_s(&tmv, &tt);
#else
        localtime_r(&tt, &tmv);
#endif
        std::strftime(stamp, sizeof stamp, "%Y%m%d_%H%M%S", &tmv);
        csv = std::string("sun_experiment_") + stamp + ".csv";
    }
    std::FILE* fp = std::fopen(csv.c_str(), "w");
    if (!fp) { std::fprintf(stderr, "cannot open %s\n", csv.c_str()); return 1; }
    std::fprintf(fp, "# EXPERIMENT METADATA\n");
    std::fprintf(fp, "# n,%d\n# k_meas,%d\n# target_j,%d\n", nq, cfg.k_meas, cfg.target_j);
    std::fprintf(fp, "# K,%d\n# L,%d\n# ell,%d\n", K, L, ell);
    std::fprintf(fp, "# edges,%s\n# ansatz,%s\n",
                 cfg.chain_edges ? "chain" : "all", cfg.full_ansatz ? "full" : "legacy");
    std::fprintf(fp, "# U_plus_gates,%d\n# U_minus_gates,%d\n", nP, nM);
    std::fprintf(fp, "# target_accuracy,%.17g\n# teps,%.17g\n", target_accuracy, teps);
    std::fprintf(fp, "# delta,%.17g\n# epsilon,%.17g\n", cfg.delta, epsilon);
    std::fprintf(fp, "# analytic_shot_bound,%.17g\n", N_analytic);
    std::fprintf(fp, "# estimator,%s\n", cfg.exact ? "exact" : "sampled");
    std::fprintf(fp, "# sign_convention,%s\n", cfg.legacy_sign ? "legacy" : "positive");
    std::fprintf(fp, "# circuits,%lld\n# shots_per_circuit,%lld\n", circuits, spc);
    std::fprintf(fp, "# precision,%s\n# seed,%llu\n", PREC_NAME,
                 (unsigned long long)cfg.seed);
    std::fprintf(fp, "# \n");

    Totals tot;
    tot.live.assign((size_t)devCount, Slot{});
    auto t_start = std::chrono::steady_clock::now();

    if (cfg.legacy_log) {
        std::printf("Output File:                    %s\n", csv.c_str());
        std::printf("------------------------\n");
        std::printf("Spawning %d dynamic workers across %d GPU(s)...\n", devCount, devCount);
        std::fflush(stdout);
    }

    std::vector<std::thread> th;
    for (int g = 0; g < devCount; ++g)
        th.emplace_back(useGlobal ? worker_global : worker, g, &ctx, &tot, &G);

    /* Progress monitor. Runs in both log modes because it also enforces
     * --stop-stderr; only the printing is suppressed in legacy mode. */
    std::thread mon([&] {
        while (!g_stop.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            long long done = g_circ_next.load();
            Slot agg;
            { std::lock_guard<std::mutex> lk(tot.mtx); agg = tot.sum_all(); }
            const Stats st = compute_stats(agg, cfg.exact, invEps2, spc);

            if (cfg.stop_stderr > 0.0 && agg.circuits > 0 && st.err > 0.0 &&
                st.err <= cfg.stop_stderr) {
                std::printf("[stop] standard error %.4g <= --stop-stderr %.4g "
                            "after %.4g shots\n", st.err, cfg.stop_stderr, st.totalN);
                std::fflush(stdout);
                g_stop.store(true);
                break;
            }
            if (done >= g_circ_total) break;
            if (!cfg.legacy_log) {
                double el = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - t_start).count();
                double rate = done / std::max(1e-9, el);
                std::printf("  %lld / %lld circuits  (%.3g circ/s, %.3g shots/s, "
                            "est %.6g +- %.3g, ETA %.0f s)\n",
                            done, g_circ_total, rate, rate * (double)spc,
                            st.mean, st.err,
                            (g_circ_total - done) / std::max(1e-9, rate));
                std::fflush(stdout);
            }
        }
    });

    for (auto& t : th) t.join();
    g_stop.store(true);
    mon.join();

    double elapsed = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t_start).count();

    std::printf("\n========================================\n");
    Slot agg;
    { std::lock_guard<std::mutex> lk(tot.mtx); agg = tot.sum_all(); }
    if (agg.circuits == 0) {
        std::printf("No circuits completed.\n");
        std::fclose(fp);
        return 0;
    }

    const Stats stt = compute_stats(agg, cfg.exact, invEps2, spc);
    const double C         = (double)agg.circuits;
    const double varC      = stt.varC;
    const double mean      = stt.mean;
    const double stderr_   = stt.err;
    const double totalN    = stt.totalN;
    const double shotTerm  = stt.shotTerm;
    const double inflation = stt.inflation;

    /* Original console layout. physical_H_norm = physical_M_norm = 1 for the
     * normalised operators, so the physical variance equals the mean, exactly
     * as in the Python. */
    if (cfg.legacy_log) {
        std::printf("--- EXPERIMENT CLOCK: Total Running Time: %.2f seconds ---\n", elapsed);
        std::printf("Experiment Complete (%.0f shots executed)\n", totalN);
        std::printf("Normalized Fractional Outcome: %.4f\n", mean);
        std::printf("Final Physical Variance:       %.4f\n", mean);
        std::printf("========================================\n");
        std::printf("Standard error:         %.3g   %s teps = %.4g\n", stderr_,
                    (stderr_ <= teps) ? "<=" : ">", teps);
        if (!cfg.exact)
            std::printf("Cluster inflation:      %.6f x  (Var_circuit %.4g, sigma^2_shot %.4g)\n",
                        inflation, varC, shotTerm);
        std::printf("Circuits simulated:     %llu  (%.4g circ/s, %.4g shots/s)\n",
                    (unsigned long long)agg.circuits, C / std::max(1e-9, elapsed),
                    totalN / std::max(1e-9, elapsed));
        std::fprintf(fp, "metric,value\n");
        std::fprintf(fp, "circuits,%llu\n", (unsigned long long)agg.circuits);
        std::fprintf(fp, "effective_shots,%.17g\n", totalN);
        std::fprintf(fp, "n_plus,%llu\n", (unsigned long long)agg.nplus);
        std::fprintf(fp, "mean,%.17g\n", mean);
        std::fprintf(fp, "stderr,%.17g\n", stderr_);
        std::fprintf(fp, "var_circuit,%.17g\n", varC);
        std::fprintf(fp, "wall_seconds,%.6f\n", elapsed);
        std::fclose(fp);
        std::printf("Wrote %s\n", csv.c_str());
        return 0;
    }

    if (!cfg.exact) {
        std::printf("Cluster inflation:      %.6f x  (1.0 = as good as independent shots)\n",
                    inflation);
        std::printf("  Var_circuit           %.6g\n", varC);
        std::printf("  sigma^2_shot          %.6g\n", shotTerm);
    }

    std::printf("Circuits simulated:     %llu   in %.2f s  (%.4g circ/s)\n",
                (unsigned long long)agg.circuits, elapsed, C / std::max(1e-9, elapsed));
    std::printf("Effective shots:        %.6g   (%.4g shots/s)\n",
                totalN, totalN / std::max(1e-9, elapsed));
    std::printf("Estimate:               %.8g  +- %.3g\n", mean, stderr_);
    std::printf("Target teps:            %.8g  -> %s\n", teps,
                (stderr_ <= teps) ? "REACHED" : "not reached");
    std::printf("========================================\n");

    std::fprintf(fp, "metric,value\n");
    std::fprintf(fp, "circuits,%llu\n", (unsigned long long)agg.circuits);
    std::fprintf(fp, "effective_shots,%.17g\n", totalN);
    std::fprintf(fp, "n_plus,%llu\n", (unsigned long long)agg.nplus);
    std::fprintf(fp, "mean,%.17g\n", mean);
    std::fprintf(fp, "stderr,%.17g\n", stderr_);
    std::fprintf(fp, "var_circuit,%.17g\n", varC);
    std::fprintf(fp, "wall_seconds,%.6f\n", elapsed);
    std::fclose(fp);
    std::printf("Wrote %s\n", csv.c_str());
    return 0;
}
