/* =============================================================================
 *  sun_large.cuh -- large-n execution path for sun_experiment.cu
 *
 *  Included from sun_experiment.cu AFTER Config / Ctx / Totals are declared;
 *  it is a separate file for readability, not a standalone translation unit.
 *
 *  WHY A SECOND PATH
 *  The default kernel keeps the whole state vector in shared memory, one block
 *  per circuit. That caps n at about 12 (4*N amplitudes against ~100 KB/SM).
 *  At n = 20 the state is 2^20 amplitudes = 8 MB in FP32, so it has to live in
 *  global memory and the circuit is driven by one kernel launch per gate GROUP
 *  over a batch of B circuits.
 *
 *  GATE-GROUP FUSION
 *  Consecutive gates sharing a flip mask act on the same amplitude pairs, so
 *  they are fused into one load / modify / store pass. build_generators() emits
 *  Hx(i,j) and Hy(i,j) adjacently -- both flip bits i and j -- and the trailing
 *  Hz gates are all diagonal (flip = 0), so a layer of K = n^2-1 gates collapses
 *  to C(n,2) + 1 passes: 191 instead of 399 at n = 20, slightly better than a 2x
 *  cut in memory traffic. Fusion does not require the gates to commute; they are
 *  applied in sequence to the registers already holding the pair.
 *
 *  COST AT n = 20
 *  L*K = 15,960 gates -> 3,820 fused passes for U+ and 3,820 for U- (the latter
 *  on two branches). At 16N bytes per pass that is ~190 GB per circuit, about
 *  0.13 s on an A100-class part, so ~10^4 circuits lands near 20 minutes.
 *
 *  COALESCING CAVEAT
 *  Thread p handles amplitude c = insert-0-at-bit-q(p), q = lowest flip bit.
 *  For q >= 5 a warp reads 32 consecutive amplitudes; for q < 5 it reads 32 of
 *  every 64, roughly halving efficiency on those gates. Since qubit j occupies
 *  bit n-1-j, this affects pairs with large j -- about 45% of gates at n = 20,
 *  for an overall penalty near 1.3x. Recovering it needs the shared-memory
 *  tiling that full statevector simulators use; not done here.
 * =============================================================================
 */

#pragma once

/* --- cos(theta/2) and sin(theta/2)*sinScale for every (gate, circuit) ------ */
__global__ void k_angles_global(real* trig, const unsigned int* __restrict__ ctr,
                                int nGates, int B, unsigned long long seed,
                                long long circ0, real sinScale)
{
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx >= (size_t)nGates * (size_t)B) return;
    const int g = (int)(idx / (size_t)B), b = (int)(idx % (size_t)B);
    const double th = rng_u(seed, circ0 + b, ctr[g]) * 2.0 * M_PI - M_PI;
    double sh, ch;
    sincos(0.5 * th, &sh, &ch);
    trig[2 * idx]     = (real)ch;
    trig[2 * idx + 1] = (real)sh * sinScale;
}

__global__ void k_init_global(real* re, real* im, int nq, int B, real amp)
{
    const size_t total = ((size_t)1 << nq) * (size_t)B;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total;
         i += (size_t)gridDim.x * blockDim.x) { re[i] = amp; im[i] = (real)0; }
}

/* --- apply gates [gstart, gstart+cnt), all sharing one flip mask ----------- */
__global__ void k_gate_group(real* __restrict__ re, real* __restrict__ im,
                             int nq, int B,
                             const unsigned short* __restrict__ applied,
                             int gstart, int cnt,
                             const real* __restrict__ trig)
{
    const unsigned int N = 1u << nq;
    const GenDesc g0 = c_gen[applied[gstart]];
    const size_t stride = (size_t)gridDim.x * blockDim.x;

    if (g0.flip == 0u) {                                    /* diagonal group */
        const size_t total = (size_t)N * (size_t)B;
        for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
             idx < total; idx += stride) {
            const unsigned int c = (unsigned int)(idx & (N - 1));
            const int b = (int)(idx >> nq);
            real vr = re[idx], vi = im[idx];
            for (int k = 0; k < cnt; ++k) {
                const GenDesc g = c_gen[applied[gstart + k]];
                const size_t ti = 2 * ((size_t)(gstart + k) * (size_t)B + b);
                const real cs = trig[ti], sn = trig[ti + 1];
                term_diag(vr, vi, cs, sn * (real)g.a0, g.g0, sigma(c, g.yz0));
                term_diag(vr, vi, cs, sn * (real)g.a1, g.g1, sigma(c, g.yz1));
            }
            re[idx] = vr; im[idx] = vi;
        }
    } else {
        const unsigned int q    = (unsigned int)(__ffs((int)g0.flip) - 1);
        const unsigned int lm   = (1u << q) - 1u;
        const unsigned int half = N >> 1;
        const size_t total = (size_t)half * (size_t)B;
        for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
             idx < total; idx += stride) {
            const unsigned int p = (unsigned int)(idx & (half - 1));
            const int b = (int)(idx >> (nq - 1));
            const unsigned int c = ((p & ~lm) << 1) | (p & lm);
            const unsigned int d = c ^ g0.flip;
            const size_t off = (size_t)b * N;
            real vcr = re[off + c], vci = im[off + c];
            real vdr = re[off + d], vdi = im[off + d];
            for (int k = 0; k < cnt; ++k) {
                const GenDesc g = c_gen[applied[gstart + k]];
                const size_t ti = 2 * ((size_t)(gstart + k) * (size_t)B + b);
                const real cs = trig[ti], sn = trig[ti + 1];
                term_pair(vcr, vci, vdr, vdi, cs, sn * (real)g.a0, g.g0,
                          sigma(c, g.yz0), sigma(d, g.yz0));
                term_pair(vcr, vci, vdr, vdi, cs, sn * (real)g.a1, g.g1,
                          sigma(c, g.yz1), sigma(d, g.yz1));
            }
            re[off + c] = vcr; im[off + c] = vci;
            re[off + d] = vdr; im[off + d] = vdi;
        }
    }
}

/* --- B <- E(-eps) A and A <- E(+eps) A, both from the same input ----------- */
__global__ void k_dup_shift_global(real* reA, real* imA, real* reB, real* imB,
                                   int nq, int B, int targetJ,
                                   real epsCos, real epsSin)
{
    const unsigned int N = 1u << nq;
    const GenDesc g = c_gen[targetJ];
    const size_t stride = (size_t)gridDim.x * blockDim.x;

    if (g.flip == 0u) {
        const size_t total = (size_t)N * (size_t)B;
        for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
             idx < total; idx += stride) {
            const unsigned int c = (unsigned int)(idx & (N - 1));
            const real s0 = sigma(c, g.yz0), s1 = sigma(c, g.yz1);
            real ar = reA[idx], ai = imA[idx], br = ar, bi = ai;
            term_diag(ar, ai, epsCos,  epsSin * (real)g.a0, g.g0, s0);
            term_diag(ar, ai, epsCos,  epsSin * (real)g.a1, g.g1, s1);
            term_diag(br, bi, epsCos, -epsSin * (real)g.a0, g.g0, s0);
            term_diag(br, bi, epsCos, -epsSin * (real)g.a1, g.g1, s1);
            reA[idx] = ar; imA[idx] = ai; reB[idx] = br; imB[idx] = bi;
        }
    } else {
        const unsigned int q    = (unsigned int)(__ffs((int)g.flip) - 1);
        const unsigned int lm   = (1u << q) - 1u;
        const unsigned int half = N >> 1;
        const size_t total = (size_t)half * (size_t)B;
        for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
             idx < total; idx += stride) {
            const unsigned int p = (unsigned int)(idx & (half - 1));
            const int b = (int)(idx >> (nq - 1));
            const unsigned int c = ((p & ~lm) << 1) | (p & lm);
            const unsigned int d = c ^ g.flip;
            const size_t off = (size_t)b * N;
            const real s0 = sigma(c, g.yz0), t0 = sigma(d, g.yz0);
            const real s1 = sigma(c, g.yz1), t1 = sigma(d, g.yz1);
            real acr = reA[off + c], aci = imA[off + c];
            real adr = reA[off + d], adi = imA[off + d];
            real bcr = acr, bci = aci, bdr = adr, bdi = adi;
            term_pair(acr, aci, adr, adi, epsCos,  epsSin * (real)g.a0, g.g0, s0, t0);
            term_pair(acr, aci, adr, adi, epsCos,  epsSin * (real)g.a1, g.g1, s1, t1);
            term_pair(bcr, bci, bdr, bdi, epsCos, -epsSin * (real)g.a0, g.g0, s0, t0);
            term_pair(bcr, bci, bdr, bdi, epsCos, -epsSin * (real)g.a1, g.g1, s1, t1);
            reA[off + c] = acr; imA[off + c] = aci;
            reA[off + d] = adr; imA[off + d] = adi;
            reB[off + c] = bcr; imB[off + c] = bci;
            reB[off + d] = bdr; imB[off + d] = bdi;
        }
    }
}

/* --- one block per (circuit, branch); f lands in out[branch*B + b] --------- */
__global__ __launch_bounds__(TPB)
void k_expval_global(const real* __restrict__ reA, const real* __restrict__ imA,
                     const real* __restrict__ reB, const real* __restrict__ imB,
                     int nq, int B, unsigned int flipM, unsigned int yzM, int gM,
                     double* out)
{
    __shared__ double scratch[TPB / 32];
    const unsigned int N = 1u << nq;
    const int col = blockIdx.x;
    const int b   = (col < B) ? col : col - B;
    const real* re = (col < B) ? reA : reB;
    const real* im = (col < B) ? imA : imB;
    const size_t off = (size_t)b * N;

    double acc = 0.0;
    for (unsigned int c = threadIdx.x; c < N; c += TPB) {
        const unsigned int d = c ^ flipM;
        real wr, wi;
        mul_ig(gM, re[off + d], im[off + d], wr, wi);
        const double s = (__popc(d & yzM) & 1) ? -1.0 : 1.0;
        acc += s * ((double)re[off + c] * (double)wr +
                    (double)im[off + c] * (double)wi);
    }
    const double t = block_sum(acc, scratch);
    if (threadIdx.x == 0) out[col] = t;
}

/* --- one block per circuit: Delta, the per-circuit value, and the shots ---- */
template <bool EXACT>
__global__ __launch_bounds__(TPB)
void k_finish_global(const double* __restrict__ fvals, int B,
                     double invEps2, double signConv, long long shotsPerCirc,
                     unsigned long long seed, long long circ0,
                     unsigned long long* nplusOut, unsigned long long* shotsOut,
                     unsigned long long* circOut, double* sumOut, double* sumsqOut,
                     double* dbg)
{
    __shared__ unsigned long long uscratch[TPB / 32];
    __shared__ double s_p;
    const int b = blockIdx.x;

    const double fA = fvals[b], fB = fvals[b + B];
    const double delta = fA - fB;
    const double v = signConv * invEps2 * 0.25 * delta * delta;
    double p = 0.5 + signConv * 0.125 * delta * delta;
    p = (p < 0.0) ? 0.0 : (p > 1.0 ? 1.0 : p);

    if (threadIdx.x == 0) {
        s_p = p;
        atomicAdd(sumOut, v);
        atomicAdd(sumsqOut, v * v);
        atomicAdd(circOut, 1ull);
        if (dbg && b == 0) { dbg[0] = fA; dbg[1] = fB; dbg[2] = v; dbg[3] = p; }
    }
    __syncthreads();

    if (!EXACT) {
        curandStatePhilox4_32_10_t st;
        curand_init(seed ^ 0x9E3779B97F4A7C15ull,
                    (unsigned long long)(circ0 + b) * TPB + threadIdx.x, 0ull, &st);
        unsigned long long tot = 0ull;
        if (shotsPerCirc <= EXACT_BINOMIAL_MAX) {
            unsigned long long local = 0ull;
            for (long long t = threadIdx.x; t < shotsPerCirc; t += TPB)
                if (curand_uniform_double(&st) < s_p) ++local;
            tot = block_sum_ull(local, uscratch);
        } else {
            /* M p (1-p) > 1e6 here, so a rounded normal is indistinguishable in
             * the mean and variance the estimator actually consumes. */
            if (threadIdx.x == 0) {
                const double M  = (double)shotsPerCirc;
                const double mu = M * s_p;
                const double sd = sqrt(M * s_p * (1.0 - s_p));
                double x = mu + sd * curand_normal_double(&st);
                if (x < 0.0) x = 0.0;
                if (x > M)   x = M;
                tot = (unsigned long long)(x + 0.5);
            }
        }
        if (threadIdx.x == 0) {
            atomicAdd(nplusOut, tot);
            atomicAdd(shotsOut, (unsigned long long)shotsPerCirc);
        }
    }
}

/* ==========================================================================
 * Host driver
 * ========================================================================== */

static void worker_global(int gpu, const Ctx* ctx, Totals* tot,
                          const std::vector<HostGen>* G)
{
    CUDA_CHECK(cudaSetDevice(gpu));
    const Config& cfg = ctx->cfg;
    const int nq = ctx->nq;
    const size_t N = (size_t)1 << nq;
    const int nP = (int)ctx->pp.gen.size();
    const int nM = (int)ctx->pm.gen.size();

    std::vector<GenDesc> gd(G->size());
    for (size_t i = 0; i < G->size(); ++i) {
        gd[i].flip = (*G)[i].flip;
        gd[i].yz0  = (*G)[i].yz0;  gd[i].yz1 = (*G)[i].yz1;
        gd[i].g0   = (*G)[i].g0;   gd[i].g1  = (*G)[i].g1;
        gd[i].a0   = (*G)[i].a0;   gd[i].a1  = (*G)[i].a1;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_gen, gd.data(), sizeof(GenDesc) * gd.size()));

    /* batch size from free VRAM: 4 state arrays plus the two angle tables */
    size_t freeB = 0, totalB = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
    const size_t perCirc = 4 * N * sizeof(real)
                         + 2 * (size_t)(nP + nM) * sizeof(real);
    int B = (int)std::min<size_t>((size_t)(freeB * 0.70) / perCirc, 256u);
    if (cfg.batch > 0) B = cfg.batch;
    B = std::max(1, B);
    B = (int)std::min<long long>(B, std::max<long long>(1, g_circ_total));

    real *reA, *imA, *reB, *imB, *trigP = nullptr, *trigM = nullptr;
    unsigned short *dP = nullptr, *dM = nullptr;
    unsigned int *dcP = nullptr, *dcM = nullptr;
    double *d_f, *d_sum, *d_sumsq, *d_dbg;
    unsigned long long *d_nplus, *d_shots, *d_circ;

    CUDA_CHECK(cudaMalloc(&reA, N * B * sizeof(real)));
    CUDA_CHECK(cudaMalloc(&imA, N * B * sizeof(real)));
    CUDA_CHECK(cudaMalloc(&reB, N * B * sizeof(real)));
    CUDA_CHECK(cudaMalloc(&imB, N * B * sizeof(real)));
    if (nP) {
        CUDA_CHECK(cudaMalloc(&trigP, 2 * (size_t)nP * B * sizeof(real)));
        CUDA_CHECK(cudaMalloc(&dP,  nP * sizeof(unsigned short)));
        CUDA_CHECK(cudaMalloc(&dcP, nP * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpy(dP,  ctx->pp.gen.data(), nP * sizeof(unsigned short), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dcP, ctx->pp.ctr.data(), nP * sizeof(unsigned int),   cudaMemcpyHostToDevice));
    }
    if (nM) {
        CUDA_CHECK(cudaMalloc(&trigM, 2 * (size_t)nM * B * sizeof(real)));
        CUDA_CHECK(cudaMalloc(&dM,  nM * sizeof(unsigned short)));
        CUDA_CHECK(cudaMalloc(&dcM, nM * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpy(dM,  ctx->pm.gen.data(), nM * sizeof(unsigned short), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dcM, ctx->pm.ctr.data(), nM * sizeof(unsigned int),   cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&d_f,     2 * (size_t)B * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sum,   sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sumsq, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dbg,   4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_nplus, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_shots, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_circ,  sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_sum,   0, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_sumsq, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_dbg,   0, 4 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_nplus, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_shots, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_circ,  0, sizeof(unsigned long long)));

    cudaDeviceProp pr;
    CUDA_CHECK(cudaGetDeviceProperties(&pr, gpu));
    const int GRID = pr.multiProcessorCount * 16;   /* grid-stride, so generous */

    if (cfg.legacy_log)
        std::printf("  [+] Worker %d bound to cuda:%d | Assigned Batch Size: %lld"
                    "  (global path, %d circuits/launch)\n",
                    gpu, gpu, (long long)B * ctx->spc, B);
    else
        std::fprintf(stderr, "  [+] cuda:%d  global path, B = %d, %.1f MB/circuit\n",
                     gpu, B, perCirc / 1048576.0);
    std::fflush(stdout);

    bool selftest_done = false;

    while (!g_stop.load()) {
        long long take = 0, start = 0;
        {
            long long cur = g_circ_next.load();
            while (cur < g_circ_total) {
                long long want = std::min<long long>(B, g_circ_total - cur);
                if (g_circ_next.compare_exchange_weak(cur, cur + want)) {
                    start = cur; take = want; break;
                }
            }
        }
        if (take <= 0) break;
        const int b = (int)take;
        const auto t0 = std::chrono::steady_clock::now();

        k_init_global<<<GRID, TPB>>>(reA, imA, nq, b, (real)(1.0 / std::sqrt((double)N)));
        if (nP)
            k_angles_global<<<((size_t)nP * b + TPB - 1) / TPB, TPB>>>(
                trigP, dcP, nP, b, cfg.seed, start, (real)1);
        if (nM)
            k_angles_global<<<((size_t)nM * b + TPB - 1) / TPB, TPB>>>(
                trigM, dcM, nM, b, cfg.seed, start, (real)(-1));

        for (size_t gi = 0; gi < ctx->pp.gstart.size(); ++gi)
            k_gate_group<<<GRID, TPB>>>(reA, imA, nq, b, dP,
                                        ctx->pp.gstart[gi], ctx->pp.gcnt[gi], trigP);

        k_dup_shift_global<<<GRID, TPB>>>(reA, imA, reB, imB, nq, b, cfg.target_j,
                                          ctx->epsCos, ctx->epsSin);

        for (size_t gi = 0; gi < ctx->pm.gstart.size(); ++gi) {
            k_gate_group<<<GRID, TPB>>>(reA, imA, nq, b, dM,
                                        ctx->pm.gstart[gi], ctx->pm.gcnt[gi], trigM);
            k_gate_group<<<GRID, TPB>>>(reB, imB, nq, b, dM,
                                        ctx->pm.gstart[gi], ctx->pm.gcnt[gi], trigM);
        }

        k_expval_global<<<2 * b, TPB>>>(reA, imA, reB, imB, nq, b,
                                        ctx->flipM, ctx->yzM, ctx->gM, d_f);
        if (cfg.exact)
            k_finish_global<true><<<b, TPB>>>(d_f, b, ctx->invEps2, ctx->signConv,
                ctx->spc, cfg.seed, start, d_nplus, d_shots, d_circ,
                d_sum, d_sumsq, cfg.selftest ? d_dbg : nullptr);
        else
            k_finish_global<false><<<b, TPB>>>(d_f, b, ctx->invEps2, ctx->signConv,
                ctx->spc, cfg.seed, start, d_nplus, d_shots, d_circ,
                d_sum, d_sumsq, cfg.selftest ? d_dbg : nullptr);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        const double secs = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - t0).count();

        if (cfg.legacy_log) {
            std::printf("[Worker %d | cuda:%d] Evaluated batch of %lld shots in %.4f seconds.\n",
                        gpu, gpu, take * ctx->spc, secs);
            std::fflush(stdout);
        }

        {   /* republish cumulative counters for the monitor */
            Slot sl;
            CUDA_CHECK(cudaMemcpy(&sl.nplus,    d_nplus, sizeof(sl.nplus),    cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&sl.shots,    d_shots, sizeof(sl.shots),    cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&sl.circuits, d_circ,  sizeof(sl.circuits), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&sl.sum,      d_sum,   sizeof(sl.sum),      cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&sl.sumsq,    d_sumsq, sizeof(sl.sumsq),    cudaMemcpyDeviceToHost));
            std::lock_guard<std::mutex> lk(tot->mtx);
            tot->live[(size_t)gpu] = sl;
        }

        if (cfg.selftest && !selftest_done) {
            selftest_done = true;
            double dbg[4];
            CUDA_CHECK(cudaMemcpy(dbg, d_dbg, sizeof(dbg), cudaMemcpyDeviceToHost));
            std::fprintf(stderr, "  [selftest cuda:%d] f(+eps)=%.12g  f(-eps)=%.12g  "
                                 "value=%.12g  p=%.12g\n", gpu, dbg[0], dbg[1], dbg[2], dbg[3]);
        }
    }

    if (trigP) CUDA_CHECK(cudaFree(trigP));
    if (trigM) CUDA_CHECK(cudaFree(trigM));
    if (dP)  CUDA_CHECK(cudaFree(dP));
    if (dM)  CUDA_CHECK(cudaFree(dM));
    if (dcP) CUDA_CHECK(cudaFree(dcP));
    if (dcM) CUDA_CHECK(cudaFree(dcM));
    CUDA_CHECK(cudaFree(reA)); CUDA_CHECK(cudaFree(imA));
    CUDA_CHECK(cudaFree(reB)); CUDA_CHECK(cudaFree(imB));
    CUDA_CHECK(cudaFree(d_f)); CUDA_CHECK(cudaFree(d_sum)); CUDA_CHECK(cudaFree(d_sumsq));
    CUDA_CHECK(cudaFree(d_dbg));
    CUDA_CHECK(cudaFree(d_nplus)); CUDA_CHECK(cudaFree(d_shots)); CUDA_CHECK(cudaFree(d_circ));
}
