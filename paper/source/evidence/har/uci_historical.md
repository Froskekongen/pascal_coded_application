# UCI HAR: historical experiments and mathematical findings

This account preserves the experiments that led to the three models in
[uci_har.md](uci_har.md). The example now implements only those retained
models. The historical parameter sweeps, alternative representations and
three-head runners are no longer supported execution paths. Their recorded
CSV/JSON results remain unchanged, so individual scores and confusion matrices
can still be inspected.

The common dataset was the official subject-disjoint split: 7,352 training
and 2,947 test windows, each with 128 observations at 50 Hz. Earlier comparisons
used ridge, RBF SVM and random forest; later comparisons fixed ridge at alpha
10. Scaling used training data only. The feature choices were repeatedly
examined on this test split. All maxima below are descriptive, not independent
validation-selected performance estimates. Percentages are test accuracy
unless explicitly labelled otherwise.

## Evidence and numerical reliability

Each directory below retains its `results.csv` and `run.json`. The latter
records settings, versions, confusion matrices and, where available, paired
prediction changes. The 2,344 rows overlap in their controls; they are not
2,344 independent tests of one claim.

| Experiment | Fits | Recorded evidence |
|---|---:|---|
| Initial signatures, lead–lag and direct moments | 234 | [CSV](results/results.csv), [metadata](results/run.json) |
| Fourier, delays and Pascal mechanisms | 252 | [CSV](temporal_results/results.csv), [metadata](temporal_results/run.json) |
| Energy envelopes and delayed products | 234 | [CSV](observable_results/results.csv), [metadata](observable_results/run.json) |
| Ridge P/depth and pretransform sweep | 85 | [CSV](pascal_results/results.csv), [metadata](pascal_results/run.json) |
| Generic feature families | 133 | [CSV](family_results/results.csv), [metadata](family_results/run.json) |
| Pooled-family and spectral ablations | 438 | [CSV](ablation_results/results.csv), [metadata](ablation_results/run.json) |
| Small pooled readouts | 414 | [CSV](pool_results/results.csv), [metadata](pool_results/run.json) |
| Ordinary value–clock graphs | 57 | [CSV](occupation_results/results.csv), [metadata](occupation_results/run.json) |
| Pascal-coded clocks | 310 | [CSV](coded_occupation_results/results.csv), [metadata](coded_occupation_results/run.json) |
| Matched ordinary-signature controls | 187 | [CSV](consolidated_results/results.csv), [metadata](consolidated_results/run.json) |

Initial family runs encountered native Cholesky crashes and cached-array
corruption. The recorded family-and-later runs use an equivalent float64
ridge solve through NumPy's general-LU route. Alpha and intercept treatment
did not change. Fresh-process LSQR checks reproduced the headline predictions;
cache checksums and confusion-matrix comparisons passed. Earlier multithreaded
temporal runs also failed, with no established root cause. Their successful
recorded runs limited numerical threads; legacy experiments retain their
original sklearn solver.

Extraction was often shared between readouts, so its reported times must not
be summed across rows. Extraction of the supplied 561 columns was never
measured: its zero in older CSVs is a placeholder. The final 187-comparison
run took 148.4 seconds, including feature extraction; its ridge fits together
took about seven seconds. The retained-only replacement now has separate,
shorter timings.

## What the supplied 561 columns motivated

The distributed `features.txt`, `features_info.txt` and `README.txt` describe
median filtering, a third-order Butterworth low-pass filter at 20 Hz and a
0.3 Hz gravity/body separation. Derived signals include acceleration jerk,
gyroscope derivatives, norms and FFTs. There are 265 time-domain columns,
289 frequency-domain columns and seven mean-vector angles.

The inventory consists of eight common statistics on 33 scalar signals
(264 columns), 17 signal-magnitude areas, 80 order-four Burg coefficients,
15 within-triad correlations, 52 peak-frequency/mean-frequency/skew/kurtosis
columns, 126 band energies and seven angles. The common statistics are mean,
std, median absolute deviation, maximum, minimum, energy, interquartile range
and entropy. The 126 bands use fourteen intervals on each axis of three FFT
triads; repeated names are distinct columns, not duplicates to remove.

This is a description rather than an executable specification: entropy,
FFT endpoints, derivative boundaries and some moment conventions are not fully
specified. The notes mention 64 FFT bins; a standard real FFT of 128 samples
has 65 including both endpoints. The supplied features are also normalized to
[-1,1]. Generic feature identities therefore concerned the underlying
statistics, not exact reproduction of all supplied column values.

## Initial signatures, quotients and direct moments

| Representation | Features | Ridge | SVM | RF |
|---|---:|---:|---:|---:|
| Supplied 561 | 561 | 95.72 | 95.49 | 92.43 |
| Ordinary anchored total-acceleration/gyro tensor | 56 | 75.57 | 87.44 | 87.92 |
| Linear-offset anchored tensor | 210 | 86.29 | 88.29 | 88.73 |
| Mean/std/covariance/roughness | 54 | 88.53 | 89.85 | 92.77 |
| Integrated lead–lag offset logsignature | 351 | 87.00 | 88.39 | 83.03 |
| 561 + ordinary anchored tensor | 617 | 97.05 | 96.91 | 92.94 |
| 561 + offset anchored tensor | 771 | 97.01 | 96.64 | 92.57 |

Restoring gravity-bearing total acceleration mattered more than merely adding
an anchor to body acceleration. Integrated lead–lag made moments algebraically
available but did not close the gap; explicitly normalized statistics were
more useful to these heads. In this first hybrid comparison, the ordinary
addition beat the matched offset addition under all three heads.

The acceleration/gyro quotient removed selected cross-modality **areas**.
It did not imply statistical independence, rotation invariance or removal of
all cross-products: tensor quotients retained symmetric endpoint products.
Quotients mostly exchanged accuracy for fewer coordinates. Extraction projected
a dense tensor, so no sparse-computation speedup was demonstrated.

For increments \(z_r\), define sampled covariation
\(Q_{ij}=\sum_rz_r^iz_r^j\). The lead-before-lag staircase satisfies
\(A_{L_i,G_j}=A_{ij}+Q_{ij}/2\), hence
\(Q_{ij}=A_{L_i,G_j}+A_{L_j,G_i}\), where \(A\) denotes signed area.
For integrated samples \(z_r=x_r/N\), \(NQ\) is the sample second-moment
matrix. For raw increments \(z_r=x_{r+1}-x_r\), it instead measures signal
roughness. An origin anchor must not count as an observed increment in the
latter statistic. Neither fixed-grid quantity is automatically a
sampling-independent stochastic quadratic variation. One cross-channel
lead–lag area alone also contains ordinary area, so is not a covariance.

## Fourier, short delays and integrated observables

| Selected historical representation | Features | Ridge | SVM | RF |
|---|---:|---:|---:|---:|
| Ordinary lag-2 logsignature | 91 | 85.34 | 90.77 | 92.06 |
| Linear-offset lag-2 logsignature | 351 | 87.21 | 91.41 | 91.99 |
| Fourier-bin-2/4 cross-spectrum | 78 | 80.62 | 90.26 | 92.87 |
| 561 + Fourier cross-spectrum | 639 | 96.67 | 97.05 | 94.10 |
| 561 + Fourier–Pascal tensor | 1,491 | 96.78 | 95.39 | 92.16 |
| Pascal gap moments | 78 | 65.35 | 80.12 | 79.71 |

Delays helped standalone signatures; lag 2 (40 ms) was generally preferable
to lag 5 (100 ms), but their hybrid gains were limited. Fourier–Pascal raised
standalone tensor ridge from 85.34% for its matched bin-4 carrier to 90.30%.
It did not improve the strongest 561 hybrids. Compact Fourier cross-products
were often preferable to every tensor coordinate.

A second trial integrated two smoothed body/gyro energy envelopes and four
lag-2/5 triad dot-product streams, with six posture means. On the combined
observable path, ordinary/Pascal/Fourier–Pascal tensor accuracies were
82.83/83.54/85.04 for ridge, 85.65/87.04/90.50 for SVM and 84.42/86.09/87.00
for RF. Fourier–Pascal products of first levels already gave
84.93/90.16/87.68, and beat full tensors under every 561-hybrid head.
The best observable hybrids reached 96.34/96.13/93.79, below the strongest
earlier hybrids. This did not support keeping that coarse observable embedding
or all directed second-level coordinates; it did not refute their mathematical
ability to encode order.

## What the Pascal and Fourier calculations establish

At fixed width, Pascal, monomial/Newton and QR polynomial codes span the same
space. An invertible code-basis change acts by tensor powers at each signature
level, preserving full-signature information and the unregularized linear
span. It changes ridge's penalty, SVM distances and forest splits. For the
128-by-3 code matrices, measured condition numbers were about 10,618 for raw
Pascal, 22.7 for normalized monomials and one for QR. Standalone P3 tensor ridge
was 84.22% in QR coordinates and 80.05% in Newton coordinates; this is not an
information gain from a basis change.

For integrated increments \(z_r=x_r/N\), positions \(t_r=r/N\), and integer
\(q\geq1\), polynomial coding gives the exact readout

\[
D_q^{ij}=\sum_{r<s}(t_s-t_r)^qz_r^iz_s^j
=\sum_{a=0}^q(-1)^a\binom qa S_{i,a;j,q-a}.
\]

The same-segment geometric terms cancel because \((t_r-t_r)^q=0\).
On an **event staircase**, whose clock is fixed during each observation
increment, \(D_q^{ij}=q!S^{i\,t^q\,j}\). This trades ordinary word depth for
code width; the identity does not apply unchanged to simultaneous clock/signal
interpolation. Shifting the numerical time origin preserves gaps; cropping
or cyclically moving samples through the boundary need not.

These readouts are not all new independent interactions. If
\(m_i^{(a)}=\sum_rt_r^az_r^i\) and \(\mu_i=m_i^{(0)}\), then

\[
D_1^{ij}-D_1^{ji}=\mu_im_j^{(1)}-m_i^{(1)}\mu_j,\qquad
D_2^{ij}+D_2^{ji}=\mu_im_j^{(2)}+m_i^{(2)}\mu_j-2m_i^{(1)}m_j^{(1)}.
\]

For a scalar mean-zero signal,
\(D_1=-N^{-1}\sum_{k=1}^{N-1}(\sum_{r<k}z_r)^2\): it is a transformed
energy statistic. A checked ten-sample example illustrates the finite-depth
separation:

```text
A = [-1, -1, -1, -1,  1,  1,  1, 1, -1, 1]
B = [-1, -1, -1,  1,  1, -1, -1, 1,  1, 1]
```

Both have mean zero, std one, the same first/last values, roughness 4/3 and
weighted first moment 0.17. Their integrated space–time depth-two signatures
agree, but \(D_1(A)=-0.045\) and \(D_1(B)=-0.037\). This is not a collision
of all 561 columns or evidence that HAR labels depend on this distinction.

Fourier modulation gives the related causal spectrum

\[
H_{ij}(\omega)=\sum_{r<s}e^{i\omega(t_s-t_r)}z_r^iz_s^j,
\qquad H_{ij}^{(q)}(0)=i^qD_q^{ij}.
\]

For \(F_i(\omega)=\sum_rz_r^ie^{i\omega t_r}\),
\(\overline F_iF_j=Q_{ij}+H_{ij}+\overline H_{ji}\). The diagonal term
cannot be omitted. Modulated gap moments describe derivatives near a chosen
carrier, and are determined by the full directed aperiodic cross-correlation
sequence. A local Taylor interpretation does not justify extrapolation across
many DFT bins. One tested carrier is not a claim to have matched activity cadence.
FFT preprocessing followed by frequency-order signatures is also different
from Fourier weighting of observations ordered in time.

## From generic generators to smaller pools

The ridge P/depth sweep did not favor increasing complexity monotonically.
The raw P3/depth-2 hybrid reached 97.12%; P2, P4 and P6 at depth two gave
97.01%, while P3/depth three gave 94.23%. Difference, energy, lag-product and
spectral-power pretransforms did not beat the raw hybrid; their strongest
hybrids were at most 96.10%.

Four generic families then separated moments, filters/delays, spectral shape
and geometric contractions. Their combined pool had 724 coordinates and
97.79% accuracy; adding raw342 produced 1,066 features and 98.07%. Adding the
supplied561 to the pool instead reached **98.37% with 1,285 features**; adding
raw again reduced this to 98.17%. Thus 98.10% is the best **retained** result,
not the highest historical score.

Concatenated P3/depth-2 family signatures without pools achieved only 93.86%,
or 94.33% with raw signatures. The normalized statistical readouts mattered.
Spectral was the strongest individual family: with its pool, ordinary P1/depth
three gave 96.27%, compared with P3/depth two at 96.06%. With raw signatures
also present, spectral P3/depth two reached 96.95%.

Reducing the 1,066-feature model to spectral+rotation+raw kept 98.07% with
815 features. Dropping the twelve lagged-cosine summaries yielded the retained
803/98.10 model, only one error better. The useful finding was dimension
reduction. Removing the separate moment/filter pools did not remove every
derivative or delay: difference spectra and delayed dot products remained.
Raw modality-quotient and separate-modality versions trailed at 97.52%/722
and 97.73%/641. A 740-coordinate spectral tensor and a 550-coordinate reduction
of its nonconstant affine span gave different ridge results despite preserving
the same linear information, illustrating the effect of coordinate penalties.

A 234-feature small pool with spectral160 and raw342 gave 736/97.93%; using
raw log171 gave 565/97.86%. Subsequent reductions used spectral log100:

| Raw log171 + spectral log100, plus | Total features | Accuracy |
|---|---:|---:|
| Nothing | 271 | 93.48% |
| Energy/cosine26 | 297 | 96.91% |
| Moment pool66 | 337 | 97.56% |
| Moment pool102 | 373 | 97.79% |
| Moment pool142 | 413 | 97.90% |
| Pool234 | 505 | 97.90% |

Broader signature-only alternatives peaked at 95.35% with 892 features.
The later near-98% results must be credited to geometry, amplitude and calibrated
moments as well as raw/spectral signatures.

## Moment readouts, ordinary graphs and coded clocks

A moment of values is not a product of means. A scalar integrated signature
depends only on its endpoint, so greater depth alone cannot recover sample
energy. One must change the observable, preserve sampled covariation, or use
a graph construction. Also distinguish moments of **amplitudes** from moments
of **frequency locations**. For bin energies \(e_k=Ep_k\),

\[
K^{-1}\sum_k(\sqrt{e_k})^r=E^{r/2}K^{-1}\sum_k p_k^{r/2}.
\]

Energy and concentration powers determine amplitude moments, but this nonlinear
reconstruction need not preserve a linear classifier's performance. Similarly,
std/skew/kurtosis are explicitly calibrated moment readouts. Zero-energy spectra
invalidate the unit-mass reductions and were rejected by the compact extractor.
Norms after axiswise training scaling do not give a rotation-invariant spectrum.

The closed-value graph identity and its proof are retained in
[uci_har.md](uci_har.md#one-graph-construction-for-moments-and-trends). Its
historical results explain the readout choice:

| Representation added to the raw/spectral log pair | Total features | Accuracy |
|---|---:|---:|
| Raw moments + log-energy | 373 | 97.01% |
| Full ordinary graphs through depth 3 + log-energy | 396 | 96.98% |
| Full ordinary graphs through depth 4 + log-energy | 501 | 97.22% |
| Calibrated graph readouts | 373 | 97.79% |
| Calibrated + standardized amplitude ordering | 393 | 97.93% |
| Calibrated + amplitude/invariant ordering | 429 | 98.00% |
| Calibrated + all available ordering coordinates | 478 | 97.86% |

Calibrated graph readouts reproduced the direct-pool predictions exactly.
Unnormalized ordering was less useful: amplitude ordering left 97.79%, and
invariant ordering gave 97.83%. Additional coding of log cumulative-energy
paths gave little: P1 endpoint373/97.79%, P2 log413/97.83%, P3 log473/97.76%.
The scalar log-energy path itself has vertices
\(\log(1+\sum_{k\le j}e_k)\), so its first level is exactly \(\log(1+E)\).

Coding only the graph clocks gives
\((\log S)^{x^rt_a}=\sum_kq_a(k)z_k^r/(r!N)\). Coding the value jumps too
would change closure and the identity; that alternative was not tested.
The coded reconstruction of the 429 model preserved every prediction. Removing
the separate energy block left calibrated82 at 353/97.86%; adding twelve
invariant linear trends gave 365/98.00%; adding three cosine trends gave the
retained 368/98.03%. A wider 463-feature choice reached 98.07% but did not
meet the compactness objective. Ordinary graph `xtt` and the P2 trend differ
only by a per-coordinate scale removed by StandardScaler; a general basis
change need not preserve ridge in this way.

## Final ordinary-signature comparisons

The final run held the other retained blocks and alpha fixed while swapping
one signature branch. It tested tensor and log readouts, clocks, anchors,
integration with/without clocks, delays 1/2/5, lead–lag, integrated lead–lag,
and paper-inspired variants. Raw time/integrated paths reached depth four;
wide lagged paths reached depth three, with branch outputs capped at 1,600
and dense intermediates at 3,000 coordinates.

| Fixed context | Retained accuracy | Best conventional raw replacement | Features | Accuracy | Extra errors |
|---|---:|---|---:|---:|---:|
| Compact | 98.03% | Lead–lag tensor, depth 2 | 377 | 97.46% | 17 |
| Reduced pool | 98.10% | Lag-1 tensor, depth 2 | 641 | 97.83% | 8 |
| Supplied561 | 97.12% | Anchored tensor, depth 2 | 615 | 97.05% | 2 |

All maxima use the matched total-acceleration/gyro source. Relative to each
retained model, those replacements corrected 15/16/26 errors and introduced
32/24/28. The supplied561 comparison is especially close and favors the ordinary
representation in dimension. Removing the raw branch entirely gave
197/96.17%, 461/97.18% and 561/95.72% in the three contexts.

For spectral controls, an ordinary frequency-clock/cumulative-mass staircase
avoided the triviality of a one-dimensional unit-mass signature. Holding the
other compact blocks fixed gave:

| Ordinary spectral replacement | Total features | Accuracy |
|---|---:|---:|
| No spectral branch | 268 | 96.78% |
| Log depth 2 | 288 | 97.49% |
| Log depth 3 | 328 | 97.79% |
| Tensor depth 3 | 428 | 97.83% |
| Log depth 4 | 388 | 97.83% |
| Tensor depth 4 | 708 | 98.00% |
| Log depth 5 | 508 | 97.93% |
| Tensor depth 5 | 1,308 | 97.83% |
| Depth-5 projection onto Pascal coordinates | 368 | 98.03% |

The depth-four tensor is only one error behind compact368, using 440 rather
than 100 spectral features. Its paired changes were eight corrections and nine
introduced errors. This chiefly supports compactness. Joint ordinary
replacements reached 96.44%/427, but used only two predetermined raw choices
and were not an exhaustive pairing search.

The projection is an exact ordinary-signature identity. With letters `t` for
frequency and `x` for cumulative mass, the monomial endpoints are
\((1,S^{tx},2S^{ttx})\), and antisymmetric differences \(D=2J\) are

\[
D_{01}=S^{xtx},\qquad
D_{02}=2(S^{txtx}+S^{xttx}),\qquad
D_{12}=2(S^{ttxtx}+S^{txttx}).
\]

After QR conversion, this independently computed projection changed zero
predictions. Explicit ordinary signatures of the raw polynomial lift likewise
reproduced all three retained predictions, and ordinary graph trends reproduced
the fifteen coded trends. The results concern useful shallow coordinates, not
information inaccessible to ordinary signatures of appropriate lifted paths.

Conditional deletion of compact calibration blocks gave 97.15% without
amplitude40, 96.34% without cosine6, 97.42% without invariant36, and 97.86%
without trends15. These remove named coordinates, not every occurrence of the
phenomena elsewhere, and are not additive or causal feature-importance scores.

### Relation to the reference paper

The single local [HAR signature paper](../../reference_papers/signature_human_activity.pdf)
uses body acceleration plus gyro and a time coordinate. Section 2.2 describes a
simultaneous one-step shift, whereas Appendix A.2 describes a doubled path and
channel count. Both simultaneous shift and conventional staircase lead–lag
were tested separately. A duplicate endpoint changes a 255-vertex staircase
into 256 points without changing its signature.

Body-source adaptations reached at most 97.35%, 97.69%, 96.17% in the three
contexts, but also changed the raw input by removing gravity. The paper's
validation-selected RF/MLP protocol was not replicated: the comparison kept
ridge and existing training input scaling. Its deepest 137,256-coordinate time
tensor was outside the experiment's fast budget. Its reported score therefore
is not a directly matched result here.

## Directions retained as possibilities, not implemented conclusions

The separation of transforms, augmentations and readouts follows the
[generalised signature method](https://arxiv.org/abs/2006.00873).

Histogram or kernel observables can supply occupation probabilities, with
entropy a nonlinear readout and quantiles generally approximated by finite
bins. Lag-product moments supply ingredients for autoregression, not an exact
implementation of Burg's algorithm. Ordering a correlation curve by lag is
different from ordering the original observations in time. Contracting a
vector signature spatially is also different from forming Gram streams first.

[Iterated sums](https://arxiv.org/abs/1906.05823) have a merged-product term,
\(I^iI^j=I^{ij}+I^{ji}+I^{[ij]}\), and geometric second level is
\(S^{ij}=I^{ij}+I^{[ij]}/2\). Higher-order quasi-shuffle features require the
corresponding genuine product corrections. This remains a separate algebraic
proposal, not a renamed geometric signature.

Empirical expected signatures of scalar straight segments give
\(M=L^{-1}\sum_k\exp(z_ke)\). Its tensor logarithm has coefficients mean,
variance/2 and third central moment/6 through degree three. This is generally
**not a path logsignature**: averaging signatures need not preserve the
geometric identities, and its logarithm need not be a Lie element. Averaging
scalar path logsignatures instead would retain only the mean.

These directions and the distinction between selected-word evaluation,
nonlinear statistical readouts and full signatures remain documented in
[tensorsig_suggestions.md](tensorsig_suggestions.md). No new equivalence,
classification guarantee or universal Pascal advantage follows from
consolidating the notes.

## Provenance

This account consolidates `EXPERIMENT_HISTORY.md`, `pascal_investigation.md`,
`feature_families.md`, `ABLATIONS.md`, `POOL_STORY.md`, `occupation_theory.md`,
`OCCUPATIONS.md`, `CODED_OCCUPATIONS.md`, the control sections of
`CONSOLIDATED.md`, and the generated results summaries. Retained definitions
and motivation from those files and `MOTIVATION.md` are incorporated in
`uci_har.md`. Repeated tables and obsolete execution instructions were removed;
exact saved measurements remain in the evidence files above.
