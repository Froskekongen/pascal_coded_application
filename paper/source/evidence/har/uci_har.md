# Retained signature models for UCI human activity recognition

A phone records both posture and movement. The direction of its average
acceleration helps distinguish sitting from standing, while the distribution
of oscillations helps distinguish walking from climbing stairs. Neither a
window average nor a spectrum alone describes how acceleration and rotation
unfold together. The retained models combine these three kinds of information:
**signal levels and temporal order, spectral shape and scale, and relative
sensor geometry**.

Among the three retained models, the strongest measured result is **98.10%
with 803 features**: a spectral and geometric pool concatenated with a raw Pascal
signature. The compact model obtains **98.03% with 368 features**, replacing
much of that pool by spectral Pascal logsignatures and selected value–clock
graph readouts. The difference is two test windows; the reduction of 435
features is the more substantial finding. A third model appends the same raw
Pascal tensor to the supplied 561 features and obtains **97.12%**.

This account explains only these retained constructions. The earlier searches,
ordinary-signature controls and rejected alternatives are collected in
[uci_historical.md](uci_historical.md). Proposed library additions remain in
[tensorsig_suggestions.md](tensorsig_suggestions.md). Notation follows the
[first levels and signed areas in the paper](../../paper_revision_15/source/main.tex).

## The observations and what changes before taking a signature

The UCI Human Activity Recognition dataset contains six activities: walking,
walking upstairs, walking downstairs, sitting, standing and lying down. A
waist-mounted phone samples acceleration and angular velocity at 50 Hz.
Windows contain 128 samples, or 2.56 seconds, with 50% overlap. We use the
supplied subject-disjoint split: 7,352 training windows and 2,947 test windows,
from 21 and 9 subjects respectively.

The files provide total acceleration, body acceleration and gyroscope signals,
each with three axes. We form gravity as total minus body acceleration. These
are already processed dataset signals, not measurements reconstructed from
an unfiltered recording. The supplied 561-column table is a separate input to
one retained model.

Two kinds of scaling serve different purposes. Raw signatures and spectral
signals use a mean and population standard deviation for each axis, fitted
across training windows and samples. This preserves differences between window
means. The geometric branch instead divides each triad by one training RMS
scalar, without centering or rescaling individual axes. That common scale
preserves norms and dot products under a shared orthogonal change of spatial coordinates.
A final training-only standardization is applied to the assembled features
before ridge classification.

## Raw Pascal signatures: retain levels, then locate their interactions

Recall the paper's first and second signature levels for path increments
\(v_k\):

\[
s=\sum_k v_k,\qquad
B=\sum_{k<l}v_k\otimes v_l+\frac12\sum_kv_k\otimes v_k,
\qquad J=\frac12(B-B^\top).
\]

The vector \(s\) records net displacement. The area \(J_{ij}\) compares the
order in which channels \(i\) and \(j\) move. The symmetric part satisfies
\(B+B^\top=ss^\top\), so a depth-two tensor signature consists of endpoints,
endpoint products and areas.

For HAR, a signature of successive differences alone would discard constant
sensor levels. That would remove useful posture information. Instead, let
\(x_k\in\mathbb R^6\) contain standardized **total acceleration and gyroscope**
samples and use \(x_k/N\) as increments, with \(N=128\). The endpoint of this
integrated path is the window mean. Integration here is a feature transform;
it is not a claim that integrating a gyroscope and an accelerometer in the
same way reconstructs a physical trajectory.

Next give every increment weighted copies. Form the Pascal columns
\(b_a(k)=\binom{k}{a}\), \(a=0,1,2\), and orthonormalize them by QR, choosing
a positive diagonal in the triangular factor. Write the resulting columns as
\(q_0,q_1,q_2\), and define

\[
v_k^{i,a}=\frac{x_k^i}{N}q_a(k),\qquad a=0,1,2.
\]

This is the paper's Pascal construction in a better conditioned basis.
Normalizing each original Pascal column as in the paper gives the same QR
basis. The three columns span constant, linear and quadratic functions of
sample position. In particular, \(q_0=1/\sqrt N\); its endpoint is a fixed
multiple of the mean, while the other endpoints locate signal levels within
the window.

An area between two coded channels is

\[
J_{(i,a),(j,b)}=
\frac1{2N^2}\sum_{k<l}
\left[q_a(k)q_b(l)x_k^ix_l^j
      -q_b(k)q_a(l)x_k^jx_l^i\right].
\]

It asks whether values in one channel precede values in another, with weights
that distinguish early and late occurrences. For one nonnegative signal and
the unorthogonalized codes \(1,u\), write \(z_k=x_k/N\). The corresponding area is
\(\frac12\sum_{k<l}(u_l-u_k)z_kz_l\): positive contributions grow when the
observations are farther apart. Signed sensor values can cancel, so the area
is a summary of weighted interaction, not a list of recovered delays.

The parameter choices have distinct roles. **P3** gives quadratic position
resolution; **depth 2** gives interactions between pairs of increments. They
are small, interpretable choices that worked in this experiment, not proven
optimal parameters.

Six signals with three copies give 18 lifted channels. The two larger models
retain \(s\) and all entries of \(B\), giving \(18+18^2=342\) features. The
compact model retains \(s\) and the upper triangle of \(J\), giving
\(18+\binom{18}{2}=171\). These determine one another through
\(B=J+ss^\top/2\), but the reconstruction is nonlinear. A linear classifier
on the full tensor can use endpoint products directly; a linear classifier
on the logsignature cannot form them automatically.

## The 803-feature model: broad spectral summaries and geometric pools

The best retained model combines **342 raw tensor coordinates, 359 spectral
summaries and 102 geometric summaries**. Its spectral branch is a pool of
explicit statistics; it does not contain the spectral Pascal branch described
in the next section.

### Spectral shape and scale

Use five triads: body acceleration, gravity, gyroscope, and first differences
of body acceleration and gyroscope. Differences are formed before their own
training scaling; no sample-rate multiplier is used. Append the norm of each
standardized triad to its three axes. This produces twenty scalar signals.
The norm is taken before the FFT: the spectrum of a norm is not the sum of
the three axis spectra.

For a scalar signal \(z_0,\ldots,z_{n-1}\), compute the one-sided discrete
Fourier transform and bin energies

\[
F_k=\sum_{r=0}^{n-1}z_r e^{-2\pi i kr/n},\qquad
e_k=w_k|F_k/n|^2,\qquad E=\sum_k e_k.
\]

The weight \(w_k\) is two for interior positive-frequency bins and one at DC
and, when present, Nyquist. Then \(E=n^{-1}\sum_r z_r^2\). Define amplitudes
\(a_k=\sqrt{e_k}\) and normalized masses \(p_k=e_k/E\) for nonzero spectra.
The frequencies \(f_k=2k/n\) are measured relative to 25 Hz. The original
signals have 65 bins; the 127-sample differences have 64 and no Nyquist bin.

Each scalar signal contributes eighteen summaries:

| Readouts | Count | What they make accessible to ridge |
|---|---:|---|
| Mean, std, skewness and kurtosis of bin amplitudes | 4 | Typical amplitude, unevenness and large peaks |
| Energy \(E\) and \(\log(1+E)\) | 2 | Absolute scale in two useful nonlinear coordinates |
| Frequency centroid and spread | 2 | Where the energy lies and how widely it is distributed |
| Spectral entropy and peak frequency | 2 | Concentration and the dominant oscillation |
| Mass in eight equal frequency bands | 8 | A coarse description of spectral shape |

Centroid and spread are \(\sum p_kf_k\) and
\(\sqrt{\sum p_k(f_k-\sum_jp_jf_j)^2}\). Entropy is
\(-\sum p_k\log p_k\), with zero terms defined by continuity. The bands
partition \([0,1]\), with the final interval including one. Kurtosis is the
standardized fourth central moment, without subtracting three; all variances
use population normalization. Thus the pool contains \(20\times18=360\)
columns before removing one column constant on the training set.

This mixture is deliberate. Normalized spectral shape distinguishes rhythms
without their overall strength; energy restores that strength. Band masses
and the peak make relatively localized frequency information available to a
linear classifier. Entropy and amplitude moments summarize different aspects
of concentration. The pool is richer than a single centroid or a single
measure of spread, but still follows one recipe for every scalar signal.

### Relative geometry and persistence

Let \(v_g(k)\), \(g=1,2,3\), denote body acceleration, gravity and gyroscope,
each divided by its single training RMS scale. Construct twelve scalar streams:

\[
\|v_g(k)\|^2,\qquad
\langle v_g(k),v_h(k)\rangle\ (g<h),\qquad
\langle v_g(k),v_g(k-\tau)\rangle\quad(\tau=2,5).
\]

There are three energies, three simultaneous products and six delayed products.
The delays are 40 and 100 milliseconds. Only valid pairs are used. A large
positive delayed product indicates aligned vectors with a large product of
magnitudes. A negative product indicates opposed directions; a small product
can reflect small magnitudes or weak alignment.

For each stream retain mean, std, mean absolute value, RMS, skewness, kurtosis,
minimum and maximum: \(12\times8=96\) features. Add mean and std of the three
simultaneous cross-triad cosines, giving six more. The cosine separates
alignment from magnitude; the unnormalized dot product keeps both.

These streams are unchanged by the same fixed orthogonal transformation of
all three triads throughout a window. The complete classifier is not rotation
invariant: its raw and axis-standardized spectral branches retain orientation.
The geometric pool contributes a complementary description of coordination.

### How far the pooling has a signature interpretation

Averaging an observable \(\phi(z_k)\) is the first level of a path whose
increments are \(\phi(z_k)/L\). Means, energies and simultaneous products fit
this construction directly. Spectral entropy and band masses similarly pool
transformed bin masses. This explains a reusable design: transform the signal,
choose a meaningful observable, then pool or retain its order through a
signature.

It does not make every statistic a shallow linear coordinate of the original
raw signature. Std, normalized moments and cosines have nonlinear readouts;
extrema and peak location remain explicit nonlinear operations. Their usefulness
to ridge is a property of the complete feature generator, including these
operations. The compact model gives a more unified signature construction for
the subset of statistics that it retains.

## The 368-feature model: spectral areas and value–clock moments

The compact model has six blocks:

| Block | Features |
|---|---:|
| Raw Pascal P3/depth-2 logsignature | 171 |
| Spectral Pascal P3/depth-2 logsignatures | 100 |
| Spectral-amplitude mean/std | 40 |
| Cross-triad cosine mean/std | 6 |
| Energy, simultaneous-dot and delayed-dot mean/std/skew | 36 |
| Standardized invariant/cosine P2 graph trends | 15 |
| **Total** | **368** |

### Replace broad spectral pooling by location moments and areas

Keep the same twenty spectra. For each one, make a three-dimensional path
whose increments are \(p_kq_a(k)\), using P3 QR codes over the frequency-bin
index. Each scalar spectrum is treated separately; no cross-spectrum areas
are retained. Bin index and frequency differ only by a fixed affine scaling
within a group, so the codes span the same degree-two polynomial space.

Before changing to QR coordinates, the codes \((1,f,f^2)\) give endpoint
\((1,\mathbb E F,\mathbb E F^2)\), where \(F\) is a frequency drawn from the
spectral masses. The first coordinate is constant. The remaining endpoints
locate the spectrum and describe its second moment.

The three areas add a different description of spread. For independent draws
\(F,G\) from that spectrum, using the paper's area convention,

\[
J_{01}=\tfrac14\mathbb E|F-G|,\qquad
J_{02}=\tfrac14\mathbb E|F^2-G^2|,\qquad
J_{12}=\tfrac14\mathbb E[FG|F-G|].
\]

For example, the first equality follows by summing
\(\frac12p_kp_l(f_l-f_k)\) over \(k<l\): two independent draws count each
unequal pair twice. These are frequency-pair separation statistics, with
increasing emphasis on high frequencies. They need no band boundaries or
peak detector.

They also contain information beyond the first two moments. On frequencies
\(0,1/3,2/3,1\), masses \((1/4,1/4,1/4,1/4)\) and
\((1/5,2/5,1/10,3/10)\) have the same first and second moments but different
mean absolute separation. Thus an area can distinguish spectra that agree
in centroid and variance.

Each spectrum supplies two varying endpoints and three areas: five features,
or one hundred overall. The implementation stores spectral areas as
\((B_{ij}-B_{ji})/\sqrt2=\sqrt2J_{ij}\), a fixed coordinate scaling removed by
feature standardization. The raw logsignature uses \(J\) itself. QR changes
which linear combinations are stored, not the polynomial space they describe.

Power discards Fourier phase, and normalization discards energy. The raw branch
and amplitude readouts supply complementary information; a signature cannot
reconstruct information already discarded by its input transform.

### One graph construction for moments and trends

For any scalar stream \(z_1,\ldots,z_L\), start a path at value \(x=0\) and
clock \(t=0\). Jump to the next observed value while keeping the clock fixed,
hold that value while the clock advances by \(1/L\), and repeat. At the end,
return the value to zero. Only the value closes; the clock ends at one.

While the value is held fixed and the clock advances, the pure-value prefix of order \(r\) is
\(x^r/r!\). Consequently

\[
S^{x^rt}=\frac1{r!L}\sum_k z_k^r
       =(\log S)^{x^rt}.
\]

The logarithm has the same coefficient because every correction for a split
of this word contains a pure-value prefix of the complete path. That prefix
vanishes when the value endpoint returns to zero. Superscripts here are
selected tensor-log word coefficients, rather than arbitrary coordinates of
a Lie basis.

The mean is available at depth 2, the second moment at depth 3, and the third
moment at depth 4. To obtain stable central moments, first compute the mean,
then build the same readouts from \(z_k-\bar z\). Std is the square root of
the second central moment; skewness divides the third by the cube of std.
These calibrations are stated nonlinear parts of the feature generator.
Selected words are evaluated through tensorsig without materializing the full
graph signature.

Apply this construction to the twenty spectral-amplitude streams, the three
cosines and the twelve energy/dot-product streams. Keep mean/std for amplitudes
and cosines, and mean/std/skew for the twelve geometric streams. This yields
\(40+6+36=82\) coordinates.

Amplitude mean and std restore spectral scale particularly economically.
With \(K\) bins,

\[
E=K\left[\operatorname{mean}(a)^2+\operatorname{std}(a)^2\right],\qquad
\operatorname{mean}(a)=\frac{\sqrt E}{K}\sum_k\sqrt{p_k}.
\]

Together they specify energy and a measure of concentration. A separate
log-energy column is unnecessary for information sufficiency, although such
a nonlinear readout can still affect a linear classifier.

Finally replace the graph clock increments by \(q_a(k)/L\). The same argument
gives weighted moments

\[
(\log S)^{x^rt_a}=\frac1{r!L}\sum_kq_a(k)z_k^r.
\]

The model adds only the linear-clock, first-moment coordinate, applied to each
of the fifteen standardized cosine/geometric streams:

\[
T(z)=\frac1L\sum_k q_1(k)\frac{z_k-\bar z}{\sigma_z}.
\]

This P2/depth-2 coordinate measures whether relatively large values occur early
or late, separately from the mean and scale retained above. Streams with
variance at most \(10^{-12}\) receive zero skewness and zero trends. The
identity is also a rescaled ordinary depth-three graph coordinate; coding
organizes the readout at shallow depth.

## The 903-feature hybrid

The third model concatenates the supplied 561 columns and the same 342 raw
Pascal tensor coordinates as the 803-feature model. The supplied table covers
filtered time signals, derivatives, magnitudes, FFT statistics, band energies,
autoregressive coefficients, correlations and mean-vector angles. We load it
unchanged and do not claim to reproduce its original extraction pipeline.

Its purpose is to retain a familiar engineered representation while giving
ridge access to coded temporal interactions. This model is useful as a stable
comparison, even though its 97.12% score is below both generic retained models.
The cost of producing its supplied 561 columns is not measured here.

## Results and their interpretation

All three models use ridge with an intercept and \(\alpha=10\). For training
feature matrix \(X\) and targets \(Y_{ic}=+1\) for the correct class and
\(-1\) otherwise, the fitted coefficients
minimize \(\|XW+\mathbf1b^\top-Y\|_F^2+10\|W\|_F^2\); prediction takes the
largest class score. Constant columns are removed using training rows only.
The numerical implementation uses float64 and a general-LU solve.

| Retained model | Features | Accuracy | Macro F1 | Errors / 2,947 |
|---|---:|---:|---:|---:|
| Reduced spectral/rotation pool + raw Pascal | 803 | **98.10%** | **98.14%** | **56** |
| Compact raw + spectral Pascal + graph readouts | 368 | 98.03% | 98.06% | 58 |
| Supplied 561 + raw Pascal | 903 | 97.12% | 97.16% | 85 |

The compact model removes 54% of the 803-model coordinates at a cost of two
additional errors. That supports it as a small default; it does not show that
the broad pool's extra statistics are useless. The two recipes also differ
in tensor versus logsignature raw coordinates, so their difference cannot be
attributed to one discarded spectral statistic.

Most residual errors concern sitting versus standing: 49 of 56 errors for the
803 model, 47 of 58 for the compact model, and 67 of 85 for the hybrid. Both
generic models classify all walking and lying windows correctly on this split.
The common limitation is therefore not principally an inability to distinguish
oscillatory gait patterns. The aggregate matrices do not establish why any
particular posture window is misclassified.

The reduced code reproduces the recorded feature dimensions and confusion
matrices. The full three-model run took **4.31 seconds**, including ZIP loading,
with **2.87 seconds** of shared feature preparation. Fits took
0.019/0.034/0.039 seconds for the compact, pooled and supplied-feature models
respectively. Shared preparation builds all three models and is not an
optimized per-model deployment measurement. Current measurements and confusion matrices are in
[retained_results/results.csv](retained_results/results.csv) and
[retained_results/run.json](retained_results/run.json); these timings exclude
the original production of the supplied 561 columns.

The mathematical claim is modest but useful. Changing the observation path
makes signal levels, spectral location, occupation moments and relative
geometry accessible through small signature families. Polynomial coding
brings selected position-weighted interactions to low depth. All these coded
objects are ordinary signatures of lifted paths, so this is not greater
expressivity than every possible ordinary-signature embedding. QR conditions
the code basis, not necessarily the entire feature matrix; basis changes and
nonlinear readouts can alter the ridge penalty and its usable coordinates.

The feature choices were explored repeatedly on the official test split, and
alpha was fixed rather than tuned separately for each representation. These
scores describe this research experiment, not a fresh estimate of performance
after model selection. The ordinary comparisons and earlier searches are
preserved in [uci_historical.md](uci_historical.md), with their scopes and
negative results.

## Definitions and provenance

The data and supplied-feature descriptions come from `README.txt`,
`features_info.txt` and `features.txt` in the cached
[official UCI archive](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).
The account reconciles the retained-model definitions, spectral-family
calculations and graph-moment identities from the earlier local notes. The
saved consolidated run establishes the historical retained scores; the new
three-model run checks the simplified implementation against them.

For implementation conventions, energy denominators are floored at
\(10^{-20}\), cosine norm products at \(10^{-12}\), and standard deviations
used in divisions at \(10^{-8}\). Constant-stream skewness and kurtosis are
zero. Normalized spectral Pascal features require spectra with energy above the numerical floor, a condition
satisfied by the retained HAR inputs. These conventions make the finite-sample
construction explicit without asserting exact equality to the dataset's own
561-feature extraction conventions.
