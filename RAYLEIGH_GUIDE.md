# Display update: stems, bracketed matrices and inline beams

- Block 3 shows QPSK symbols as discrete I/Q stems.
- Block 4 shows all 128 pre-IFFT frequency bins as stems, including DC, guards and padding. The x axis is signed, −64…63. Choose **User layer** or **BS antenna 1…8** to inspect either a user's mapped symbols or an actual precoded antenna IFFT input. Selecting a BS antenna shows that same antenna in both panels; it carries contributions from both users.
- Block 5 uses the full-width signal area for bracketed W (8×2), s (2×1), x (8×1), and both H matrices (2×8). Noise and received vectors are also bracketed. Highlighting steps through H's eight Tx columns. Scroll the matrix area vertically on shorter windows.
- Both transmit beam plots are now always inside the main animation window. Dashed vertical markers identify the sampled strongest-lobe direction for each user. A peak direction is not necessarily a physical UE location; arbitrary/multipath channels may have multiple lobes. Curves are independently normalized, not absolute gains.
- Selecting a tone or applying channels/precoding updates the inline beams. SNR alone does not change ZF/MRT beam directions.

For each antenna t, W creates x_t = W_t1 s_1 + W_t2 s_2. Its two columns are the users' beamforming weight vectors. ZF chooses them to suppress interference after the modeled receive combiner; it is not simply steering toward a single supplied angle.

---

# Three-path Rayleigh OFDM: channel first, receiver studies next

## What changed

`python animate.py --paused` now starts in **rayleigh** mode with 17 stages. Formulas are typeset directly on the Tk canvas with fraction bars, radicals, and raised/lowered scripts. No extra installation is required.

Select **Channel = rayleigh**, choose a **Rayleigh realization seed**, and click **Apply settings**. Uncheck **Receiver noise** to isolate fading, or leave it enabled to study fading plus AWGN. Fading and noise are separate impairments: Gaussian receiver noise does not become Rayleigh noise.

Open **Block 4: Frequency grid** to inspect the payload allocation. Choose an OFDM symbol q and signed Tone in the top controls. Then open **Block 5: FD channel matrix**. It is an equivalent frequency-domain preview calculated from the full run, not a claim that the physical channel acts before the IFFT. Playback progressively sums the eight Tx-antenna contributions to Hx.

The matrix view shows, separately for UE1 and UE2:

- The selected tone's actual 2×8 channel H_u[k].
- Both 8×1 precoder columns, displayed transposed for readability.
- Both user symbols and x=W s, the eight antenna values on this resource element.
- The partial and complete Hx, the actual FFT-domain noise, and the actual receiver FFT Y.
- The numerical error in Y−Hx−N, and the three sampled path gains/delays.

Values in the animation are rounded to one decimal to fit; `output.txt` and the NumPy archive retain full precision. H rows are receive antennas, not different users. Each panel belongs to one user.

## Where the two payloads live

Let q=0,1,2 denote OFDM symbols. There are 72 assigned signed tones, ordered −36…−1 then +1…+36. Let j=0…71 denote a position in that list. The payload QPSK index is m=72q+j, carrying bits 2m and 2m+1 when m is inside the payload.

| OFDM symbol q | UE1: Naruto | UE2: Sasuke |
|---|---|---|
| 0 | 72 QPSK symbols, bits 0–143 | 72 QPSK symbols, bits 0–143 |
| 1 | 72 QPSK symbols, bits 144–287 | 72 QPSK symbols, bits 144–287 |
| 2 | 48 QPSK symbols, bits 288–383 | 8 QPSK symbols, bits 288–303 |

UE1's third symbol uses tones −36…−1 and +1…+12. UE2's third symbol uses tones −36…−29. Other assigned positions are zero amplitude. All symbols have a DC hole and 55 outer guard bins.

Both users occupy the **same time-frequency positions in distinct spatial layers**; they are not assigned separate frequency bands. For example, q=0 and tone −36 carry UE1's first QPSK symbol and UE2's first QPSK symbol simultaneously:

$$\mathbf s[q,k]=\begin{bmatrix}s_1[q,k]\\s_2[q,k]\end{bmatrix},\qquad
\mathbf x[q,k]=\mathbf W[k]\mathbf s[q,k].$$

The 8×2 precoder creates eight antenna-grid entries from those two layer entries. Each antenna has its own IFFT and CP. The time waveform contains a sum of all its subcarriers; a bit is not confined to one isolated time sample after the IFFT.

## The three-path fading model

We use a sparse geometric NLOS model with three complex Gaussian path gains per user:

$$\alpha_{u,\ell}\sim\mathcal{CN}(0,p_\ell),\qquad\ell=0,1,2,$$

$$p_\ell=\frac{10^{P_\ell/10}}{\sum_m10^{P_m/10}},\qquad P=[0,-5,-10]\ \mathrm{dB}.$$

Each path's envelope is Rayleigh distributed. This is a spatially correlated sparse channel, **not an i.i.d. Rayleigh matrix**. The path angles are illustrative fixed parameters, as in the earlier geometric preview. There is no deterministic LOS path.

For unit-norm ULA steering vectors:

$$\mathbf A_{u,\ell}=\sqrt{8\cdot2}\,\alpha_{u,\ell}\,
\mathbf a_r(\phi_{u,\ell})\mathbf a_t(\theta_{u,\ell})^H.$$

Each A is a 2×8 matrix. Default discrete delays are d=[0,1,3] samples. With fs=7.68 MHz, they correspond to 0, 130.208, and 390.625 ns. These sample-aligned delays make the discrete-time convolution exact. They replace the earlier preview's approximately 130/390 ns values.

The maximum delay is 3 samples, less than the 10-sample CP. Path gains stay fixed across all three OFDM symbols; changing the realization seed draws a new channel. This is **block fading**, with no Doppler/time-selective fading within the frame. Sampling a new gain on every bit would be a different model and would generally disrupt subcarrier orthogonality.

## Physical channel and frequency-domain equivalent

The program really performs causal matrix convolution on the eight transmitted, CP-prefixed antenna waveforms:

$$\mathbf y_u[n]=\sum_{\ell=0}^{2}\mathbf A_{u,\ell}\mathbf x_{BS}[n-d_\ell]+\mathbf n_u[n].$$

For the physical receiver, remove CP and FFT each of the two antenna signals. The channel is diagonal across OFDM tones when its memory fits the CP and timing is perfect:

$$\mathbf H_u[k]=\sum_{\ell=0}^{2}\mathbf A_{u,\ell}e^{-j2\pi k d_\ell/N},\qquad N=128,$$

$$\mathbf Y_u[q,k]=\mathbf H_u[k]\mathbf X[q,k]+\mathbf N_u[q,k].$$

Therefore H_u[k] is 2×8, and differs across k. The code verifies this equation against the independently time-convolved samples; the identity is not merely generated twice with the same multiplication. The received tail after the last frame is saved separately.

The sampled gains absorb carrier-phase effects. This is an equivalent-baseband model with 28 GHz array context, ideal half-wavelength elements, and fixed angular paths; no full RF passband is sampled.

## Receiver baseline and fair comparisons

For now the existing perfect-CSI baseline is retained **per tone**: choose each UE's dominant left singular vector c_u[k], form G[k] with rows c_u[k]ᴴ H_u[k], then obtain a ZF or MRT precoder. Each precoder column has unit norm. The receiver combines and divides by its desired effective gain after the FFT:

$$z_u[q,k]=\frac{\mathbf c_u[k]^H\mathbf Y_u[q,k]}{D_{uu}[k]},\qquad
\mathbf D[k]=\mathbf G[k]\mathbf W[k].$$

That is a baseline, not a claim that receiver design is complete. Future comparisons can keep the exact same channel seed, payloads, powers and noise draws while changing the receiver. Channel estimation, pilots, imperfect CSI, alternative equalizers, and coding are not added in this milestone.

Use **Rx1/Rx2** to view physical received signals. **Combined / equalized** time-domain views are reconstructed from the per-tone equalized grid for illustration; their CP is synthetic. They are not physical time-domain receive combining. Stage 12 shows discrete FFT stems; subsequent stages operate on the combined/equalized tones.

## Saved arrays and controls

- H1/H2: (128,2,8), all per-tone channels.
- path_taps: (2,3,2,8), user/path/Rx/Tx.
- path_gains: (2,3); delay_samples: (3,).
- precoder_fd: (128,8,2); combiners_fd: (2,128,2).
- bs_grid: (8,3,128); bs_antenna_tx: (8,414).
- ue1/ue2 clean_fd, predicted_fd, noise_fd, antenna_fft: (2,3,128).

Legacy flat-shaped precoder/combiners aliases refer only to the first allocated tone; use the `_fd` arrays for this model. Beam plots use the currently selected tone, so beam shapes can vary across frequency. The channel editor's H1/H2 text fields apply only in **flat** mode; in Rayleigh mode matrices are generated from the three paths.

Saved `channel_config.json` can be edited to set `delays_samples`, `powers_db`, `channel_seed`, `noise_enabled`, and `channel_model`, then loaded with `--config`. Delays must be three distinct nonnegative integers no larger than 10 samples; the model rejects unsupported CP lengths rather than silently using a diagonal FD model in an ISI regime.

References: [MathWorks OFDM transmitter and receiver](https://www.mathworks.com/help/comm/ug/ofdm-transmitter-and-receiver.html), [OFDM channel response](https://www.mathworks.com/help/comm/ref/ofdmchannelresponse.html), and [fading channel models](https://www.mathworks.com/help/comm/ug/fading-channels.html).
