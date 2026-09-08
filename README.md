# Hidden Leaf 5G simulation — phase 1

A Python learning project that follows two messages through an uncoded QPSK/OFDM baseband transmitter and receiver. The intended destination is a downlink from an 8-element BS uniform linear array to two UEs with 2 receive elements each, one data layer per UE.

**Current boundary:** both users have independent AWGN loopbacks. Their streams are not combined, precoded, radiated, or passed through the MIMO channels yet. The program generates sparse H1/H2 channel previews separately. This is a deliberately small NR-inspired CP-OFDM exercise, not an NR-compliant PDSCH implementation.

## Run from the beginning

Python 3.10+ and NumPy are sufficient. No MATLAB is needed.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python simulate.py
python -m unittest -v
```

To compare another noise level without replacing the reference results:

```bash
python simulate.py --snr-db 5 --seed 555 --output results_5db
```

The fixed seed makes the experiment reproducible. It does not imply that a real channel or receiver repeats the same noise.

- `simulate.py`: source, with functions matching the stages below.
- `results/summary.json`: counts, decoded messages, BER, EVM, and configuration.
- `results/ue1_bits.txt`, `results/ue2_bits.txt`: every payload bit, in order.
- `results/ue1_trace.json`, `results/ue2_trace.json`: byte encodings and every QPSK symbol, with its OFDM symbol and signed subcarrier index.
- `results/waveforms_and_channels.npz`: complex arrays at intermediate stages and channel previews. Generated locally; rerun Python to obtain this binary archive.
- `results/baseband.svg`: constellation and waveform diagnostic.

## 1. Scenario and explicit assumptions

| Parameter | Initial value |
|---|---|
| Intended topology | 1 BS, 2 UEs |
| BS elements | 8, ULA |
| UE receive elements | 2 each, assumed ULA for channel preview |
| Intended carrier | 28 GHz |
| Modulation | Gray QPSK, equivalent to square 4-QAM |
| Bits per QPSK symbol | 2 |
| Subcarrier spacing | 60 kHz |
| FFT size | 128 |
| Potential data tones per OFDM symbol | 72; signed indices -36…-1, +1…+36 |
| CP length | 10 samples (a teaching choice, not NR normal-CP timing) |
| Requested SNR convention | 15 dB QPSK Es/N0 on occupied resource elements |
| Channel for current payload recovery | Unity gain plus complex AWGN, independently per user |
| Sparse channel preview | 3 paths per UE, zero-mean complex Gaussian path gains, fixed during the frame |

4 or 6 GHz lies in NR FR1; 28 GHz is an FR2 mmWave choice. Carrier frequency is metadata in this equivalent-baseband phase. It will enter antenna spacing in metres and the physical link budget later. Changing FC alone does not change these baseband samples.

The 72-tone allocation and small FFT are chosen for visibility, not a standardized FR2 carrier bandwidth. There is a DC hole. The sum of 72 individual subcarrier bandwidths is 4.32 MHz; the allocation also has a DC gap, and the rectangularly windowed waveform has sidelobes outside that nominal allocation.

## 2. Each text becomes its own bitstream

We preserve case, spaces, and punctuation exactly. The surrounding quotation marks are not payload. UTF-8 encoding produces one byte per character for these ASCII-only messages. This is text encoding, not error-correcting channel coding.

For byte value c, most-significant-bit-first serialization is

$$b_{8j+r}=\left\lfloor\frac{c_j}{2^{7-r}}\right\rfloor\bmod2,\quad r=0,\ldots,7.$$

**UE1:** `Hello you are Naruto. welcome to this simulation`

48 bytes → 384 bits → 192 QPSK symbols.

First five characters:

```text
H         e         l         l         o
01001000  01100101  01101100  01101100  01101111
```

**UE2:** `Konnichiwa Sasuke, welcome to this sim`

38 bytes → 304 bits → 152 QPSK symbols.

First five characters:

```text
K         o         n         n         i
01001011  01101111  01101110  01101110  01101001
```

No scrambling, CRC, LDPC coding, rate matching, or interleaving is performed. The two payloads are held in separate arrays throughout.

## 3. Choose QPSK and map the bits

“2-QAM” would usually mean a two-point constellation equivalent to BPSK. We use 4-QAM/QPSK because its I/Q components are particularly easy to see.

Shannon's ideal AWGN capacity bound is

$$C=B\log_2(1+\gamma),\qquad \gamma=10^{15/10}=31.6228,$$

so the ideal scalar-channel bound at an SNR of 15 dB is approximately 5.0278 bits/s/Hz. This is not a rule that selects an M-QAM constellation. Modulation and coding depend on target error rate, coding, channel estimation, and actual post-equalization SINR. It is not the capacity of our future two-user MIMO channel. We use the formula as a reference only, with its SNR convention stated.

For each consecutive bit pair:

$$s_m=\frac{(1-2b_{2m})+j(1-2b_{2m+1})}{\sqrt2}.$$

| Bits | Symbol |
|---|---|
| 00 | (1+j)/√2 |
| 01 | (1-j)/√2 |
| 10 | (-1+j)/√2 |
| 11 | (-1-j)/√2 |

Every symbol has unit energy: |s|² = 1. The real component is I, the imaginary component is Q; they jointly represent one complex baseband symbol, not two users.

UE1 begins with H = `01 00 10 00`, so its first four symbols are `(1-j)/√2`, `(1+j)/√2`, `(-1+j)/√2`, `(1+j)/√2`.

UE2 begins with K = `01 00 10 11`, so its first four symbols are `(1-j)/√2`, `(1+j)/√2`, `(-1+j)/√2`, `(-1-j)/√2`.

## 4. Map each stream to an OFDM frequency grid

A resource element here is one subcarrier during one OFDM symbol. For user u, create a separate grid X_u[q,k] of shape (3,128). q indexes OFDM time symbols; k indexes FFT bins.

Map payload QPSK symbols into signed tones -36 through -1, then +1 through +36. Signed tone r is stored at Python FFT bin r modulo 128. Null DC, outer guard bins, and unused payload positions are complex zeros.

| User | OFDM symbol 0 | OFDM symbol 1 | OFDM symbol 2 |
|---|---|---|---|
| UE1 | 72 QPSK symbols | 72 QPSK symbols | 48 QPSK symbols + 24 zero REs |
| UE2 | 72 QPSK symbols | 72 QPSK symbols | 8 QPSK symbols + 64 zero REs |

Zero REs are zero amplitude, **not** QPSK symbols formed from `00` bits. No payload bits are appended. The receiver knows the original payload lengths in this exercise and discards the unused positions.

The same subcarrier indices appear in both separate grids. That alone does not implement multiuser multiplexing; spatial precoding must be added before jointly transmitting those overlapping resources.

## 5. IFFT: frequency-domain symbols become a waveform

For each row of each user's grid:

$$x_u[q,n]=\frac1{\sqrt N}\sum_{k=0}^{N-1}X_u[q,k]e^{j2\pi kn/N},\quad N=128.$$

This is `np.fft.ifft(..., norm='ortho')`. Each sample is the coherent sum of the subcarrier sinusoids. This interference over time produces the waveform's peaks and troughs. It is different from spatial interference between antenna elements that creates beams.

The unitary scaling preserves energy:

$$\sum_n|x_u[q,n]|^2=\sum_k|X_u[q,k]|^2.$$

Subcarrier spacing determines useful symbol duration and sample rate:

$$T_{\mathrm{useful}}=1/\Delta f=16.6667\ \mu s,$$
$$f_s=N\Delta f=7.68\ \mathrm{Msamples/s},\qquad T_s=130.2083\ \mathrm{ns}.$$

The subcarriers are orthogonal over the useful FFT interval because

$$\sum_{n=0}^{N-1}e^{j2\pi(k-l)n/N}=N\delta_{kl}.$$

## 6. Add a cyclic prefix and serialize

Copy the last 10 useful samples to the front of each OFDM symbol:

$$x_{u,CP}[q]=[x_u[q,118:128],\ x_u[q,0:128]].$$

The copy protects the FFT interval from preceding-symbol multipath when the effective channel memory fits inside the CP and timing is correct. With appropriate synchronization this permits a per-subcarrier channel representation. It is not a noise remover.

$$T_{CP}=10/f_s=1.30208\ \mu s,$$
$$T_{symbol}=138/f_s=17.96875\ \mu s.$$

Each user's serialized waveform contains 3×138 = 414 complex samples, spanning 53.90625 µs. Useful-time efficiency is 128/138 ≈ 92.75%, before null tones and payload padding. This constant 10-sample CP is not the exact normal-CP schedule of NR.

## 7. What happens between baseband and passband?

Our saved x samples are a complex envelope. They do not oscillate at 28 GHz. In a physical transmitter the DAC and reconstruction filters create I(t) and Q(t), then ideal quadrature upconversion forms

$$x_{RF}(t)=\sqrt2\Re\{x_{BB}(t)e^{j2\pi f_ct}\}
=\sqrt2[I(t)\cos(2\pi f_ct)-Q(t)\sin(2\pi f_ct)].$$

With the same power convention, ideal coherent downconversion is

$$y_{BB}(t)=\operatorname{LPF}\{\sqrt2\,y_{RF}(t)e^{-j2\pi f_ct}\}.$$

An ADC then samples this envelope. These equations assume aligned oscillators and ideal filtering. Phase noise, carrier-frequency offset, IQ imbalance, amplifier nonlinearities, and ADC effects are not simulated.

Equivalent-baseband simulation folds ideal up/downconversion into a complex channel. It avoids directly sampling a 28 GHz RF oscillation at tens of gigasamples per second. The 7.68 MHz baseband sample rate is appropriate for the small envelope allocation, not for direct real-valued RF sampling at 28 GHz. No numerical RF up/downconversion is claimed in phase 1.

## 8. Add receiver noise, independently for each user

Hidden Leaf is our scenario label. No measured geometry or interference environment has been specified. Start with independent additive white Gaussian noise representing an idealized receiver thermal-noise contribution:

$$y_u[n]=x_{u,CP}[n]+w_u[n],$$
$$w_u[n]\sim\mathcal{CN}(0,\sigma^2),\qquad
w_u[n]=\sqrt{\sigma^2/2}(z_I[n]+jz_Q[n]),$$

where z_I and z_Q are independent standard normal samples. Different draws are used for UE1 and UE2.

We define the requested 15 dB as **Es/N0 per occupied QPSK resource element in the unity-gain loopback**:

$$E_s=1,\quad\sigma^2=10^{-15/10}=0.0316228.$$

The unitary FFT preserves this noise variance. Each real noise component has variance 0.0158114. This is not necessarily 15 dB measured over all time-domain samples: guard tones, padding, and CP affect the time-domain average signal power. It is also not a transmit-power/link-budget specification or future per-user SINR.

For uncoded QPSK, Eb = Es/2, so Eb/N0 here is approximately 11.99 dB. A physical thermal-noise power model would use kTB times receiver noise factor; we would need bandwidth, temperature, noise figure, path loss, and transmit power to connect it to absolute received powers. Buildings affect propagation; other transmitters cause interference and need a separate model.

## 9. Receiver: remove CP, FFT, decide, decode

For each stream independently:

1. Reshape 414 samples into 3 blocks of 138 samples.
2. Remove the first 10 samples of each block.
3. Apply a unitary FFT, the inverse of the transmitter IFFT.
4. Extract the same signed subcarrier positions in the same order.
5. Keep the first 192 symbols for UE1 and first 152 for UE2.
6. Make hard QPSK decisions from signs.
7. Repack bits into bytes and UTF-8 decode.

$$\hat b_{2m}=\mathbf1[\Re\{\hat s_m\}<0],\qquad
\hat b_{2m+1}=\mathbf1[\Im\{\hat s_m\}<0].$$

There is no equalizer because the present payload channel has gain one. Timing, payload length, and resource allocation are known exactly. Channel estimation and synchronization will be separate steps later.

$$BER=\frac{\text{incorrect payload bits}}{\text{payload bits}},$$
$$EVM_{RMS}=\sqrt{\frac{\sum_m|\hat s_m-s_m|^2}{\sum_m|s_m|^2}}.$$

With seed 555 and 15 dB, both messages recover exactly. UE1 has 0/384 errors and 17.2084% EVM; UE2 has 0/304 errors and 17.4344% EVM. Expected RMS EVM is √σ² ≈ 17.7828%; a finite random sample differs. Zero errors in these short messages does not establish a general BER of zero.

## 10. H1 and H2: generated now, applied in the next milestone

The program separately constructs a frequency-selective geometric channel for each user:

$$H_u[k]=\sqrt{N_tN_r}\sum_{\ell=0}^{L-1}\alpha_{u,\ell}
\,a_r(\phi_{u,\ell})a_t(\theta_{u,\ell})^H
\,e^{-j2\pi f_k\tau_\ell},$$

where Nt=8, Nr=2, L=3, and H_u[k] has shape 2×8. Row r corresponds to receive element r; column t corresponds to transmit element t. In the saved archive, H1 and H2 each have shape (128,2,8).

- Delays: 0, 130, 390 ns, comfortably below the teaching CP. These are continuous delays in the frequency-response preview; no rounded discrete tap filter is applied.
- Relative average ray powers: 0, -5, -10 dB, normalized to sum to one.
- α_l ~ CN(0,p_l), so individual path envelopes are Rayleigh distributed. This is a sparse, spatially correlated NLOS model, not an i.i.d. Rayleigh matrix, and not a calibrated 3GPP channel.
- Angles and delays are fixed, path gains are held fixed during the frame; no Doppler or deterministic line-of-sight component. A dominant deterministic LOS contribution would motivate a Rician extension.
- Carrier-dependent path phase is absorbed into α; f_k is the baseband tone frequency.
- The normalization makes the ensemble-average Frobenius energy of H equal to NtNr=16. We do not normalize every random channel realization to the same power.

Provisional half-wavelength ULA steering vectors, with angles measured from broadside, are

$$a_N(\theta)=\frac1{\sqrt N}[1,e^{j\pi\sin\theta},\ldots,e^{j\pi(N-1)\sin\theta}]^T.$$

At 28 GHz, λ=c/fc≈10.707 mm and d=λ/2≈5.353 mm. This assumes ideal elements, far-field plane waves, and no mutual coupling. It introduces only enough geometry to make the channel preview meaningful; beam-pattern design is deferred.

Transmit departure angles are (-25,-8,13) degrees for UE1 and (30,48,7) for UE2. Arrival angles in each receiver's own array frame are (10,-30,45) and (-15,25,-50). These are illustrative angular parameters, not a ray-traced reconstruction of a village.

With unitary DFT matrices Fr and Ft, beamspace is

$$H_{b,u}[k]=F_r^H H_u[k]F_t.$$

A few physical rays motivate a compact angular representation. Off-grid angles leak across DFT bins, so these beamspace matrices are generally not exactly sparse; with only two receive elements, angular resolution is especially coarse.

The future multiuser signal equation is

$$s[k]=[s_1[k],s_2[k]]^T,\quad x[k]=W[k]s[k],$$
$$y_u[k]=H_u[k]w_1[k]s_1[k]+H_u[k]w_2[k]s_2[k]+n_u[k].$$

W is 8×2, x has 8 antenna-port values, and y_u has 2 receive-element values. Each UE needs receive combining. For UE1 the second term is inter-user interference; for UE2 the first is interference. The eventual IFFT/CP processing is performed on each of the eight precoded antenna-port grids. The present two-stream waveforms demonstrate the OFDM operations before that extension.

Eight antennas do not automatically require eight independent messages, and two receive antennas do not automatically require two layers per UE. Our starting design is one layer per user.

## References and scope

- [3GPP TS 38.211 via ETSI, physical channels and modulation](https://etsi.org/deliver/etsi_ts/138200_138299/138211/15.02.00_60/ts_138211v150200p.pdf): QPSK mapping and NR waveform foundations. This project omits the full NR processing chain.
- [MathWorks NR downlink carrier configuration](https://www.mathworks.com/help/5g/ref/nrdlcarrierconfig.html): FR1/FR2 and subcarrier-spacing context.
- [Alkhateeb et al., Channel Estimation and Hybrid Precoding for Millimeter Wave Cellular Systems](https://arxiv.org/abs/1401.7426): sparse angular mmWave modeling motivation.
- [MathWorks fading channels](https://www.mathworks.com/help/comm/ug/fading-channels.html): Rayleigh/Rician and multipath distinctions.
- [MathWorks SNR conversion](https://www.mathworks.com/help/comm/ref/convertsnr.html): distinction between per-subcarrier and sample SNR.

Next: build intuition for ULA phase progression, array factors, main lobes, sidelobes, and nulls; then connect precoding and combining to these payloads and inspect desired-signal power, inter-user interference, and noise separately.
