# Configurable MU-MIMO and beam visualization

## Start and edit

```bash
python animate.py --paused
```

1. Enter **Reference SNR (dB)** and click **Apply SNR / precoder**. Valid range: −40 to 100 dB. Lower SNR increases noise amplitude.
2. Select **ZF** (default) or **MRT**, then apply.
3. Select **Rx1**, **Rx2**, or **Combined / equalized**. This affects the noise, received waveform, CP removal, and receiver FFT views. The later symbol decisions always use the combined/equalized stream.
4. Click **Channels & beams**. Edit H1 and H2, then click **Apply channels and update beams**. The two plots, weights, received samples, and decoded results are recomputed together; playback pauses at the start of the selected block.
5. Select **Receiver FFT** to see I and Q as discrete stems with sample markers, not a continuous curve. FFT bins are unshifted (0…127), including DC and guard bins.

A channel entry may be real or complex. Both `i` and `j` are accepted. JSON strings are convenient for complex values, but this exact mathematical-style matrix is also accepted:

```text
[[1,2,i,1,-1,-i,-i,i],[1,2,1,2,i,2i,i,2i]]
```

This is the default **H1**. H2 is an explicit, editable synthetic default with different phases. Its second row is a complex multiple of its first row, representing a rank-one single-direction example. The default H2 direction is +30° in the chosen BS ULA convention. H1 is an arbitrary supplied matrix; it is not interpreted as a unique list of scatterer angles.

The complete applied configuration is saved to `results/channel_config.json`. Relaunch that configuration with:

```bash
python animate.py --config results/channel_config.json --paused
```

`--snr-db` overrides the config's SNR when supplied to animate.py. The same configuration can generate outputs without a window:

```bash
python simulate.py --config results/channel_config.json
```

In simulate.py's config mode, the configuration supplies the SNR. Its legacy `--snr-db` option applies to the independent-AWGN baseline.

## Signal dimensions and processing

Each UE has one data layer and two receive elements. On an occupied tone, the two independent unit-energy QPSK layers form s=[s1,s2]ᵀ, a 2×1 vector. The base station uses an 8×2 weight matrix W:

$$x=W s,\qquad y_u=H_u x+n_u=H_u w_1s_1+H_u w_2s_2+n_u.$$

H_u is 2×8: rows are receiver elements; columns are BS elements. Therefore each receiver sample is a sum of eight differently weighted antenna signals, not merely a scalar gain times one isolated user's data.

For this extension, both H matrices and W are constant across tones and the whole frame (flat fading). Consequently applying W to the serialized layer OFDM waveforms is mathematically equivalent to precoding each tone before the eight antenna IFFTs. This equivalence would not hold for tone-varying W; the earlier sparse multipath preview is not applied in this flat-channel mode.

Each receiver combiner c_u is the dominant left singular vector of H_u and has unit norm. Form the two-row effective channel G:

$$G_u=c_u^H H_u.$$

For **ZF**, calculate W0=G† (Moore–Penrose pseudoinverse). For **MRT**, W0=Gᴴ. Normalize each column:

$$w_u=W_{0,u}/\|W_{0,u}\|_2.$$

This uses one unit of transmit power per independent unit-energy layer; the expected sum transmit power is 2 on jointly loaded tones. The short deterministic text payload and padding need not have exactly that empirical sample-average power. Unequal realized received powers are retained.

Let D=GW. Combine and equalize:

$$z_u=\frac{c_u^H y_u}{D_{uu}}=s_u+\frac{D_{uv}}{D_{uu}}s_v+\frac{c_u^H n_u}{D_{uu}},\quad v\ne u.$$

With full row-rank G and perfect CSI, ZF gives D_uv≈0. MRT generally leaves interference. Singular or extremely ill-conditioned ZF channels are rejected with an explanation; the program does not silently claim successful separation. All-zero desired channels cannot be equalized.

The animation's blocks 5–7 show the user-layer waveforms. Actual eight-port transmitted samples are stored as `bs_antenna_tx`. Receiver blocks can display either raw Rx element or the combined/equalized result. This is fully digital precoding with ideal elements; it does not model hybrid RF-chain constraints.

## Noise and actual SINR

Each receiver element gets independent noise:

$$n_{u,r}\sim\mathcal{CN}(0,\sigma^2),\qquad\sigma^2=10^{-\mathrm{SNR}_{ref}/10}.$$

The reference SNR is 1/σ², using Es=1 per input layer. Each real component has standard deviation √(σ²/2). A 10 dB SNR increase reduces noise power by ten and amplitude by √10. The same random seed is reused across edits so SNR comparisons scale the same noise realization.

This is **not** a guarantee that each antenna or decoded stream has the entered SNR. Gains, precoding and interference change received quality. For independent, fully loaded unit-power layers and unit-norm combiners:

$$\mathrm{SINR}_u=\frac{|D_{uu}|^2}{|D_{uv}|^2+\sigma^2}.$$

The report and beam window display this predicted full-load post-combining SINR. Padding means the actual short-frame interference can differ, particularly with MRT. EVM and BER are measured from the actual payload. BER=0 on a short text is not evidence of a generally error-free link.

Changing only reference SNR does not change ZF or MRT weights, so their transmit beam shapes should remain unchanged; noise and SINR change. Changing H or the precoder can change the beams.

## What the beam diagram means

Assume a far-field 8-element ULA with ideal isotropic elements, half-wavelength spacing, 28 GHz carrier, and angle θ measured from broadside. Its unit-norm steering vector is

$$a(\theta)=\frac1{\sqrt8}[1,e^{j\pi\sin\theta},\ldots,e^{j7\pi\sin\theta}]^T.$$

Each user's solid curve is the **transmit beam shape**:

$$P_u(\theta)=|a(\theta)^H w_u|^2.$$

This shows how its precoder distributes power by angle in this array convention. Main lobes, sidelobes and nulls arise from coherent addition of antenna contributions.

Each dashed curve is a **channel response to a steering-vector input**:

$$R_u(\theta)=\|H_u a(\theta)\|_2^2.$$

It answers “how strongly would this matrix respond to that transmit steering vector?” It is not the emitted beam or an inferred physical arrival-angle distribution. An arbitrary 2×8 matrix alone cannot identify a unique physical multipath geometry. Array geometry and propagation assumptions are required to interpret it.

Each curve is independently normalized to its own peak and shown as 10 log10(P/Pmax), clipped at −40 dB. This compares shapes, not absolute power or directivity in dBi. The −90°…90° cut does not resolve the front/back ambiguity of a linear array.

## Outputs and validation

Every applied setting regenerates `summary.json`, `channel_config.json`, all block outputs, and the NumPy archive. In MIMO mode the archive includes H1/H2, W (`precoder`), c (`combiners`), D (`effective_coupling`), all eight transmitted antenna waveforms, raw two-antenna received samples, per-antenna noise/FFT, and beam curves. The complete text report includes applied MIMO arrays in its appendix. Legacy sparse-channel-only keys are removed in this mode.

```bash
python -m unittest -v
python animate.py --check
```

The GUI requires desktop Python with working Tk. Headless tests check numerical processing and drawing commands; they do not replace a visual desktop test.

References: [MathWorks multiuser precoding](https://www.mathworks.com/help/wlan/ug/802-11ac-multi-user-mimo-precoding.html) and [array radiation and response patterns](https://www.mathworks.com/help/phased/ug/element-and-array-radiation-patterns-and-responses.html). This remains an educational model, not a full standardized NR PDSCH implementation.
