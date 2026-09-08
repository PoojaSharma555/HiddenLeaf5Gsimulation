"""Phase 1: two independent QPSK/OFDM baseband chains; sparse channel preview."""
from pathlib import Path
import argparse
import json
import math
import numpy as np
from block_output import write_block_outputs

MESSAGES = ('Hello you are Naruto. welcome to this simulation',
            'Konnichiwa Sasuke, welcome to this sim')
NFFT, NCP, SCS, FC = 128, 10, 60_000., 28e9
# FFT ordering: negative tones wrap to upper indices. DC and guard tones are zero.
TONES = np.r_[np.arange(-36, 0), np.arange(1, 37)]
BINS = TONES % NFFT
FS = NFFT * SCS


def text_bits(message):
    return np.unpackbits(np.frombuffer(message.encode('utf-8'), dtype=np.uint8))


def qpsk(bits):
    pairs = np.asarray(bits, dtype=float).reshape(-1, 2)
    return ((1 - 2*pairs[:, 0]) + 1j*(1 - 2*pairs[:, 1])) / np.sqrt(2)


def decisions(symbols):
    return np.stack((symbols.real < 0, symbols.imag < 0), axis=-1).astype(np.uint8).ravel()


def ofdm_tx(symbols, n_symbols):
    filled = np.zeros(n_symbols * len(BINS), dtype=complex)
    filled[:len(symbols)] = symbols
    grid = np.zeros((n_symbols, NFFT), dtype=complex)
    grid[:, BINS] = filled.reshape(n_symbols, -1)
    useful = np.fft.ifft(grid, axis=1, norm='ortho')
    return grid, np.concatenate((useful[:, -NCP:], useful), axis=1).ravel()


def ofdm_rx(waveform):
    blocks = waveform.reshape(-1, NFFT + NCP)
    return np.fft.fft(blocks[:, NCP:], axis=1, norm='ortho')[:, BINS].ravel()


def steering(n, angle_deg):
    # Half-wavelength spacing; angle measured from broadside; unit norm.
    return np.exp(1j*np.pi*np.arange(n)*np.sin(np.deg2rad(angle_deg))) / np.sqrt(n)


def sparse_channels(rng):
    """Toy NLOS block fading, Gaussian ray gains; not a calibrated 3GPP model."""
    delays = np.array([0., 130e-9, 390e-9])
    power = 10. ** (np.array([0., -5., -10.]) / 10.)
    power /= power.sum()
    tx_angles = [[-25., -8., 13.], [30., 48., 7.]]
    rx_angles = [[10., -30., 45.], [-15., 25., -50.]]
    freq = np.fft.fftfreq(NFFT, 1/FS)
    channels, gains = [], []
    for tx, rx in zip(tx_angles, rx_angles):
        alpha = np.sqrt(power/2) * (rng.normal(size=3) + 1j*rng.normal(size=3))
        h = np.zeros((NFFT, 2, 8), dtype=complex)
        for l in range(3):
            spatial = np.outer(steering(2, rx[l]), steering(8, tx[l]).conj())
            h += np.sqrt(16)*alpha[l]*np.exp(-2j*np.pi*freq*delays[l])[:, None, None]*spatial
        channels.append(h)
        gains.append(alpha)
    ft = np.fft.fft(np.eye(8), axis=0, norm='ortho')
    fr = np.fft.fft(np.eye(2), axis=0, norm='ortho')
    channels = np.array(channels)
    beamspace = np.einsum('ab,ukbc,cd->ukad', fr.conj().T, channels, ft)
    return channels, beamspace, np.array(gains), delays


def svg_plot(path, original, received, wave, snr_db):
    """Dependency-free, labeled diagnostic figure in SVG."""
    items = ['<svg xmlns="http://www.w3.org/2000/svg" width="980" height="460" viewBox="0 0 980 460">',
             '<rect width="980" height="460" fill="#fff"/>',
             '<g font-family="Arial" font-size="13" fill="#17212b">',
             '<text x="40" y="28" font-size="20">UE1: QPSK and complex baseband OFDM</text>',
             f'<text x="95" y="57">Received QPSK symbols ({snr_db:g} dB Es/N0)</text>',
             '<text x="540" y="57">First OFDM symbol, including cyclic prefix</text>']
    x0, y0, w, h = 65, 85, 345, 290
    items += [f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="none" stroke="#77818b"/>']
    extent = max(1.5, 1.1*max(np.max(abs(received.real)), np.max(abs(received.imag))))
    for v in [-extent, 0, extent]:
        x, y = x0+(v+extent)/(2*extent)*w, y0+(extent-v)/(2*extent)*h
        items += [f'<text x="{x}" y="395" text-anchor="middle">{v:g}</text>',
                  f'<text x="55" y="{y+4}" text-anchor="end">{v:g}</text>']
    for z in received:
        x, y = x0+(z.real+extent)/(2*extent)*w, y0+(extent-z.imag)/(2*extent)*h
        items.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.4" fill="#1769aa" opacity="0.65"/>')
    for z in np.unique(original):
        x, y = x0+(z.real+extent)/(2*extent)*w, y0+(extent-z.imag)/(2*extent)*h
        items.append(f'<path d="M{x-5},{y}h10 M{x},{y-5}v10" stroke="#c43b28" stroke-width="2"/>')
    items += ['<text x="230" y="424">I (normalized)</text>', '<text x="17" y="245" transform="rotate(-90 17 245)">Q (normalized)</text>']
    x0, w = 525, 410
    sample = wave[:NFFT+NCP]
    lim = 1.15*max(np.max(np.abs(sample.real)), np.max(np.abs(sample.imag)))
    items += [f'<rect x="{x0}" y="85" width="{w}" height="290" fill="none" stroke="#77818b"/>',
              f'<rect x="{x0}" y="85" width="{w*NCP/(len(sample)-1):.2f}" height="290" fill="#999" opacity="0.15"/>']
    for data, color in [(sample.real, '#1769aa'), (sample.imag, '#c43b28')]:
        pts = ' '.join(f'{x0+i/(len(sample)-1)*w:.2f},{230-v/lim*145:.2f}' for i,v in enumerate(data))
        items.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.3"/>')
    for i in [0, 64, 137]:
        items.append(f'<text x="{x0+i/137*w}" y="395" text-anchor="middle">{i/FS*1e6:.2f}</text>')
    for v in [-lim, 0, lim]:
        items.append(f'<text x="515" y="{230-v/lim*145+4}" text-anchor="end">{v:.2f}</text>')
    items += ['<text x="705" y="424">Time (µs)</text>', '<text x="470" y="265" transform="rotate(-90 470 265)">Amplitude (normalized)</text>',
              '<text x="570" y="447" fill="#1769aa">I: blue</text><text x="655" y="447" fill="#c43b28">Q: red</text><text x="740" y="447">Shading: CP</text>', '</g></svg>']
    path.write_text('\n'.join(items))


def run(snr_db=15., seed=555, output=Path('results')):
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    bits = [text_bits(m) for m in MESSAGES]
    n_symbols = math.ceil(max(map(len, bits))/(2*len(BINS)))
    noise_var = 10. ** (-snr_db/10)  # Es=1 per occupied QPSK resource element
    arrays, rows = {}, []
    for u, (message, b) in enumerate(zip(MESSAGES, bits), 1):
        s = qpsk(b)
        grid, tx = ofdm_tx(s, n_symbols)
        noise = np.sqrt(noise_var/2)*(rng.normal(size=tx.size)+1j*rng.normal(size=tx.size))
        rx = tx + noise
        cp_blocks = tx.reshape(n_symbols, NFFT + NCP)
        rx_no_cp = rx.reshape(n_symbols, NFFT + NCP)[:, NCP:]
        rx_grid = np.fft.fft(rx_no_cp, axis=1, norm='ortho')
        detected = ofdm_rx(rx)[:len(s)]
        rb = decisions(detected)[:len(b)]
        decoded = np.packbits(rb).tobytes().decode('utf-8', errors='replace')
        errors = int(np.count_nonzero(rb != b))
        evm = float(np.sqrt(np.mean(abs(detected-s)**2)))
        rows.append(dict(user=f'UE{u}', message=message, bytes=len(b)//8, bits=len(b),
                         qpsk_symbols=len(s), ofdm_symbols=n_symbols,
                         zero_padding_resource_elements=n_symbols*len(BINS)-len(s),
                         recovered_text=decoded, bit_errors=errors, ber=errors/len(b),
                         rms_evm_percent=100*evm,
                         measured_es_n0_db=float(-20*np.log10(evm))))
        arrays.update({f'ue{u}_bits': b, f'ue{u}_qpsk': s, f'ue{u}_grid':grid,
                       f'ue{u}_tx':tx, f'ue{u}_rx':rx, f'ue{u}_received_qpsk':detected,
                       f'ue{u}_bytes':np.frombuffer(message.encode('utf-8'), dtype=np.uint8),
                       f'ue{u}_ifft':cp_blocks[:, NCP:], f'ue{u}_cp_blocks':cp_blocks,
                       f'ue{u}_noise':noise, f'ue{u}_rx_no_cp':rx_no_cp,
                       f'ue{u}_rx_grid':rx_grid, f'ue{u}_rx_allocated':rx_grid[:, BINS].ravel(),
                       f'ue{u}_recovered_bits':rb, f'ue{u}_recovered_bytes':np.packbits(rb)})
        (output/f'ue{u}_bits.txt').write_text(''.join(map(str, b.tolist()))+'\n')
        trace = dict(message=message,
                     bytes=[dict(index=i, decimal=int(v), binary=f'{v:08b}')
                            for i,v in enumerate(message.encode('utf-8'))],
                     symbols=[dict(index=i, bits=b[2*i:2*i+2].tolist(),
                                   I=float(z.real), Q=float(z.imag),
                                   ofdm_symbol=i//len(BINS),
                                   signed_subcarrier=int(TONES[i%len(BINS)]))
                              for i,z in enumerate(s)])
        (output/f'ue{u}_trace.json').write_text(json.dumps(trace, indent=2)+'\n')
        if u == 1:
            svg_plot(output/'baseband.svg', s, detected, tx, snr_db)
    # A separate RNG keeps the channel preview independent of payload length/noise draws.
    channels, beamspace, gains, delays = sparse_channels(np.random.default_rng(seed+1))
    arrays.update(H1=channels[0], H2=channels[1], H_beamspace=beamspace,
                  path_gains=gains, path_delays_s=delays, active_bins=BINS)
    np.savez_compressed(output/'waveforms_and_channels.npz', **arrays)
    report = dict(stage='Independent baseband AWGN checks; H1/H2 preview is NOT applied',
                  carrier_hz=FC, scs_hz=SCS, fft_size=NFFT, cp_samples=NCP,
                  sample_rate_hz=FS, useful_symbol_us=1e6/SCS, cp_us=NCP/FS*1e6,
                  frame_duration_us=n_symbols*(NFFT+NCP)/FS*1e6,
                  snr_definition='QPSK Es/N0 per occupied resource element before any spatial channel',
                  requested_snr_db=snr_db, complex_noise_variance=noise_var,
                  shannon_awgn_bound_bits_per_s_hz=float(np.log2(1+10**(snr_db/10))),
                  seed=seed, users=rows)
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    write_block_outputs(output, report, arrays)
    print(json.dumps(report, indent=2))
    print(f'Complete block outputs: {(output / "output.txt").resolve()}')
    for u in (1, 2):
        print(f'UE{u} complex OFDM samples: {(output / f"ue{u}_ofdm_waveform.txt").resolve()}')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snr-db', type=float, default=15.)
    p.add_argument('--seed', type=int, default=555)
    p.add_argument('--output', type=Path, default=Path(__file__).parent/'results')
    args = p.parse_args()
    run(args.snr_db, args.seed, args.output)
