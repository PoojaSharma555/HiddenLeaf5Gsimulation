"""Write complete, untruncated block outputs for the teaching simulation."""
import json
import numpy as np


def write_block_outputs(output, report, arrays):
    def array_block(handle, title, formula, data):
        handle.write(f'\n{title}\nFormula: {formula}\n')
        handle.write(f'Shape: {data.shape}; dtype: {data.dtype}\n')
        handle.write(np.array2string(data, threshold=np.inf, precision=17,
                                    max_line_width=120, floatmode='unique') + '\n')

    with (output/'output.txt').open('w', encoding='utf-8') as handle:
        handle.write('HIDDEN LEAF — COMPLETE BLOCK OUTPUTS\n')
        handle.write('Indices start at zero. Complex numbers use j = sqrt(-1).\n')
        handle.write('I = real part; Q = imaginary part. Amplitudes are normalized.\n')
        handle.write('All array entries are printed; no truncated arrays.\n')
        handle.write('Two independent AWGN loopbacks; MIMO channel previews are NOT applied.\n')
        handle.write(json.dumps({k:v for k,v in report.items() if k != 'users'}, indent=2)+'\n')
        for u, row in enumerate(report['users'], 1):
            prefix = f'ue{u}_'
            handle.write(f'\n{"="*78}\nUE{u}: {row["message"]}\n{"="*78}\n')
            array_block(handle, 'BLOCK 1 — UTF-8 bytes',
                        'c = UTF8(message)', arrays[prefix+'bytes'])
            array_block(handle, 'BLOCK 2 — Payload bits (MSB first)',
                        'b[8*j+r] = floor(c[j] / 2**(7-r)) mod 2', arrays[prefix+'bits'])
            array_block(handle, 'BLOCK 3 — QPSK symbols',
                        's[m] = ((1-2*b[2*m]) + j*(1-2*b[2*m+1])) / sqrt(2)', arrays[prefix+'qpsk'])
            array_block(handle, 'BLOCK 4 — Frequency grid X[q,k], including every null bin',
                        'X[q, signed_tone mod NFFT] = payload symbol; otherwise 0', arrays[prefix+'grid'])
            array_block(handle, 'BLOCK 5 — COMPLEX BASEBAND OFDM WAVEFORM BEFORE CP, x[q,n]',
                        'x[q,n] = sum_k X[q,k]*exp(j*2*pi*k*n/NFFT) / sqrt(NFFT)', arrays[prefix+'ifft'])
            array_block(handle, 'BLOCK 6 — CP blocks x_CP[q,n]',
                        'x_CP[q] = concatenate(x[q,-NCP:], x[q,:])', arrays[prefix+'cp_blocks'])
            array_block(handle, 'BLOCK 7 — Serialized COMPLEX BASEBAND OFDM WAVEFORM',
                        'tx = x_CP.reshape(-1), OFDM symbol by OFDM symbol', arrays[prefix+'tx'])
            array_block(handle, 'BLOCK 8 — AWGN samples',
                        'w = sqrt(noise_variance/2)*(z_I + j*z_Q)', arrays[prefix+'noise'])
            array_block(handle, 'BLOCK 9 — Received waveform',
                        'rx = tx + w', arrays[prefix+'rx'])
            array_block(handle, 'BLOCK 10 — Received samples after CP removal',
                        'rx_useful = rx.reshape(number_of_symbols,NFFT+NCP)[:,NCP:]', arrays[prefix+'rx_no_cp'])
            array_block(handle, 'BLOCK 11 — Full receiver FFT grid Y[q,k]',
                        'Y[q,k] = sum_n rx_useful[q,n]*exp(-j*2*pi*k*n/NFFT) / sqrt(NFFT)', arrays[prefix+'rx_grid'])
            array_block(handle, 'BLOCK 12 — Allocated tones including noisy padding positions',
                        'allocated = Y[:,active_bins].reshape(-1)', arrays[prefix+'rx_allocated'])
            array_block(handle, 'BLOCK 13 — Payload symbols after discarding padding',
                        'received_qpsk = allocated[:number_of_payload_symbols]', arrays[prefix+'received_qpsk'])
            array_block(handle, 'BLOCK 14 — Hard-decided payload bits',
                        'b_hat[2*m] = (real(s_hat[m])<0); b_hat[2*m+1] = (imag(s_hat[m])<0)', arrays[prefix+'recovered_bits'])
            array_block(handle, 'BLOCK 15 — Recovered bytes',
                        'c_hat = packbits(b_hat), MSB first', arrays[prefix+'recovered_bytes'])
            handle.write('\nBLOCK 16 — Decoded text and error measurements\n')
            handle.write('Formula: BER = incorrect_bits / payload_bits\n')
            handle.write('Formula: EVM = sqrt(sum(abs(s_hat-s)**2) / sum(abs(s)**2))\n')
            handle.write(json.dumps(row, indent=2)+'\n')

            # A simple indexed table is easier to inspect than a long array dump.
            with (output/f'ue{u}_ofdm_waveform.txt').open('w', encoding='utf-8') as wave:
                wave.write(f'UE{u} COMPLEX BASEBAND OFDM SAMPLES — normalized amplitudes\n')
                wave.write('Time origin is the beginning of the first transmitted cyclic prefix.\n')
                for title, key, includes_cp in [('BEFORE CP (IFFT output)', 'ifft', False),
                                                ('AFTER CP (serialized transmitter waveform)', 'cp_blocks', True)]:
                    wave.write(f'\n{title}\n')
                    wave.write('index\tofdm_symbol\tsample_in_block\ttime_s\tis_cp\tI\tQ\tcomplex_value\n')
                    data = arrays[prefix+key]
                    nfft, ncp, fs = report['fft_size'], report['cp_samples'], report['sample_rate_hz']
                    for q in range(data.shape[0]):
                        for n, z in enumerate(data[q]):
                            index = q*data.shape[1]+n
                            # Useful-only samples retain their positions on the transmitted timeline.
                            t = (q*(nfft+ncp)+n+(0 if includes_cp else ncp))/fs
                            wave.write(f'{index}\t{q}\t{n}\t{t:.17g}\t{int(includes_cp and n<ncp)}\t'
                                       f'{z.real:.17g}\t{z.imag:.17g}\t{z.real:.17g}{z.imag:+.17g}j\n')
        handle.write('\nAPPENDIX — SPARSE CHANNEL PREVIEW ONLY; NOT PART OF PAYLOAD LOOPBACK\n')
        for name in ['active_bins','path_delays_s','path_gains','H1','H2','H_beamspace']:
            array_block(handle, name, 'See README section 10 for geometric channel and DFT beamspace equations', arrays[name])
