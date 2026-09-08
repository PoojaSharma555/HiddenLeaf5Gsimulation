"""Configurable flat-fading MU-MIMO extension with fully digital precoding."""
from pathlib import Path
import ast
import contextlib
import io
import json
import re
import numpy as np


def parse_matrix(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            expression = value.replace('i', 'j')
            expression = re.sub(r'(?<![\w.])j\b', '1j', expression)
            value = ast.literal_eval(expression)
    matrix = np.array([[complex(str(x).replace('i', 'j')) for x in row] for row in value])
    if matrix.shape != (2, 8) or not np.isfinite(matrix).all():
        raise ValueError('Each channel must be a finite 2 by 8 matrix.')
    return matrix


def serial_matrix(h):
    return [[str(complex(z)).replace('j', 'i') for z in row] for row in h]


def defaults():
    return dict(snr_db=15., precoder='ZF',
                H1=[[1,2,'i',1,-1,'-i','-i','i'],[1,2,1,2,'i','2i','i','2i']],
                H2=[[1,'-i',-1,'i',1,'-i',-1,'i'],
                    ['0.7+0.2i','-0.2-0.7i','-0.7-0.2i','0.2+0.7i',
                     '0.7+0.2i','-0.2-0.7i','-0.7-0.2i','0.2+0.7i']])


def weights(channels, method):
    # Fix each receiver combiner to its dominant left singular vector.
    combiners = np.array([np.linalg.svd(h, full_matrices=False)[0][:,0] for h in channels])
    effective = np.array([c.conj() @ h for c,h in zip(combiners, channels)])
    if method == 'ZF':
        if np.linalg.matrix_rank(effective) < 2 or np.linalg.cond(effective) > 1e8:
            raise ValueError('ZF cannot reliably separate these channels. Change H1/H2 or select MRT.')
        w = np.linalg.pinv(effective)
    elif method == 'MRT':
        w = effective.conj().T
    else:
        raise ValueError('Precoder must be ZF or MRT.')
    norms = np.linalg.norm(w, axis=0)
    if np.any(norms < 1e-12):
        raise ValueError('Each user must have a nonzero channel.')
    # Unit power per layer: E||x||²=2 with independent unit-energy symbols.
    w /= norms
    coupling = effective @ w
    if np.any(abs(np.diag(coupling)) < 1e-10):
        raise ValueError('A desired effective channel gain is zero; cannot equalize.')
    return w, combiners, coupling


def beam_patterns(channels, w):
    angle = np.linspace(-90.,90.,721)
    a = np.exp(1j*np.pi*np.arange(8)[:,None]*np.sin(np.deg2rad(angle)))/np.sqrt(8)
    tx = abs(a.conj().T @ w).T**2
    response = np.array([np.sum(abs(h @ a)**2,axis=0) for h in channels])
    # Normalize each curve independently to show shape, not absolute link gain.
    db = lambda p: 10*np.log10(np.maximum(p / np.maximum(p.max(axis=1,keepdims=True),1e-30),1e-4))
    return angle, db(tx), db(response)


def run_mimo(config=None, seed=555, output=Path('results')):
    from simulate import run, NFFT, NCP, BINS
    from block_output import write_block_outputs
    config = {**defaults(), **(config or {})}
    snr = float(config['snr_db'])
    if not np.isfinite(snr) or not -40 <= snr <= 100:
        raise ValueError('SNR must be a finite value between -40 and 100 dB.')
    channels = np.array([parse_matrix(config[k]) for k in ('H1','H2')])
    w,c,coupling = weights(channels, config['precoder'])
    output=Path(output)
    with contextlib.redirect_stdout(io.StringIO()):
        report=run(snr,seed,output)
    with np.load(output/'waveforms_and_channels.npz') as data:
        arrays={k:data[k].copy() for k in data.files}
    streams=np.array([arrays[f'ue{u}_tx'] for u in (1,2)])
    bs_tx=w @ streams
    noise_var=10**(-snr/10)
    rng=np.random.default_rng(seed+10)
    for u in range(2):
        key=f'ue{u+1}_'
        noise=np.sqrt(noise_var/2)*(rng.normal(size=(2,streams.shape[1]))+1j*rng.normal(size=(2,streams.shape[1])))
        raw=channels[u] @ bs_tx+noise
        combined=c[u].conj() @ raw / coupling[u,u]
        combined_noise=c[u].conj() @ noise / coupling[u,u]
        useful=combined.reshape(3,NFFT+NCP)[:,NCP:]
        grid=np.fft.fft(useful,axis=1,norm='ortho')
        allocated=grid[:,BINS].ravel()
        symbols=allocated[:len(arrays[key+'qpsk'])]
        bits=np.stack((symbols.real<0,symbols.imag<0),axis=1).astype(np.uint8).ravel()
        raw_no_cp=raw.reshape(2,3,NFFT+NCP)[:,:,NCP:]
        arrays.update({key+'rx':combined,key+'noise':combined_noise,key+'rx_no_cp':useful,
                       key+'rx_grid':grid,key+'rx_allocated':allocated,key+'received_qpsk':symbols,
                       key+'recovered_bits':bits,key+'recovered_bytes':np.packbits(bits),
                       key+'antenna_rx':raw,key+'antenna_noise':noise,
                       key+'antenna_rx_no_cp':raw_no_cp,
                       key+'antenna_fft':np.fft.fft(raw_no_cp,axis=2,norm='ortho')})
        errors=int(np.count_nonzero(bits!=arrays[key+'bits']))
        evm=float(np.sqrt(np.mean(abs(symbols-arrays[key+'qpsk'])**2)))
        sinr=abs(coupling[u,u])**2/(abs(coupling[u,1-u])**2+noise_var)
        report['users'][u].update(recovered_text=np.packbits(bits).tobytes().decode('utf-8',errors='replace'),
             bit_errors=errors,ber=errors/len(bits),rms_evm_percent=100*evm,
             measured_es_n0_db=None,evm_based_sinr_db=float(-20*np.log10(max(evm,1e-30))),
             full_load_sinr_db=float(10*np.log10(sinr)))
    angles,tx_db,channel_db=beam_patterns(channels,w)
    arrays.update(bs_antenna_tx=bs_tx,precoder=w,combiners=c,effective_coupling=coupling,
                  H1=channels[0],H2=channels[1],beam_angles_deg=angles,
                  transmit_beams_db=tx_db,channel_response_db=channel_db)
    # Remove old preview artifacts to avoid confusing applied flat H with sparse H[k].
    for key in ('H_beamspace','path_gains','path_delays_s'):
        arrays.pop(key,None)
    config.update(snr_db=snr,H1=serial_matrix(channels[0]),H2=serial_matrix(channels[1]))
    report.update(stage='Joint 8-Tx / two 2-Rx users; applied flat channels, digital precoding and combining',
                  snr_definition='Reference per-layer Es / per-Rx-antenna noise variance; Es=1, total layer power=2. Actual post-combining SINR is reported separately.',
                  channel_model='User-configurable, static flat fading; same H on every tone. Perfect CSI.',
                  precoder_method=config['precoder'],channel_config=config)
    report.pop('shannon_awgn_bound_bits_per_s_hz',None)
    np.savez_compressed(output/'waveforms_and_channels.npz',**arrays)
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'channel_config.json').write_text(json.dumps(config,indent=2)+'\n')
    write_block_outputs(output,report,arrays)
    from simulate import svg_plot
    svg_plot(output/'baseband.svg',arrays['ue1_qpsk'],arrays['ue1_received_qpsk'],arrays['ue1_tx'],snr,snr_label='reference SNR')
    return report,arrays
