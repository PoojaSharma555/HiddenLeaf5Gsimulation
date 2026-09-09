"""Three physical matrix taps, time-domain convolution, and verified OFDM FD model."""
from pathlib import Path
import contextlib
import io
import json
import numpy as np


def channel_taps(config, seed):
    from simulate import NFFT, NCP, FS, steering
    delays=np.asarray(config.get('delays_samples',[0,1,3]),dtype=float)
    powers=np.asarray(config.get('powers_db',[0,-5,-10]),dtype=float)
    if delays.shape!=(3,) or powers.shape!=(3,) or not np.isfinite(delays).all() or not np.isfinite(powers).all():
        raise ValueError('Provide exactly three finite delays and three average path powers.')
    if np.any(delays!=np.floor(delays)) or np.any(delays<0) or np.any(delays>NCP):
        raise ValueError('Tap delays must be integer samples between 0 and CP length 10.')
    delays=delays.astype(int)
    if len(np.unique(delays))!=3:
        raise ValueError('Use three distinct delay taps.')
    power=10**((powers-powers.max())/10);power/=power.sum()
    rng=np.random.default_rng(int(config.get('channel_seed',seed+1)))
    gains=np.sqrt(power/2)*(rng.normal(size=(2,3))+1j*rng.normal(size=(2,3)))
    tx_angles=[[-25,-8,13],[30,48,7]]
    rx_angles=[[10,-30,45],[-15,25,-50]]
    taps=np.empty((2,3,2,8),complex)
    for u in range(2):
        for l in range(3):
            taps[u,l]=4*gains[u,l]*np.outer(steering(2,rx_angles[u][l]),steering(8,tx_angles[u][l]).conj())
    phase=np.exp(-2j*np.pi*np.arange(NFFT)[:,None]*delays[None,:]/NFFT)
    h=np.einsum('kl,ulrt->ukrt',phase,taps)
    return taps,h,gains,delays,power


def convolve_mimo(taps, delays, tx):
    """Causal linear matrix convolution; keep received frame and full channel tail."""
    received=np.zeros((2,tx.shape[1]+int(max(delays))),complex)
    for tap,delay in zip(taps,delays):
        received[:,delay:delay+tx.shape[1]] += tap @ tx
    return received


def run_multipath(config,seed,output):
    from simulate import run,NFFT,NCP,FS,BINS,svg_plot
    from mimo import weights,beam_patterns
    from block_output import write_block_outputs
    output=Path(output)
    snr=float(config.get('snr_db',15))
    if not np.isfinite(snr) or not -40<=snr<=100:raise ValueError('SNR must be between -40 and 100 dB.')
    taps,h,gains,delays,power=channel_taps(config,seed)
    method=config.get('precoder','ZF')
    ws,cs,ds=[],[],[]
    for k in range(NFFT):
        w,c,d=weights(h[:,k],method)
        ws.append(w);cs.append(c);ds.append(d)
    ws=np.array(ws);cs=np.array(cs).transpose(1,0,2);ds=np.array(ds)
    with contextlib.redirect_stdout(io.StringIO()):report=run(snr,seed,output)
    with np.load(output/'waveforms_and_channels.npz') as source:
        a={k:source[k].copy() for k in source.files}
    # User grids (2,3,128) -> antenna grids (8,3,128), then 8 IFFTs and CPs.
    s=np.array([a[f'ue{u}_grid'] for u in (1,2)])
    x=np.einsum('ktu,uqk->tqk',ws,s)
    useful=np.fft.ifft(x,axis=-1,norm='ortho')
    tx=np.concatenate([useful[:,:,-NCP:],useful],axis=-1).reshape(8,-1)
    noise_var=10**(-snr/10) if config.get('noise_enabled',True) else 0.
    rng=np.random.default_rng(seed+10)
    for u in range(2):
        key=f'ue{u+1}_'
        full=convolve_mimo(taps[u],delays,tx)
        clean=full[:,:tx.shape[1]]
        noise=np.sqrt(noise_var/2)*(rng.normal(size=clean.shape)+1j*rng.normal(size=clean.shape))
        raw=clean+noise
        raw_no_cp=raw.reshape(2,3,NFFT+NCP)[:,:,NCP:]
        raw_fft=np.fft.fft(raw_no_cp,axis=-1,norm='ortho')
        clean_fft=np.fft.fft(clean.reshape(2,3,NFFT+NCP)[:,:,NCP:],axis=-1,norm='ortho')
        predicted=np.einsum('krt,tqk->rqk',h[u],x)
        error=float(np.max(abs(clean_fft-predicted)))
        noise_fft=np.fft.fft(noise.reshape(2,3,NFFT+NCP)[:,:,NCP:],axis=-1,norm='ortho')
        equalizer=cs[u].conj()/ds[:,u,u,None]
        grid=np.einsum('kr,rqk->qk',equalizer,raw_fft)
        equalized_noise=np.einsum('kr,rqk->qk',equalizer,noise_fft)
        allocated=grid[:,BINS].ravel();symbols=allocated[:len(a[key+'qpsk'])]
        bits=np.stack([symbols.real<0,symbols.imag<0],axis=1).astype(np.uint8).ravel()
        # Post-equalization IFFT is a diagnostic, NOT raw time-domain combining.
        combined_useful=np.fft.ifft(grid,axis=-1,norm='ortho')
        diagnostic=np.concatenate([combined_useful[:,-NCP:],combined_useful],axis=-1).ravel()
        n_time=np.fft.ifft(equalized_noise,axis=-1,norm='ortho')
        n_diag=np.concatenate([n_time[:,-NCP:],n_time],axis=-1).ravel()
        a.update({key+'rx':diagnostic,key+'noise':n_diag,key+'rx_no_cp':combined_useful,
                  key+'rx_grid':grid,key+'rx_allocated':allocated,key+'received_qpsk':symbols,
                  key+'recovered_bits':bits,key+'recovered_bytes':np.packbits(bits),
                  key+'antenna_rx':raw,key+'antenna_noise':noise,key+'antenna_rx_no_cp':raw_no_cp,
                  key+'antenna_fft':raw_fft,key+'clean_fd':clean_fft,key+'predicted_fd':predicted,
                  key+'noise_fd':noise_fft,key+'channel_tail':full[:,tx.shape[1]:]})
        interference=abs(ds[:,u,1-u])**2
        sinr=abs(ds[:,u,u])**2/np.maximum(interference+noise_var,1e-30)
        errors=int(np.count_nonzero(bits!=a[key+'bits']))
        evm=float(np.sqrt(np.mean(abs(symbols-a[key+'qpsk'])**2)))
        report['users'][u].update(recovered_text=np.packbits(bits).tobytes().decode('utf-8',errors='replace'),
            bit_errors=errors,ber=errors/len(bits),rms_evm_percent=100*evm,measured_es_n0_db=None,
            evm_based_sinr_db=float(-20*np.log10(max(evm,1e-30))),
            full_load_sinr_db=float(10*np.log10(np.median(sinr[BINS]))),
            fd_identity_max_error=error)
        a[key+'sinr_per_tone_db']=10*np.log10(sinr)
    # Retain all tone-dependent matrices; old flat-shaped aliases select first allocated tone only.
    selected=int(BINS[0]);angle,bp,cp=beam_patterns(h[:,selected],ws[selected])
    a.update(H1=h[0],H2=h[1],path_taps=taps,path_gains=gains,path_delays_s=delays/FS,
             delay_samples=delays,path_powers=power,precoder_fd=ws,combiners_fd=cs,coupling_fd=ds,
             precoder=ws[selected],combiners=cs[:,selected],effective_coupling=ds[selected],
             bs_antenna_tx=tx,bs_grid=x,beam_angles_deg=angle,transmit_beams_db=bp,channel_response_db=cp)
    a.pop('H_beamspace',None)
    config={**config,'channel_model':'rayleigh','channel_seed':int(config.get('channel_seed',seed+1)),
            'delays_samples':delays.tolist(),'powers_db':list(config.get('powers_db',[0,-5,-10])),
            'noise_enabled':bool(config.get('noise_enabled',True))}
    report.update(stage='Three-path sparse Rayleigh MIMO: causal time convolution and per-tone FD processing',
                  channel_model='Three spatially correlated Gaussian ray gains per UE; constant within frame; no LOS or Doppler.',
                  channel_config=config,precoder_method=method,complex_noise_variance=noise_var,
                  snr_definition='Reference Es=1 per layer divided by per-antenna noise variance. Noise can be disabled.',
                  sinr_summary='Median full-load SINR across allocated tones, dB; noise-free values use numerical floor.',
                  cp_valid=True,maximum_delay_samples=int(max(delays)),
                  diagnostic_rx_note='Combined time waveform is reconstructed AFTER per-tone equalization; CP is synthetic.')
    report.pop('shannon_awgn_bound_bits_per_s_hz',None)
    np.savez_compressed(output/'waveforms_and_channels.npz',**a)
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'channel_config.json').write_text(json.dumps(config,indent=2)+'\n')
    write_block_outputs(output,report,a)
    svg_plot(output/'baseband.svg',a['ue1_qpsk'],a['ue1_received_qpsk'],a['ue1_tx'],snr,snr_label='reference SNR')
    return report,a
