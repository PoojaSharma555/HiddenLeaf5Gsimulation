"""Inspect the actual frequency-domain MU-MIMO multiplication on one resource element."""
import numpy as np
from simulate import BINS,TONES


def render_fd(app,c,u):
    w,h=c.winfo_width(),c.winfo_height()
    q=int(app.symbol.get())
    tone=int(app.tone.get()) if hasattr(app,'tone') else int(TONES[0])
    k=tone%128
    a=app.arrays
    hmat=a[f'H{u}'];hmat=hmat[k] if hmat.ndim==3 else hmat
    weights=a['precoder_fd'][k] if 'precoder_fd' in a else a['precoder']
    s=np.array([a[f'ue{v}_grid'][q,k] for v in (1,2)])
    x=weights@s
    noise=a[f'ue{u}_noise_fd'][:,q,k] if f'ue{u}_noise_fd' in a else np.fft.fft(a[f'ue{u}_antenna_noise'].reshape(2,3,138)[:,:,10:],axis=-1,norm='ortho')[:,q,k]
    y=a[f'ue{u}_antenna_fft'][:,q,k]
    cols=min(8,int(min(app.progress,1)*8))
    partial=hmat[:,:cols]@x[:cols]
    full=hmat@x
    fmt=lambda z:f'{z.real:+.1f}{z.imag:+.1f}j'
    app.text(c,15,75,f'q={q} • signed tone {tone:+d} • FFT bin {k} • {tone*60} kHz',size=11)
    def vector(label,values,ypos,color='#e8eef7'):
        app.text(c,15,ypos,label,size=10,color=color)
        space=(w-85)/8
        for i,z in enumerate(values):
            c.create_text(80+i*space,ypos,text=fmt(z),anchor='nw',fill=color,font=('Menlo',8))
    vector('H row 0',hmat[0],110)
    vector('H row 1',hmat[1],133)
    vector('w₁ᵀ',weights[:,0],170,'#65baff')
    vector('w₂ᵀ',weights[:,1],193,'#ffb569')
    vector('x = Ws',x,230)
    app.text(c,15,260,f's₁={fmt(s[0])}   s₂={fmt(s[1])}   (zero = no payload on this RE)',size=10)
    app.text(c,15,287,f'Hx: adding {cols}/8 Tx contributions → [{fmt(partial[0])}, {fmt(partial[1])}]',size=11)
    app.text(c,15,314,f'Full Hx = [{fmt(full[0])}, {fmt(full[1])}]',size=11)
    app.text(c,15,339,f'N = [{fmt(noise[0])}, {fmt(noise[1])}]',size=11,color='#ffb569')
    app.text(c,15,364,f'Actual FFT Y = [{fmt(y[0])}, {fmt(y[1])}]',size=11,color='#83d6ac')
    err=np.max(abs(y-full-noise))
    app.text(c,15,391,f'Check max |Y − Hx − N| = {err:.2e} • display rounded to 1 decimal',size=10)
    if 'path_gains' in a:
        delays=a['delay_samples'];gains=a['path_gains'][u-1]
        app.text(c,15,418,'Rays: '+', '.join(f'd={d}: α={fmt(g)}' for d,g in zip(delays,gains)),size=10,width=w-30)
