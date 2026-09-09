"""Explicit receive combining and scalar channel equalization views."""
import numpy as np
from fd_view import matrix


def receiver_values(a,u,q,k):
    index=u-1
    c=a['combiners_fd'][index,k] if 'combiners_fd' in a else a['combiners'][index]
    d=a['coupling_fd'][k] if 'coupling_fd' in a else a['effective_coupling']
    y=a[f'ue{u}_antenna_fft'][:,q,k]
    terms=c.conj()*y
    r=terms.sum();gain=d[index,index]
    s=a[f'ue{u}_grid'][q,k];other=a[f'ue{2 if u==1 else 1}_grid'][q,k]
    desired=gain*s;interference=d[index,1-index]*other
    noise=r-desired-interference
    return c,y,terms,r,gain,r/gain,s,desired,interference,noise


def render_receiver(app,canvas,u,equalize=False):
    q=int(app.symbol.get());k=int(app.tone.get())%128 if hasattr(app,'tone') else 92
    c,y,terms,r,g,z,s,desired,interference,noise=receiver_values(app.arrays,u,q,k)
    width=canvas.winfo_width();scale=min(1.,width/570)
    fmt=lambda v:f'{v.real:+.4f}{v.imag:+.4f}j'
    text=lambda xx,yy,value,color='#e8eef7',size=11:app.text(canvas,xx*scale,yy,value,color,size,width=width-30)
    text(15,78,f'OFDM symbol {q} • tone {k if k<64 else k-128:+d}')
    if not equalize:
        text(15,110,'Two raw antenna FFT outputs → one combined value')
        matrix(canvas,c.conj()[None,:],15*scale,165,75*scale,28,size=10)
        text(22,137,'cᴴ (1 × 2)');text(184,181,'×',size=18)
        matrix(canvas,y,215*scale,151,90*scale,28,size=10)
        text(222,124,'Y (2 × 1)');text(331,181,'=',size=18)
        matrix(canvas,np.array([r]),362*scale,165,110*scale,28,size=10)
        text(370,137,'r (scalar)')
        text(15,233,'r = conj(c₀)Y₀ + conj(c₁)Y₁')
        shown=1 if min(app.progress,1)<.5 else 2
        text(15,261,f'Rx1 contribution: {fmt(terms[0])}',color='#65baff')
        if shown==2:text(15,287,f'Rx2 contribution: {fmt(terms[1])}',color='#ffb569')
        text(15,322,f'Sum shown: {fmt(terms[:shown].sum())}')
        text(15,352,'This combines antennas; it has NOT removed the desired gain.',color='#ffb569')
    else:
        text(15,110,'Divide the combined value by the desired effective gain')
        text(15,145,f'Before: r = {fmt(r)}',color='#ffb569',size=13)
        text(15,184,f'dᵤᵤ = cᴴ H wᵤ = {fmt(g)}',size=13)
        text(15,223,f'After: z = r / dᵤᵤ = {fmt(z)}',color='#83d6ac',size=13)
        text(15,256,f'Transmitted reference sᵤ = {fmt(s)}',color='#65baff')
        text(15,289,f'Desired term / gain: {fmt(desired/g)}')
        text(15,315,f'Other-user interference / gain: {fmt(interference/g)}')
        text(15,341,f'Noise / gain: {fmt(noise/g)}')
        text(15,376,'Equalization corrects gain/phase; it does not remove noise.\nZF suppresses modeled interference; MRT may leave it.',size=10)
