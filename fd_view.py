"""Bracketed, dimension-labelled matrices for the actual FD signal path."""
import numpy as np
from simulate import TONES


def matrix(c, data, x, y, cell_w=65, cell_h=25, size=10, highlight=None):
    data=np.asarray(data)
    if data.ndim==1:data=data[:,None]
    rows,cols=data.shape
    width,height=cols*cell_w,rows*cell_h
    for edge,direction in [(x,1),(x+width+10,-1)]:
        c.create_line(edge+direction*6,y,edge,y,edge,y+height,edge+direction*6,y+height,fill='#e8eef7',width=2)
    for r in range(rows):
        for col in range(cols):
            z=data[r,col]
            color='#ffb569' if highlight==col else '#e8eef7'
            text=f'{z.real:.1f}{z.imag:+.1f}j'
            c.create_text(x+5+(col+.5)*cell_w,y+(r+.5)*cell_h,text=text,anchor='center',fill=color,font=('Menlo',size))
    return width+10,height


def render_fd(app,c,u=None):
    c.delete('all')
    width,height=c.winfo_width(),c.winfo_height()
    q=int(app.symbol.get());tone=int(app.tone.get()) if hasattr(app,'tone') else int(TONES[0]);k=tone%128
    a=app.arrays
    w=a['precoder_fd'][k] if 'precoder_fd' in a else a['precoder']
    s=np.array([a[f'ue{v}_grid'][q,k] for v in (1,2)]);x=w@s
    # Shared wide canvas shows W and both users' channels together.
    scale=max(.45,min(1.,width/1180))
    text=lambda xx,yy,value,color='#e8eef7',size=12:app.text(c,xx*scale,yy,value,color,size)
    text(18,12,f'ONE RESOURCE ELEMENT: q={q}, tone={tone:+d}, f={tone*60} kHz')
    text(18,40,'1. Precoder: two user symbols → eight complex antenna weights/signals',size=11)
    matrix(c,w,18*scale,104,68*scale,26,size=max(8,int(10*scale)))
    text(25,76,'W  (8 × 2)')
    text(170,186,'×',size=20)
    matrix(c,s,199*scale,178,72*scale,26,size=10)
    text(202,147,'s  (2 × 1)')
    text(292,186,'=',size=20)
    matrix(c,x,325*scale,104,72*scale,26,size=10)
    text(325,76,'x  (8 × 1)')
    text(18,328,'Each row: xₜ = Wₜ₁ s₁ + Wₜ₂ s₂',size=11)
    text(18,350,'Each column of W is one user’s beam.',size=11)
    text(18,373,'W controls amplitude and phase at all 8 antennas.',size=11)
    active=min(7,int(min(app.progress,.999)*8))
    text(18,398,f'Channel sum: highlighted Tx column {active}',color='#ffb569',size=11)
    for user in (1,2):
        h=a[f'H{user}'];h=h[k] if h.ndim==3 else h
        offset=75+(user-1)*170
        text(465,offset,f'2. UE{user}: H{user}[k] (2 × 8) × x (8 × 1)',size=12)
        matrix(c,h,465*scale,offset+30,75*scale,25,size=max(8,int(10*scale)),highlight=active)
        noise=a.get(f'ue{user}_noise_fd')
        if noise is None:
            noise=np.fft.fft(a[f'ue{user}_antenna_noise'].reshape(2,3,138)[:,:,10:],axis=-1,norm='ortho')
        noise=noise[:,q,k];y=a[f'ue{user}_antenna_fft'][:,q,k]
        text(465,offset+101,'Hx + N = Y',size=12)
        matrix(c,h@x,585*scale,offset+90,78*scale,22,size=9)
        text(683,offset+106,'+',size=18)
        matrix(c,noise,714*scale,offset+90,78*scale,22,size=9)
        text(812,offset+106,'=',size=18)
        matrix(c,y,842*scale,offset+90,78*scale,22,size=9)
        text(945,offset+96,f'Error: {np.max(abs(y-h@x-noise)):.1e}',size=10)
    text(465,424,'Brackets show actual matrix dimensions. Values rounded to 1 decimal; full precision is saved.',size=10)
