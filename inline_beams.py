"""Compact beam-direction views inside the main animation window."""
import numpy as np
from mimo import beam_patterns


def draw_inline_beams(app):
    c=app.inline_beams
    c.delete('all')
    width,height=c.winfo_width(),c.winfo_height()
    if width<100 or height<100:return
    k=int(app.tone.get())%128
    a=app.arrays
    hs=np.array([a[f'H{u}'][k] if a[f'H{u}'].ndim==3 else a[f'H{u}'] for u in (1,2)])
    w=a['precoder_fd'][k] if 'precoder_fd' in a else a['precoder']
    angles,power,_=beam_patterns(hs,w)
    for u in range(2):
        left=55+u*width/2;right=(u+1)*width/2-25;top=42;bottom=height-45
        px=lambda theta:left+(theta+90)/180*(right-left)
        py=lambda db:bottom-(db+40)/40*(bottom-top)
        color=['#65baff','#ffb569'][u]
        peak=float(angles[np.argmax(power[u])])
        c.create_text(left,7,text=f'UE{u+1} beam • peak {peak:+.1f}° • {app.config["precoder"]} • tone {app.tone.get()}',anchor='nw',fill=color,font=('Helvetica',11))
        c.create_rectangle(left,top,right,bottom,outline='#536175')
        for v in [-90,-45,0,45,90]:c.create_text(px(v),bottom+12,text=str(v),fill='#aab9ca',font=('Helvetica',9))
        for v in [-40,-20,0]:c.create_text(left-20,py(v),text=str(v),fill='#aab9ca',font=('Helvetica',9))
        coords=[z for theta,p in zip(angles,power[u]) for z in (px(theta),py(p))]
        c.create_line(*coords,fill=color,width=2)
        c.create_line(px(peak),top,px(peak),bottom,fill=color,dash=(3,3))
        c.create_text((left+right)/2,height-9,text='Angle from broadside (°) • relative power (dB), peak = 0',fill='#aab9ca',font=('Helvetica',9))
