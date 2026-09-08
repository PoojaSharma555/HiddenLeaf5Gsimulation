"""Tk controls for channel editing and geometry-dependent beam plots."""
import json
from pathlib import Path
from mimo import defaults, parse_matrix, serial_matrix, run_mimo


def install_controls(app, parent):
    import tkinter as tk
    from tkinter import ttk, messagebox
    app.config=app.report.get('channel_config', defaults())
    app.receiver=tk.StringVar(value='Rx1')
    app.snr=tk.StringVar(value=str(app.config['snr_db']))
    app.method=tk.StringVar(value=app.config['precoder'])
    app.beam_window=None
    row=ttk.Frame(parent)
    row.pack(fill='x',pady=(8,0))
    ttk.Label(row,text='Reference SNR (dB)').pack(side='left')
    ttk.Entry(row,textvariable=app.snr,width=7).pack(side='left',padx=6)
    box=ttk.Combobox(row,textvariable=app.method,values=['ZF','MRT'],state='readonly',width=5)
    box.pack(side='left',padx=6)
    def apply(channels=False):
        try:
            new={**app.config,'snr_db':float(app.snr.get()),'precoder':app.method.get()}
            if channels:
                for key,editor in app.editors.items():
                    new[key]=serial_matrix(parse_matrix(editor.get('1.0','end')))
            report,arrays=run_mimo(new,app.report['seed'],Path(app.output))
            app.report,app.arrays,app.config=report,arrays,report['channel_config']
            app.progress,app.playing=0.,False
            app.render()
            if app.beam_window is not None and app.beam_window.winfo_exists():
                draw_beams(app)
        except (ValueError,TypeError,SyntaxError,OverflowError) as exc:
            messagebox.showerror('Check configuration',str(exc),parent=app.root)
    ttk.Button(row,text='Apply SNR / precoder',command=apply).pack(side='left',padx=6)
    ttk.Label(row,text='Receiver view').pack(side='left',padx=(10,4))
    receiver=ttk.Combobox(row,textvariable=app.receiver,values=['Rx1','Rx2','Combined / equalized'],state='readonly',width=21)
    receiver.pack(side='left')
    receiver.bind('<<ComboboxSelected>>',lambda _: app.render())
    def open_beams():
        if app.beam_window is not None and app.beam_window.winfo_exists():
            app.beam_window.lift()
            return
        win=tk.Toplevel(app.root)
        app.beam_window=win
        win.title('Channels, precoding and ULA beams')
        win.geometry('1080x760')
        win.minsize(900,680)
        win.configure(bg='#101722')
        ttk.Label(win,text='8-element ULA • d = λ/2 • angle measured from broadside',padding=12).pack(anchor='w')
        ttk.Label(win,text='Edit H1 / H2, then Apply channels. Rows = Rx elements; columns = BS elements. Use i or j for complex values.',padding=(12,0)).pack(anchor='w')
        editors=ttk.Frame(win,padding=12)
        editors.pack(fill='x')
        app.editors={}
        for key in ('H1','H2'):
            frame=ttk.Frame(editors)
            frame.pack(side='left',fill='x',expand=True,padx=5)
            ttk.Label(frame,text=key).pack(anchor='w')
            editor=tk.Text(frame,height=5,width=45,font=('Menlo',11),wrap='word')
            editor.pack(fill='x')
            editor.insert('1.0',json.dumps(app.config[key]))
            app.editors[key]=editor
        ttk.Button(win,text='Apply channels and update beams',command=lambda:apply(True)).pack(anchor='w',padx=16)
        app.beam_canvas=tk.Canvas(win,bg='#182333',highlightthickness=0)
        app.beam_canvas.pack(fill='both',expand=True,padx=12,pady=12)
        app.beam_canvas.bind('<Configure>',lambda _:draw_beams(app))
        ttk.Label(win,text='Shapes are normalized separately to 0 dB. Dashed channel response is not a reconstructed multipath/arrival-angle map.',padding=(12,0,12,12)).pack(anchor='w')
        draw_beams(app)
    ttk.Button(row,text='Channels & beams',command=open_beams).pack(side='left',padx=8)
    app.open_beams=open_beams


def draw_beams(app):
    c=app.beam_canvas
    c.delete('all')
    w,h=c.winfo_width(),c.winfo_height()
    if w<100 or h<100:return
    angles=app.arrays['beam_angles_deg']
    for u in range(2):
        left=60+u*w/2
        right=(u+1)*w/2-20
        top,bottom=60,h-95
        def x(v):return left+(v+90)/180*(right-left)
        def y(v):return bottom-(v+40)/40*(bottom-top)
        c.create_text(left,15,text=f'UE{u+1} • {app.config["precoder"]} transmit beam and channel response',anchor='nw',fill='#e8eef7',font=('Helvetica',12))
        c.create_rectangle(left,top,right,bottom,outline='#536175')
        for v in [-90,-45,0,45,90]:
            c.create_text(x(v),bottom+15,text=str(v),fill='#aab9ca')
        for v in [-40,-20,0]:
            c.create_text(left-20,y(v),text=str(v),fill='#aab9ca')
        c.create_text((left+right)/2,bottom+40,text='Angle from broadside (degrees)',fill='#aab9ca')
        c.create_text(left,top-15,text='Relative power (dB)',anchor='w',fill='#aab9ca')
        for key,color,dash in [('transmit_beams_db','#65baff',()),('channel_response_db','#ffb569',(5,3))]:
            coords=[z for theta,v in zip(angles,app.arrays[key][u]) for z in (x(theta),y(v))]
            c.create_line(*coords,fill=color,width=2,dash=dash)
        sinr=app.report['users'][u]['full_load_sinr_db']
        c.create_text(left,bottom+64,text=f'Solid: |aᴴw|²   Dashed: ‖Ha‖²   SINR: {sinr:.2f} dB',anchor='w',fill='#e8eef7')
