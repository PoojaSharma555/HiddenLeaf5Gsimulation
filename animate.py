"""Launch a native, interactive animation of the actual phase-1 simulation data.

Run: python animate.py
No matplotlib or browser required. Tkinter must be installed with Python.
Playback is deliberately slowed down; it is not a physical real-time receiver.
"""
from pathlib import Path
import argparse
import contextlib
import io
import time
import numpy as np
from simulate import run, NFFT, NCP, FS, BINS, TONES

STAGES = [
    ('UTF-8 bytes', 'c = UTF8(message)', 'Each character becomes a byte. The two messages stay separate.'),
    ('Payload bits', 'b[8j+r] = floor(c[j] / 2^(7-r)) mod 2', 'Read each byte from its most significant bit to its least significant bit.'),
    ('QPSK mapping', 's[m] = ((1−2b[2m]) + j(1−2b[2m+1])) / √2', 'Each pair of bits selects one complex symbol. I and Q belong to the same user.'),
    ('Frequency grid', 'X[q, tone mod 128] = s[m]; unused bins = 0', 'Fill 72 allocated tones per OFDM symbol. Grey cells remain zero amplitude.'),
    ('IFFT synthesis', 'x[q,n] = Σₖ X[q,k] exp(j2πkn/128) / √128', 'Watch the active subcarriers add up. The final sum is the complex baseband waveform.'),
    ('Cyclic prefix', 'x_CP[q] = [x[q,118:128], x[q,0:128]]', 'Copy the final 10 useful samples to the beginning. Shading marks the copied prefix.'),
    ('Serialize waveform', 'tx = concatenate(x_CP[0], x_CP[1], x_CP[2])', 'Three OFDM symbols become 414 complex samples on one time axis.'),
    ('Generate AWGN', 'w = √(σ²/2) (z_I + jz_Q), σ² = 10^(−EsN0_dB/10)', 'Each user gets independent complex Gaussian receiver noise.'),
    ('Received waveform', 'rx[n] = tx[n] + w[n]', 'The receiver observes the transmitted envelope plus noise. H1/H2 are not applied.'),
    ('Remove CP', 'rx_useful[q] = rx_blocks[q,10:138]', 'Discard the copied prefix before the FFT. Timing is assumed perfectly known.'),
    ('Receiver FFT', 'Y[q,k] = Σₙ rx_useful[q,n] exp(−j2πkn/128) / √128', 'The FFT separates the subcarriers. These are FFT results revealed in bin order.'),
    ('Select allocated tones', 'allocated = Y[:, active_bins].reshape(−1)', 'Keep the 72 assigned bins, including unused payload positions that now contain noise.'),
    ('Discard padding', 's_hat = allocated[:payload_symbol_count]', 'Keep 192 payload symbols for Naruto and 152 for Sasuke.'),
    ('Hard decisions', 'b_hat[2m] = Re(s_hat[m])<0; b_hat[2m+1] = Im(s_hat[m])<0', 'The signs of I and Q recover the two bits. Red cells mark actual bit errors.'),
    ('Recovered bytes', 'c_hat = packbits(b_hat), most significant bit first', 'Group each set of eight recovered bits into a byte.'),
    ('Recovered message', 'BER = incorrect bits / payload bits', 'Decode the bytes as UTF-8. The error counts are from this exact run.'),
]
BG, PANEL, INK, MUTED = '#101722', '#182333', '#e8eef7', '#aab9ca'
BLUE, ORANGE, GREEN, RED = '#65baff', '#ffb569', '#83d6ac', '#ff7f8f'


def prepare(output, snr_db=15., seed=555):
    """Generate and load one consistent run, using the same processing as simulate.py."""
    with contextlib.redirect_stdout(io.StringIO()):
        report = run(snr_db, seed, output)
    with np.load(output/'waveforms_and_channels.npz') as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    return report, arrays


def partial_ifft(grid_row, count):
    """Actual partial Fourier synthesis in ascending signed-tone order."""
    count = max(0, min(len(BINS), int(count)))
    n = np.arange(NFFT)
    return (grid_row[BINS[:count], None] *
            np.exp(2j*np.pi*TONES[:count, None]*n/NFFT)).sum(axis=0)/np.sqrt(NFFT)


class Animation:
    def __init__(self, root, report, arrays, autoplay=True):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.root, self.report, self.arrays = tk, root, report, arrays
        self.stage, self.progress, self.playing = 0, 0., autoplay
        self.last_time, self.seek_update, self.closed = time.monotonic(), False, False
        root.title('Hidden Leaf • OFDM signal journey')
        root.geometry('1180x820')
        root.minsize(940, 710)
        root.configure(bg=BG)
        root.protocol('WM_DELETE_WINDOW', self.close)
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('TFrame', background=BG)
        style.configure('TLabel', background=BG, foreground=INK, font=('Helvetica', 11))
        style.configure('TButton', padding=6)
        top = ttk.Frame(root, padding=14)
        top.pack(fill='x')
        ttk.Label(top, text='HIDDEN LEAF  /  SIGNAL JOURNEY', font=('Helvetica', 18, 'bold')).pack(anchor='w')
        ttk.Label(top, text=f'Two independent baseband chains • QPSK • 60 kHz SCS • {report["requested_snr_db"]:g} dB Es/N0', foreground=MUTED).pack(anchor='w', pady=4)
        ttk.Label(top, text='Slowed educational replay of actual samples • 53.906 µs simulated frame • no spatial transmission yet', foreground=MUTED).pack(anchor='w')
        controls = ttk.Frame(root, padding=(14, 0, 14, 8))
        controls.pack(fill='x')
        ttk.Button(controls, text='Restart', command=self.restart).pack(side='left', padx=(0, 6))
        ttk.Button(controls, text='◀ Previous', command=lambda: self.step(-1)).pack(side='left', padx=6)
        self.play_button = ttk.Button(controls, text='Pause' if autoplay else 'Play', command=self.toggle)
        self.play_button.pack(side='left', padx=6)
        ttk.Button(controls, text='Next ▶', command=lambda: self.step(1)).pack(side='left', padx=6)
        ttk.Label(controls, text='Speed').pack(side='left', padx=(20, 5))
        self.speed = tk.StringVar(value='1×')
        ttk.Combobox(controls, textvariable=self.speed, values=['0.5×', '1×', '2×', '4×'], state='readonly', width=5).pack(side='left')
        ttk.Label(controls, text='OFDM symbol to inspect').pack(side='left', padx=(20, 5))
        self.symbol = tk.StringVar(value='0')
        symbol_box = ttk.Combobox(controls, textvariable=self.symbol, values=['0', '1', '2'], state='readonly', width=4)
        symbol_box.pack(side='left')
        symbol_box.bind('<<ComboboxSelected>>', lambda _: self.render())
        select_row = ttk.Frame(root, padding=(14, 0, 14, 8))
        select_row.pack(fill='x')
        self.stage_name = tk.StringVar(value='1. '+STAGES[0][0])
        stages = ttk.Combobox(select_row, textvariable=self.stage_name,
                             values=[f'{i+1}. {s[0]}' for i,s in enumerate(STAGES)], state='readonly', width=30)
        stages.pack(side='left')
        stages.bind('<<ComboboxSelected>>', lambda _: self.choose(int(self.stage_name.get().split('.')[0])-1))
        self.seek = tk.DoubleVar(value=0)
        ttk.Scale(select_row, from_=0, to=100, variable=self.seek, command=self.scrub).pack(side='left', fill='x', expand=True, padx=15)
        self.status = ttk.Label(select_row, text='', width=20)
        self.status.pack(side='right')
        self.formula = ttk.Label(root, text='', font=('Helvetica', 15), padding=(14, 8))
        self.formula.pack(anchor='w')
        self.explanation = ttk.Label(root, text='', wraplength=1120, padding=(14, 0, 14, 10))
        self.explanation.pack(anchor='w')
        panes = ttk.Frame(root, padding=(14, 0, 14, 0))
        panes.pack(fill='both', expand=True)
        self.canvases = []
        for u in (1, 2):
            canvas = tk.Canvas(panes, bg=PANEL, highlightthickness=0, width=550, height=410)
            canvas.pack(side='left', fill='both', expand=True, padx=(0, 8) if u==1 else (0, 0))
            canvas.bind('<Configure>', lambda _: self.render())
            self.canvases.append(canvas)
        ttk.Label(root, text='Blue = I (real)   Orange = Q (imaginary)   •   Space: play/pause   ←/→: change block',
                  foreground=MUTED, padding=14).pack(anchor='w')
        root.bind('<space>', lambda _: self.toggle())
        root.bind('<Left>', lambda _: self.step(-1))
        root.bind('<Right>', lambda _: self.step(1))
        self.render()
        root.after(40, self.tick)

    def close(self):
        self.closed = True
        self.root.destroy()

    def toggle(self):
        if self.stage == 15 and self.progress >= 1 and not self.playing:
            self.stage, self.progress = 0, 0.
        self.playing = not self.playing
        self.last_time = time.monotonic()
        self.render()

    def restart(self):
        self.stage, self.progress, self.playing = 0, 0., True
        self.last_time = time.monotonic()
        self.render()

    def choose(self, stage):
        self.stage, self.progress, self.playing = stage, 0., False
        self.render()

    def step(self, delta):
        self.choose(max(0, min(15, self.stage+delta)))

    def scrub(self, value):
        if not self.seek_update:
            self.progress, self.playing = float(value)/100, False
            self.render()

    def tick(self):
        if self.closed:
            return
        now = time.monotonic()
        if self.playing:
            self.progress += min(now-self.last_time, .25)*float(self.speed.get().rstrip('×'))/6
            # A brief hold at completion lets the finished block be inspected.
            if self.progress >= 1.18:
                if self.stage < 15:
                    self.stage, self.progress = self.stage+1, 0.
                else:
                    self.progress, self.playing = 1., False
            self.render()
        self.last_time = now
        self.root.after(40, self.tick)

    def text(self, c, x, y, text, color=INK, size=12, anchor='nw', width=None):
        kwargs = dict(fill=color, font=('Helvetica', size), anchor=anchor)
        if width is not None:
            kwargs['width'] = width
        c.create_text(x, y, text=text, **kwargs)

    def waveform(self, c, data, count, xlabel, cp=False, reference=None):
        w, h = c.winfo_width(), c.winfo_height()
        left, right, top, bottom = 62, w-24, 115, h-85
        limit = max(.15, float(np.max(np.abs(np.r_[data.real, data.imag])))*1.12)
        if reference is not None:
            limit = max(limit, float(np.max(np.abs(np.r_[reference.real, reference.imag])))*1.12)
        def px(i): return left+i/max(1,len(data)-1)*(right-left)
        def py(v): return (top+bottom)/2-v/limit*(bottom-top)/2
        if cp:
            c.create_rectangle(left, top, px(NCP-1), bottom, fill='#33433e', outline='')
            self.text(c, left+2, top+2, 'CP', GREEN, 10)
        c.create_rectangle(left, top, right, bottom, outline='#536175')
        c.create_line(left, py(0), right, py(0), fill='#38465a')
        for v in (-limit, 0, limit):
            self.text(c, left-7, py(v), f'{v:.2f}', MUTED, 10, 'e')
        for i in (0, (len(data)-1)//2, len(data)-1):
            label = f'{i/FS*1e6:.2f}' if xlabel.startswith('Time') else str(i)
            self.text(c, px(i), bottom+8, label, MUTED, 10, 'n')
        self.text(c, (left+right)/2, bottom+32, xlabel, MUTED, 11, 'n')
        self.text(c, left, top-22, 'Normalized amplitude', MUTED, 10)
        if reference is not None:
            for values in (reference.real, reference.imag):
                points = [p for i,v in enumerate(values) for p in (px(i), py(v))]
                c.create_line(*points, fill='#465165', dash=(3, 3))
        for values, color in [(data.real, BLUE), (data.imag, ORANGE)]:
            points = [p for i,v in enumerate(values[:count]) for p in (px(i), py(v))]
            if len(points)>=4:
                c.create_line(*points, fill=color, width=2)
        if count:
            z = data[min(count-1, len(data)-1)]
            self.text(c, left, h-24, f'Last displayed: {z.real:+.6f} {z.imag:+.6f}j', INK, 11)

    def constellation(self, c, data, count):
        w, h = c.winfo_width(), c.winfo_height()
        size = min(w-110, h-195)
        left, top = (w-size)/2, 115
        limit = max(1.3, float(np.max(np.abs(np.r_[data.real, data.imag])))*1.12)
        def px(v): return left+(v+limit)/(2*limit)*size
        def py(v): return top+(limit-v)/(2*limit)*size
        c.create_rectangle(left,top,left+size,top+size,outline='#536175')
        c.create_line(px(0), top, px(0),top+size,fill='#536175')
        c.create_line(left,py(0),left+size,py(0),fill='#536175')
        for a,b in [(0,0),(0,1),(1,0),(1,1)]:
            z = ((1-2*a)+1j*(1-2*b))/np.sqrt(2)
            x,y = px(z.real),py(z.imag)
            c.create_oval(x-5,y-5,x+5,y+5,outline=GREEN,width=2)
            self.text(c,x+8,y-15,f'{a}{b}',GREEN,10)
        for z in data[:count]:
            x,y = px(z.real),py(z.imag)
            c.create_oval(x-2,y-2,x+2,y+2,fill=BLUE,outline='')
        for v in [-limit,0,limit]:
            self.text(c,px(v),top+size+5,f'{v:.1f}',MUTED,10,'n')
            self.text(c,left-7,py(v),f'{v:.1f}',MUTED,10,'e')
        self.text(c,left+size/2,top+size+26,'I (real)',MUTED,11,'n')
        self.text(c,left,top-20,'Q (imaginary)',MUTED,11)
        if count:
            z=data[count-1]
            self.text(c,20,h-35,f'Symbol {count-1}: {z.real:+.6f} {z.imag:+.6f}j',size=11)

    def render_user(self, c, u):
        c.delete('all')
        w,h = c.winfo_width(),c.winfo_height()
        if w < 100 or h < 100:
            return
        p = min(self.progress,1.)
        row = self.report['users'][u-1]
        a = lambda key: self.arrays[f'ue{u}_{key}']
        q = int(self.symbol.get())
        self.text(c,20,16,f'UE{u}  /  '+('NARUTO' if u==1 else 'SASUKE'),GREEN,15)
        self.text(c,20,45,f'{row["bits"]} bits → {row["qpsk_symbols"]} QPSK symbols → 3 OFDM symbols',MUTED,11)
        stage = self.stage
        if stage in (0,14,15):
            data = a('bytes' if stage==0 else 'recovered_bytes')
            count = min(len(data),int(p*len(data)))
            content = bytes(data[:count]).decode('utf-8', errors='replace')
            self.text(c,20,90,content+'▌',size=22,width=w-40)
            if stage!=15:
                rows = [f'[{i:02d}]  {int(v):3d}   {int(v):08b}' for i,v in enumerate(data[:count])]
                # Keep the latest byte visible as the stream advances.
                capacity = max(1,int((h-245)/19))
                shown = rows[-capacity:]
                self.text(c,20,190,'Index  Byte  Binary (latest bytes)',MUTED,11)
                self.text(c,20,217,'\n'.join(shown),size=12)
            else:
                self.text(c,20,200,f'Bit errors: {row["bit_errors"]}/{row["bits"]}\nBER: {row["ber"]:.6f}\nRMS EVM: {row["rms_evm_percent"]:.2f}%',size=19)
                self.text(c,20,h-55,'Payload recovery only; no beamforming/channel propagation.',MUTED,11,width=w-40)
        elif stage in (1,13):
            data = a('bits' if stage==1 else 'recovered_bits')
            count = int(p*len(data))
            cols=32
            cell=(w-40)/cols
            height=min(25,(h-155)/int(np.ceil(len(data)/cols)))
            for i in range(len(data)):
                x,y=20+(i%cols)*cell,105+(i//cols)*height
                active=i<count
                wrong=stage==13 and data[i]!=a('bits')[i]
                color=RED if wrong else (BLUE if data[i] else GREEN)
                c.create_rectangle(x,y,x+cell-2,y+height-2,fill=color if active else '#283548',outline='')
                if active:
                    self.text(c,x+cell/2,y+height/2,str(data[i]),BG,10,'center')
            self.text(c,20,78,f'{count}/{len(data)} bits • read left to right, then next row',MUTED,11)
        elif stage in (2,11,12):
            key = {2:'qpsk',11:'rx_allocated',12:'received_qpsk'}[stage]
            data=a(key)
            count=int(p*len(data))
            self.text(c,20,78,f'{count}/{len(data)} symbols • green rings = ideal QPSK',MUTED,11)
            self.constellation(c,data,count)
        elif stage==3:
            count=int(p*row['qpsk_symbols'])
            grid=a('grid')
            cell=(w-45)/72
            spacing=max(48,(h-190)/3)
            for symbol in range(3):
                y=110+symbol*spacing
                self.text(c,20,y-22,f'OFDM symbol {symbol}',MUTED,11)
                for j,k in enumerate(BINS):
                    active=symbol*72+j<count and grid[symbol,k]!=0
                    x=20+j*cell
                    c.create_rectangle(x,y,x+cell-1,y+28,fill=BLUE if active else '#334155',outline='')
            self.text(c,20,h-72,'72 allocated tones: −36…−1, +1…+36\nDC and 55 outer guard bins are zero.\nFinal-symbol padding remains grey (zero amplitude).',MUTED,11,width=w-40)
        elif stage==4:
            count=int(p*len(BINS))
            data=partial_ifft(a('grid')[q],count)
            self.text(c,20,78,f'OFDM symbol {q} • {count}/72 tone contributions • dashed: final waveform',MUTED,11)
            self.waveform(c,data,NFFT,'Sample n within useful OFDM symbol',reference=a('ifft')[q])
        elif stage==10:
            data=a('rx_grid')[q]
            count=int(p*NFFT)
            self.text(c,20,78,f'OFDM symbol {q} • {count}/128 FFT bins (unshifted order)',MUTED,11)
            self.waveform(c,data,count,'FFT bin k (unshifted order)')
        else:
            key={5:'cp_blocks',6:'tx',7:'noise',8:'rx',9:'rx_no_cp'}[stage]
            data=a(key)
            if data.ndim==2:
                data=data[q]
            count=int(p*len(data))
            self.text(c,20,78,f'{count}/{len(data)} complex samples'+(f' • OFDM symbol {q}' if stage in (5,9) else ''),MUTED,11)
            self.waveform(c,data,count,'Time from start of displayed block (µs)',cp=stage==5)

    def render(self):
        if self.closed:
            return
        title,formula,explanation=STAGES[self.stage]
        self.stage_name.set(f'{self.stage+1}. {title}')
        self.formula.configure(text=formula)
        self.explanation.configure(text=explanation)
        self.status.configure(text=f'Block {self.stage+1}/16 • {min(self.progress,1):.0%}')
        self.play_button.configure(text='Pause' if self.playing else 'Play')
        self.seek_update=True
        self.seek.set(min(self.progress,1)*100)
        self.seek_update=False
        for u,c in enumerate(self.canvases,1):
            self.render_user(c,u)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snr-db',type=float,default=15.)
    parser.add_argument('--seed',type=int,default=555)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results')
    parser.add_argument('--paused',action='store_true',help='Open paused for manual stepping')
    parser.add_argument('--check',action='store_true',help='Verify data and Fourier synthesis without opening a GUI')
    args=parser.parse_args()
    report,arrays=prepare(args.output,args.snr_db,args.seed)
    if args.check:
        for u in (1,2):
            for q in range(3):
                np.testing.assert_allclose(partial_ifft(arrays[f'ue{u}_grid'][q],72),arrays[f'ue{u}_ifft'][q],atol=3e-14)
        print('Animation data verified: both users, all 3 OFDM symbols, all 16 blocks available.')
        return
    try:
        import tkinter as tk
    except ImportError:
        parser.exit(1,'Tkinter is missing. Use a desktop Python installation with Tcl/Tk support.\n')
    try:
        root=tk.Tk()
    except tk.TclError:
        parser.exit(1,'Cannot open the Tk window in this environment. Run python animate.py from a graphical desktop terminal with a working Tcl/Tk installation. Use --check for numerical verification without a display.\n')
    app=Animation(root,report,arrays,not args.paused)
    root.mainloop()


if __name__=='__main__':
    main()
