"""Native Tk typesetting: real fractions, radicals, and raised/lowered scripts.

No TeX installation, image assets, or extra Python dependency is required.
"""
import tkinter as tk
from tkinter import font


def row(*nodes):return ('row',nodes)
def sub(base,value):return ('script',base,None,value)
def sup(base,value):return ('script',base,value,None)
def both(base,upper,lower):return ('script',base,upper,lower)
def frac(top,bottom):return ('frac',top,bottom)
def root(value):return ('root',value)

FORMULAS=[
    row('c = UTF8(message)'),
    row(sub('b','8j+r'),' = ⌊',frac(sub('c','j'),sup('2','7−r')),'⌋ mod 2'),
    row(sub('s','m'),' = ',frac(row('(1 − 2',sub('b','2m'),') + j(1 − 2',sub('b','2m+1'),')'),root('2'))),
    row(sub('S','u'),'[q, k] = ',sub('s','u'),'[m]     (allocated tones; otherwise 0)'),
    row(sub('H','u'),'[k] = ',both('Σ','2','ℓ=0'),sub('A','u,ℓ'),sup('e',row('−j2πk',sub('d','ℓ'),'/N')),
        '       ',sub('Y','u'),' = ',sub('H','u'),' W s + ',sub('N','u')),
    row('x[q,n] = ',frac('1',root('N')),both('Σ','N−1','k=0'),'X[q,k]',sup('e','j2πkn/N')),
    row(sub('x','CP'),'[q] = [x[q,N−L:N], x[q,0:N]]'),
    row(sub('x','BS'),' = W s     (per subcarrier before antenna IFFTs)'),
    row(sub('n','u,r'),' = ',root(frac(sup('σ','2'),'2')),'(',sub('z','I'),' + j',sub('z','Q'),')'),
    row(sub('y','u'),'[n] = ',both('Σ','2','ℓ=0'),sub('A','u,ℓ'),sub('x','BS'),'[n − ',sub('d','ℓ'),'] + ',sub('n','u'),'[n]'),
    row(sub('y','useful'),'[q,n] = ',sub('y','CP'),'[q,n+L]'),
    row('Y[q,k] = ',frac('1',root('N')),both('Σ','N−1','n=0'),'y[q,n]',sup('e','−j2πkn/N')),
    row(sub('z','u'),'[q,k] = ',frac(row(both('c','H','u'),'[k]',sub('Y','u'),'[q,k]'),row(sub('D','uu'),'[k]'))),
    row(sub('ŝ','u'),' = allocated[0 : payload symbol count]'),
    row(sub('b̂','2m'),' = 1{Re(',sub('ŝ','m'),') < 0}     ',sub('b̂','2m+1'),' = 1{Im(',sub('ŝ','m'),') < 0}'),
    row('ĉ = packbits(b̂), most significant bit first'),
    row('BER = ',frac('incorrect payload bits','total payload bits')),
]

FORMULAS[12] = row('allocated = z[q, assigned tones]')
FORMULAS[12:12] = [row(sub('r','u'),'[q,k] = ',both('c','H','u'),'[k]',sub('Y','u'),'[q,k]'), row(sub('z','u'),'[q,k] = ',frac(sub('r','u'),sub('d','uu')),'     ',sub('d','uu'),' = ',both('c','H','u'),sub('H','u'),sub('w','u'))]

class MathCanvas(tk.Canvas):
    def __init__(self,parent,**kwargs):
        super().__init__(parent,height=84,bg='#101722',highlightthickness=0,**kwargs)
        self.stage=None
        self.fonts={}
        self.bind('<Configure>',lambda _:self.redraw())

    def face(self,size):
        size=max(9,int(size))
        if size not in self.fonts:self.fonts[size]=font.Font(family='Helvetica',size=size)
        return self.fonts[size]

    def layout(self,node,size):
        # Return width, height, commands relative to the top left.
        if isinstance(node,str):
            f=self.face(size)
            return f.measure(node),f.metrics('linespace'),[('text',0,0,node,size)]
        tag=node[0]
        if tag=='row':
            parts=[self.layout(n,size) for n in node[1]]
            height=max(p[1] for p in parts);x=0;commands=[]
            for w,h,cmd in parts:
                commands+=self.shift(cmd,x,(height-h)/2);x+=w
            return x,height,commands
        if tag=='frac':
            a=self.layout(node[1],size*.85);b=self.layout(node[2],size*.85)
            width=max(a[0],b[0])+10
            commands=self.shift(a[2],(width-a[0])/2,0)+[('line',0,a[1]+3,width,a[1]+3)]
            commands+=self.shift(b[2],(width-b[0])/2,a[1]+7)
            return width,a[1]+b[1]+7,commands
        if tag=='root':
            w,h,cmd=self.layout(node[1],size)
            return w+18,h+5,self.shift(cmd,17,5)+[('line',1,h*.6,5,h*.6+3),('line',5,h*.6+3,9,h+3),('line',9,h+3,15,2),('line',15,2,w+18,2)]
        if tag=='script':
            a=self.layout(node[1],size)
            upper=self.layout(node[2],size*.65) if node[2] is not None else (0,0,[])
            lower=self.layout(node[3],size*.65) if node[3] is not None else (0,0,[])
            top=upper[1]*.65
            cmd=self.shift(a[2],0,top)+self.shift(upper[2],a[0],0)+self.shift(lower[2],a[0],top+a[1]*.65)
            return a[0]+max(upper[0],lower[0])+2,top+a[1]+lower[1]*.5,cmd
        raise ValueError(tag)

    @staticmethod
    def shift(commands,x,y):
        result=[]
        for command in commands:
            if command[0]=='text':
                tag,a,b,text,size=command;result.append((tag,a+x,b+y,text,size))
            else:
                tag,a,b,c,d=command;result.append((tag,a+x,b+y,c+x,d+y))
        return result

    def set_stage(self,stage):
        if self.stage!=stage:self.stage=stage;self.redraw()

    def redraw(self):
        if self.stage is None:return
        self.delete('all')
        size=20
        width,height,commands=self.layout(FORMULAS[self.stage],size)
        while width>max(300,self.winfo_width()-30) and size>12:
            size-=1;width,height,commands=self.layout(FORMULAS[self.stage],size)
        for command in self.shift(commands,14,max(0,(84-height)/2)):
            if command[0]=='text':
                _,x,y,text,size=command
                self.create_text(x,y,text=text,font=self.face(size),fill='#e8eef7',anchor='nw')
            else:self.create_line(*command[1:],fill='#e8eef7',width=1.5)
