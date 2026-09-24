"""Render submission figures from released measurements; all text >=9.2 pt."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from matplotlib import font_manager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[1]/'data')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    def read(name): return json.loads((args.data_dir/name).read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9.3,
        'axes.labelsize':9.3, 'xtick.labelsize':9.3, 'ytick.labelsize':9.3,
        'legend.fontsize':9.3, 'pdf.fonttype':42, 'axes.spines.top':False,
        'axes.spines.right':False, 'axes.linewidth':.7})
    blue, red, grey = '#4477AA', '#CC4455', '#BBBBBB'
    def save(fig, name):
        fig.savefig(args.out/(name+'.pdf'))
        fig.savefig(args.out/(name+'.png'), dpi=180)
        plt.close(fig)
    # A 244 pt canvas matches the manuscript column; no tight bounding-box rescaling.
    for name, weight in [("Diagram", "normal"), ("DiagramBold", "bold")]:
        pdfmetrics.registerFont(TTFont(name, font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight=weight))))
    c = canvas.Canvas(str(args.out/'fig_stack.pdf'), pagesize=(244,108), initialFontName='Diagram', initialFontSize=9.3)
    c.setFont('DiagramBold',9.3)
    c.drawString(5,97,'Observed failure'); c.drawString(132,97,'Calibration check')
    rows=[('AAC path: +64 ms','Test extraction lag'),('Argmax / foldback','Validate readout range'),('AV-Align near chance','Test gain and chance'),('Length / content floor','Measure spread / floor')]
    for n,(a,b) in enumerate(rows):
        assert pdfmetrics.stringWidth(a, "Diagram", 9.3) <= 122
        assert pdfmetrics.stringWidth(b, "Diagram", 9.3) <= 110
        y=75-22*n
        c.setStrokeColorRGB(.75,.8,.85); c.line(3,y-6,241,y-6)
        c.setFont('Diagram',9.3); c.drawString(5,y,a);c.drawString(132,y,b)
    c.save()
    fig,(ax,bx)=plt.subplots(1,2,figsize=(244/72,128/72))
    fig.subplots_adjust(left=.145,right=.985,bottom=.24,top=.87,wspace=.48)
    ct=read('clicktrain_xcorr.json')
    for tag,color in [('lossless',blue),('aac_noedit',red)]:
        x=np.asarray(ct[tag]['lags'])/16; y=np.asarray(ct[tag]['vals']);m=(x>=-12)&(x<=108)
        ax.plot(x[m],y[m],color=color,lw=1)
    ax.set(xlim=(-12,108),ylim=(-.25,1.12),xticks=[0,64],yticks=[0,1],xlabel='Lag (ms)',ylabel='Correlation')
    ax.set_title('(a) 64 ms lag',fontsize=9.3,pad=4)
    ec=read('ref05_envelope_case.json')
    for tag,color,off in [('lossless',blue,1.15),('aac',red,0)]:
        bx.plot(ec[tag]['t'],np.asarray(ec[tag]['env'])/ec[tag]['env_max']+off,color=color,lw=.6)
    bx.text(59,2.10,'292 onsets',color=blue,ha='right',fontsize=9.3)
    bx.text(59,.85,'3 onsets',color=red,ha='right',fontsize=9.3)
    bx.set(xlim=(-2,60),ylim=(0,2.4),xticks=[0,30,60],yticks=[],xlabel='Time (s)')
    bx.set_title('(b) Onset envelope',fontsize=9.3,pad=4)
    save(fig,'fig_bias')
    exts=sorted(args.data_dir.glob('ref*_desync_calib_ext.json'))
    if len(exts)!=14: raise ValueError(f'Expected 14 reference curves, found {len(exts)}')
    E,A=[],[]
    for p in exts:
        d=json.loads(p.read_text());ks=sorted(d['per_offset'],key=float);inj=np.asarray(list(map(float,ks)))
        E.append([d['per_offset'][k]['mean'] for k in ks]);A.append([np.median(d['curves'][k]['argmax']) for k in ks])
    fig,ax=plt.subplots(figsize=(244/72,140/72));fig.subplots_adjust(left=.18,right=.98,bottom=.25,top=.92)
    ax.axvspan(-3.3,-2,color=red,alpha=.08);ax.axvspan(2,3.3,color=red,alpha=.08)
    ax.plot([-3.3,3.3],[-3.3,3.3],'k--',lw=.6)
    for row in E: ax.plot(inj,row,color=grey,alpha=.55,lw=.5)
    ax.step(inj,np.median(A,axis=0),where='mid',color=red,lw=1.1)
    ax.plot(inj,np.median(E,axis=0),color=blue,lw=1.4,marker='o',ms=2)
    ax.text(-3.05,2.25,'Argmax',color=red);ax.text(.2,-2.6,'Expectation',color=blue)
    ax.set(xlim=(-3.3,3.3),ylim=(-3.3,3.3),xticks=[-3,-2,-1,0,1,2,3],yticks=[-3,0,3],xlabel='Injected offset (s)',ylabel='Reading (s)')
    save(fig,'fig_teaser')
    v=read('avalign_offset_calib_v2.json');mc=read('null_model_mc.json');ns=read('nscaling_realref.json')
    offs=sorted(v['offsets']);keys=[f'{x:+.3f}' for x in offs]
    fig,(ax,bx)=plt.subplots(1,2,figsize=(244/72,134/72));fig.subplots_adjust(left=.15,right=.98,bottom=.25,top=.86,wspace=.7)
    for d in v['per_video'].values():ax.plot(offs,[d['offsets'][k]['full_iou'] for k in keys],color=grey,lw=.5,alpha=.6)
    ax.axhline(mc['chance_mean'],color='black',ls='--',lw=.8)
    ax.plot(offs,[v['summary'][k]['full_mean'] for k in keys],color=blue,lw=1.3)
    ax.plot([-.064],[.105],'s',color=red,ms=4)
    ax.set(xlim=(-1.15,1.15),ylim=(0,.3),xticks=[-1,0,1],yticks=[0,.1,.2,.3],xlabel='Offset (s)',ylabel='AV-Align')
    ax.set_title('(a) Offset sweep',fontsize=9.3)
    pairs=sorted((float(k),d['within_video_sd']) for k,d in ns['summary'].items() if d.get('within_video_sd') is not None)
    ls,sig=np.array(pairs).T; xx=np.array([3,40])
    bx.plot(xx,sig[0]*np.sqrt(ls[0]/xx),'k--',lw=.8);bx.plot(ls,sig,'o-',color=blue,ms=3,lw=1.3)
    bx.set(xlim=(3,32),ylim=(.02,.085),xticks=[4,15,30],yticks=[.03,.05,.07],xlabel='Length (s)',ylabel='SD')
    bx.set_title('(b) Window SD',fontsize=9.3)
    save(fig,'fig_avalign')

if __name__=='__main__':main()
