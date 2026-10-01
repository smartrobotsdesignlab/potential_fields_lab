import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'potential_fields_lab'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, math
from pole_detection import detect_poles
from synth_lds import raycast, noisy, ANGLE_MIN, ANGLE_INC
rng = np.random.default_rng(1)
R = 0.05
WALLS = (-2.6, 2.8, -2.4, 2.5)   # office walls ~2.5 m away

print("A) single pole, sweep range (200 random bearings each)")
print(" range | detect% | mean err | 95% err | beams")
rows=[]
for rr in [0.3,0.5,0.75,1.0,1.25,1.5,1.75]:
    det=0; errs=[]; beams=[]
    for _ in range(200):
        th=rng.uniform(-np.pi,np.pi); cx,cy=rr*np.cos(th),rr*np.sin(th)
        s=noisy(raycast([(cx,cy,R)],WALLS),rng)
        d=detect_poles(s,ANGLE_MIN,ANGLE_INC)
        best=[q for q in d if math.hypot(q['x']-cx,q['y']-cy)<0.25]
        if best:
            det+=1; q=best[0]; errs.append(math.hypot(q['x']-cx,q['y']-cy)); beams.append(q['n'])
    e=np.array(errs) if errs else np.array([np.nan])
    rows.append((rr,det/2,np.mean(e),np.percentile(e,95),np.mean(beams) if beams else 0))
    print(f" {rr:4.2f}  | {det/2:6.1f} | {np.mean(e)*100:5.1f} cm | {np.percentile(e,95)*100:5.1f} cm | {np.mean(beams) if beams else 0:4.1f}")

print("\nB) office clutter: 2 poles + 2 chairs (2.5 cm legs) + 5 cm table leg + walls, 300 scans")
poles=[(0.9,0.25,R),(1.2,-0.45,R)]
chairs=[]
for (ox,oy) in [(1.9,0.9),(-1.2,1.4)]:
    for dx in (0,0.42):
        for dy in (0,0.42): chairs.append((ox+dx,oy+dy,0.0125))
table=[(2.1,-1.2,0.025)]
fp=0; miss=0; errs=[]
for _ in range(300):
    s=noisy(raycast(poles+chairs+table,WALLS),rng)
    d=detect_poles(s,ANGLE_MIN,ANGLE_INC)
    for (px,py,_r) in poles:
        m=[q for q in d if math.hypot(q['x']-px,q['y']-py)<0.25]
        if m: errs.append(math.hypot(m[0]['x']-px,m[0]['y']-py))
        else: miss+=1
    fp+=sum(1 for q in d if min(math.hypot(q['x']-px,q['y']-py) for px,py,_ in poles)>=0.25)
print(f" poles missed: {miss}/600   false detections: {fp} in 300 scans   mean err {np.mean(errs)*100:.1f} cm")

print("\nB2) same but table leg moved INSIDE 1.6 m (worst case, 5 cm leg at 1.3 m)")
fp=0
for _ in range(300):
    s=noisy(raycast(poles+[(0.2,1.3,0.025)],WALLS),rng)
    d=detect_poles(s,ANGLE_MIN,ANGLE_INC)
    fp+=sum(1 for q in d if min(math.hypot(q['x']-px,q['y']-py) for px,py,_ in poles)>=0.25)
print(f" false detections from the 5 cm leg: {fp}/300 scans")

print("\nC) averaging over scans (EMA alpha=0.3, 10 scans), pole at 1.0 and 1.5 m")
for rr in [1.0,1.5]:
    fin=[]
    for _ in range(100):
        th=rng.uniform(-np.pi,np.pi); cx,cy=rr*np.cos(th),rr*np.sin(th); est=None
        for k in range(10):
            d=detect_poles(noisy(raycast([(cx,cy,R)],WALLS),rng),ANGLE_MIN,ANGLE_INC)
            m=[q for q in d if math.hypot(q['x']-cx,q['y']-cy)<0.25]
            if not m: continue
            z=np.array([m[0]['x'],m[0]['y']]); est=z if est is None else 0.7*est+0.3*z
        if est is not None: fin.append(math.hypot(est[0]-cx,est[1]-cy))
    print(f" {rr} m: mean {np.mean(fin)*100:.1f} cm, 95% {np.percentile(fin,95)*100:.1f} cm")

print("\nD) systematic +5% range scale error (accuracy spec), pole at 1.0 m")
errs=[]
for _ in range(200):
    th=rng.uniform(-np.pi,np.pi); cx,cy=np.cos(th),np.sin(th)
    d=detect_poles(noisy(raycast([(cx,cy,R)],WALLS),rng,scale_bias=0.05),ANGLE_MIN,ANGLE_INC)
    m=[q for q in d if math.hypot(q['x']-cx,q['y']-cy)<0.25]
    if m: errs.append(math.hypot(m[0]['x']-cx,m[0]['y']-cy))
print(f" mean err {np.mean(errs)*100:.1f} cm  (bias, not noise: averaging won't remove it)")

