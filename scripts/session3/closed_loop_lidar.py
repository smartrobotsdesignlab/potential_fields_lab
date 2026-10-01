import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'potential_fields_lab'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, math
from pole_detection import detect_poles, PoleTracker
from synth_lds import raycast, noisy, ANGLE_MIN, ANGLE_INC
rng=np.random.default_rng(3)
def wrap(a): return math.atan2(math.sin(a),math.cos(a))
def field(p,g,obs,ka=1.0,kr=0.5,d0=0.7):
    d=g-p;n=np.linalg.norm(d); F=(d/n if n>1 else d)*ka
    for o in obs:
        v=p-np.array(o); r=np.linalg.norm(v)
        if 1e-6<r<d0: F=F+kr*(1/r-1/d0)/r**2*(v/r)
    return F
def run(true_poles, mode, goal=(2.0,0.0), T=60, walls=(-2.6,2.8,-2.4,2.5)):
    g=np.array(goal); x=y=th=0.0; wp=0.0; ts=0; stuck=0
    trk=PoleTracker(); obs=[] if mode=='lidar' else list(true_poles); path=[(0,0)]; det_err=[]
    t=0.0; dt=0.02
    while t<T:
        k=int(round(t/dt))
        if mode=='lidar' and k%10==0:                       # 5 Hz scan
            circ=[]
            for (px,py) in true_poles:                      # world -> laser frame
                dx,dy=px-x,py-y; c,s=math.cos(th),math.sin(th)
                circ.append((c*dx+s*dy,-s*dx+c*dy,0.05))
            wl=None
            sc=noisy(raycast(circ,None),rng)
            # walls in world: approximate by adding them via a rotated raycast is overkill here;
            # clutter rejection was tested separately in step 1
            pts=[]
            for d in detect_poles(sc,ANGLE_MIN,ANGLE_INC):
                c,s=math.cos(th),math.sin(th)
                pts.append((x+c*d['x']-s*d['y'], y+s*d['x']+c*d['y']))
            obs=trk.update(pts,t)
            for o in obs: det_err.append(min(math.hypot(o[0]-a,o[1]-b) for a,b in true_poles))
        if k%5==0:                                           # 10 Hz control
            p=np.array([x,y])
            if np.linalg.norm(g-p)<0.15: return np.array(path),'GOAL',t,det_err
            F=field(p,g,obs); fm=np.linalg.norm(F)
            if fm<0.05:
                stuck+=1
                if stuck>=15: return np.array(path),'STUCK',t,det_err
                v=w=0.0
            else:
                stuck=0; e=wrap(math.atan2(F[1],F[0])-th)
                if abs(e)<0.05: w=0.0; ts=0
                elif abs(e)>2.8:
                    ts=ts or (1 if e>0 else -1); w=ts*1.0
                else: ts=0; w=max(-1,min(1,1.5*e))
                w=wp+max(-0.25,min(0.25,w-wp)); wp=w; v=0.15*max(0,math.cos(e))
        x+=v*math.cos(th)*dt; y+=v*math.sin(th)*dt; th=wrap(th+w*dt); t+=dt
        if k%5==0: path.append((x,y))
    return np.array(path),'TIMEOUT',t,det_err
res={}
for name,poles in [('single',[(1.0,0.25)]),('gap',[(1.0,0.3),(1.0,-0.3)])]:
    for mode in ['yaml','lidar']:
        P,out,t,de=run(poles,mode); res[(name,mode)]=(P,out)
        extra=f"  tracked-pole error mean {np.mean(de)*100:.1f} cm, max {np.max(de)*100:.1f} cm" if de else ""
        print(f"{name:7s} {mode:5s}: {out:6s} t={t:5.1f}s end=({P[-1][0]:.2f},{P[-1][1]:.2f}){extra}")
    a,b=res[(name,'yaml')][0],res[(name,'lidar')][0]
    m=min(len(a),len(b)); print(f"         max path difference yaml vs lidar: {np.max(np.hypot(*(a[:m]-b[:m]).T))*100:.1f} cm")

