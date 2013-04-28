import numpy as np
import pylab as pl
import time

def plot_exact_solution():
    # Advection velocity:
    a=1

    x=np.linspace(-1,1,1000)
    pl.clf()
    pl.hold(False)
    for t in np.linspace(0,1,100):
        pl.plot(x,eta(x-a*t))
        pl.draw()
        time.sleep(0.0001)


def eta(x,IC='gaussian'):
    # Initial condition:
    x=np.remainder(x,1.)
    if IC=='gaussian':
        return np.exp(-100*(x-0.5)**2)
    elif IC=='square':
        return 1.0*(x>0.05)*(x<0.25)


def FD_solution(method='Lax-Friedrichs',a=1.,cflnum=0.9,m=50,IC='gaussian'):
    method=method.lower()

    x=np.linspace(0,1,m+1);
    u=eta(x,IC)

    h=x[1]-x[0]
    k=cflnum*h/a

    if method=='leapfrog':
        # Take one step with Euler:
        uold=u.copy()
        u=eta(x-a*k,IC); u[0]=eta(1.-a*k,IC)
        u[1:-1]=u[1:-1]-0.5*cflnum*(u[2:]-u[:-2])
        u[-1]=u[-1]-0.5*cflnum*(u[0]-u[-2])
        u[0] =u[0] -0.5*cflnum*(u[1]-u[-1])
        unew=np.zeros(np.size(u))

    pl.clf()
    pl.hold(False)
    T=2.
    N=int(round(T/k))

    for n in xrange(N+1):
        t=n*k
        pl.plot(x,u,'-o',x,eta(x-t*a,IC),'-k')
        pl.ylim([-0.1,1.])
        pl.legend(['computed','exact'],loc=2)
        pl.draw()
        time.sleep(0.0001)
        if method=='centered':
            u[1:-1]=u[1:-1]-0.5*cflnum*(u[2:]-u[:-2])
            u[0] = u[0] -0.5*cflnum*(u[1]-u[-1])
            u[-1]= u[-1]-0.5*cflnum*(u[0]-u[-2])
        elif method=='leapfrog':
            unew[1:-1]=uold[1:-1]-cflnum*(u[2:]-u[:-2])
            unew[0]=uold[0]-cflnum*(u[1]-u[-1])
            unew[-1]=uold[-1]-cflnum*(u[0]-u[-2])
            uold=u.copy()
            u=unew.copy()
        elif method=='lax-friedrichs':
            u[1:-1]=0.5*(u[2:]+u[:-2])-0.5*cflnum*(u[2:]-u[:-2])
            u[0] =  0.5*(u[1] +u[-1]) -0.5*cflnum*(u[1]-u[-1])
            u[-1]=  0.5*(u[0] +u[-2]) -0.5*cflnum*(u[0]-u[-2])
        elif method=='lax-wendroff':
            u[1:-1]=u[1:-1]-0.5*cflnum*(u[2:]-u[:-2])+0.5*cflnum**2 * (u[2:]-2*u[1:-1]+u[:-2])
            u[0]=   u[0]   -0.5*cflnum*(u[1]-u[-1])+0.5*cflnum**2 * (u[1]-2*u[0]   +u[-1])
            u[-1]=  u[-1]  -0.5*cflnum*(u[0]-u[-2])+0.5*cflnum**2 * (u[0]-2*u[-1]   +u[-2])
        elif method=='beam-warming':
            u[2:]=u[2:]-0.5*cflnum*(3*u[2:]-4*u[1:-1]+u[:-2])+0.5*cflnum**2 * (u[2:]-2*u[1:-1]+u[:-2])
            #u[0]=u[0]-0.5*cflnum*(3*u[0]-4*u[-1]+u[-2])+0.5*cflnum**2 * (u[0]-2*u[-1]+u[:-2])
        elif method=='upwind':
            u[1:]=u[1:]-cflnum*(u[1:]-u[:-1])
            u[0]= u[0] -cflnum*(u[0] -u[-1])
        else: 
            raise Exception('unrecognized method')

    t=(N+1)*k
    #pl.plot(x,u)
    pl.plot(x,u,'-o',x,eta(x-t*a,IC),'-k')
    pl.ylim([-0.1,1.])
    pl.legend(['computed','exact'],loc=2)
    pl.draw()
