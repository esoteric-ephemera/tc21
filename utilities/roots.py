import numpy as np

def bracket(fun,bds,nstep=500,vector=False,args=(),kwargs={}):

    step = (bds[1]-bds[0])/nstep
    ivals = []
    if vector:
        tmpl = np.arange(bds[0],bds[1],step)
        funl = fun(tmpl,*args,**kwargs)
        ofun = funl[0]
        for istep in range(1,nstep):
            if ofun*funl[istep] <= 0:
                ivals.append([tmpl[istep-1],tmpl[istep]])
            ofun = funl[istep]
    else:
        tmp = bds[0]
        for istep in range(nstep):
            cfun = fun(tmp,*args,**kwargs)
            if istep == 0:
                ofun = cfun
            if ofun*cfun <= 0:
                ivals.append([tmp-step,tmp])
            ofun = cfun
            tmp += step
    return ivals

def bisect(fun,bds,tol=1.e-6,maxstep=100,args=(),kwargs={}):

    """
        Routine adapted from
        W.H. Press, S.A. Teukolsky, W.T. Vetterling, and B.P. Flannery,
        ``Numerical Recipes in Fortran 77:
        The Art of Scientific Computing'',
        2nd ed., Cambridge University Press 1992
        ISBN 0-521-43064-X
    """

    def wrap_fun(arg):
        return fun(arg,*args,**kwargs)
    tmp0 = wrap_fun(bds[0])
    tmp1 = wrap_fun(bds[1])
    if tmp0*tmp1 > 0:
        raise ValueError('No root in bracket')
    if tmp0 < 0.0:
        regs = [bds[0],bds[1]]
    else:
        regs = [bds[1],bds[0]]
    for iter in range(maxstep):
        mid = (regs[0] + regs[1])/2.0
        spc = mid - regs[0]
        tmpm = wrap_fun(mid)
        if tmpm < 0.0:
            regs[0] = mid
        else:
            regs[1] = mid
        if abs(tmpm)<tol or abs(spc)<1.e-14:#abs(spc) < tol*abs(mid) or
            break
    return mid,tmpm
