import numpy as np

source_dir = './eps_c_data_corr/'

fls = {'ALDA': 'ALDA', 'RPA': 'RPA', 'MCP07 (static)': 'MCP07_static','MCP07 (dynamic)': 'MCP07'}
fnls = list(fls.keys())
eps_c = {}

for fl in fls:
    eps_c[fl] = {}
    targ_fl = source_dir+'epsilon_C_'+fls[fl]+'.csv'
    dat = np.genfromtxt(targ_fl,delimiter=',',skip_header=1)
    if fl == 'RPA':
        rs_l = dat[:,0]
        ref = dat[:,2]
    eps_c[fl]['epsc'] = dat[:,1]
    eps_c[fl]['pe'] = dat[:,-1]

with open(source_dir+'eps_c_comparison.tex','w+') as ofl:
    hstr = ['$r_s$','PZ81']
    for fnl in fnls:
        hstr.append(fnl)
        hstr.append(fnl + ' Percent error')
    ofl.write(('{:} & '*(len(hstr)-1) + '{:} \\\\ \hline \n').format(*hstr))
    for irs,rs in enumerate(rs_l):
        tstr = [rs,ref[irs]]
        for fnl in fnls:
            if irs > len(eps_c[fnl]['pe'])-1:
                tstr.append(0.0)
                tstr.append(0.0)
            else:
                tstr.append(eps_c[fnl]['epsc'][irs])
                tstr.append(eps_c[fnl]['pe'][irs])
        ofl.write(('{:} & {:.4f} & ' + '{:.4f} & {:.2f} & '*(len(fnls)-1) + '{:.4f} & {:.2f} \\\\ \n').format(*tstr))
