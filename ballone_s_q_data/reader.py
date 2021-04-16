import numpy as np
from glob import glob
from os import system,path

def convert_and_write():

    dirs = ['./GofR/','./SofK/']

    gr_l = glob(dirs[0]+'gofr*')
    sk_l = glob(dirs[1]+'Sk*')

    if not path.isdir('./data_files/'):
        system('mkdir ./data_files')

    hdr = 'r (bohr), g(r)'
    for gr in gr_l:
        nelec =  gr.split('.')[-2].split('N')[-1]
        rs_str = gr.split('.')[-1].split('rs')[-1]
        rs = int(rs_str)
        if not path.isdir('./data_files/rs_'+rs_str):
            system('mkdir ./data_files/rs_'+rs_str)
        r_sc,grfn = np.transpose(np.genfromtxt(gr,delimiter=''))
        r = r_sc*rs
        np.savetxt('./data_files/rs_'+rs_str+'/gr_N-'+nelec+'_rs-'+rs_str+'.csv',np.transpose((r,grfn)),delimiter=',',header=hdr,fmt='%.6f,%.6f')

    hdr = 'k (1/bohr), S(k)'
    for sk in sk_l:
        nelec =  sk.split('.')[-2].split('N')[-1]
        rs_str = sk.split('.')[-1].split('rs')[-1]
        rs = int(rs_str)
        if not path.isdir('./data_files/rs_'+rs_str):
            system('mkdir ./data_files/rs_'+rs_str)
        k_sc,skfn = np.transpose(np.genfromtxt(sk,delimiter=''))
        k = k_sc/rs
        np.savetxt('./data_files/rs_'+rs_str+'/sk_N-'+nelec+'_rs-'+rs_str+'.csv',np.transpose((k,skfn)),delimiter=',',header=hdr,fmt='%.6f,%.6f')
    return

if __name__ == "__main__":

    convert_and_write()
