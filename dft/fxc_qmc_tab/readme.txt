This file explains the content of the fxc files in fxc.tgz
The filenames are fxc_XXX_corr_wp.txt, where XXX=int(100*rs), encodes the density.

The first few lines, marked with # ..., are comment lines, where some information on the system are given, e.g. Fermi energy, ALDA fxc, ...

The actual data start from the 7th line and are separated in blocks. The blocks are separated by empty lines. The first block is for q=0.1kF, the second for q=0.2kF and so forth up to qmax=6.4kF.

In one block the energy dependence is resolved by \Delta \omega=0.02 \omega_p, where \omega_p is the plasma frequency. The energy range starts from zero up to the maximal \omega, which is 3.98 \omega_p, i.e. 200 points.

The coloums contain the data:
1.	q/kF
2.	\omega/\omega_p
3.	Im[\chi] where \chi is the density density response of the HEG
4.	Re[\chi]
5.	Test m0 sumrule
6.	Test f sumrule
7.	Im[\eps] i.e. the imaginary part of the dielectric function
8.	Re[\eps]
9.	Im[f_xc]
10.	Re[f_xc]

License information:

This 2p2h-kernel is made available under the Open Database License: http://opendatacommons.org/licenses/odbl/1.0/.
Any rights in individual contents of the database are licensed under the Database Contents License: http://opendatacommons.org/licenses/dbcl/1.0/
For any public use of the database, or works produced from the database please cite the article: M. Panholzer, M. Gatti and L. Reining, Nonlocal and Nonadiabatic Effects in the Charge-Density Response of Solids: A Time-Dependent Density-Functional Approach. Phys. Rev. Lett. 120, 166402 (2018)
