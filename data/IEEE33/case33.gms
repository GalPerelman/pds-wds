*We request that publications derived from the use of the codes provided in this online depository, explicitly acknowledge that fact by citing the papers, the code(s) and the appropriate software.

*Papers:
*[1] S. H. Dolatabadi, M. Ghorbanian, P. Siano, and N. D. Hatziargyriou “An Enhanced IEEE 33 Bus Benchmark Test System for Distribution System Studies,” IEEE Trans. Power Syst., 2020. 
*[2] M. Ghorbanian, S. H. Dolatabadi, and P. Siano, “Game theory-based energy-management method considering autonomous demand response and distributed generation interactions in smart distribution systems,” IEEE Syst. J., pp. 1–10, 2020. 
*[3] This GitHub code repository including DOI
*Software:
*[4] R. D. Zimmerman and C. E. Murillo-Sanchez, “Matpower [software],” 2018. [Online]. Available: https://matpower.org
*[5] G. D. Corporation, “General algebraic modeling system (gams) release 27.1.0,” Fairfax, VA, USA, 2019. [Online]. Available: https://www.gams.com/
*[6] “Digsilent powerfactory,” 2019. [Online]. Available: https://www.digsilent.de/
*[7] MathWorks, “Matlab/simulink 9.3,” 2019. [Online]. Available: https://www.mathworks.com
*[8] J. Y. Wong, “IEEE 33 Bus System,” 2020, MATLAB Central File Exchange. Retrieved September 15, 2020. [Online]. Available: https://www.mathworks.com/matlabcentral/fileexchange/73127-ieee-33-bus-system



set b 'buses' /b1*b33/;
set l 'lines' /l1*l35/;


parameter Vbase /12.66/; *kV
parameter Sbase /100/; *MVA



parameter Pd(b)  *MW
/
b1	0
b2	0.1
b3	0.09
b4	0.12
b5	0.06
b6	0.06
b7	0.2
b8	0.2
b9	0.06
b10	0.06
b11	0.045
b12	0.06
b13	0.06
b14	0.12
b15	0.06
b16	0.06
b17	0.06
b18	0.09
b19	0.09
b20	0.09
b21	0.09
b22	0.09
b23	0.09
b24	0.42
b25	0.42
b26	0.06
b27	0.06
b28	0.06
b29	0.12
b30	0.2
b31	0.15
b32	0.21
b33	0.06
/;

parameter Qd(b)  *MVAR
/
b1	0
b2	0.06
b3	0.04
b4	0.08
b5	0.03
b6	0.02
b7	0.1
b8	0.1
b9	0.02
b10	0.02
b11	0.03
b12	0.035
b13	0.035
b14	0.08
b15	0.01
b16	0.02
b17	0.02
b18	0.04
b19	0.04
b20	0.04
b21	0.04
b22	0.04
b23	0.05
b24	0.2
b25	0.2
b26	0.025
b27	0.025
b28	0.02
b29	0.07
b30	0.6
b31	0.07
b32	0.1
b33	0.04
/;

parameter Gs(b)
/
b1	0
b2	0
b3	0
b4	0
b5	0
b6	0
b7	0
b8	0
b9	0
b10	0
b11	0
b12	0
b13	0
b14	0
b15	0
b16	0
b17	0
b18	0
b19	0
b20	0
b21	0
b22	0
b23	0
b24	0
b25	0
b26	0
b27	0
b28	0
b29	0
b30	0
b31	0
b32	0
b33	0
/;

parameter Bs(b)  *MVAR
/
b1	0
b2	0
b3	0
b4	0
b5	0
b6	0
b7	0
b8	0
b9	0
b10	0
b11	0
b12	0
b13	0
b14	0
b15	0
b16	0
b17	0
b18	0   *Bs: 0 MVAr for meshed and 0.4 MVAr (capacitive) for radial
b19	0
b20	0
b21	0
b22	0
b23	0
b24	0
b25	0
b26	0
b27	0
b28	0
b29	0
b30	0
b31	0
b32	0
b33	0   *Bs: 0 MVAr for meshed and 0.6 MVAr (capacitive) for radial
/;

parameter Vmax(b)  *p.u.
/
b1	1          *1.05
b2	1.05
b3	1.05
b4	1.05
b5	1.05
b6	1.05
b7	1.05
b8	1.05
b9	1.05
b10	1.05
b11	1.05
b12	1.05
b13	1.05
b14	1.05
b15	1.05
b16	1.05
b17	1.05
b18	1.05
b19	1.05
b20	1.05
b21	1.05
b22	1.05
b23	1.05
b24	1.05
b25	1.05
b26	1.05
b27	1.05
b28	1.05
b29	1.05
b30	1.05
b31	1.05
b32	1.05
b33	1.05
/;

parameter Vmin(b)  *p.u.
/
b1	1          *0.95
b2	0.95
b3	0.95
b4	0.95
b5	0.95
b6	0.95
b7	0.95
b8	0.95
b9	0.95
b10	0.95
b11	0.95
b12	0.95
b13	0.95
b14	0.95
b15	0.95
b16	0.95
b17	0.95
b18	0.95
b19	0.95
b20	0.95
b21	0.95
b22	0.95
b23	0.95
b24	0.95
b25	0.95
b26	0.95
b27	0.95
b28	0.95
b29	0.95
b30	0.95
b31	0.95
b32	0.95
b33	0.95
/;

parameter angmin(b) *degree
/
b1	0          *-5
b2	-360
b3	-360
b4	-360
b5	-360
b6	-360
b7	-360
b8	-360
b9	-360
b10	-360
b11	-360
b12	-360
b13	-360
b14	-360
b15	-360
b16	-360
b17	-360
b18	-360
b19	-360
b20	-360
b21	-360
b22	-360
b23	-360
b24	-360
b25	-360
b26	-360
b27	-360
b28	-360
b29	-360
b30	-360
b31	-360
b32	-360
b33	-360
/;

parameter angmax(b) *degree
/
b1	0          *5
b2	360
b3	360
b4	360
b5	360
b6	360
b7	360
b8	360
b9	360
b10	360
b11	360
b12	360
b13	360
b14	360
b15	360
b16	360
b17	360
b18	360
b19	360
b20	360
b21	360
b22	360
b23	360
b24	360
b25	360
b26	360
b27	360
b28	360
b29	360
b30	360
b31	360
b32	360
b33	360
/;


parameter Pgenmax(b)  *MW
/
b1	4
b2	0
b3	0
b4	0
b5	0
b6	0
b7	0
b8	0
b9	0
b10	0
b11	0
b12	0
b13	0
b14	0
b15	0
b16	0
b17	0
b18	0.2 *Very High Renewable Generation: Pgenmax 0.8
b19	0
b20	0
b21	0
b22	0.2 *Very High Renewable Generation: Pgenmax 0.8
b23	0
b24	0
b25	0.2 *Very High Renewable Generation: Pgenmax 0.8
b26	0
b27	0
b28	0
b29	0
b30	0
b31	0
b32	0
b33	0.2 *Very High Renewable Generation: Pgenmax 0.8
/;

parameter Pgenmin(b)  *MW
/
b1	0
b2	0
b3	0
b4	0
b5	0
b6	0
b7	0
b8	0
b9	0
b10	0
b11	0
b12	0
b13	0
b14	0
b15	0
b16	0
b17	0
b18	0
b19	0
b20	0
b21	0
b22	0
b23	0
b24	0
b25	0
b26	0
b27	0
b28	0
b29	0
b30	0
b31	0
b32	0
b33	0
/;

parameter Qgenmax(b)  *MVAR
/
b1	2.5
b2	0
b3	0
b4	0
b5	0
b6	0
b7	0
b8	0
b9	0
b10	0
b11	0
b12	0
b13	0
b14	0
b15	0
b16	0
b17	0
b18	0
b19	0
b20	0
b21	0
b22	0
b23	0
b24	0
b25	0
b26	0
b27	0
b28	0
b29	0
b30	0
b31	0
b32	0
b33	0
/;

parameter Qgenmin(b)  *MVAR
/
b1	-2.5
b2	0
b3	0
b4	0
b5	0
b6	0
b7	0
b8	0
b9	0
b10	0
b11	0
b12	0
b13	0
b14	0
b15	0
b16	0
b17	0
b18	0
b19	0
b20	0
b21	0
b22	0
b23	0
b24	0
b25	0
b26	0
b27	0
b28	0
b29	0
b30	0
b31	0
b32	0
b33	0
/;

parameter Genstatus(b)
/
b1	1
b2	0
b3	0
b4	0
b5	0
b6	0
b7	0
b8	0
b9	0
b10	0
b11	0
b12	0
b13	0
b14	0
b15	0
b16	0
b17	0
b18	1
b19	0
b20	0
b21	0
b22	1
b23	0
b24	0
b25	1
b26	0
b27	0
b28	0
b29	0
b30	0
b31	0
b32	0
b33	1
/;

parameter costcoeffA *($/MW^2h)
/
b1	0.003
b2	0
b3	0
b4	0
b5	0
b6	0
b7	0
b8	0
b9	0
b10	0
b11	0
b12	0
b13	0
b14	0
b15	0
b16	0
b17	0
b18	0.0026
b19	0
b20	0
b21	0
b22	0.0026
b23	0
b24	0
b25	0.0026
b26	0
b27	0
b28	0
b29	0
b30	0
b31	0
b32	0
b33	0.0026
/;

parameter costcoeffB *($/MWh)
/
b1	12
b2	0
b3	0
b4	0
b5	0
b6	0
b7	0
b8	0
b9	0
b10	0
b11	0
b12	0
b13	0
b14	0
b15	0
b16	0
b17	0
b18	10.26
b19	0
b20	0
b21	0
b22	10.26
b23	0
b24	0
b25	10.26
b26	0
b27	0
b28	0
b29	0
b30	0
b31	0
b32	0
b33	10.26
/;

parameter costcoeffC *($/h)
/
b1	240
b2	0
b3	0
b4	0
b5	0
b6	0
b7	0
b8	0
b9	0
b10	0
b11	0
b12	0
b13	0
b14	0
b15	0
b16	0
b17	0
b18	210
b19	0
b20	0
b21	0
b22	210
b23	0
b24	0
b25	210
b26	0
b27	0
b28	0
b29	0
b30	0
b31	0
b32	0
b33	210
/;

parameter r(l) *Ohm
/
l1	0.0922
l2	0.493
l3	0.366
l4	0.3811
l5	0.819
l6	0.1872
l7	0.7114
l8	1.03
l9	1.044
l10	0.1966
l11	0.3744
l12	1.468
l13	0.5416
l14	0.591
l15	0.7463
l16	1.289
l17	0.732
l18	0.164
l19	1.5042
l20	0.4095
l21	0.7089
l22	0.4512
l23	0.898
l24	0.896
l25	0.203
l26	0.2842
l27	1.059
l28	0.8042
l29	0.5075
l30	0.9744
l31	0.3105
l32	0.341
l33	2
l34	2
l35	0.5
/;

parameter x(l) *Ohm
/
l1	0.047
l2	0.2511
l3	0.1864
l4	0.1941
l5	0.707
l6	0.6188
l7	0.2351
l8	0.74
l9	0.74
l10	0.065
l11	0.1238
l12	1.155
l13	0.7129
l14	0.526
l15	0.545
l16	1.721
l17	0.574
l18	0.1565
l19	1.3554
l20	0.4784
l21	0.9373
l22	0.3083
l23	0.7091
l24	0.7011
l25	0.1034
l26	0.1447
l27	0.9337
l28	0.7006
l29	0.2585
l30	0.963
l31	0.3619
l32	0.5302
l33	2
l34	2
l35	0.5
/;

parameter Linestatus(l)
/
l1	1
l2	1
l3	1
l4	1
l5	1
l6	1
l7	1
l8	1
l9	1
l10	1
l11	1
l12	1
l13	1
l14	1
l15	1
l16	1
l17	1
l18	1
l19	1
l20	1
l21	1
l22	1
l23	1
l24	1
l25	1
l26	1
l27	1
l28	1
l29	1
l30	1
l31	1
l32	1
l33	0  *Linestatus: 0 for radial and 1 for meshed
l34	0  *Linestatus: 0 for radial and 1 for meshed
l35	0  *Linestatus: 0 for radial and 1 for meshed
/;


table Alinetobus(l,b) *lines to buses incidence matrix
	b1	b2	b3	b4	b5	b6	b7	b8	b9	b10	b11	b12	b13	b14	b15	b16	b17	b18	b19	b20	b21	b22	b23	b24	b25	b26	b27	b28	b29	b30	b31	b32	b33
l1	1	-1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
l2	0	1	-1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
l3	0	0	1	-1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
l4	0	0	0	1	-1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
l5	0	0	0	0	1	-1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
l6	0	0	0	0	0	1	-1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
l7	0	0	0	0	0	0	1	-1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
l8	0	0	0	0	0	0	0	1	-1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
l9	0	0	0	0	0	0	0	0	1	-1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
l10	0	0	0	0	0	0	0	0	0	1	-1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
l11	0	0	0	0	0	0	0	0	0	0	1	-1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
l12	0	0	0	0	0	0	0	0	0	0	0	1	-1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
l13	0	0	0	0	0	0	0	0	0	0	0	0	1	-1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
l14	0	0	0	0	0	0	0	0	0	0	0	0	0	1	-1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
l15	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	-1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
l16	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	-1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
l17	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	-1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
l18	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	-1	0	0	0	0	0	0	0	0	0	0	0	0	0	0
l19	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	-1	0	0	0	0	0	0	0	0	0	0	0	0	0
l20	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	-1	0	0	0	0	0	0	0	0	0	0	0	0
l21	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	-1	0	0	0	0	0	0	0	0	0	0	0
l22	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	-1	0	0	0	0	0	0	0	0	0	0
l23	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	-1	0	0	0	0	0	0	0	0	0
l24	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	-1	0	0	0	0	0	0	0	0
l25	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	-1	0	0	0	0	0	0	0
l26	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	-1	0	0	0	0	0	0
l27	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	-1	0	0	0	0	0
l28	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	-1	0	0	0	0
l29	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	-1	0	0	0
l30	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	-1	0	0
l31	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	-1	0
l32	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	-1
l33	0	0	0	0	0	0	0	-1	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0
l34	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	-1	0	0	0	0	0	0	0	0	0	0	0
l35	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	-1	0	0	0	0
;