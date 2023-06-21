%We request that publications derived from the use of the codes provided in this online depository, explicitly acknowledge that fact by citing the papers, the code(s) and the appropriate software.

%Papers:
%[1] S. H. Dolatabadi, M. Ghorbanian, P. Siano, and N. D. Hatziargyriou “An Enhanced IEEE 33 Bus Benchmark Test System for Distribution System Studies,” IEEE Trans. Power Syst., 2020. 
%[2] M. Ghorbanian, S. H. Dolatabadi, and P. Siano, “Game theory-based energy-management method considering autonomous demand response and distributed generation interactions in smart distribution systems,” IEEE Syst. J., pp. 1–10, 2020. 
%[3] This GitHub code repository including DOI
%Software:
%[4] R. D. Zimmerman and C. E. Murillo-Sanchez, “Matpower [software],” 2018. [Online]. Available: https://matpower.org
%[5] G. D. Corporation, “General algebraic modeling system (gams) release 27.1.0,” Fairfax, VA, USA, 2019. [Online]. Available: https://www.gams.com/
%[6] “Digsilent powerfactory,” 2019. [Online]. Available: https://www.digsilent.de/
%[7] MathWorks, “Matlab/simulink 9.3,” 2019. [Online]. Available: https://www.mathworks.com
%[8] J. Y. Wong, “IEEE 33 Bus System,” 2020, MATLAB Central File Exchange. Retrieved September 15, 2020. [Online]. Available: https://www.mathworks.com/matlabcentral/fileexchange/73127-ieee-33-bus-system




function mpc = case33																					
%CASE33  Power flow data for IEEE 33 bus distribution system																					
%    Please see CASEFORMAT for details on the case file format.																					
																					
																					
%% MATPOWER Case Format : Version 2																					
mpc.version = '2';																					
																					
%%-----  Power Flow Data  -----%%																					
%% system MVA base																					
mpc.baseMVA = 100;																					
																					
%% bus data																					
%	bus_i	type	Pd	Qd	Gs	Bs	area	Vm	Va	baseKV	zone	Vmax	Vmin								
mpc.bus = [  %% (Pd and Qd are specified in kW & kVAr here, converted to MW & MVAr below)																					
	1	3	0	0	0	0	1	1	0	12.66	1	1	1;								
	2	1	100	60	0	0	1	1	0	12.66	1	1.05	0.95;								
	3	1	90	40	0	0	1	1	0	12.66	1	1.05	0.95;								
	4	1	120	80	0	0	1	1	0	12.66	1	1.05	0.95;								
	5	1	60	30	0	0	1	1	0	12.66	1	1.05	0.95;								
	6	1	60	20	0	0	1	1	0	12.66	1	1.05	0.95;								
	7	1	200	100	0	0	1	1	0	12.66	1	1.05	0.95;								
	8	1	200	100	0	0	1	1	0	12.66	1	1.05	0.95;								
	9	1	60	20	0	0	1	1	0	12.66	1	1.05	0.95;								
	10	1	60	20	0	0	1	1	0	12.66	1	1.05	0.95;								
	11	1	45	30	0	0	1	1	0	12.66	1	1.05	0.95;								
	12	1	60	35	0	0	1	1	0	12.66	1	1.05	0.95;								
	13	1	60	35	0	0	1	1	0	12.66	1	1.05	0.95;								
	14	1	120	80	0	0	1	1	0	12.66	1	1.05	0.95;								
	15	1	60	10	0	0	1	1	0	12.66	1	1.05	0.95;								
	16	1	60	20	0	0	1	1	0	12.66	1	1.05	0.95;								
	17	1	60	20	0	0	1	1	0	12.66	1	1.05	0.95;								
	18	2	90	40	0	0	1	1	0	12.66	1	1.05	0.95;	%Bs: 0 MVAr for meshed and 0.4 MVAr (capacitive) for radial	%PV or PQ							
	19	1	90	40	0	0	1	1	0	12.66	1	1.05	0.95;								
	20	1	90	40	0	0	1	1	0	12.66	1	1.05	0.95;								
	21	1	90	40	0	0	1	1	0	12.66	1	1.05	0.95;								
	22	1	90	40	0	0	1	1	0	12.66	1	1.05	0.95;	%PV or PQ							
	23	1	90	50	0	0	1	1	0	12.66	1	1.05	0.95;								
	24	1	420	200	0	0	1	1	0	12.66	1	1.05	0.95;								
	25	1	420	200	0	0	1	1	0	12.66	1	1.05	0.95;	%PV or PQ							
	26	1	60	25	0	0	1	1	0	12.66	1	1.05	0.95;								
	27	1	60	25	0	0	1	1	0	12.66	1	1.05	0.95;								
	28	1	60	20	0	0	1	1	0	12.66	1	1.05	0.95;								
	29	1	120	70	0	0	1	1	0	12.66	1	1.05	0.95;								
	30	1	200	600	0	0	1	1	0	12.66	1	1.05	0.95;								
	31	1	150	70	0	0	1	1	0	12.66	1	1.05	0.95;								
	32	1	210	100	0	0	1	1	0	12.66	1	1.05	0.95;								
	33	2	60	40	0	0	1	1	0	12.66	1	1.05	0.95;	%Bs: 0 MVAr for meshed and 0.6 MVAr (capacitive) for radial	%PV or PQ						
];																					
																					
																					
%% generator data																					
%	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin	Pc1	Pc2	Qc1min	Qc1max	Qc2min	Qc2max	ramp_agc	ramp_10	ramp_30	ramp_q	apf
mpc.gen = [																					
	1	0	0	2.5	-2.5	1	100	1	4	0	0	0	0	0	0	0	0	0	0	0	0;
	18	0.2	0	0	0	1	100	1	0.2	0	0	0	0	0	0	0	0	0	0	0	0;%Very High Renewable Generation: Pg 0.8,  Pmax 0.8
	22	0.2	0	0	0	1	100	1	0.2	0	0	0	0	0	0	0	0	0	0	0	0;%Very High Renewable Generation: Pg 0.8,  Pmax 0.8
	25	0.2	0	0	0	1	100	1	0.2	0	0	0	0	0	0	0	0	0	0	0	0;%Very High Renewable Generation: Pg 0.8,  Pmax 0.8
	33	0.2	0	0	0	1	100	1	0.2	0	0	0	0	0	0	0	0	0	0	0	0;%Very High Renewable Generation: Pg 0.8,  Pmax 0.8
    ];									%MW												
																					
%% branch data																					
%	fbus	tbus	r	x	b	rateA	rateB	rateC	ratio	angle	status	angmin	angmax								
mpc.branch = [  %% (r and x specified in ohms here, converted to p.u. below)																					
	1	2	0.0922	0.047	0	0	0	0	0	0	1	-360	360;	%ratio -0.05 to 0.05	angle -5 to 5 degrees		
	2	3	0.493	0.2511	0	0	0	0	0	0	1	-360	360;								
	3	4	0.366	0.1864	0	0	0	0	0	0	1	-360	360;								
	4	5	0.3811	0.1941	0	0	0	0	0	0	1	-360	360;								
	5	6	0.819	0.707	0	0	0	0	0	0	1	-360	360;								
	6	7	0.1872	0.6188	0	0	0	0	0	0	1	-360	360;								
	7	8	0.7114	0.2351	0	0	0	0	0	0	1	-360	360;								
	8	9	1.03	0.74	0	0	0	0	0	0	1	-360	360;								
	9	10	1.044	0.74	0	0	0	0	0	0	1	-360	360;								
	10	11	0.1966	0.065	0	0	0	0	0	0	1	-360	360;								
	11	12	0.3744	0.1238	0	0	0	0	0	0	1	-360	360;								
	12	13	1.468	1.155	0	0	0	0	0	0	1	-360	360;								
	13	14	0.5416	0.7129	0	0	0	0	0	0	1	-360	360;								
	14	15	0.591	0.526	0	0	0	0	0	0	1	-360	360;								
	15	16	0.7463	0.545	0	0	0	0	0	0	1	-360	360;								
	16	17	1.289	1.721	0	0	0	0	0	0	1	-360	360;								
	17	18	0.732	0.574	0	0	0	0	0	0	1	-360	360;								
	2	19	0.164	0.1565	0	0	0	0	0	0	1	-360	360;								
	19	20	1.5042	1.3554	0	0	0	0	0	0	1	-360	360;								
	20	21	0.4095	0.4784	0	0	0	0	0	0	1	-360	360;								
	21	22	0.7089	0.9373	0	0	0	0	0	0	1	-360	360;								
	3	23	0.4512	0.3083	0	0	0	0	0	0	1	-360	360;								
	23	24	0.898	0.7091	0	0	0	0	0	0	1	-360	360;								
	24	25	0.896	0.7011	0	0	0	0	0	0	1	-360	360;								
	6	26	0.203	0.1034	0	0	0	0	0	0	1	-360	360;								
	26	27	0.2842	0.1447	0	0	0	0	0	0	1	-360	360;								
	27	28	1.059	0.9337	0	0	0	0	0	0	1	-360	360;								
	28	29	0.8042	0.7006	0	0	0	0	0	0	1	-360	360;								
	29	30	0.5075	0.2585	0	0	0	0	0	0	1	-360	360;								
	30	31	0.9744	0.963	0	0	0	0	0	0	1	-360	360;								
	31	32	0.3105	0.3619	0	0	0	0	0	0	1	-360	360;								
	32	33	0.341	0.5302	0	0	0	0	0	0	1	-360	360;								
	21	8	2	2	0	0	0	0	0	0	0	-360	360;	%status: 0 for radial and 1 for meshed								
	12	22	2	2	0	0	0	0	0	0	0	-360	360;	%status: 0 for radial and 1 for meshed								
	25	29	0.5	0.5	0	0	0	0	0	0	0	-360	360;	%status: 0 for radial and 1 for meshed								
];																					
																					
%%-----  OPF Data  -----%%																					
%% generator cost data																					
%	1	startup	shutdown	n	x1	y1	...	xn	yn												
%	2	startup	shutdown	n	c(n-1)	...	c0														
mpc.gencost = [																					
	2	0	0	3	0.003	12	240;														
	2	0	0	3	0.0026	10.26	210;														
	2	0	0	3	0.0026	10.26	210;														
	2	0	0	3	0.0026	10.26	210;														
	2	0	0	3	0.0026	10.26	210;														
];																					
																					
																					
%% convert branch impedances from Ohms to p.u.																					
[PQ, PV, REF, NONE, BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, ...																					
    VA, BASE_KV, ZONE, VMAX, VMIN, LAM_P, LAM_Q, MU_VMAX, MU_VMIN] = idx_bus;																					
[F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, ...																					
    TAP, SHIFT, BR_STATUS, PF, QF, PT, QT, MU_SF, MU_ST, ...																					
    ANGMIN, ANGMAX, MU_ANGMIN, MU_ANGMAX] = idx_brch;																					
Vbase = mpc.bus(1, BASE_KV) * 1e3;      %% in Volts																					
Sbase = mpc.baseMVA * 1e6;              %% in VA																					
mpc.branch(:, [BR_R BR_X]) = mpc.branch(:, [BR_R BR_X]) / (Vbase^2 / Sbase);																					
																					
%% convert loads from kW to MW																					
mpc.bus(:, [PD, QD]) = mpc.bus(:, [PD, QD]) / 1e3;																					
