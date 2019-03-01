'''
Assorted ground motion prediction equation tools
Diego Melgar, May, 2016
'''


def PGAr_calc(M, Rjb, U, SS, RS, NS,italy=False):
    '''
    Calculate reference PGA
    '''
    
    from numpy import log,exp,ones,where
    
    # GMPE coefficients from the PEER spreadsheet: 
    # http://peer.berkeley.edu/ngawest2/wp-content/uploads/2016/02/NGAW2_GMPE_Spreadsheets_v5.7_041415_Protected.zip
    # in the "BSSA14_Coeffs sheet, T(s)=0 corresponds to PGA, T(s)=-1 is PGV
    coefficients=[0.4473,0.4856,0.2459,0.4539,1.431,0.05053,-0.1662,5.5,-1.13400,0.19170,-0.00809,4.5,
                        1.,4.5,0.000000,0.002860,-0.002550,-0.6000,1500.00,760,0.,0.1,-0.1500,-0.00701,-9.900,
                        -9.900,110.000,270.000,0.100,0.070,225.,300.,0.6950,0.4950,0.3980,0.3480]
    
    #Assign each coefficient
    e0 = coefficients[0]
    e1 = coefficients[1]
    e2 = coefficients[2]
    e3 = coefficients[3]
    e4 = coefficients[4]
    e5 = coefficients[5]
    e6 = coefficients[6]
    Mh = coefficients[7]
    c1 = coefficients[8]
    c2 = coefficients[9]
    c3 = coefficients[10]
    Mref = coefficients[11]
    Rref = coefficients[12]
    h = coefficients[13]
    Dc3 = coefficients[14]
    Dc3chtur = coefficients[15]
    Dc3jpit = coefficients[16]
    
    Mharray=Mh*ones(Rjb.shape)
    Uarray=U.copy()
    SSarray=SS.copy()
    NSarray=NS.copy()
    RSarray=RS.copy()
    harray=h*ones(Rjb.shape)
    c1array=c1*ones(Rjb.shape)
    c2array=c2*ones(Rjb.shape)
    Mrefarray=Mref*ones(Rjb.shape)
    c3array=c3*ones(Rjb.shape)
    Dc3array=Dc3*ones(Rjb.shape)
    Dc3jpitarray=Dc3jpit*ones(Rjb.shape)
    Rrefarray=Rref*ones(Rjb.shape)
    
    fm=ones(Rjb.shape)
    i=where((M<Mharray)==True)[0]
    #if M <= Mh:
    fm[i] = e0*Uarray[i] + e1*SSarray[i] + e2*NSarray[i] + e3*RSarray[i] + e4*(M[i] - Mharray[i]) + e5*(M[i] - Mharray[i])**2
    #else:
    i=where((M<Mharray)==False)[0]
    fm[i] = e0*Uarray[i] + e1*SSarray[i] + e2*NSarray[i] + e3*RSarray[i] + e6*(M[i] - Mharray[i])
        
    
    R = (Rjb**2 + harray**2)**0.5
    
    #region term
    #CA
    fp = (c1array + c2array * (M - Mrefarray)) * log(R / Rrefarray) + (c3array + Dc3array) * (R - Rrefarray)
    #ITLAY
    if italy==True:
        fp = (c1array + c2array * (M - Mrefarray)) * log(R / Rrefarray) + (c3array + Dc3jpitarray) * (R - Rrefarray)
    
    #Calculate PGAr
    PGAr = exp(fm + fp)
    
    return PGAr




def PGAr_calc_one_station(M, Rjb, U, RS, NS):
    '''
    Calculate reference PGA
    '''
    
    from numpy import log,exp,ones
    
    # GMPE coefficients from the PEER spreadsheet: 
    # http://peer.berkeley.edu/ngawest2/wp-content/uploads/2016/02/NGAW2_GMPE_Spreadsheets_v5.7_041415_Protected.zip
    # in the "BSSA14_Coeffs sheet, T(s)=0 corresponds to PGA, T(s)=-1 is PGV
    coefficients=[0.4473,0.4856,0.2459,0.4539,1.431,0.05053,-0.1662,5.5,-1.13400,0.19170,-0.00809,4.5,
                        1.,4.5,0.000000,0.002860,-0.002550,-0.6000,1500.00,760,0.,0.1,-0.1500,-0.00701,-9.900,
                        -9.900,110.000,270.000,0.100,0.070,225.,300.,0.6950,0.4950,0.3980,0.3480]
    
    #Assign each coefficient
    e0 = coefficients[0]
    e1 = coefficients[1]
    e2 = coefficients[2]
    e3 = coefficients[3]
    e4 = coefficients[4]
    e5 = coefficients[5]
    e6 = coefficients[6]
    Mh = coefficients[7]
    c1 = coefficients[8]
    c2 = coefficients[9]
    c3 = coefficients[10]
    Mref = coefficients[11]
    Rref = coefficients[12]
    h = coefficients[13]
    Dc3 = coefficients[14]
    Dc3chtur = coefficients[15]
    Dc3jpit = coefficients[16]
    
    if NS == 0 and RS == 0 and U == 0:
        SS = 1
    else:
        SS = 0
    

    if M <= Mh:
        fm = e0*U + e1*SS + e2*NS + e3*RS + e4*(M - Mh) + e5*(M - Mh)**2
    else:
        fm = e0*U + e1*SS + e2*NS + e3*RS + e6*(M - Mh)
    
    R = (Rjb**2 + h**2)**0.5
    
    #region term
    fp = (c1 + c2 * (M - Mref)) * log(R / Rref) + (c3 + Dc3) * (R - Rref)
    
    #Calculate PGAr
    PGAr = exp(fm + fp)
    
    return PGAr


        

def bssa14(M, Rjb, Vs30,U=None,SS=None,RS=None,NS=None,Z1=None,intensity_measure='PGA',italy=False):
    '''
    Calculate ground motion intensity using the BSSA14 GMPE
    
    Parameters:
        M - Moment magnitude
        Rjb - Distance to surface projection of fault in km
        U - is 1 if unspecified faulting style
        RS - is 1 if reverse faulting
        NS - is 1 if normal fault
        Vs30 - Vs30 in m/s
        Z1 - Depth to Vs=1km/s, if unknown use Z1=None
        
    Returns:
        Y - the desired ground motion intensity, PGA in g or PGV in cm/s
        
    Notes: For strike slip faulting (default) set U=NS=RS=0
    '''
    
    from numpy import log,exp,sqrt,array,ones,where,zeros
    
    # GMPE coefficients from the PEER spreadsheet: 
    # http://peer.berkeley.edu/ngawest2/wp-content/uploads/2016/02/NGAW2_GMPE_Spreadsheets_v5.7_041415_Protected.zip
    # in the "BSSA14_Coeffs sheet, T(s)=0 corresponds to PGA, T(s)=-1 is PGV
    
    #Convert input to floats
    Vs30=Vs30.astype(float)
    Rjb=Rjb.astype(float)
    M=M.astype(float)

    
    if intensity_measure.upper()=='PGA':
        coefficients=[0.4473,0.4856,0.2459,0.4539,1.431,0.05053,-0.1662,5.5,-1.13400,0.19170,-0.00809,4.5,
                        1.,4.5,0.000000,0.002860,-0.002550,-0.6000,1500.00,760,0.,0.1,-0.1500,-0.00701,-9.900,
                        -9.900,110.000,270.000,0.100,0.070,225.,300.,0.6950,0.4950,0.3980,0.3480]
    elif intensity_measure.upper()=='PGV':
        coefficients=[5.037,5.078,4.849,5.033,1.073,-0.1536,0.2252,6.2,-1.24300,0.14890,-0.00344,4.5,1.,5.3,
                        0.000000,0.004350,-0.000330,-0.8400,1300.00,760,0.,0.1,-0.1000,-0.00844,-9.900,-9.900,
                        105.000,272.000,0.082,0.080,225.,300.,0.6440,0.5520,0.4010,0.3460]
    elif intensity_measure.upper()=='SA0.3':
        coefficients=[1.2217,1.2401,1.0246,1.2653,0.95676,-0.1959,-0.092855,6.14,-1.09480,0.13388,-0.00548,4.5,
                        1,4.93,0.000000,0.002200,-0.003300,-0.8417,1308.47,760,0,0.1,-0.2191,-0.00670,-9.9,-9.9,
                        103.150,268.590,0.138,0.050,225,300,0.6750,0.5610,0.3630,0.2290]
    elif intensity_measure.upper()=='SA1.0':
        coefficients=[0.3932,0.4218,0.207,0.4124,1.5004,-0.18983,0.17895,6.2,-1.19300,0.10248,-0.00121,4.5,1,5.74,
                        0.000000,0.002920,-0.002090,-1.0500,1109.95,760,0,0.1,-0.1052,-0.00844,0.367,0.208,116.390,
                        270.000,0.098,0.020,225,300,0.5530,0.6250,0.4980,0.2980]
    elif intensity_measure.upper()=='SA3.0':
        coefficients=[-1.1898,-1.142,-1.23,-1.2664,2.1323,-0.04332,0.62694,6.2,-1.21790,0.09764,0.00000,4.5,1,6.93,
                        0.000000,0.002620,-0.001190,-1.0112,922.43,760,0,0.1,-0.0136,-0.00183,1.135,0.516,130.360,
                        195.000,0.088,0.000,225,300,0.5340,0.6190,0.5370,0.3440]
    else:
        print 'ERROR: Unknown intensity measure'
        #return

    #Assign each coefficient
    e0 = coefficients[0]
    e1 = coefficients[1]
    e2 = coefficients[2]
    e3 = coefficients[3]
    e4 = coefficients[4]
    e5 = coefficients[5]
    e6 = coefficients[6]
    Mh = coefficients[7]
    c1 = coefficients[8]
    c2 = coefficients[9]
    c3 = coefficients[10]
    Mref = coefficients[11]
    Rref = coefficients[12]
    h = coefficients[13]
    Dc3 = coefficients[14]
    Dc3chtur = coefficients[15]
    Dc3jpit = coefficients[16]
    C = coefficients[17]
    Vc = coefficients[18]
    Vref = coefficients[19]
    f1 = coefficients[20]
    f3 = coefficients[21]
    f4 = coefficients[22]
    f5 = coefficients[23]
    f6 = coefficients[24]
    f7 = coefficients[25]
    R1 = coefficients[26]
    R2 = coefficients[27]
    Dfr = coefficients[28]
    Dfv = coefficients[29]
    V1 = coefficients[30]
    V2 = coefficients[31]
    phi1 = coefficients[32]
    phi2 = coefficients[33]
    tau1 = coefficients[34]
    tau2 = coefficients[35]


    Uarray=U.copy()
    NSarray=NS.copy()
    SSarray=SS.copy()
    RSarray=RS.copy()

    # Hinge magnitude term and conversion to arrays
    Mharray=Mh*ones(Rjb.shape)

    fm=zeros(Rjb.shape)
    
    i=where((M<=Mharray)==True)[0]
    #if M <= Mh:
    fm[i] = e0*Uarray[i] + e1*SSarray[i] + e2*NSarray[i] + e3*RSarray[i] + e4*(M[i] - Mharray[i]) + e5*(M[i] - Mharray[i])**2
    #else:
    i=where((M<=Mharray)==False)[0]
    fm[i] = e0*Uarray[i] + e1*SSarray[i] + e2*NSarray[i] + e3*RSarray[i] + e6*(M[i] - Mharray[i])
    
    #Disance term
    harray=h*ones(Rjb.shape)
    R = (Rjb**2 + harray**2)**0.5

    # Region term
    c1array=c1*ones(Rjb.shape)
    c2array=c2*ones(Rjb.shape)
    Mrefarray=Mref*ones(Rjb.shape)
    Rrefarray=Rref*ones(Rjb.shape)
    c3array=c3*ones(Rjb.shape)
    Dc3array=Dc3*ones(Rjb.shape)
    Dc3jpitarray=Dc3jpit*ones(Rjb.shape)
    if italy==True:
        fp = (c1array + c2array*(M - Mrefarray))*log(R/Rrefarray) + (c3array + Dc3jpitarray)*(R - Rrefarray)
    else:
        fp = (c1array + c2array*(M - Mrefarray))*log(R/Rrefarray) + (c3array + Dc3array)*(R - Rrefarray)

    #Linear Site Term
    Carray=C*ones(Rjb.shape)
    Vcarray=Vc*ones(Rjb.shape)
    Vrefarray=Vref*ones(Rjb.shape)
    i_vs30=(Vs30 <= Vc)
    i=where(i_vs30==True)[0]
    if len(i)>0:
        flin = Carray*log(Vs30[i]/Vrefarray[i])
    i=where(i_vs30==False)[0]
    if len(i)>0:
        flin = Carray*log(Vcarray[i]/Vrefarray[i])

    #Nonlinear Site Term    
    minVarray=Vs30.copy()
    i=where(Vs30>760)[0]
    if len(i)>0:
        minVarray[i_vs30] = 760

    #Combine terms
    PGAr=PGAr_calc(M, Rjb, U,SS, RS, NS)
    
    f5array=f5*ones(Rjb.shape)
    f1array=f1*ones(Rjb.shape)
    f3array=f3*ones(Rjb.shape)
    f4array=f4*ones(Rjb.shape)
    
    f2array = f4array*((exp(f5array*(minVarray - 360))) - exp(f5array*(760 - 360)))
    fnl = f1array + f2array*log((PGAr + f3array)/f3array)
    fnl = f1array + (f4array*((exp(f5array*(minVarray - 360)))-exp(f5*(760 - 360))))*log((PGAr + f3array)/f3array)

    #Basin Depth Term
    mz1 = exp(-7.15/4*log((Vs30**4 + 570.94**4)/(1360**4 + 570.94**4)))/1000

    #Final correction
    if Z1 == None:
        dz1 = zeros(Vs30.shape)
    else:
        dz1 = Z1 - mz1

    fz1 = zeros(Vs30.shape)
    
    #elif dz1 <= f7/f6:
    #    fz1 = f6*dz1
    #elif dz1 > f7/f6:
    #    fz1 = f7
    #else:
    #    fz1 = 0

    if Z1 == None:
        fz1 = zeros(Vs30.shape)
    else:
        fz1 = fz1

    #Site Term
    fs = flin + fnl #in ln units

    #Model Prediction in ln units
    
    Y = exp(fm + fp + fs + fz1)
    
    #Stdev
    sigma=bssa14_stdev(M,Rjb,Vs30,intensity_measure=intensity_measure)
    
    return Y,sigma

def bssa14_one_station(M, Rjb, Vs30, U=0, RS=0, NS=0, SS=0,Z1=None,intensity_measure='PGA'):
    '''
    Calculate ground motion intensity using the BSSA14 GMPE
    
    Parameters:
        M - Moment magnitude
        Rjb - Distance to surface projection of fault in km
        U - is 1 if unspecified faulting style
        RS - is 1 if reverse faulting
        NS - is 1 if normal fault
        Vs30 - Vs30 in m/s
        Z1 - Depth to Vs=1km/s, if unknown use Z1=None
        
    Returns:
        Y - the desired ground motion intensity, PGA in g or PGV in cm/s
        
    Notes: For strike slip faulting (default) set U=NS=RS=0
    '''
    
    from numpy import log,exp,sqrt
    
    # GMPE coefficients from the PEER spreadsheet: 
    # http://peer.berkeley.edu/ngawest2/wp-content/uploads/2016/02/NGAW2_GMPE_Spreadsheets_v5.7_041415_Protected.zip
    # in the "BSSA14_Coeffs sheet, T(s)=0 corresponds to PGA, T(s)=-1 is PGV
    
    #Convert input to floats
    Vs30=float(Vs30)
    Rjb=float(Rjb)
    M=float(M)
    
    if intensity_measure.upper()=='PGA':
        coefficients=[0.4473,0.4856,0.2459,0.4539,1.431,0.05053,-0.1662,5.5,-1.13400,0.19170,-0.00809,4.5,
                        1.,4.5,0.000000,0.002860,-0.002550,-0.6000,1500.00,760,0.,0.1,-0.1500,-0.00701,-9.900,
                        -9.900,110.000,270.000,0.100,0.070,225.,300.,0.6950,0.4950,0.3980,0.3480]
    elif intensity_measure.upper()=='PGV':
        coefficients=[5.037,5.078,4.849,5.033,1.073,-0.1536,0.2252,6.2,-1.24300,0.14890,-0.00344,4.5,1.,5.3,
                        0.000000,0.004350,-0.000330,-0.8400,1300.00,760,0.,0.1,-0.1000,-0.00844,-9.900,-9.900,
                        105.000,272.000,0.082,0.080,225.,300.,0.6440,0.5520,0.4010,0.3460]
    else:
        print 'ERROR: Unknown intensity measure'
        #return

    #Assign each coefficient
    e0 = coefficients[0]
    e1 = coefficients[1]
    e2 = coefficients[2]
    e3 = coefficients[3]
    e4 = coefficients[4]
    e5 = coefficients[5]
    e6 = coefficients[6]
    Mh = coefficients[7]
    c1 = coefficients[8]
    c2 = coefficients[9]
    c3 = coefficients[10]
    Mref = coefficients[11]
    Rref = coefficients[12]
    h = coefficients[13]
    Dc3 = coefficients[14]
    Dc3chtur = coefficients[15]
    Dc3jpit = coefficients[16]
    C = coefficients[17]
    Vc = coefficients[18]
    Vref = coefficients[19]
    f1 = coefficients[20]
    f3 = coefficients[21]
    f4 = coefficients[22]
    f5 = coefficients[23]
    f6 = coefficients[24]
    f7 = coefficients[25]
    R1 = coefficients[26]
    R2 = coefficients[27]
    Dfr = coefficients[28]
    Dfv = coefficients[29]
    V1 = coefficients[30]
    V2 = coefficients[31]
    phi1 = coefficients[32]
    phi2 = coefficients[33]
    tau1 = coefficients[34]
    tau2 = coefficients[35]

    # Magnitude Scaling Term
    if NS == 0 and RS == 0 and U == 0:
        SS = 1
    else:
        SS = 0

    # Hinge magnitude term
    if M <= Mh:
        fm = e0*U + e1*SS + e2*NS + e3*RS + e4*(M - Mh) + e5*(M - Mh)**2
    else:
        fm = e0*U + e1*SS + e2*NS + e3*RS + e6*(M - Mh)  
    
    #Disance term
    R = (Rjb**2 + h**2)**0.5

    # Region term
    fp = (c1 + c2*(M - Mref))*log(R/Rref) + (c3 + Dc3)*(R - Rref)

    #Linear Site Term
    if Vs30 <= Vc:
        flin = C*log(Vs30/Vref)
    else:
        flin = C*log(Vc / Vref)

    #Nonlinear Site Term
    if Vs30 < 760:
        minV = Vs30
    else:
        minV = 760

    #Combine terms
    PGAr=PGAr_calc_one_station(M, Rjb, U, RS, NS)
    
    f2 = f4*((exp(f5*(minV - 360))) - exp(f5*(760 - 360)))
    fnl = f1 + f2*log((PGAr + f3)/f3)
    fnl = f1 + (f4*((exp(f5*(minV - 360)))-exp(f5*(760 - 360))))*log((PGAr + f3)/f3)

    #Basin Depth Term
    mz1 = exp(-7.15/4*log((Vs30**4 + 570.94**4)/(1360**4 + 570.94**4)))/1000

    #Final correction
    if Z1 == None:
        dz1 = 0
    else:
        dz1 = Z1 - mz1

    fz1 = 0
    
    #elif dz1 <= f7/f6:
    #    fz1 = f6*dz1
    #elif dz1 > f7/f6:
    #    fz1 = f7
    #else:
    #    fz1 = 0

    if Z1 == None:
        fz1 = 0
    else:
        fz1 = fz1

    #Site Term
    fs = flin + fnl #in ln units

    #Model Prediction in ln units
    
    Y = exp(fm + fp + fs + fz1)
    
    #Standard deviation
    sigma=bssa14_stdev_one_station(M,Rjb,Vs30,intensity_measure=intensity_measure)
    
    return Y,sigma


def bssa14_stdev_one_station(M,Rjb,Vs30,intensity_measure='PGA'):
    '''
    Get GMPE standar deviation
    '''
    from numpy import log,exp,sqrt,array,ones,where,zeros
    
    # GMPE coefficients from the PEER spreadsheet: 
    # http://peer.berkeley.edu/ngawest2/wp-content/uploads/2016/02/NGAW2_GMPE_Spreadsheets_v5.7_041415_Protected.zip
    # in the "BSSA14_Coeffs sheet, T(s)=0 corresponds to PGA, T(s)=-1 is PGV
    
    #Convert input to floats
    Vs30=float(Vs30)
    Rjb=float(Rjb)
    M=float(M)
    
    if intensity_measure.upper()=='PGA':
        coefficients=[0.4473,0.4856,0.2459,0.4539,1.431,0.05053,-0.1662,5.5,-1.13400,0.19170,-0.00809,4.5,
                        1.,4.5,0.000000,0.002860,-0.002550,-0.6000,1500.00,760,0.,0.1,-0.1500,-0.00701,-9.900,
                        -9.900,110.000,270.000,0.100,0.070,225.,300.,0.6950,0.4950,0.3980,0.3480]
    elif intensity_measure.upper()=='PGV':
        coefficients=[5.037,5.078,4.849,5.033,1.073,-0.1536,0.2252,6.2,-1.24300,0.14890,-0.00344,4.5,1.,5.3,
                        0.000000,0.004350,-0.000330,-0.8400,1300.00,760,0.,0.1,-0.1000,-0.00844,-9.900,-9.900,
                        105.000,272.000,0.082,0.080,225.,300.,0.6440,0.5520,0.4010,0.3460]
    else:
        print 'ERROR: Unknown intensity measure'
        #return

    #Assign each coefficient
    e0 = coefficients[0]
    e1 = coefficients[1]
    e2 = coefficients[2]
    e3 = coefficients[3]
    e4 = coefficients[4]
    e5 = coefficients[5]
    e6 = coefficients[6]
    Mh = coefficients[7]
    c1 = coefficients[8]
    c2 = coefficients[9]
    c3 = coefficients[10]
    Mref = coefficients[11]
    Rref = coefficients[12]
    h = coefficients[13]
    Dc3 = coefficients[14]
    Dc3chtur = coefficients[15]
    Dc3jpit = coefficients[16]
    C = coefficients[17]
    Vc = coefficients[18]
    Vref = coefficients[19]
    f1 = coefficients[20]
    f3 = coefficients[21]
    f4 = coefficients[22]
    f5 = coefficients[23]
    f6 = coefficients[24]
    f7 = coefficients[25]
    R1 = coefficients[26]
    R2 = coefficients[27]
    Dfr = coefficients[28]
    Dfv = coefficients[29]
    V1 = coefficients[30]
    V2 = coefficients[31]
    phi1 = coefficients[32]
    phi2 = coefficients[33]
    tau1 = coefficients[34]
    tau2 = coefficients[35]


    if M <= 4.5:
        tauM = tau1
    elif M > 4.5 and M < 5.5:
        tauM = tau1 + (tau2 - tau1) * (M - 4.5)
    else:
        tauM = tau2
    
    if M <= 4.5:
        phiM = phi1
    elif M > 4.5 and M < 5.5:
        phiM = phi1 + (phi2 - phi1) * (M - 4.5)
    else:
        phiM = phi2

    
    if Rjb <= R1:
        phiMR = phiM
    elif Rjb > R1 and Rjb <= R2:
        phiMR = phiM + Dfr * (log(Rjb / R1) / (log(R2 / R1)))
    else:
        phiMR = phiM + Dfr
    
    if Vs30 >= V2:
        phi = phiMR
    elif Vs30 >= V1 and Vs30 <= V2:
        phi = phiMR - Dfv * (log(V2 / Vs30) / (log(V2 / V1)))
    else:
        phi = phiMR - Dfv
    
    #Model Prediction in ln units
    sigma = (tauM**2 + phi**2)**0.5
    
    return sigma
    
    
def bssa14_stdev(M,Rjb,Vs30,intensity_measure='PGA'):
    '''
    Get GMPE standar deviation
    '''
    from numpy import log,exp,sqrt,array,ones,where,zeros
    
    # GMPE coefficients from the PEER spreadsheet: 
    # http://peer.berkeley.edu/ngawest2/wp-content/uploads/2016/02/NGAW2_GMPE_Spreadsheets_v5.7_041415_Protected.zip
    # in the "BSSA14_Coeffs sheet, T(s)=0 corresponds to PGA, T(s)=-1 is PGV
    
    #Convert input to floats
    Vs30=Vs30.astype('float')
    Rjb=Rjb.astype('float')
    M=M.astype('float')
    
    if intensity_measure.upper()=='PGA':
        coefficients=[0.4473,0.4856,0.2459,0.4539,1.431,0.05053,-0.1662,5.5,-1.13400,0.19170,-0.00809,4.5,
                        1.,4.5,0.000000,0.002860,-0.002550,-0.6000,1500.00,760,0.,0.1,-0.1500,-0.00701,-9.900,
                        -9.900,110.000,270.000,0.100,0.070,225.,300.,0.6950,0.4950,0.3980,0.3480]
    elif intensity_measure.upper()=='PGV':
        coefficients=[5.037,5.078,4.849,5.033,1.073,-0.1536,0.2252,6.2,-1.24300,0.14890,-0.00344,4.5,1.,5.3,
                        0.000000,0.004350,-0.000330,-0.8400,1300.00,760,0.,0.1,-0.1000,-0.00844,-9.900,-9.900,
                        105.000,272.000,0.082,0.080,225.,300.,0.6440,0.5520,0.4010,0.3460]
    elif intensity_measure.upper()=='SA0.3':
        coefficients=[1.2217,1.2401,1.0246,1.2653,0.95676,-0.1959,-0.092855,6.14,-1.09480,0.13388,-0.00548,4.5,
                        1,4.93,0.000000,0.002200,-0.003300,-0.8417,1308.47,760,0,0.1,-0.2191,-0.00670,-9.9,-9.9,
                        103.150,268.590,0.138,0.050,225,300,0.6750,0.5610,0.3630,0.2290]
    elif intensity_measure.upper()=='SA1.0':
        coefficients=[0.3932,0.4218,0.207,0.4124,1.5004,-0.18983,0.17895,6.2,-1.19300,0.10248,-0.00121,4.5,1,5.74,
                        0.000000,0.002920,-0.002090,-1.0500,1109.95,760,0,0.1,-0.1052,-0.00844,0.367,0.208,116.390,
                        270.000,0.098,0.020,225,300,0.5530,0.6250,0.4980,0.2980]
    elif intensity_measure.upper()=='SA3.0':
        coefficients=[-1.1898,-1.142,-1.23,-1.2664,2.1323,-0.04332,0.62694,6.2,-1.21790,0.09764,0.00000,4.5,1,6.93,
                        0.000000,0.002620,-0.001190,-1.0112,922.43,760,0,0.1,-0.0136,-0.00183,1.135,0.516,130.360,
                        195.000,0.088,0.000,225,300,0.5340,0.6190,0.5370,0.3440]
    else:
        print 'ERROR: Unknown intensity measure'
        #return

    #Assign each coefficient
    e0 = coefficients[0]
    e1 = coefficients[1]
    e2 = coefficients[2]
    e3 = coefficients[3]
    e4 = coefficients[4]
    e5 = coefficients[5]
    e6 = coefficients[6]
    Mh = coefficients[7]
    c1 = coefficients[8]
    c2 = coefficients[9]
    c3 = coefficients[10]
    Mref = coefficients[11]
    Rref = coefficients[12]
    h = coefficients[13]
    Dc3 = coefficients[14]
    Dc3chtur = coefficients[15]
    Dc3jpit = coefficients[16]
    C = coefficients[17]
    Vc = coefficients[18]
    Vref = coefficients[19]
    f1 = coefficients[20]
    f3 = coefficients[21]
    f4 = coefficients[22]
    f5 = coefficients[23]
    f6 = coefficients[24]
    f7 = coefficients[25]
    R1 = coefficients[26]
    R2 = coefficients[27]
    Dfr = coefficients[28]
    Dfv = coefficients[29]
    V1 = coefficients[30]
    V2 = coefficients[31]
    phi1 = coefficients[32]
    phi2 = coefficients[33]
    tau1 = coefficients[34]
    tau2 = coefficients[35]
    
    tauM=zeros(M.shape)
    #if M <= 4.5:
    i=where(M<=4.5)
    tauM[i] = tau1
    #elif M > 4.5 and M < 5.5:
    i=where((M>4.5) & (M<5.5))[0]
    tauM[i] = tau1 + (tau2 - tau1) * (M[i] - 4.5)
    #else:
    i=where(M>=5.5)[0]
    tauM[i] = tau2
    
    phiM=zeros(M.shape)
    #if M <= 4.5:
    i=where(M<=4.5)
    phiM[i] = phi1
    #elif M > 4.5 and M < 5.5:
    i=where((M>4.5) & (M<5.5))[0]
    phiM[i] = phi1 + (phi2 - phi1) * (M[i] - 4.5)
    #else:
    i=where(M>=5.5)[0]
    phiM[i] = phi2

    phiMR=zeros(M.shape)
    #if Rjb <= R1:
    i=where(Rjb<=R1)[0]
    phiMR[i] = phiM[i]
    #elif Rjb > R1 and Rjb <= R2:
    i=where((Rjb>R1) & (Rjb<=R2))[0]
    phiMR[i] = phiM[i] + Dfr * (log(Rjb[i] / R1) / (log(R2 / R1)))
    #else:
    i=where(Rjb>R2)[0]
    phiMR[i] = phiM[i] + Dfr
    
    phi=zeros(M.shape)
    #if Vs30 >= V2:
    i=where(Vs30>=V2)[0]
    phi[i] = phiMR[i]
    #elif Vs30 >= V1 and Vs30 <= V2:
    i=where((Vs30>=V1) & (Vs30<=V2))[0]
    phi[i] = phiMR[i] - Dfv * (log(V2 / Vs30[i]) / (log(V2 / V1)))
    #else:
    i=where(Vs30<V1)[0]
    phi[i] = phiMR[i] - Dfv
    
    
    #Model Prediction in ln units
    sigma = (tauM**2 + phi**2)**0.5
    
    return sigma
    
def sample_gmpe(median_motion,stdev):
    '''
    Obtain ground motion from lognormal distribution
    '''
    
    from numpy.random import randn
    from numpy import log,exp
    
    x=randn(len(median_motion))
    mean_pga=log(median_motion) #because GMPE's work in Ln space
    #Scale to lognormal
    z_pga=x*stdev+mean_pga
    #And back to physical units
    sampled_motion=exp(z_pga)
    
    return sampled_motion




def cua_envelope(M,dist_in_km,times,ptime,stime,Pcoeff=0,Scoeff=12):
    '''
    Cua envelopes, modified from Ran Nof's Cua2008 module
    '''
    from numpy import where,sqrt,exp,log10,arctan,pi,zeros
    
    a = [0.719, 0.737, 0.801, 0.836, 0.950, 0.943, 0.745, 0.739, 0.821, 0.812, 0.956, 0.933,
            0.779, 0.836, 0.894, 0.960, 1.031, 1.081, 0.778, 0.751, 0.900, 0.882, 1.042, 1.034]
    b = [-3.273e-3, -2.520e-3, -8.397e-4, -5.409e-4, -1.685e-6, -5.171e-7, -4.010e-3, -4.134e-3,
                -8.543e-4, -2.652e-6, -1.975e-6, -1.090e-7, -2.555e-3, -2.324e-3, -4.286e-4, -8.328e-4,
                -1.015e-7, -1.204e-6, -2.66e-5, -2.473e-3, -1.027e-5,- 5.41e-4, -1.124e-5, -4.924e-6]
    d = [-1.195, -1.26, -1.249, -1.284, -1.275, -1.161, -1.200, -1.199, -1.362, -1.483, -1.345, -1.234,
                -1.352, -1.562, -1.440, -1.589, -1.438, -1.556, -1.385, -1.474, -1.505, -1.484, -1.367, -1.363]
    c1 = [1.600, 2.410, 0.761, 1.214, 2.162, 2.266, 1.752, 2.030, 1.148, 1.402, 1.656, 1.515,
                1.478, 2.423, 1.114, 1.982, 1.098, 1.946, 1.763, 1.593, 1.388, 1.530, 1.379, 1.549]
    c2 = [1.045, 0.955, 1.340, 0.978, 1.088, 1.016, 1.091, 1.972, 1.100, 0.995, 1.164, 1.041,
                1.105, 1.054, 1.110, 1.067, 1.133, 1.091, 1.112, 1.106, 1.096, 1.04, 1.178, 1.082]
    e = [-1.065, -1.051, -3.103, -3.135, -4.958, -5.008, -0.955, -0.775, -2.901, -2.551, -4.799, -4.749,
                -0.645, -0.338, -2.602, -2.351, -4.342, -4.101, -0.751, -0.355, -2.778, -2.537, -4.738, -4.569]
    sig_uncorr = [0.307, 0.286, 0.268, 0.263, 0.284, 0.301, 0.288, 0.317, 0.263, 0.298, 02.83, 0.312,
                0.308, 0.312, 0.279, 0.296, 0.277, 0.326, 0.300, 0.300, 0.250, 0.270, 0.253, 0.286]
    sig_corr = [0.233, 0.229, 0.211, 0.219, 0.239, 0.247, 0.243, 0.256, 0.231, 0.239, 0.254, 0.248,
                0.243, 0.248, 0.230, 0.230, 0.233, 0.236, 0.238, 0.235, 0.220, 0.221, 0.232, 0.230]
    
    # Coefficienstime for eqn: log(env_param) = alpha*M + beta*R + delta*logR + mu
    # Coefficienstime and equation for t_rise (rise time):
    
    alpha_t_rise = [0.06, 0.07, 0.06, 0.07, 0.05, 0.05, 0.06, 0.06, 0.06, 0.06, 0.08, 0.067,
                0.064, 0.055, 0.093, 0.087, 0.109, 0.12, 0.069, 0.059, 0.116, 0.11, 0.123, 0.124]  
    beta_t_rise = [5.5e-4, 1.2e-3, 1.33e-3, 4.35e-4, 1.29e-3, 1.19e-3, 7.45e-4, 5.87e-4, 7.32e-4, 1.08e-3, 1.64e-3, 1.21e-3,
                0, 1.21e-3, 0, 4.0e-4, 7.68e-4, 0, 0, 2.18e-3, 0, 1.24e-3, 1.3e-3, 0]
    delta_t_rise = [0.27, 0.24, 0.23, 0.47, 0.27, 0.47, 0.37, 0.23, 0.25, 0.22, 0.13, 0.28,
                0.48, 0.34, 0.48, 0.49, 0.38, 0.45, 0.49, 0.26, 0.503, 0.38, 0.257, 0.439]
    mu_t_rise = [-0.37, -0.38, -0.34, -0.68, -0.34, -0.58, -0.51, -0.37, -0.37, -0.36, -0.33, -0.46,
                -0.89, -0.66, -0.96, -0.98, -0.87,-0.89,-0.97, -0.66, -1.14, -0.91, -0.749, -0.82]
    
    # Coefficienstime and equation for delta_t (wave duration):
    
    alpha_delta_t = [0, 0.03, 0.054, 0.03, 0.047, 0.051, 0, 0, 0.046, 0.031, 0.058, 0.043,
                0, 0.028, 0.02, 0.028, 0.04, 0.03, 0.03, 0.03, 0.018, 0.017, 0.033, 0.023]
    beta_delta_t = [2.58e-3, 2.37e-3, 1.93e-3, 2.03e-3, 0, 1.12e-3, 2.75e-3, 1.76e-3, 2.61e-3, 1.7e-3, 2.02e-3, 9.94e-4,
                -4.87e-4, 0, 0, 0, 1.1e-3, 0, -1.4e-3, -1.78e-3, 0, -6.93e-4, 2.6e-4, -7.18e-4]
    delta_delta_t = [0.21, 0.39, 0.16, 0.289, 0.45, 0.33, 0.165, 0.36, 0, 0.26, 0, 0.19,
                0.13, 0.07, 0, 0.046, -0.15, 0.037, 0.22, 0.307, 0, 0.119, 0, 0.074]
    mu_delta_t = [-0.22, -0.59, -0.36, -0.45, -0.68, -0.59, -0.245, -0.48, -0.213, -0.52, -0.253, -0.42,
                0.0024, -0.102, 0.046, -0.083, 0.11, -0.066, -0.17, -0.66, -0.072, -0.05, -0.015, -0.005]
    
    # Coefficienstime and equation for tau (decay):
    
    alpha_tau = [0.047, 0.087, 0.054, 0.0403, 0, 0.035, 0.03, 0.057, 0.03, 0.0311, 0.05, 0.052,
                0.037, 0.0557, 0.029, 0.045, 0.029, 0.038, 0.031, 0.06, 0.04, 0.051, 0.024, 0.022]  
    beta_tau = [0, -1.89e-3, 5.37e-5, -1.26e-3, 0, -1.27e-3, 2.75e-3, -1.36e-3, 8.6e-4, -6.4e-4, 8.9e-4, 0,
                0, -8.2e-4, 8.0e-4, -5.46e-4, 0, -1.34e-3, 0, -1.45e-3, 9.4e-4, -1.41e-3, 0, -1.65e-3]
    delta_tau = [0.48, 0.58, 0.41, 0.387, 0.19, 0.19, 0.58, 0.63, 0.35, 0.44, 0.16, 0.12,
                0.39, 0.51, 0.25, 0.46, 0.36, 0.48, 0.34, 0.51, 0.25, 0.438, 0.303, 0.44]
    gamma_tau = [0.82, 0.58, 0.73, 0.58, 0, 0, 0, 0, 0, 0, 0, 0, 1.73, 1.63, 1.61, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mu_tau = [-0.75, -0.87, -0.51, -0.372, -0.07, -0.03, -0.97, -0.96, -0.62, -0.55, -0.387, -0.166,
                -0.59, -0.68, -0.31, -0.55, -0.38, -0.39, -0.44, -0.60, -0.34, -0.368, -0.22, -0.19]
    avg_gamma = 0.15

    
    # Coefficienstime and equation for gamma (decay):
    alpha_gamma = [-0.032, -0.048, -0.044, -0.0403, -0.062, -0.061, -0.027, -0.024, -0.039, -0.037, -0.052, -0.066,
                -0.014, -0.015, -0.024, -0.031, -0.025, -2.67e-2, -0.0149, -0.0197, -0.028, -0.0334, -0.015, -0.0176] #<--should be =-0.048 for i=1? not =-0.48?
    beta_gamma = [-1.81e-3, -1.42e-3, -1.65e-3, -2.0e-3, -2.3e-3, -1.9e-3, -1.75e-3, -1.6e-3, -1.88e-3, -2.23e-3, -1.67e-3, -2.5e-3,
                -5.28e-4, -5.89e-4, -1.02e-3, -4.61e-4, -4.22e-4, 2.0e-4, -4.64e-4, 0, -8.32e-4, 0, 0, 5.65e-4]
    delta_gamma = [-0.1, -0.13, -0.16, 0, 0, 0.11, -0.18, -0.24, -0.18, -0.14, -0.21, 0,
                -0.11, -0.163, -0.055, -0.162, -0.145, -0.217, -0.122, -0.242, -0.123, -0.21, -0.229, -0.25]
    tau_gamma = [0.27, 0.26, 0.33, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0.38, 0.39, 0.36, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mu_gamma = [0.64, 0.71, 0.72, 0.578, 0.61, 0.39, 0.74, 0.84, 0.76, 0.71, 0.849, 0.63,
                0.26, 0.299, 0.207, 0.302, 0.262, 0.274, 0.255, 0.378, 0.325, 0.325, 0.309, 0.236]
    avg_gamma = 0.15
    

    stat_err = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    sta_corr =  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # coefficienstime
    t_rise_p = 10**(alpha_t_rise[Pcoeff] * M + beta_t_rise[Pcoeff] * dist_in_km + delta_t_rise[Pcoeff] * log10(dist_in_km) + mu_t_rise[Pcoeff])
    t_rise_s = 10**(alpha_t_rise[Scoeff] * M + beta_t_rise[Scoeff] * dist_in_km + delta_t_rise[Scoeff] * log10(dist_in_km) + mu_t_rise[Scoeff])
    delta_t_p = 10**(alpha_delta_t[Pcoeff] * M + beta_delta_t[Pcoeff] * dist_in_km + delta_delta_t[Pcoeff] * log10(dist_in_km) + mu_delta_t[Pcoeff])
    delta_t_s = 10**(alpha_delta_t[Scoeff] * M + beta_delta_t[Scoeff] * dist_in_km + delta_delta_t[Scoeff] * log10(dist_in_km) + mu_delta_t[Scoeff])
    tau_p = 10**(alpha_tau[Pcoeff] * M + beta_tau[Pcoeff] * dist_in_km + delta_tau[Pcoeff] * log10(dist_in_km) + mu_tau[Pcoeff])
    tau_s = 10**(alpha_tau[Scoeff] * M + beta_tau[Scoeff] * dist_in_km + delta_tau[Scoeff] * log10(dist_in_km) + mu_tau[Scoeff])
    gamma_p = 10**(alpha_gamma[Pcoeff] * M + beta_gamma[Pcoeff] * dist_in_km + delta_gamma[Pcoeff] * log10(dist_in_km) + mu_gamma[Pcoeff])
    gamma_s = 10**(alpha_gamma[Scoeff] * M + beta_gamma[Scoeff] * dist_in_km + delta_gamma[Scoeff] * log10(dist_in_km) + mu_gamma[Scoeff])
    
    # Other variable (turn on saturation for larger evenstime?)
    C_p = (arctan(M-5) + (pi/2))*(c1[Pcoeff]*exp(c2[Pcoeff] * (M-5)))
    C_s = (arctan(M-5) + (pi/2))*(c1[Scoeff]*exp(c2[Scoeff] * (M-5)))
    R1 = sqrt(dist_in_km**2 + 9)
    
    # Basic AMplitudes
    A_p = 10**(a[Pcoeff]*M + b[Pcoeff]*(R1 + C_p) + d[Pcoeff]*log10(R1+C_p) + e[Pcoeff]+(sta_corr[Pcoeff]) + stat_err[Pcoeff])
    A_s = 10**(a[Scoeff]*M + b[Scoeff]*(R1 + C_s) + d[Scoeff]*log10(R1+C_s) + e[Scoeff]+(sta_corr[Scoeff]) + stat_err[Scoeff])
    
    # calculate envelope (ENV)
    envelope = zeros(len(times))

    # P envelope
    indx = where((times>=ptime) & (times<ptime+t_rise_p)) # between trigger and rise time
    if len(indx): envelope[indx] = (A_p/t_rise_p*(times[indx]-ptime)) # make sure we have data in that time frame and get envelope
    indx = where((times>=ptime+t_rise_p) & (times<ptime+t_rise_p+delta_t_p)) # flat area
    if len(indx): envelope[indx] = A_p # make sure we have data in that time frame and get envelope
    indx = where(times>ptime+t_rise_p+delta_t_p) # coda
    if len(indx): envelope[indx] = (A_p/((times[indx]-ptime-t_rise_p-delta_t_p+tau_p)**gamma_p)) # make sure we have data in that time frame and get envelope
    
    # S envelope
    indx = where((times>=stime) & (times<stime+t_rise_s)) # between trigger and rise time
    if len(indx): envelope[indx] += (A_s/t_rise_s*(times[indx]-stime)) # make sure we have data in that time frame and get envelope
    indx = where((times>=stime+t_rise_s) & (times<stime+t_rise_s+delta_t_s)) # flat area
    if len(indx): envelope[indx] += A_s # make sure we have data in that time frame and get envelope
    indx = where(times>stime+t_rise_s+delta_t_s) # coda
    if len(indx): envelope[indx] += (A_s/((times[indx]-stime-t_rise_s-delta_t_s+tau_s)**gamma_s)) # make sure we have data in that time frame and get envelope
    
    return envelope