# Copyright (c) 2024, Mike Nico Pionteck, Claudio Attaccalite and Myrta Grüning
# All rights reserved.
#
# This file is part of the yambopy project
# Calculate linear response from real-time calculations (yambo_nl)
#
import numpy as np
from yambopy.units import ha2ev,fs2aut, SVCMm12VMm1,AU2VMm1
from yambopy.nl.external_efield import Divide_by_the_Field
from yambopy.nl.harmonic_analysis import update_T_range
from tqdm import tqdm
import scipy as sci
from scipy import optimize
import scipy.linalg
import sys
import os

def LS_fit_diff(c,order,f1,f2,t,s):
    output = np.zeros(len(t))
    n = 0
    for ii in range(order+1):
        for jj in range(order+1):
            if (ii+jj > order):
                continue
            if (ii+jj == 0):
                output[:] = output[:] + c[n]
                n = n + 1
            else:
                f = f1*ii + f2*jj 
                output[:] = output[:] + c[n]*np.cos(f*t[:]) - c[n+1]*np.sin(f*t[:])
                n = n + 2
    for ii in range(1,order+1):
        for jj in range(ii,order+1):
            if (ii+jj > order):
                continue
            f = abs(f1*ii - f2*jj)
            output[:] = output[:] + c[n]*np.cos(f*t[:]) - c[n+1]*np.sin(f*t[:])
            n = n + 2
    return s-output
#
def Sampling(P,T_range,T_step,mesh,efield,SAMP_MOD):
    i_t_start = int(np.round(T_range[0]/T_step)) 
    i_deltaT  = int(np.round((T_range[1]-T_range[0])/T_step)/mesh)

    # Memory allocation 
    P_i      = np.zeros(mesh, dtype=np.double)
    T_i      = np.zeros(mesh, dtype=np.double)
    Sample = np.zeros((mesh,2), dtype=np.double)
    # Calculation of  T_i and P_i
    if SAMP_MOD=='linear':
        for i_t in range(mesh):
            T_i[i_t] = (i_t_start + i_deltaT * i_t)*T_step - efield["initial_time"]
            P_i[i_t] = P[i_t_start + i_deltaT * i_t]
    elif SAMP_MOD=='log':
        T_i=np.geomspace(i_t_start*T_step, T_range[1], mesh, endpoint=False)
        for i1 in range(mesh):
            i_t=int(np.round(T_i[i1]/T_step))
            T_i[i1]=i_t*T_step
            P_i[i1]=P[i_t]
    elif SAMP_MOD=='random':
        T_i=np.random.uniform(i_t_start*T_step, T_range[1], mesh)
        for i1 in range(mesh):
            i_t=int(np.round(T_i[i1]/mesh))
            T_i[i1]=i_t*T_step
            P_i[i1]=P[i_t]
    Sample[:,0]=T_i
    Sample[:,1]=P_i
    return Sample
#
def find_dec(a):
  # Convert the value to a string
  value_str = str(a)
  
  # Find the position of the decimal point
  if '.' in value_str:
      decimal_pos = value_str.find('.')
      # Calculate the number of decimal places
      num_decimals = len(value_str) - decimal_pos - 1
  else:
      num_decimals = 0
  
  return num_decimals
#
def fundamental_frequency_and_time_period(f1_eV, f2_eV):
    dec = max(find_dec(f1_eV), find_dec(f2_eV))
    # Calculate the greatest common divisor (GCD) of the two frequencies
    gcd_frequency_eV = np.gcd(int(f1_eV * 10**dec), int(f2_eV * 10**dec)) / 10**dec

    # Calculate the fundamental time period T = 1 / frequency
    hbar_eVs = 6.582119569e-16  # Reduced Planck's constant in eV⋅s
    fundamental_time_period_fs = hbar_eVs / gcd_frequency_eV * 1e15  # Convert seconds to femtoseconds

    return gcd_frequency_eV, fundamental_time_period_fs
#
def find_coeff_LS(order,P,f1,f2,T_range,T_step,mesh,efield,SAMP_MOD,xtol,gtol,ftol):
    N = 2*sum(range(order+2)) -1 + 2*sum(range(1+order%2,order,2))
    c = np.zeros(N)
    c[1] = 1 #10**(-13)
    c[2*(order+1)] = 1 #10**(-13)
    M = int((N-1)/2+1)
    copt  = np.zeros(M,dtype=np.cdouble)
    t = Sampling(P,T_range,T_step,mesh,efield,SAMP_MOD)[:,0]
    s = Sampling(P,T_range,T_step,mesh,efield,SAMP_MOD)[:,1]
    coeff = sci.optimize.least_squares(LS_fit_diff,c,args=(order,f1,f2,t,s),xtol=xtol,gtol=gtol,ftol=ftol)
    copt[0] = coeff.x[0]
    #print(coeff.optimality)
    #print(coeff.success)
    for ii in range(1,M):
        copt[ii] = 0.5*(coeff.x[2*(ii-1)+1] + 1j*coeff.x[2*(ii-1)+2])
    return copt, coeff.optimality
#
def LS_SF_Analysis(dec,nldb, X_order=2, period=30, mesh=1000,prn_Peff=False,prn_FFT=False,prn_Fundamentals=False,prn_Xhi=True,SAMP_MOD='linear',xtol=1e-8,gtol=1e-15,ftol=1e-8):
    # Time series 
    time  =nldb.IO_TIME_points
    # Time step of the simulation
    T_step=nldb.IO_TIME_points[1]-nldb.IO_TIME_points[0]
    # External field of the first run
    efield=nldb.Efield[0]
    # Numer of exteanl laser frequencies
    n_frequencies=len(nldb.Polarization)
    # Array of polarizations for each laser frequency
    polarization=nldb.Polarization

    print("\n* * * Sum/difference frequency generation: harmonic analysis * * *\n")

    freqs=np.zeros(n_frequencies,dtype=np.double)

    if efield["name"] != "SIN" and efield["name"] != "SOFTSIN" and efield["name"] != "ANTIRES":
        raise ValueError("Harmonic analysis works only with SIN or SOFTSIN fields")

    if(nldb.Efield_general[1]["name"] == "SIN" or nldb.Efield_general[1]["name"] == "SOFTSIN"):
        # frequency of the second and third laser, respectively)
        pump_freq=nldb.Efield_general[1]["freq_range"][0] 
        print("Frequency of the second field : "+str(pump_freq*ha2ev)+" [eV] \b")
    elif(nldb.Efield_general[1]["name"] == "none"):
        raise ValueError("Only one field present, please use standard harmonic_analysis.py for SHG,THG, etc..!")
    else:
        raise ValueError("Fields different from SIN/SOFTSIN are not supported ! ")
    
    if(nldb.Efield_general[2]["name"] != "none"):
        raise ValueError("Three fields not supported yet ! ")

    print("Number of frequencies : %d " % n_frequencies)
    # Smaller frequency
    W_step=sys.float_info.max
    max_W =sys.float_info.min

    for count, efield in enumerate(nldb.Efield):
        freqs[count]=efield["freq_range"][0]
        if efield["freq_range"][0]<W_step:
            W_step=efield["freq_range"][0]
        if efield["freq_range"][0]>max_W:
            max_W=efield["freq_range"][0]
    print("Minimum frequency : ",str(W_step*ha2ev)," [eV] ")
    print("Maximum frequency : ",str(max_W*ha2ev)," [eV] ")
    
    # Period of the incoming laser
    T_period=2.0*np.pi/W_step
    print("Effective max time period for field1 ",str(T_period/fs2aut)+" [fs] ")

    if prn_Fundamentals:
        #threshhold_period = 10**3
        #filtered_freqs = []
        #comment_freqs = []
        # Calculate the fundamental frequency and time period for each frequency
        print("Print fundamental frequency and time period for each frequency...")
        for i_f in range(len(freqs)):
            f, T = fundamental_frequency_and_time_period(freqs[i_f]*ha2ev, pump_freq*ha2ev)
            print("Frequency",i_f,":", f,"eV;", T,"fs")
            #if T <= threshhold_period:
            #    filtered_freqs.append(freqs[i_f])
            #    print(freqs[i_f], f, T)
            #else:
            #    comment_freqs.append(freqs[i_f])
            #    #print(freqs[i_f], f, T)
        print("End of fundamental frequency and time period for each frequency...")
        #n_frequencies = len(filtered_freqs)
        #freqs = np.array(filtered_freqs)


    T_range=np.zeros(2,dtype=np.double)
    if (time[-1]>period*fs2aut):
        T_range[0]=time[-1]-period*fs2aut
        T_range[1]=time[-1]
    else:
        raise ValueError("Your time range is too long !")
    
    print("Initial time range : ",str(T_range[0]/fs2aut),'-',str(T_range[1]/fs2aut)," [fs] ")
    print("Pump frequency : ",str(pump_freq*ha2ev),' [eV] ')

    mapping = []
    for ii in range(X_order+1):
        for jj in range(X_order+1):
            if (ii+jj > X_order):
                continue
            mapping.append((ii,jj))
    for ii in range(1,X_order+1):
        for jj in range(ii,X_order+1):
            if (ii+jj > X_order):
                continue
            mapping.append((ii,-jj))

    V_size=int((2*sum(range(X_order+2)) -1 + 2*sum(range(1+X_order%2,X_order,2))-1)/2+1)

    X_effective       =np.zeros((V_size,n_frequencies,3),dtype=np.cdouble)
    Optimality        =np.zeros((V_size,n_frequencies,3),dtype=np.cdouble)
    Susceptibility    =np.zeros((V_size,n_frequencies,3),dtype=np.cdouble)
    plot_sampling     =np.zeros((mesh,2,n_frequencies,3),dtype=np.double)
    
    print("Loop in frequecies...")
    # Find the Fourier coefficients by inversion
    for i_f in tqdm(range(n_frequencies)):
        for i_d in range(3):
            X_effective[:,i_f,i_d], Optimality[:,i_f,i_d]=find_coeff_LS(X_order, polarization[i_f][i_d,:],freqs[i_f],pump_freq,T_range,T_step,mesh,efield,SAMP_MOD,xtol,gtol,ftol)
            plot_sampling[:,:,i_f,i_d]=Sampling(polarization[i_f][i_d,:],T_range,T_step,mesh,efield,SAMP_MOD)
    
    # Calculate Susceptibilities from X_effective
    for i_v in range(V_size):
        i_order1, i_order2 = mapping[i_v]
        for i_f in range(n_frequencies):
            Susceptibility[i_v,i_f,:]=X_effective[i_v,i_f,:]
            D2=1.0
            if i_order1!=0:
                D2*=Divide_by_the_Field(nldb.Efield[0],abs(i_order1))
            if i_order2!=0:
                D2*=Divide_by_the_Field(nldb.Efield[1],abs(i_order2))
            if i_order1==0 and i_order2==0:
                D2=Divide_by_the_Field(nldb.Efield[0],abs(i_order1))*Divide_by_the_Field(nldb.Efield[1],abs(i_order2))
            Susceptibility[i_v,i_f,:]*=D2
        
    # Print time dependent polarization
    P=np.zeros((n_frequencies,3,len(time)),dtype=np.cdouble)
    for i_f in tqdm(range(n_frequencies)):
        for i_d in range(3):
            for i_v in range(V_size):
                i_order1, i_order2 = mapping[i_v]
                P[i_f,i_d,:]+=2*X_effective[i_v,i_f,i_d]*np.exp(1j * (i_order1*freqs[i_f]+i_order2*pump_freq) * time[:])
    if(prn_Peff):
        print("Reconstruct effective polarizations ...")
        header2="[fs]            "
        header2+="Px     "
        header2+="Py     "
        header2+="Pz     "
        footer2='Time dependent polarization reproduced from Fourier coefficients'
        footerSampling='Sampled polarization'
        footerError='Error in reconstructed polarization'
        headerError="[eV]            "
        headerError+="err[Px]     "
        headerError+="err[Py]     "
        headerError+="err[Pz]     "
        i_t_start = int(np.round(T_range[0]/T_step)) 
        valuesError=np.zeros((n_frequencies,4),dtype=np.double)
        N=len(P[i_f,i_d,:])-i_t_start
        for i_f in range(n_frequencies):
            # Print reconstructed polarization
            values=np.c_[time.real/fs2aut]
            values=np.append(values,np.c_[P[i_f,0,:].real],axis=1)
            values=np.append(values,np.c_[P[i_f,1,:].real],axis=1)
            values=np.append(values,np.c_[P[i_f,2,:].real],axis=1)
            output_file2='o.YamboPy-pol_reconstructed_F'+str(i_f+1)
            np.savetxt(output_file2,values,header=header2,delimiter=' ',footer=footer2)
        
            # Print sampling point
            valuesSampling=np.c_[plot_sampling[:,0,i_f,0]/fs2aut]
            valuesSampling=np.append(valuesSampling,np.c_[plot_sampling[:,1,i_f,0]],axis=1)
            valuesSampling=np.append(valuesSampling,np.c_[plot_sampling[:,1,i_f,1]],axis=1)
            valuesSampling=np.append(valuesSampling,np.c_[plot_sampling[:,1,i_f,2]],axis=1)
            output_file3='o.YamboPy-sampling_F'+str(i_f+1)
            np.savetxt(output_file3,valuesSampling,header=header2,delimiter=' ',footer=footerSampling)

            # Print error in reconstructed polarization in all frequencies
            valuesError[i_f,0]=freqs[i_f]*ha2ev
            for i_d in range(3):
                valuesError[i_f,i_d+1]=np.sqrt(np.sum((P[i_f,i_d,i_t_start:].real-polarization[i_f][i_d,i_t_start:]))**2)/N
        output_file4='o.YamboPy-errP'
        np.savetxt(output_file4,values,header=header2,delimiter=' ',footer=footer2)

    if(prn_FFT):
        print("Calculate FFT of the difference between the original and reproduced polarization ...")
        FFT_header="[fs]            "
        FFT_header+="FFTx     "
        FFT_header+="FFTy     "
        FFT_header+="FFTz     "
        FFT_footer='Time dependent FFT of the difference between the original and reproduced polarization in the sampling region'
        i_t_start = int(np.round(T_range[0]/T_step))
        for i_f in range(n_frequencies):
            FFTtime = sci.fft.fftfreq(int(np.round(time[-1]/T_step)), T_step)[i_t_start-1:]
            FFTx = sci.fft.fft(polarization[i_f][0,i_t_start:]-P[i_f,0,i_t_start:])
            FFTy = sci.fft.fft(polarization[i_f][1,i_t_start:]-P[i_f,1,i_t_start:])
            FFTz = sci.fft.fft(polarization[i_f][2,i_t_start:]-P[i_f,2,i_t_start:])
            FFTvalues=np.c_[FFTtime]
            FFTvalues=np.append(FFTvalues,np.c_[2/(int(np.round(time[-1]/T_step)))*np.abs(FFTx)],axis=1)
            FFTvalues=np.append(FFTvalues,np.c_[2/(int(np.round(time[-1]/T_step)))*np.abs(FFTy)],axis=1)
            FFTvalues=np.append(FFTvalues,np.c_[2/(int(np.round(time[-1]/T_step)))*np.abs(FFTz)],axis=1)
            FFToutput_file='o.YamboPy-FFT_difference_F'+str(i_f+1)
            np.savetxt(FFToutput_file,FFTvalues,header=FFT_header,delimiter=' ',footer=FFT_footer)
    
    # Print the result
    for i_v in range(V_size):
        i_order1, i_order2 = mapping[i_v]
        if i_order1==0 and i_order2==0: 
            Unit_of_Measure = SVCMm12VMm1/AU2VMm1
        else:
            Unit_of_Measure = np.power(SVCMm12VMm1/AU2VMm1,abs(i_order1)+abs(i_order2)-1,dtype=np.double)
            Susceptibility[i_v,:,:]=Susceptibility[i_v,:,:]*Unit_of_Measure
            output_file='o.YamboPy-SF_LSF_probe_order_'+str(i_order1)+'_'+str(i_order2)
            if i_order1 == 0 or (i_order1 == 1 and i_order2 == 0) or (i_order1 == 0 and i_order2 == 1):
                header="E [eV]            X/Im(x)            X/Re(x)            X/Im(y)            X/Re(y)            X/Im(z)            X/Re(z)            Optimality(x)            Optimality(y)            Optimality(z)"
            else:
                header="[eV]            "
                header+="X/Im[cm/stV]^%d     X/Re[cm/stV]^%d     " % (abs(i_order1)+abs(i_order2)-1,abs(i_order1)+abs(i_order2)-1)
                header+="X/Im[cm/stV]^%d     X/Re[cm/stV]^%d     " % (abs(i_order1)+abs(i_order2)-1,abs(i_order1)+abs(i_order2)-1)
                header+="X/Im[cm/stV]^%d     X/Re[cm/stV]^%d     " % (abs(i_order1)+abs(i_order2)-1,abs(i_order1)+abs(i_order2)-1)
                header+="Optimality(x)     "
                header+="Optimality(y)     "
                header+="Optimality(z)     "
            values=np.c_[freqs*ha2ev]
            values=np.append(values,np.c_[Susceptibility[i_v,:,0].imag],axis=1)
            values=np.append(values,np.c_[Susceptibility[i_v,:,0].real],axis=1)
            values=np.append(values,np.c_[Susceptibility[i_v,:,1].imag],axis=1)
            values=np.append(values,np.c_[Susceptibility[i_v,:,1].real],axis=1)
            values=np.append(values,np.c_[Susceptibility[i_v,:,2].imag],axis=1)
            values=np.append(values,np.c_[Susceptibility[i_v,:,2].real],axis=1)
            values=np.append(values,np.c_[Optimality[i_v,:,0].real],axis=1)
            values=np.append(values,np.c_[Optimality[i_v,:,1].real],axis=1)
            values=np.append(values,np.c_[Optimality[i_v,:,2].real],axis=1)
            footer='Non-linear response analysis performed using YamboPy'
            if prn_Xhi:
                np.savetxt(output_file,values,header=header,delimiter=' ',footer=footer)

    return Susceptibility,freqs
