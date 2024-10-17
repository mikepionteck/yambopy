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
from scipy.optimize import minimize
import scipy.linalg
import sys
import os

def required_sampling_points(frequency1_eV, frequency2_eV, sampling_time_fs, safety_factor=1):

    h = 4.135667696e-15  # Planck's constant in eV·s

    # Convert eV to Hz
    frequency1_Hz = frequency1_eV / h
    frequency2_Hz = frequency2_eV / h
    
    # Identify the maximum frequency
    max_frequency_Hz = max(frequency1_Hz, frequency2_Hz)
    
    # Calculate the Nyquist rate
    nyquist_rate = 2 * max_frequency_Hz  # Nyquist criterion

    # Apply safety factor
    sampling_rate = safety_factor * nyquist_rate

    # Convert sampling time from fs to seconds
    sampling_time_s = sampling_time_fs * 1e-15  # Convert fs to seconds
    
    # Calculate the number of sampling points
    num_sampling_points = int(np.ceil(sampling_rate * sampling_time_s))
    
    return num_sampling_points
#
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
# Ridge regularization in nearly degenerate cases for least squares fitting
def LS_fit_diff_ridge(c,order,f1,f2,t,s,lambda_ridge):
    residual = LS_fit_diff(c,order,f1,f2,t,s)
    ridge = 0
    diff = abs(f1*ha2ev-f2*ha2ev)
    if diff < 1e-2:
        ridge = lambda_ridge
    reg_term = ridge * c
    res = np.concatenate((residual, reg_term))
    return res
#
def Sampling(P,T_range,T_step,mesh,SAMP_MOD):
    i_t_start = int(np.round(T_range[0]/T_step))
    i_deltaT  = int(np.round((T_range[1]-T_range[0])/T_step)/mesh)

    # Memory allocation 
    P_i      = np.zeros(mesh, dtype=np.double)
    T_i      = np.zeros(mesh, dtype=np.double)
    Sample = np.zeros((mesh,2), dtype=np.double)
    # Calculation of  T_i and P_i
    if SAMP_MOD=='linear':
        for i_t in range(mesh):
            T_i[i_t] = (i_t_start + i_deltaT * i_t)*T_step #- efield["initial_time"]
            #print(efield["initial_time"]/fs2aut)
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
    h_eVs = 4.135667696e-15  # Planck's constant
    fundamental_time_period_fs = h_eVs / gcd_frequency_eV * 1e15  # Convert seconds to femtoseconds

    return gcd_frequency_eV, fundamental_time_period_fs
#
def find_coeff_LS(order,P,f1,f2,T_range,T_step,mesh,SAMP_MOD,xtol,gtol,ftol,lambda_ridge):
    # Number of Fourier coefficients
    N = 2*sum(range(order+2)) -1 + 2*sum(range(1+order%2,order,2))
    # Memory allocation
    c = np.zeros(N)
    c[1] = 1
    c[2*(order+1)] = 1
    M = int((N-1)/2+1)
    copt  = np.zeros(M,dtype=np.cdouble)
    # Sampling
    t = Sampling(P,T_range,T_step,mesh,SAMP_MOD)[:,0]
    s = Sampling(P,T_range,T_step,mesh,SAMP_MOD)[:,1]
    # Find the coefficients
    coeff = sci.optimize.least_squares(LS_fit_diff_ridge,c,args=(order,f1,f2,t,s,lambda_ridge),xtol=xtol,gtol=gtol,ftol=ftol)
    copt[0] = coeff.x[0]
    #print(coeff.optimality)
    #print(coeff.success)
    for ii in range(1,M):
        copt[ii] = 0.5*(coeff.x[2*(ii-1)+1] + 1j*coeff.x[2*(ii-1)+2])
    return copt, coeff.optimality
#
def LS_SF_Analysis(nldb, X_order=2,T_range=[-1, -1],prn_Peff=False,prn_FFT=False,prn_Fundamentals=False,prn_Xhi=True,SAMP_MOD='linear',safety_factor=1,xtol=1e-8,gtol=1e-15,ftol=1e-8,lambda_ridge=1e-8):
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

    for count, efield in enumerate(nldb.Efield):
        freqs[count]=efield["freq_range"][0]
    
    print("Frequency range of the first field : "+str(freqs[0]*ha2ev)+" - "+str(freqs[-1]*ha2ev)+" [eV] \b")

    print("Pump frequency : ",str(pump_freq*ha2ev),' [eV] ')

    if prn_Fundamentals:
        # Calculate the fundamental frequency and time period for each frequency
        print("Print fundamental frequency and time period for each frequency...")
        for i_f in range(len(freqs)):
            f, T = fundamental_frequency_and_time_period(freqs[i_f]*ha2ev, pump_freq*ha2ev)
            print("Fundamental frequency and time period for frequency %d: %.3f eV, %.3f fs" % (i_f+1, f, T))
        print("End of fundamental frequency and time period for each frequency...")

    if T_range[0] <= 0.0:
        T_range[0]=2.0/nldb.NL_damping*6.0
    if T_range[1] <= 0.0:
        T_range[1]=time[-1]

    period = T_range[1]-T_range[0]
    
    # Number of sampling points
    mesh = np.zeros(n_frequencies, dtype=np.int64)
    for i_f in range(n_frequencies):
        # In degenerate case use the maximum number of sampling points
        if abs(freqs[i_f]*ha2ev-pump_freq*ha2ev) < 1e-2:
            mesh[i_f] = T_range[1]/T_step
        else:
            mesh[i_f] = required_sampling_points(freqs[i_f]*ha2ev, pump_freq*ha2ev, period, safety_factor)
        print("Number of sampling points for frequency %d: %d" % (i_f+1, mesh[i_f]))
    #mesh = max(mesh_array)

    print("Initial time range : ",str(T_range[0]/fs2aut),'-',str(T_range[1]/fs2aut)," [fs] ")
    print("Minimum and maximum number of initial sampling points : ",str(min(mesh)),'-',str(max(mesh)))

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
    
    print("Loop in frequecies...")
    # Find the Fourier coefficients by inversion
    for i_f in tqdm(range(n_frequencies)):
        for i_d in range(3):
            X_effective[:,i_f,i_d],Optimality[:,i_f,i_d]=find_coeff_LS(X_order, polarization[i_f][i_d,:],freqs[i_f],pump_freq,T_range,T_step,mesh[i_f],SAMP_MOD,xtol,gtol,ftol,lambda_ridge)

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
        N=len(P[i_f,i_d,i_t_start:])
        for i_f in range(n_frequencies):
            valuesError[i_f,0]=freqs[i_f]*ha2ev
            for i_d in range(3):
        
                # Call the Sampling function once and store the result
                sampling_result = Sampling(polarization[i_f][i_d, :], T_range, T_step, mesh[i_f], SAMP_MOD)

                # Calculate the number of sampling points
                sampling_points = len(sampling_result[:, 0])

                # Populate plot_sampling with the result
                plot_sampling = np.zeros((sampling_points, 2), dtype=np.double)
                plot_sampling[:, :] = sampling_result

                # Print reconstructed polarization
                values=np.c_[time.real/fs2aut]
                values=np.append(values,np.c_[P[i_f,i_d,:].real],axis=1)
                output_file2='o.YamboPy-pol_reconstructed_F'+str(i_f+1)+'_D'+str(i_d+1)
                np.savetxt(output_file2,values,header=header2,delimiter=' ',footer=footer2)

                # Print sampling point
                valuesSampling=np.c_[plot_sampling[:,0]/fs2aut]
                valuesSampling=np.append(valuesSampling,np.c_[plot_sampling[:,1]],axis=1)
                output_file3='o.YamboPy-sampling_F'+str(i_f+1)+'_D'+str(i_d+1)
                np.savetxt(output_file3,valuesSampling,header=header2,delimiter=' ',footer=footerSampling)

                # Print error in reconstructed polarization in all frequencies
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
