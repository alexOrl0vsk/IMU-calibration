import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

def allanDeviation(dataArr,fs,maxNumM: int=100):
	""" (from Michael Wrona's Blog, based on matlab's implementaion), Computes the Allan deviation. 
		@param  dataArr 	-- data in a column vector 
		@param  fs 			-- sampling frequency
		@param  maxNumM 	-- number of samples for logspace (high values can't make plot more accurate, limited by the length of the recorded data)
		returns a tuple of time bins and Allan deviations 
	"""
	ts = 1/fs 
	N = len(dataArr)
	Mmax = 2**np.floor(np.log2(N / 2))
	M = np.logspace(np.log10(1), np.log10(Mmax), num=maxNumM)
	M = np.ceil(M)				# integer 
	M = np.unique(M)			# remove duplicates
	taus = M * ts				# cluster durations
	allanVar = np.zeros(len(M))
	for i, mi in enumerate(M):
		twoMi = int(2*mi)
		mi = int(mi)
		allanVar[i] = np.sum(
			(dataArr[twoMi:N] - (2.0 * dataArr[mi:N-mi]) + dataArr[0:N-twoMi])**2
		)
	allanVar /= (2.0 * taus**2) * (N - (2.0 * M))
	return taus, np.sqrt(allanVar)

def allanPlot(df,resample_freq,save_plot_name: str = ''):
	"""	Plots the allanDeviation results 
		@param df 				-- um7 pd dataframe
		@param resample_freq 	-- resampling frequency
		@param save_plot_name 	-- to saves the Allan plot
	"""
	scale_factor = 2000 / 32768.0
	fs = resample_freq				# raw data sampling period is non constant, resampling at this rate	
	time = df['gyro_raw_time']
	print("min t :: ",min(time))
	print("max t :: ",max(time))
	print("diff t :: ",max(time)-min(time))
	time_resampled = np.arange(min(time), max(time), 1/fs)
	# resample so fs is constant 
	x_resampled = np.interp(time_resampled,time,df['gyro_raw_x'] * scale_factor)
	y_resampled = np.interp(time_resampled,time,df['gyro_raw_y'] * scale_factor)
	z_resampled = np.interp(time_resampled,time,df['gyro_raw_z'] * scale_factor)
	# integrate angles 
	theta_x = np.cumsum(x_resampled) / fs
	theta_y = np.cumsum(y_resampled) / fs
	theta_z = np.cumsum(z_resampled) / fs
	# get deviations for each axis
	tausX, stdX = allanDeviation(theta_x,fs,1000)
	tausY, stdY = allanDeviation(theta_y,fs,1000)
	tausZ, stdZ = allanDeviation(theta_z,fs,1000)
	# in [deg/h]
	stdX *= 3600
	stdY *= 3600
	stdZ *= 3600
	# plot 
	plt.loglog(tausX,stdX,label='X-axis',c='r')	
	plt.loglog(tausY,stdY,label='Y-axis',c='g')
	plt.loglog(tausZ,stdZ,label='Z-axis',c='b')	
	plt.title(f'Gyroscope Allan deviation resampled at {fs} Hz')
	plt.grid(True, which="both",ls="dotted",lw=0.5, color='0.65')
	plt.xlabel(r'$\tau$ [sec]')
	plt.ylabel(r'$\sigma(\tau)$ [deg/h]')
	plt.axis('equal')
	plt.legend()
	#plt.savefig(f'{save_plot_name}', dpi=300)
	plt.show()

