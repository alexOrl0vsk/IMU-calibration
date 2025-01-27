import sys
import os
import re
import pandas as pd
import numpy as np
import scipy as sp
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def getPolyReg(X,Y, order: int):
	''' Polynomial regression 
		@param 	X 		-- input values
		@param 	Y 		-- true output values
		@param order	-- order of regressor 
		returns the fitted predicted output 
	'''
	poly = PolynomialFeatures(degree=order, include_bias=False)
	poly_features = poly.fit_transform(np.array(X).reshape(-1, 1))
	poly_reg_model = LinearRegression()
	poly_reg_model.fit(poly_features, Y)
	y_predicted = poly_reg_model.predict(poly_features)
	r2 = r2_score(Y,y_predicted)
	print(f"poly_reg_order_{order}\tR2_score :: {r2:.4f}\tcoeffs :: {', '.join(f'{x:.8f}'for x in poly_reg_model.coef_)}\t\t intercept :: {poly_reg_model.intercept_:.8f}")
	return y_predicted

def plotData(df,which,saved_name=''):
	"""	Plots the all_raw_packet data. If doesn't work check the column indices. 
		@param 	df 			- dataframe
		@param 	which 		- 'gyro', 'acc', 'mag'
		@param 	saved_name 	- dataframe name
	"""
	match which:
		case 'gyro':
			xx, yy, zz, tt = 1, 2, 3, 4 
			plot_title = 'Raw gyro data'
			y_label = "Angular velocity [-]"
		case 'acc':
			xx, yy, zz, tt = 5, 6, 7, 8 
			plot_title = 'Raw accelerometer data'#R
			y_label = "Specific force [-]"
		case 'mag':
			xx, yy, zz, tt = 9, 10, 11, 12 
			plot_title = 'Raw magnetometer data'
			y_label = "Magnetic field [-]"
		case _: 
			print("Wrong option : acc, gyro, mag")
			return  
	temps = df.iloc[:,13]
	plt.figure(figsize=(8,6))
	plt.plot(df.iloc[:,tt],df.iloc[:,xx],color='r',linewidth=0.5,marker= '.' ,markersize=1,label = 'X-axis')		
	plt.plot(df.iloc[:,tt],df.iloc[:,yy],color='g',linewidth=0.5,marker= '.' ,markersize=1,label = 'Y-axis')		
	plt.plot(df.iloc[:,tt],df.iloc[:,zz],color='b',linewidth=0.5,marker= '.' ,markersize=1,label = 'Z-axis')	
	plt.legend(loc='upper right')
	plt.xlabel("Time [s]")
	plt.ylabel(y_label)
	plt.title(f'{saved_name}, {plot_title}')
	#plt.annotate(f'Mean T : {np.mean(temps):.2f} [C],   Min T : {min(temps):.2f} [C],   Max T : {max(temps):.2f} [C]' ,
	#			xy=(0.05, 0.95), xycoords='figure fraction')
	plt.grid(True)	
	#plt.savefig(f'{saved_name}.png', format='png', dpi=300)
	plt.show()

def plotPID(pid_df,title,save: bool = False):
	""" Plots the pid control signal vs IMU temperature
		@param pid_df -- dataframe with PID data
		@param title  -- title for the plot
		@param save   -- saves the plot
	"""
	fig, ax = plt.subplots(2,1 ,figsize=(8,6))
	ax[0].plot(pid_df['Time'],pid_df['Temperature'] , label='measured',color='b')
	ax[0].plot(pid_df['Time'], pid_df['Setpoint'], label='target',linestyle='-.',color='r')
	ax[0].set_ylabel('Temperature [deg C]')	
	ax[0].grid()
	ax[0].legend()
	ax[1].plot(pid_df['Time'], pid_df['Input_voltage'], label='input signal',color='r')
	ax[1].set_xlabel('Time [s]')
	ax[1].set_ylabel('Voltage [V]')
	ax[1].legend()
	ax[1].grid()
	plt.suptitle(f'PID controller {title}')
	if(save):
		plt.savefig(f'{title}_pid.png', format='png', dpi=300)
	plt.show()	

def all_pids(root):
	""" Opens all PID plots from a root folder containing other folder with actual data"""
	print(root)
	for folder in os.listdir(root):
		path = os.path.join(root, folder)
		if os.path.isdir(path) and folder != "IGNORE":
			for file in os.listdir(path):
				if file == 'pid_data.csv':
					df = pd.read_csv(os.path.join(path,file))
					df = df[df['Setpoint'] == int(folder[-2:])]		# for gyro_X data the controller history is overlapping due to a mistake (history was not reset)
					plotPID(df,folder,save=False)

def plotAccCalibRes(raw_scaled_data,A,b,plot_name):
	"""	Plots the results of accelerometer scalar calibration (magnitudes)
		@param 	raw_scaled_data - non calibrated data in [g]
		@param 	A - cholesky factor of matrix mapping a sphere to an ellipsoid
		@param 	b - bias vector
		@param 	plot_name - name for the plots
	"""
	data_calib = A @ (raw_scaled_data - b.reshape(3,-1))		# calibrated data
	mag_data_calib = np.sqrt(np.sum(data_calib**2, axis=0))		# absolute magnitude of calibrated data

	# plot the measurement sphere (raw vs. calibrated data)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(raw_scaled_data[0, :], raw_scaled_data[1, :], raw_scaled_data[2, :], color='r', label='Raw data')
	ax.scatter(data_calib[0, :], data_calib[1, :], data_calib[2, :], color='g', label='Calibrated data')
	ax.legend()
	plt.title(f"{plot_name}, Raw vs. calibrated magnitudes")
	ax.set_xlabel('A_x [g]')
	ax.set_ylabel('A_y [g]')
	ax.set_zlabel('A_z [g]')
	ax.set_box_aspect([1, 1, 1])
	#plt.savefig(f'{plot_name}_calib_vs_raw_mags.png', format='png', dpi=300)
	#plt.show()

	# plot the error (calibrated data vs reference magnitude)
	refAcc = 1
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	nPnts = data_calib.shape[1]
	err = mag_data_calib - refAcc
	col = (1 + err / np.max(np.abs(err))) / 2
	for iP in range(nPnts):
		ax.scatter(data_calib[0, iP], data_calib[1, iP], data_calib[2, iP], 
	               color=(col[iP], 0.1, 0))
	ax.set_box_aspect([1, 1, 1])
	    
	# reference sphere
	u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
	xs = refAcc * np.cos(u) * np.sin(v)
	ys = refAcc * np.sin(u) * np.sin(v)
	zs = refAcc * np.cos(v)
	ax.plot_surface(xs, ys, zs, color='g', alpha=0.1)
	plt.title(f'{plot_name}, Calibrated data on the unit sphere')
	ax.set_xlabel('A_x [g]')
	ax.set_ylabel('A_y [g]')
	ax.set_zlabel('A_z [g]')
	#plt.show()

	fig2 = plt.figure()
	plt.hist(mag_data_calib, bins=40,color='purple',edgecolor = "black")
	plt.ticklabel_format(useOffset=False)
	#plt.grid()
	plt.xlabel('A magnitude [g]')
	plt.ylabel('Number of samples [-]')
	plt.title(f"{plot_name}, Calibration results histrogram")
	#plt.show()

	np.set_printoptions(precision=5, suppress=True)
	print(f"\nA_cal_mag (mu): {np.mean(mag_data_calib)}")
	print(f"A_cal_mag (sigma): {np.std(mag_data_calib)}")
	print(f"A_cal_mag (CoV): {np.std(mag_data_calib) / np.mean(mag_data_calib)}")
	K=np.diag(np.sqrt(np.diag(np.linalg.inv(A.T @ A))))
	T = np.linalg.inv(A @ K)
	print("\nT  is \n",T )
	a12 = np.arccos(T[0,1] * T[1,1] + T[0,2] * T[1,2])	
	a23 = np.arccos(T[1,2] )
	a13 = np.arccos(T[0,2])
	print(f"\nScaling factors :: \n{K}")
	print(f"\nBias :: {b}")
	print(f"\nEstimated angles between the axes in [rad] :: a12 = {a12:.3f}\ta13 = {a13:.3f}\ta23 = {a23:.3f}\n")
	# error is 90 - deg(a12)

	fig3 = plt.figure()
	plt.ticklabel_format(useOffset=False)
	plt.plot(mag_data_calib, 'b-',marker='o',markersize=1.75)
	plt.title(f"{plot_name}, Calibrated data absolute magnitude")
	plt.grid()
	plt.xlabel('Sample number')
	plt.ylabel('A magnitude [g]')
	plt.show()


def plotSampleTimes(df,plot_name):
	"""	Plots the gyro sampling periods
		@param 	df 			-- dataframe with all_raw_packets
		@param 	plot_name 	-- name for plot title 
	"""
	dts = np.diff(df['gyro_raw_time']) * 1000 		# in [ms]
	print(plot_name)
	print(f'Median period [ms]\t:: {np.median(dts)}')
	print(f'Mean period [ms]\t:: {np.mean(dts)}')
	print(f'STD [ms]\t\t:: {np.std(dts,ddof=1)}')

	plt.figure(figsize=(8,6))
	plt.grid()
	plt.scatter(df['gyro_raw_time'][:-1],dts,s=1 )
	plt.title(f'{plot_name[:3]} Hz, {plot_name[-10:-4]} baud, Sample periods')
	plt.ylabel('Sampling interval [ms]')
	plt.xlabel('Time [ms]')
	plt.hlines(
		y=np.mean(dts),
		xmin=min(df['gyro_raw_time']),
		xmax=max(df['gyro_raw_time']),
		linewidth=1, color='g',label='Mean sample period'
	)
	plt.hlines(
		y=np.median(dts),
		xmin=min(df['gyro_raw_time']),
		xmax=max(df['gyro_raw_time']),
		linewidth=1, color='r',label='Median sample period'
	)
	plt.legend()
	plt.figure()
	plt.grid()
	plt.hist(1000*dts,bins=20)
	plt.title(f'{plot_name[:3]} Hz, {plot_name[-10:-4]} baud, Sample period histogram')
	plt.ylabel('Samples [-]')
	plt.xlabel('Sample time [ms]')
	#plt.savefig(f'{plot_name}.png', format='png', dpi=1200)
	plt.show()

