import os
import re
import pandas as pd
import numpy as np
import scipy as sp
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from utils import * 

def OLS(X):
	""" Ordinary Least Squares ellipsoid fitting.
		@param X  -- data in column vectors
		returns the ellipsoid parameters A_e and c (scales and offset)
	"""
	X = X.reshape(3,-1)					# column vectors
	D = np.zeros((X.shape[1],10))		# design matrix (N x 10)
	# D_i = [x^2 	x*y 	y^2 	xz 		yz		z^2 	x 	y 	z 	1]
	for i in range(0,X.shape[1]):
		D[i,:] = [X[0,i]**2 , 2*X[0,i] * X[1,i], X[1,i]**2 ,
				  2*X[0,i] * X[2,i] , 2*X[1,i] * X[2,i] , X[2,i]**2 ,
				  X[0,i] , X[1,i] , X[2,i] , 1 ]

	U, s, Vh = sp.linalg.svd(D)
	beta = Vh[-1,:]
	norm = np.linalg.norm(beta,axis=0)
	beta_norm = beta / norm	
	A = np.array([											
		[beta_norm[0], beta_norm[1],beta_norm[3]],
		[beta_norm[1], beta_norm[2],beta_norm[4]],
		[beta_norm[3], beta_norm[4],beta_norm[5]]
	])									# sym pos def matrix 	
	b = beta_norm[ 6 : 9]				# n-dimensional b-vector (offset from origin)
	d = beta_norm[-1]					# scalar 'radius' value 
	c = -0.5 * np.linalg.inv(A) @ b 
	A_e = A / (c.T @ A @ c - d )
	return A_e, c

def accel_get_averaged(data):
	''' Take averages for the accelerometer experiment.
		@param 	data  --  pandas dataframe with all_raw_data packets 
		returns the averaged data as a dataframe 
	'''
	dts = np.diff(data['accel_raw_time'])
	borders = np.where(dts >(np.mean(dts) + np.std(dts)))[0]	
	slices = []
	start = 0
	for border in borders:
		#print(start, " to ",border + 1)
		slices.append(data.iloc[start : border + 1,:])  # from start to border (last index not included)
		start = border + 1  							# move start 
	slices.append(data.iloc[start:,:])					# last piece to the end 
	df_means = pd.DataFrame(columns=['accel-mean-x','accel-mean-y','accel-mean-z'])
	for slic in slices:
		new_row = pd.Series({	
			'accel-mean-x' : np.mean(slic['accel_raw_x']),
			'accel-mean-y' : np.mean(slic['accel_raw_y']),
			'accel-mean-z' : np.mean(slic['accel_raw_z']),
		})
		df_means = pd.concat([df_means , new_row.to_frame().T], ignore_index=True)
	#print(df_means)
	return df_means

def accel_processing(directory):
	""" Saves the averaged accelerometer data to csv in a separate folder.
		@ param  directory	-- path to root directory containing folders with accel_data.csv files (raw data), 
		last two characters of folder are the temperature.
	"""
	for folder in os.listdir(directory):
		path = os.path.join(directory,folder)
		for file in os.listdir(path):
			if file.endswith("accel_data.csv"):		# files should be named like this 	
				temp = folder[-2:]					# last two characters of folder name should be temperature
				df_raw = pd.read_csv(os.path.join(path,file))
				averaged_df = accel_get_averaged(df_raw)
				savepath = Path(directory + 'accel_processed/averaged_' + temp +'.csv')
				savepath.parent.mkdir(parents=True, exist_ok=True)
				averaged_df.to_csv(savepath,index=False)

def accel_average_check(directory):
	""" Plots what happens in accel_processing 
		@ param  directory	-- path to root directory containing folders with accel_data.csv files (raw data),
		last two characters of folder are the temperature.
	"""
	for folder in os.listdir(directory):
		temp = folder[-2:]
		path = os.path.join(directory,folder)
		if os.path.isdir(path) and path != "IGNORE":
			for file in os.listdir(path):
				if file.endswith("accel_data.csv"):
					data = pd.read_csv(os.path.join(path, file))	
					dts = np.diff(data['accel_raw_time'])
					borders = np.where(dts >(np.mean(dts) + np.std(dts)))[0]
					slices = []
					start = 0
					for border in borders:
						#print(start, " to ",border + 1)
						slices.append(data.iloc[start : border + 1,:])  
						start = border + 1  
					slices.append(data.iloc[start:,:])		# last piece to the end
					plt.figure(figsize=(8,6))
					for slic in slices:
						plt.plot(slic['accel_raw_time'],slic['accel_raw_x'],color='r',linewidth=0.5)
						plt.plot(slic['accel_raw_time'],slic['accel_raw_y'],color='g',linewidth=0.5)
						plt.plot(slic['accel_raw_time'],slic['accel_raw_z'],color='b',linewidth=0.5)
						plt.hlines(np.mean(slic['accel_raw_x']),min(slic['accel_raw_time']),max(slic['accel_raw_time']),color='k',linestyles='-')
						plt.hlines(np.mean(slic['accel_raw_y']),min(slic['accel_raw_time']),max(slic['accel_raw_time']),color='k',linestyles='--')
						plt.hlines(np.mean(slic['accel_raw_z']),min(slic['accel_raw_time']),max(slic['accel_raw_time']),color='k',linestyles='-.')
						plt.title(f"{temp} [deg C], Segmented raw data : {len(slices)} pieces")
						plt.xlabel("Time [s]")
						plt.ylabel('Specific force [-]')
						plt.legend(['AccX','AccY','AccZ'])
					plt.grid()
					#plt.savefig(f'segmented_data.png', format='png', dpi=300)		
					plt.show()
					plotData(data,'acc',saved_name=f'{temp} [deg C]')

def plot_accel_results(directory):
	""" Plots the results of accelerometer thermal calibration"""
	temps = []
	params_for_temps = {}
	for file in os.listdir(directory):
		if file.endswith(".csv"):
			temps.append(int(file[-6:-4]))
			data = pd.read_csv(os.path.join(directory, file))
			data = data.to_numpy().T 					# column vectors
			scaled_data = data * 8.0 / 32768.0 
			#print(data)
			A_e, b = OLS(scaled_data)		
			params_for_temps[file[9:-4]] = {			# last 2 digits is temperature
				'params': (A_e, b),						# (A,b) tuple 
				'raw_data': scaled_data
			}
			#A = np.linalg.cholesky(A_e)
			#plotAccCalibRes(scaled_data,A,b,f'{temps[-1]} [deg C]')
	x_bias, y_bias, z_bias = [],[],[]
	x_scale, y_scale, z_scale = [],[],[]
	error_a12, error_a23, error_a13 = [],[],[]
	all_raw_mags , all_calb_mags = [], []
	save_calib_res = {"Temp":[], "Raw_RMSE" : [], "Calib_RMSE" : [],"Calib_mag_mean" : [], "Calib_mag_std" : [],"Calib_mag_CoV" : []}	
	for key in params_for_temps:								# key is temp
		print(f"\n-----------------{key}-----------------")
		raw_data = params_for_temps[key]['raw_data']
		A_e , b = params_for_temps[key]['params']
		A = sp.linalg.cholesky(A_e)
		K = np.diag(np.sqrt(np.diag(np.linalg.inv(A_e))))		# from Bonnet 2009 page 2
		T = np.linalg.inv(A @ K)	
		a12 = np.arccos(T[0,1] * T[1,1] + T[0,2] * T[1,2])	
		a23 = np.arccos(T[1,2] )
		a13 = np.arccos(T[0,2])

		x_bias.append(b[0])
		y_bias.append(b[1])
		z_bias.append(b[2])
		x_scale.append(K[0,0])
		y_scale.append(K[1,1])
		z_scale.append(K[2,2])
		
		error_a12.append(90 - a12*180/np.pi)
		error_a23.append(90 - a23*180/np.pi)
		error_a13.append(90 - a13*180/np.pi)
		
		print(f"\nEstimated angles between the sensor frame axes in [rad] :: a12 = {a12:.4f}\ta13 = {a13:.4f}\ta23 = {a23:.4f}")
		print(f"Misalignment error between the axes in [deg] :: a12 = {error_a12[-1]}\ta13 = {error_a13[-1]}\ta23 = {error_a23[-1]}\n")

		data_calibrated = A @ (raw_data - b.reshape(3,-1))
		calib_abs_magnitude = 1000 * np.sqrt(np.sum(data_calibrated**2, axis=0))	# absolute magnitudes of calibrated data in [mg]
		raw_abs_magnitude = 1000 * np.sqrt(np.sum(raw_data**2, axis=0))			# absolute magnitudes of raw data in [mg]
		
		raw_mean_mag = np.mean(raw_abs_magnitude)
		calib_mean_mag = np.mean(calib_abs_magnitude)
		calib_mag_std = np.std(calib_abs_magnitude)
		calib_mag_CoV = calib_mag_std / calib_mean_mag

		print(f"\nA_cal_mag (mu): {calib_mean_mag}")
		print(f"A_cal_mag (sigma): {calib_mag_std}")
		print(f"A_cal_mag (CoV): {calib_mag_CoV}")

		raw_RMSE = np.sqrt(np.mean((1000 - raw_abs_magnitude)**2))
		calib_RMSE = np.sqrt(np.mean((1000 - calib_abs_magnitude)**2))
		refs = np.full_like(raw_abs_magnitude,1000)
		print(f"\nRaw data MANUAL RMSE :: {raw_RMSE}\t\t SKLEARN :: {mean_squared_error(refs,raw_abs_magnitude)**0.5}")
		print(f"Calibrated MANUAL RMSE :: {calib_RMSE}\t\t SKLEARN :: {mean_squared_error(refs,calib_abs_magnitude)**0.5}")

		save_calib_res["Temp"].append(key)
		save_calib_res["Raw_RMSE"].append(raw_RMSE)						# all in [mg]
		save_calib_res["Calib_RMSE"].append(calib_RMSE)					
		save_calib_res["Calib_mag_mean"].append(calib_mean_mag)
		save_calib_res["Calib_mag_std"].append(calib_mag_std)
		save_calib_res["Calib_mag_CoV"].append(calib_mag_CoV)

	#pd.DataFrame(save_calib_res).to_csv('accel_calib_magnitudes.csv')		# calibration results
	
	# plot magnitude resitual RMSE 
	fig, ax1 = plt.subplots(figsize=(8,6))
	plt.xticks(np.arange(min(temps), max(temps)+1, step=1)) 
	color = 'tab:blue'
	ax1.set_xlabel('Temperature [deg C]')
	ax1.set_ylabel('RMSE [mg]')	#, color=color
	line1 = ax1.plot(temps, save_calib_res["Raw_RMSE"],marker='.', color=color,label='Raw data')
	ax1.tick_params(axis='y', labelcolor=color)
	ax2 = ax1.twinx()  
	color = 'tab:red'
	ax2.set_ylabel('RMSE [mg]')	#, color=color  
	line2 = ax2.plot(temps, save_calib_res["Calib_RMSE"],marker='.', color=color,label='Calibrated data ')
	ax2.tick_params(axis='y', labelcolor=color)
	ax1.grid(True)
	plt.title('Accelerometer magnitude residual RMSE')
	lines = line1 + line2
	labels = [l.get_label() for l in lines]
	ax2.legend(lines, labels,loc='center right')	#
	#plt.savefig('NO_FAN_AccelerometerRMSE.png', format='png', dpi=300) 
	fig.tight_layout()  
	#plt.show()

	# plot raw vs calibrated mean magnitudes 
	# plt.figure(figsize=(8,8))
	# # plt.plot(save_calib_res["Temp"],save_calib_res["Raw_mag_mean"],c='k',label='Raw data')
	# # plt.plot(save_calib_res["Temp"],save_calib_res["Calib_mag_mean"],c='r',label='Calibrated data')
	# plt.boxplot([all_raw_mags,all_calb_mags] ,labels=["mag_raw", "mag_calb"])
	# plt.xlabel('Temperature [deg C]')
	# plt.ylabel('Mean magnitude [mg]')
	# plt.grid()
	# plt.title('Accelerometer mean magnitudes')
	# plt.legend()
	#plt.savefig('Accelerometer mean magnitudes PID.png', format='png', dpi=600)
	
	# plot accelerometer bias vs temperature
	plt.figure(figsize=(8,8))
	print("Bias regression (AccX,AccY,AccZ) [g/deg C]")
	for Y,lab,co, in zip([x_bias, y_bias, z_bias],['AccX','AccY','AccZ'],['r','g','b']):
		y_predicted = getPolyReg(temps,Y,1)
		plt.scatter(temps,Y,c=co,marker='d')
		plt.plot(temps,y_predicted,label=lab,c=co)
	plt.xlabel('Temperature [deg C]')
	plt.ylabel('Bias [g]')
	plt.grid()
	plt.title('Accelerometer Bias')
	plt.legend()
	plt.xticks(np.arange(min(temps), max(temps)+1, step=1)) 
	#plt.savefig('AccelerometerBiases.png', format='png', dpi=300)
	
	# plot accelerometer scale factors vs temperature	
	plt.figure(figsize=(8,8))
	print("Scales regression (AccX,AccY,AccZ) [-/deg C]")
	for Y,lab,co, in zip([x_scale, y_scale, z_scale],['AccX','AccY','AccZ'],['r','g','b']):
		y_predicted = getPolyReg(temps,Y,1)
		plt.scatter(temps,Y,c=co,marker='d')
		plt.plot(temps,y_predicted,label=lab,c=co)
	plt.xlabel('Temperature [deg C]')
	plt.ylabel('Scale factor [-]')
	plt.grid()
	plt.title('Accelerometer Scale factors')
	plt.legend()
	plt.xticks(np.arange(min(temps), max(temps)+1, step=1)) 
	#plt.savefig('AccelerometerScales.png', format='png', dpi=300)
	
	# plot accelerometer misalignment angles vs temperature	
	plt.figure(figsize=(8,8))
	print("Misalignment regression (XY,YZ,XZ) [deg/deg C]")
	for Y,lab,co, in zip([error_a12, error_a23, error_a13],['XY','YZ','XZ'],['r','g','b']):
		y_predicted = getPolyReg(temps,Y,1)
		plt.scatter(temps,Y,c=co,marker='d')
		plt.plot(temps,y_predicted,label=lab,c=co)
	plt.xlabel('Temperature [deg C]')
	plt.ylabel('Misalignment angle [deg]')
	plt.grid()
	plt.title('Accelerometer misalignment angles')
	plt.legend()
	plt.xticks(np.arange(min(temps), max(temps)+1, step=1)) 
	#plt.savefig('AccelerometerMisalignment.png', format='png', dpi=300)
	plt.show()

