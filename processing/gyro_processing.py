import sys
import os
import re
import pandas as pd
import scipy as sp
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from utils import * 

def gyro_sort_by_speed(folder):
	""" Removes the corresponding biases from all data in a single-temperature folder 
		(bias-x on day X from gyro_x data, bias-y on day X from gyro_y data, etc)
		@param 	folder 	-- 1 folder from GYROS_3
		return a nested dict with data sorted by speed. 
	"""
	data = []
	for file in os.listdir(folder):
		df = pd.read_csv(os.path.join(folder,file))
		vel = re.search(r"-?\d+", file).group(0)		# get signed decimals (velocity) values
		data.append({
			'sens_axis': file[0],
			'velocity': int(vel),
			'x_mean': df['gyro_raw_x'].mean() * 2000 / 32768.0,
			'y_mean': df['gyro_raw_y'].mean() * 2000 / 32768.0,
			'z_mean': df['gyro_raw_z'].mean() * 2000 / 32768.0,
		})
	df = pd.DataFrame(data)
	bias_df = df[df['velocity'] == 0]
	unbiased_data = []
	for axis in ['X', 'Y', 'Z']:
		bias_axis = bias_df[bias_df['sens_axis'] == axis]
		for index, row in df[df['sens_axis'] == axis].iterrows():
			unbiased_data.append({
				'sens_axis': row['sens_axis'],
				'velocity': row['velocity'],
				'x_mean': row['x_mean'],
				'y_mean': row['y_mean'],
				'z_mean': row['z_mean'],
				'x_unbiased': row['x_mean'] - bias_axis['x_mean'].values[0],
				'y_unbiased': row['y_mean'] - bias_axis['y_mean'].values[0],
				'z_unbiased': row['z_mean'] - bias_axis['z_mean'].values[0],
			})
	unbiased_df = pd.DataFrame(unbiased_data)
	#print(unbiased_df)
	velocities = [ 20, 40, 80, 100, 200]
	sorted_dict = {'velocity' : [], 'data': []}
	for velocity in velocities:
		sorted_dict['velocity'].append(velocity)
		data = unbiased_df.where((abs(unbiased_df['velocity']) == velocity) | (unbiased_df['velocity'] == 0 )).dropna()
		sorted_dict['data'].append(data.reset_index(drop=True))
	#print(sorted_list)
	return sorted_dict

def gyroLS(folder):
	""" Computes the least squares for each temprature point.
		@param 	folder 	-- 1 folder from GYROS_3
		returns the scale factor matrix, LS residual2 , and the calibrated data
	"""
	sorted_dict = gyro_sort_by_speed(folder)
	ref_matrix = []
	gyro_data = []
	for velocity, data in zip(sorted_dict['velocity'], sorted_dict['data']):
		speed = int(velocity)
		OMEGAS_ref = np.array([
			[speed, 0, 0, 
			-speed, 0, 0,
			0, speed, 0,
			0,-speed, 0, 
			0, 0, speed, 
			0, 0,-speed ]])
		ref_matrix.append(OMEGAS_ref)
		gyro_row = data[data['velocity'] != 0 ].iloc[:,-3:].to_numpy()
		gyro_data.append(gyro_row)

	ref_inputs = np.vstack(ref_matrix).reshape(-1,1)
	full_data = np.vstack(gyro_data)					# row vectors
	N = full_data.shape[0]
	Omegas = [] 
	for i in range(N):
		wx, wy, wz = full_data[i]
		Omega_slice= np.array([
			[wx, 0,  0,  wy, wz, 0 , 0 , 0 , 0 ],
			[0,  wy, 0,  0 , 0 , wx, wz, 0 , 0 ],
			[0,  0,  wz, 0 , 0 , 0 , 0 , wx, wy]
		])
		Omegas.append(Omega_slice)
	Omegas_full = np.vstack(Omegas)
	solution, residuals, rank, s = np.linalg.lstsq(Omegas_full, ref_inputs, rcond=None)	#lstsq(a,b) --> b = a @ x
	scales = np.array([[ solution[0],solution[3],solution[4]],
						[solution[5],solution[1],solution[6]],
						[solution[7],solution[8],solution[2]]]).reshape(3,3)
	
	data_calibrated = full_data @ scales.T 		# transpose because data in row vectors
	calib_abs_mag = np.sqrt(np.sum(data_calibrated**2, axis=1))
	#error = data_calibrated - ref_inputs.reshape(30,3)
	calib_df = pd.DataFrame({
			"RefX": ref_inputs.reshape(30,3)[:,0],
			"RefY": ref_inputs.reshape(30,3)[:,1],
			"RefZ": ref_inputs.reshape(30,3)[:,2],
			"CalbX" : data_calibrated[:,0],
			"CalbY" : data_calibrated[:,1],
			"CalbZ" : data_calibrated[:,2],
			"CalbMag" : calib_abs_mag
		})
	return scales, residuals, calib_df

def gyro_temperatures(directory):
	""" Computes gyroLS for each tempearture 
		@param  directory	-- path to root directory GYROS_3 containing gyro data folders, 
		last two characters of folder are the temperature.
		returns a list of dicts with calibration results
	"""
	full_data = []	
	for folder in os.listdir(directory):
		path = os.path.join(directory,folder)
		if os.path.isdir(path):
			scales, residuals, calibrated_data = gyroLS(path)
			data = {
				'Temperature' : folder[-2:],
				'Scales' : scales,
				'Fit_residuals' : residuals,
				'calibrated_data' : calibrated_data
			}
			full_data.append(data)
	
	return full_data

def plot_gyro_scales(directory):
	""" Plots the gyro scale factors against temperature
 		@param  directory	-- path to GYROS_3, 
 	"""
	full_data = gyro_temperatures(directory)
	# with pd.option_context('display.max_rows', None, 'display.max_columns', None,'display.width', 2000):
	# 	pprint(full_data)
	flat_data = []
	for point in full_data:
		temp = point['Temperature']
		fit = point['Fit_residuals']
		scaleX = point['Scales'][0,0]
		scaleY = point['Scales'][1,1]
		scaleZ = point['Scales'][2,2]
		flat_data.append({"Temperature": int(temp), "LS_residual": fit, "ScaleX": scaleX, "ScaleY": scaleY, "ScaleZ": scaleZ})
	scales_df = pd.DataFrame(flat_data).sort_values(by=['Temperature']).reset_index(drop=True)
	plt.figure(figsize=(10,8))
	plt.grid()
	r2_list = []
	print("Scale factors regression (GyroX,GyroY,GyroZ) [- /deg C]")
	for Y,lab,col, in zip([scales_df['ScaleX'], scales_df['ScaleY'], scales_df['ScaleZ']],['GyroX','GyroY','GyroZ'],['r','g','b']):
		X = scales_df['Temperature']
		y_predicted = getPolyReg(X,Y,1)
		r2_list.append(r2_score(Y,y_predicted))
		plt.scatter(X, Y, marker='d', color=col)
		plt.plot(X,y_predicted, color=col,label=lab)	
	print(scales_df)
	print('Regression R2 :: ',r2_list)
	plt.xlabel("Temperature [deg C]")
	plt.ylabel("Scale factor [-]")
	plt.title(f'Gyroscope scale factors ')
	plt.legend()
	#plt.savefig('Gyroscope scale factors,linreg.png', format='png', dpi=300)
	plt.show()

def plot_gyro_off_diag(directory):
	""" Plots the gyro cross-axis factors against temperature
 		@param  directory	-- path to GYROS_3, 
 	"""
	full_data = gyro_temperatures(directory)
	off_diag_data = []
	for point in full_data:
		temp = point['Temperature']
		off_diag_data.append({
			"Temperature": int(temp),
			"XY": point['Scales'][0,1],
			"XZ": point['Scales'][0,2],
			"YX": point['Scales'][1,0],
			"YZ": point['Scales'][1,2],
			"ZX": point['Scales'][2,0],
			"ZY": point['Scales'][2,1],
		})	
	scales_df = pd.DataFrame(off_diag_data).sort_values(by=['Temperature']).reset_index(drop=True)
	just_data = scales_df.drop(columns=['Temperature'])

	plt.figure(figsize=(10,8))
	plt.grid()
	r2_list = []
	print("Misalignment factors regression (XY,XZ,YX,YZ,ZX,ZY) [- /deg C]")
	for Y,lab,col, in zip(just_data.T.values,['GyroXY','GyroXZ','GyroYX','GyroYZ','GyroZX','GyroZY'],['r','g','b','m','c','k']):
		#print(Y)
		X = scales_df['Temperature']
		y_predicted = getPolyReg(X,Y,1)
		r2_list.append(r2_score(Y,y_predicted))
		plt.scatter(X, Y, marker='d', color=col)
		plt.plot(X,y_predicted, color=col,label=lab)
	
	#print(scales_df)
	#print('Regression R2 :: ',r2_list)
	plt.xlabel("Temperature [deg C]")
	plt.ylabel("Misalignment factor [-]")
	plt.title(f'Gyroscope misalignment error factors ')
	plt.legend(loc='lower right')
	#plt.savefig('Gyroscope misalignments,linreg.png', format='png', dpi=300)
	plt.show()
	
def plot_gyro_biases(roots):
	""" Plots the gyro biases against temperature
 		@param  roots	-- list of GYRO_X,Y,Z roots.
 	"""
	for root in roots:
		gyro_biases = {field : [] for field in ['Temperature','X_bias','Y_bias','Z_bias','gyro_x_std','gyro_y_std','gyro_z_std']}
		for folder in os.listdir(root):
			path = os.path.join(root, folder)
			if os.path.isdir(path) and path != "IGNORE":
				for file in os.listdir(path):
					#print(os.path.join(root, folder))
					if file == 'vel_0_gyro_data.csv':				# bias file should be named vel_0 
						df = pd.read_csv(os.path.join(path,file))
						#print(f"\t\t{root}")
						#print(df)
						# plt.figure(figsize=(8,6))
						# plt.plot(df['gyro_raw_x']* 2000 / 32768.0,label='GyroX',c='r')		#bias_df['Temperature'],
						# plt.plot(df['gyro_raw_y']* 2000 / 32768.0,label='GyroY',c='g')		#bias_df['Temperature'],
						# plt.plot(df['gyro_raw_z']* 2000 / 32768.0,label='GyroZ',c='b')		#bias_df['Temperature'],
						# plt.grid()
						# plt.xlabel('Sample [-]')
						# plt.ylabel('Bias [deg/s]')
						# plt.title(f'{path}')
						# plt.legend()
						# plt.show()
						gyro_biases['Temperature'].append(int(folder[-2:]))
						gyro_biases['X_bias'].append(np.mean(df['gyro_raw_x'] * 2000 / 32768.0 ))
						gyro_biases['Y_bias'].append(np.mean(df['gyro_raw_y'] * 2000 / 32768.0 ))
						gyro_biases['Z_bias'].append(np.mean(df['gyro_raw_z'] * 2000 / 32768.0 ))
						gyro_biases['gyro_x_std'].append(np.std(df['gyro_raw_x'] * 2000 / 32768.0 , ddof=1 ))	# sample standard deviation
						gyro_biases['gyro_y_std'].append(np.std(df['gyro_raw_y'] * 2000 / 32768.0 , ddof=1 ))
						gyro_biases['gyro_z_std'].append(np.std(df['gyro_raw_z'] * 2000 / 32768.0 , ddof=1 ))
		bias_df = pd.DataFrame(gyro_biases).sort_values(by='Temperature')
		print("-------------------------------------------")
		print(f"\t\t{root}")
		print(bias_df)
		print("-------------------------------------------")
		X = np.array(bias_df['Temperature'])
		plt.figure(figsize=(8,6))
		plt.grid()
		print("Bias regression (GyroX,GyroY,GyroZ) [deg/s /deg C]")
		for Y,lab,co, in zip([bias_df['X_bias'], bias_df['Y_bias'], bias_df['Z_bias']],['GyroX','GyroY','GyroZ'],['r','g','b']):
			y_predicted = getPolyReg(X,Y, 1)		# linear regression 
			plt.scatter(bias_df['Temperature'],Y,c=co,marker='d')
			plt.plot(bias_df['Temperature'],y_predicted,label=lab,c=co)
		plt.xlabel('Temperature [deg C]')
		plt.ylabel('Bias [deg/s]')
		plt.ylim([-0.7, 2])
		plt.title(f'Gyroscope Biases, rotation about {root[-1]}')
		plt.legend()
		#plt.savefig(f'Gyroscope Biases - sensing {root[-1]}, 5000 samples.png', format='png', dpi=300)
	plt.show()

def gyro_bias_drift(directory):
	""" Plots the gyro turn-on-turn-off bias drift. This data not is not in the archive due to being uninteresting.
 		@param  directory	-- path to root directory with gyro data
 	"""
	bias_df =  {field : [] for field in ['Temperature','X_bias','Y_bias','Z_bias']}
	for file in os.listdir(directory):
		if file.endswith('.csv'):
			df = pd.read_csv(os.path.join(directory,file))
			bias_df['Temperature'].append(np.mean(df['temperature']  ))

			bias_df['X_bias'].append(np.mean(df['gyro_raw_x'] * 2000 / 32768.0 ))
			bias_df['Y_bias'].append(np.mean(df['gyro_raw_y'] * 2000 / 32768.0 ))
			bias_df['Z_bias'].append(np.mean(df['gyro_raw_z'] * 2000 / 32768.0 ))

	print(f"Mean T :: {np.mean(bias_df['Temperature'])}")
	print(f"X_bias mean :: {np.mean(bias_df['X_bias'])}\t\tstd :: {np.var(bias_df['X_bias'],ddof=1)}")
	print(f"Y_bias mean :: {np.mean(bias_df['Y_bias'])}\t\tstd :: {np.var(bias_df['Y_bias'],ddof=1)}")
	print(f"Z_bias mean :: {np.mean(bias_df['Z_bias'])}\t\tstd :: {np.var(bias_df['Z_bias'],ddof=1)}")

	plt.figure(figsize=(8,6))
	plt.plot(bias_df['X_bias'],label='GyroX',c='r',marker='d')		# bias_df['Temperature'],
	plt.plot(bias_df['Y_bias'],label='GyroY',c='g',marker='d')		# bias_df['Temperature'],
	plt.plot(bias_df['Z_bias'],label='GyroZ',c='b',marker='d')		# bias_df['Temperature'],
	plt.grid()
	plt.xlabel('Sample [-]')
	plt.ylabel('Bias [deg/s]')
	plt.title(f'Gyroscope turn-on bias stability ')		# against temperature
	plt.legend()
	#plt.savefig(f'Gyroscope turn-on bias, 5000 samples_JAN_16.png', format='png', dpi=300)
	hist_bin_count = 40			# number of bins for histograms
	plt.figure(figsize=(5,4))
	plt.hist(bias_df['X_bias'],bins=hist_bin_count,color='red',edgecolor='black')		
	plt.xlabel('Bias [deg/s]')
	plt.ylabel('Count [-]')
	plt.title(f'Gyro X bias histogram')	
	#plt.savefig(f'Gyro_turn-on_hist_X, 5000 samples_JAN_16.png', format='png', dpi=300)
	plt.figure(figsize=(5,4))
	plt.hist(bias_df['Y_bias'],bins=hist_bin_count,color='green',edgecolor='black')		
	plt.xlabel('Bias [deg/s]')
	plt.ylabel('Count [-]')
	plt.title(f'Gyro Y bias histogram')	
	#plt.savefig(f'Gyro_turn-on_hist_Y, 5000 samples_JAN_16.png', format='png', dpi=300)
	plt.figure(figsize=(5,4))
	plt.hist(bias_df['Z_bias'],bins=hist_bin_count,color='blue',edgecolor='black')		
	plt.xlabel('Bias [deg/s]')
	plt.ylabel('Count [-]')
	plt.title(f'Gyro Z bias histogram')	
	#plt.savefig(f'Gyro_turn-on_hist_Z, 5000 samples_JAN_16.png', format='png', dpi=300)
	plt.show()

def plot_gyro_stage_one_axis(full_path,which):
	""" Plots the sensor (one axis) vs stage data for multiple velocities
		@param 	full_path	-- path to folder with sensor and stage data (e.g., X_ax_cool_gyro_job__temp__39)
		@param	which		-- 'x', 'y', or 'z'
	"""
	num_unique = len(os.listdir(full_path))//2
	colors = cm.tab20(np.linspace(0, 1,num_unique))
	col_id = 0
	fig, ax = plt.subplots(figsize=(10,6),nrows=1, ncols=2)
	for file in os.listdir(full_path):
		if (file.endswith('gyro_data.csv')):
			um7_df_one = pd.read_csv(f'{full_path}/{file}')
			match which:
				case 'x':
					data = um7_df_one['gyro_raw_x'] * 2000 / 32768.0 	# in [deg/s]
				case 'y':
					data = um7_df_one['gyro_raw_y'] * 2000 / 32768.0
				case 'z':
					data = um7_df_one['gyro_raw_z'] * 2000 / 32768.0
			cur_label = re.search(r"-?\d+", file).group(0)		# find signed decimal in file name (velocity)
			ax[0].plot(
				um7_df_one['gyro_raw_time'],
				data,
				color=colors[col_id%num_unique],		# plot sensor and stage with same color
				linewidth=0.5,marker= ',',markersize=0.55,
				label=f'{cur_label} deg/s '
			)
			print(f"CASE {cur_label} :: mean = {(np.mean(data)):.3f},\tvariance = {(data.var(ddof=1)):.4f},\tstd = {(data.std(ddof=1)):.4f}")
			ax[0].hlines(
				y=np.mean(data),
				xmin=min(um7_df_one['gyro_raw_time']),
				xmax=max(um7_df_one['gyro_raw_time']),
				linewidth=1, color='k'
			)
		if (file.endswith('stage_data.csv')):
			stage_df = pd.read_csv(f'{full_path}/{file}')
			ax[1].plot(
				stage_df['Time'],
				stage_df['Velocity'],
				color=colors[col_id%num_unique],
				linewidth=1,marker='d',markersize=2
			) 
			col_id += 1
	#ax[0].set_yticks([-200,-100,-80,-40,-20,0,20,40,80,100,200]) 		
	ax[0].grid(True)	
	ax[1].grid(True)	
	leg = ax[0].legend()
	for line in leg.get_lines():
		line.set_linewidth(2)
	ax[0].set_title(f'Gyro raw {which.upper()}-axis data [deg/s]')	
	ax[1].set_title('Stage data [deg/s]')
	fig.supxlabel('Time [s]')
	plt.suptitle(f'Angular velocity sensor vs stage data')
	#plt.savefig(f'Y_Gyro_Stage_data.png', format='png', dpi=300)
	plt.show()


