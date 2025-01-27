from utils import *
from accel_processing import *
from gyro_processing import *
from allan import *

def run_sample_times():
	""" Plots the sampling periods at selected settings (20000 samples)."""
	directory = './UM7_data/baud_v_sample'
	for file in os.listdir(directory):
		if file.endswith(".csv"):
			plotSampleTimes(pd.read_csv(os.path.join(directory, file)),file)

def run_all_accel_results():	
	""" Plot accelerometer calibration results, choose one experiment (fan or no fan)."""
	#directory = './UM7_data/accel_data/accel_with_fan/accel_processed'		# with fan 
	directory = './UM7_data/accel_data/accel_no_fan/accel_processed'		# NO fan 
	plot_accel_results(directory)

def run_all_gyro_results():
	""" Plot gyro calibration results, bias calls the separate folders, scales/cross-factors call the same data sorted by temperature."""
	path = Path('./UM7_data/gyro_data/GYRO_X/X_ax_cool_gyro_job__temp__40')
	plot_gyro_stage_one_axis(path,'x')				# stage vs sensor data

	rootX = './UM7_data/gyro_data/GYRO_X'
	rootY = './UM7_data/gyro_data/GYRO_Y'
	rootZ = './UM7_data/gyro_data/GYRO_Z'
	roots = [rootX, rootY, rootZ]
	plot_gyro_biases(roots)
	directory = './UM7_data/gyro_data/GYROS_3'		# same data as in roots, sorted by temperature
	plot_gyro_scales(directory)
	plot_gyro_off_diag(directory)


def run_allan():
	""" Gyro Allan variance plot without extreme outliers."""
	df = pd.read_csv(f'./UM7_data/allan_100_Hz.csv',sep=";", on_bad_lines='warn' )		
	print("Before filtering :: ",df.shape)
	for data in [df['gyro_raw_x'],df['gyro_raw_y'],df['gyro_raw_z']]:
		Q3 = np.quantile(data, 0.75)
		Q1 = np.quantile(data, 0.25)
		IQR = Q3 - Q1 
		low = Q1 - 3*IQR
		high = Q3 + 3*IQR
		df = df.where((data > low ) & (data < high)).dropna()
	print("After filtering :: ",df.shape)
	#print("Mead sampling period :: ", np.mean(df['gyro_raw_time'])))
	print("Median sampling period :: ", np.median(np.diff(df['gyro_raw_time'])))
	print("1 // Median sampling period :: ", 1//np.median(np.diff(df['gyro_raw_time'])))
	resample_f = int(1//np.median(np.diff(df['gyro_raw_time'])))
	allanPlot(df,resample_f,f'gyro Allan resampled at {resample_f}')

def run_all_pid_plots():
	""" Controller input signal vs IMU temperature plots, choose accel or gyro roots."""
	rootX = './UM7_data/gyro_data/GYRO_X'
	rootY = './UM7_data/gyro_data/GYRO_Y'
	rootZ = './UM7_data/gyro_data/GYRO_Z'

	root_accel_fan = './UM7_data/accel_data/accel_with_fan'
	root_accel_no_fan = './UM7_data/accel_data/accel_no_fan'
	
	roots_gyro = [rootX, rootY, rootZ]
	roots_accel = [root_accel_fan, root_accel_no_fan]
	
	for root in roots_gyro:
		all_pids(root)


def main():
	#run_allan()					# gyro Allan variance plot
	#run_all_pid_plots()				# all pid plots, choose gyro or accel roots
	#run_all_accel_results()			# accelerometer calibration, uncomment one of directories (fan or no fan)
	#run_all_gyro_results()				# gyro results: biases, scales, cross-factors

if __name__ == '__main__':
	main()
