import os
import time 
import pandas as pd
from pathlib import Path
from threading import Thread,Event, Lock

from ka3005p import PowerSupply
from simple_pid import PID
# from rsl_comm_py import UM7Serial
# import libximc.highlevel as ximc

from um7 import Sensor
from korad import KoradPower
from rotaryStages import RotaryStage

class PowerController:
	def pidLoad(self, pid_mode:str, target_temp):
		"""	Loads the simple_pid controller, called in constructor 
			@param 	pid_mode 	-- 'cool' or 'heat'
			@param 	target_temp	-- target temperature for the controller  
			returns the pid controller object 
		"""
		match pid_mode:
			case 'cool':
				# default cooling PID setup
				pid = PID(Kp=-2.5, Ki=-0.005, Kd=-5, setpoint=target_temp)
				pid.output_limits = (0.5, 4)								# in [V]
				pid.sample_time = 5 										# update every 5 sec
				pid.starting_output = 3.5 									# good value to start from so we get to steady state faster
				return pid
			case 'heat':
				# default heating PID setup
				pid = PID(Kp=2.9, Ki=0.005, Kd=15, setpoint=target_temp)	
				pid.output_limits = (0, 3.5)								# in [V]		
				pid.sample_time = 3 										# update every 3 sec
				pid.differential_on_measurement = True						# use delta input for derivative term (library default)
				return pid
			case _:
				print(" * pidLoad() :: wrong controller mode choose 'cool' or 'heat'.")
				return None

	def __init__(self, pid_mode:str, target_temp, job_name):
		#self.power_supply = KoradPower()									# ka3005p 		PowerSupply object
		self.controller = self.pidLoad(pid_mode,target_temp)				# simple_pid 	PID object
		self.controller_mode = pid_mode										# 'cool' or 'heat'
		self.sensor = Sensor()												# um7py 		UM7Serial object
		self.rotary_stages = RotaryStage.loadAxes()							# static method, returns a list of stage objects		
		self.temp_error_tollerance = 0.005 									# 0.5% relative error allowed by default
		self.ready_for_job = Event()										# signal to start the experiment
		self.job_done = Event()												# signal to stop the PID and job
		self.lock = Lock()													# lock for thread safety
		self.controller_start_time = None 									# for saving the controller history
		self.job_name = job_name											# experiment data folder name 
		self.root = './'													# path for all saved data, job folders are placed here 
		self.controller_history = {											
			'Setpoint': [],
			'Temperature': [],
			'Time': [],
			'Input_voltage': []
		}

	def updateHistory(self,control_temp,new_voltage):
		""" Updates the PID history.
			@param	control_temp	-- last temperature recieved from sensor
			@param	new_voltage		-- newest input voltage from PID
		"""
		self.controller_history['Setpoint'].append(self.controller.setpoint)
		self.controller_history['Temperature'].append(control_temp)
		self.controller_history['Time'].append(time.time() - self.controller_start_time)
		self.controller_history['Input_voltage'].append(new_voltage)

	def saveHistory(self):
		"""Saves the PID history as a csv file"""
		filepath = Path(f'{self.root}{str(self.job_name)}__temp__{str(self.controller.setpoint)}/pid_data.csv')
		filepath.parent.mkdir(parents=True, exist_ok=True)
		history_df = pd.DataFrame(self.controller_history)
		history_df.to_csv(filepath,index=False)

	def doPID(self,control_temp):
		""" Updates the power supply voltage from PID.
			@param	control_temp	-- last temperature recieved from sensor
		"""
		new_voltage = self.controller(control_temp)
		self.power_supply.changeVoltage(new_voltage)
		self.updateHistory(control_temp,new_voltage)

	def pidLoop(self):
		"""	Gets the sensor temperature and updates the controller.
			After ready_for_job is set, Lock must be used to access sensor data.
		"""
		try:
			curr_temp = self.sensor.getTemperature()					# get um7 temperature
			print(f"Current temp :: {curr_temp:.3f} [deg C]")
			# while error is above the allowed threshold, do PID loop
			while(abs((curr_temp - self.controller.setpoint) / float(self.controller.setpoint)) > self.temp_error_tollerance):
				curr_temp = self.sensor.getTemperature()					# get um7 temperature		
				self.doPID(curr_temp)
				print(f"Current temp = {curr_temp:.3f} [deg C] (target = {self.controller.setpoint})\tPID internals = [{', '.join(f'{x:.3f}' for x in self.controller.components)}]")
			# after the temperature has stabilized, we are almost ready to start
			print(f"Temperature has stabilized, current PID internals = [{', '.join(f'{x:.3f}' for x in self.controller.components)}]")
			# reset the PID internals and increase the penalty for integral action
			self.controller.reset()
			self.controller.Ki = self.controller.Ki * 10.0
			self.doPID(self.sensor.getTemperature())
			# now set the signal for other thread to start
			print(f"Updated PID internals = [{', '.join(f'{x:.3f}' for x in self.controller.components)}]")
			print("Starting the job!")	
			self.ready_for_job.set()
			# continue running PID until the job_done signal is recieved from the experiment
			while not self.job_done.is_set():
				with self.lock:										# here data race with sensor can occur, so use lock
					curr_temp = self.sensor.getTemperature()		# get um7 temperature
					print(f"Temp = {curr_temp:.3f} [deg C] (target = {self.controller.setpoint})\tPID internals = [{', '.join(f'{x:.3f}' for x in self.controller.components)}]")
				self.doPID(curr_temp)
				time.sleep(0.5)

		except Exception as e:
			print(f" * pidLoop() :: Exception {e} raised.")
			self.job_done.set()
			self.power_supply.powerOFF()	

	def startPID(self):
		""" Checks if the setpoint temperature is possible to achieve, starts the PID loop.
			After PID loop returns, saves the history and turn power off.
		"""
		try:
			temp_diff = self.sensor.getTemperature() - self.controller.setpoint 
			if((self.controller_mode=='cool' and temp_diff < 0) or (self.controller_mode=='heat' and temp_diff > 0) ):
				raise Exception(f'Temperature out of range, Setpoint = {self.controller.setpoint} [deg C] // Current temp = {self.sensor.getTemperature():.3f} [deg C]')
			self.controller_start_time = time.time()
			print("Start PID loop!")
			self.pidLoop()
			# once the job is over, save the controller history	
			self.saveHistory()
			self.power_supply.powerOFF()
			print("PID history saved, power turned off!")
		except Exception as e:
			print(f" * startPID({self.controller_mode} mode) :: {e}")
			self.job_done.set()
			self.power_supply.powerOFF()

	def doJob(self,fun_handle,*args):
		""" Implements the waiting logic for the job function start synchronization with the PID thread.
			@param 	fun_handle	-- some job to do after target temperature is reached
			@param 	*args 		-- arguments to pass to fun_handle
		"""
		try:
			# wait for temperature to stabilize
			while not self.ready_for_job.is_set():
				if self.job_done.is_set():
					# exit point in case of exception raised in PID loop
					return 
				time.sleep(2)
			fun_handle(*args)	# this function is responsible for saving its own data 
			# after function has executed tell PID to stop
			with self.lock:
				print("Job is done!")	
				self.job_done.set()					
		except Exception as e:
			print(f"doJob() :: Exception {e} raised.")
			self.power_supply.powerOFF()
			self.job_done.set()  
			return

	def gyroJob(self,velocities):
		""" Gyro calibration data acquisition procedure. Collects data at different velocities of Stage2.
			@param	velocities 	-- list of velocities
			Saves the sensor and rotary stage data as csv.
		"""
		try :
			Stage2, Stage1 = self.rotary_stages
			#static_data = pd.DataFrame()
			with self.lock:
				print("gyroJob() :: getting stationary data...")
				static_data = self.sensor.getAllRawData(5000)				# in samples
			
			# save the static (bias) data 
			filepath = Path(f'{self.root}{self.job_name}__temp__{self.controller.setpoint}/vel_0_gyro_data.csv')
			filepath.parent.mkdir(parents=True, exist_ok=True)
			static_data.to_csv(filepath, index=False)
		
			for velocity in velocities:
				Stage2.moveVelocity(velocity)
				# measureStageData() returns um7 and stage data (10 x 500 samples)
				um7_df, stage_df = Stage2.measureStageData(10, self.lock, self.sensor.getAllRawData, 500)		
				# save for each velocity 
				filepath = Path(f'{self.root}{self.job_name}__temp__{self.controller.setpoint}/vel_{str(velocity)}_gyro_data.csv')
				filepath.parent.mkdir(parents=True, exist_ok=True)
				um7_df.to_csv(filepath, index=False)
				filepath = Path(f'{self.root}{self.job_name}__temp__{self.controller.setpoint}/vel_{str(velocity)}_stage_data.csv')
				filepath.parent.mkdir(parents=True, exist_ok=True)
				stage_df.to_csv(filepath, index=False)
		
		except Exception as e:
			for stage in self.rotary_stages:
				stage.goZero()

	def accelJob(self,num_samples):
		"""	Accelerometer calibration data acquisition procedure. Collects varying number of orientations depending on the elevation level.
			Positive directions when looking at the stages in zero position : 
			Stage1 -> counter clock-wise , Stage2 -> clock-wise	
			@param 	num_samples  -- number of samples to collect from the sensor 
			Saves the sensor data as csv.
		"""
		try:
			df_full = pd.DataFrame(columns=['gyro_raw_x','gyro_raw_y' ,	'gyro_raw_z' ,	'gyro_raw_time',
											'accel_raw_x','accel_raw_y', 'accel_raw_z' , 'accel_raw_time',
											'mag_raw_x', 'mag_raw_y' ,'mag_raw_z' , 'mag_raw_time','temperature' ,'temperature_time' ])
			Stage2 , Stage1 = self.rotary_stages
			num_azimuth_points = [1,4,8,12,8,4,1]			# number of orientations at each elevation level, 12 at horizontal level 
			Stage1.shiftPosition(-120)						# get to start position	
			for st1_iter,st2_iter in zip(range(1,8),num_azimuth_points):		
				Stage1.shiftPosition(30)
				for azi in range(1,st2_iter+1):
					Stage2.shiftPosition((-1)**(st1_iter) * 360 / st2_iter)		# switch direction of rotation, so stages go home quicker
					print(f"accelJob() :: Current position :: Stage1 -> {Stage1.getPosition():.2f}\tStage2 -> {Stage2.getPosition():.2f}")
					with self.lock:
						print("accelJob() :: getting data ...")
						df_full = pd.concat([df_full, self.sensor.getAllRawData(num_samples)])			
			filepath = Path(f'{self.root}{str(self.job_name)}__temp__{str(self.controller.setpoint)}/accel_data.csv')
			filepath.parent.mkdir(parents=True, exist_ok=True)
			df_full.to_csv(filepath,index=False)
		except Exception as e:
			print(f"accelJob() :: Exception {e} raised.")
			for stage in self.rotary_stages:
				stage.goZero()


	def threadingManager(self,fun_handle,*args):
		"""	Manages the execution of the controller and active experiment.
			Whenever a job function calls for sensor data, it first acquires a lock, otherwise you get data race with PID and segmentation fault. 
			@param 	fun_handle	-- some job to do after target temperature is reached
			@param 	*args 		-- arguments to pass to fun_handle
		"""
		try:
			thread1 = Thread(target=self.startPID, daemon=True) 
			thread2 = Thread(target=self.doJob, args =(fun_handle,*args), daemon=True)
			thread1.start()
			thread2.start()
			thread1.join()
			thread2.join()
		except Exception as e:
			print(f" * threadingManager() Exception :: {e}")
			 