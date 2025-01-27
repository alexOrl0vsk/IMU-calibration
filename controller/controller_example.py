import time
import pandas as pd
from pathlib import Path
from controller import PowerController

def heat_gyro_job():
	name = 'Z_ax_heat_gyro_job'#test_cool
	temp = 50
	myController = PowerController(pid_mode='heat',target_temp=temp,job_name=name)

	heat_temps = [43,44,45,46,47,48]
	heat_min_voltages = [0,0.25,0.25,0.4,0.6,0.7]

	velocities =[-200,-100,-80,-40,-20,20,40,80,100,200]
	#myController.gyroJob(velocities)									# run without temperature control 

	# current_temp = myController.sensor.getTemperature()				# check if the temperature is ok
	# while(current_temp > min(heat_temps)):
	# 	current_temp = myController.sensor.getTemperature()
	# 	print(f"Temp is {current_temp:.3f} C")

	for stage in myController.rotary_stages:							# return to home positions
	 	stage.goZero()
	# 	stage.configureSettings(set_home=True, save_flash=True)			# saves the home position, move the stages to the desired positin before running this line
	for temp,min_voltage in zip(heat_temps,heat_min_voltages):
		myController.controller.setpoint = temp
		myController.controller.output_limits = (min_voltage, 3.5)
		myController.controller.Ki = 0.005								# -0.005 for cool job 
		myController.threadingManager(myController.gyroJob,velocities)
		for stage in myController.rotary_stages:
			stage.goZero()
		myController.job_done.clear()
		myController.ready_for_job.clear()
		myController.controller_history = {											
			'Setpoint': [],
			'Temperature': [],
			'Time': [],
			'Input_voltage': []}

def main():
	heat_gyro_job()


if __name__ == "__main__":
	main()