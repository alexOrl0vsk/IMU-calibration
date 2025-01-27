from rsl_comm_py import UM7Serial
from um7py.um7_broadcast_packets import UM7AllRawPacket
import json
import sys
import pandas as pd
import numpy as np
from time import sleep

class Sensor:
	def __init__(self,port_name: str ='COM5'):	
		self.sensor = UM7Serial(port_name=port_name)
		self.sensor.port.baudrate = 460800		# don't change baud rate here, first use the changeSetting() method 

	def changeSetting(self):
		""" Configures the um7 and saves the settings to flash memory.
			To change the baud rate, set 1) & 2) to the same value.
			Some baud rates: 5 = 115200 (default), 6 = 128000, 10 = 460800, 11 = 921600
		"""
		print("\nCurrent pySerial settings ::")
		print("um7.port.baudrate :: ",  self.sensor.port.baudrate)
		print('um7.port.bytesize :: ', self.sensor.port.bytesize )
		print('um7.port.stopbits :: ', self.sensor.port.stopbits )
		print('um7.port.parity :: ', self.sensor.port.parity)
		print("\nCurrent um7 creg_com_settings: {}".format(self.sensor.creg_com_settings))
		# um7 ALL_RAW_RATES [in Hz]
		self.sensor.creg_com_rates2 = 255													
		creg_com_settings, *_ = self.sensor.creg_com_settings	
		# 1) changes the um7 baud rate 
		creg_com_settings.set_field_value(BAUD_RATE=10) 									 
		self.sensor.creg_com_settings = creg_com_settings.raw_value
		time.sleep(2)
		self.sensor.port.close()
		# 2) changes the pySerial baud rate, then change __init__ value
		self.sensor.port.baudrate = 460800 													
		self.sensor.port.open()
		# save to flash
		self.sensor.flash_commit = 1 	

	def getTemperature(self) -> float:
		"""Returns the um7 temperature."""
		for packet in self.sensor.recv_all_raw_broadcast(flush_buffer_on_start=True):
			return packet.temperature 

	def getAllRawData(self,num_packets: int) -> pd.DataFrame:
		""" Returns the um7 raw data in a dataframe.
			@param	num_packets		-- number of samples to receive
		"""
		all_raw_fields = ['gyro_raw_x', 'gyro_raw_y', 'gyro_raw_z', 'gyro_raw_time',
						'accel_raw_x', 'accel_raw_y', 'accel_raw_z', 'accel_raw_time',
						'mag_raw_x', 'mag_raw_y', 'mag_raw_z', 'mag_raw_time',
						'temperature', 'temperature_time']					
		data = {field: [] for field in all_raw_fields}
		iterations = 0
		for packet in self.sensor.recv_all_raw_broadcast(num_packets=num_packets,flush_buffer_on_start=True):
			data['gyro_raw_x'].append(packet.gyro_raw_x)
			data['gyro_raw_y'].append(packet.gyro_raw_y)
			data['gyro_raw_z'].append(packet.gyro_raw_z)
			data['gyro_raw_time'].append(packet.gyro_raw_time)
			data['accel_raw_x'].append(packet.accel_raw_x)
			data['accel_raw_y'].append(packet.accel_raw_y)
			data['accel_raw_z'].append(packet.accel_raw_z)
			data['accel_raw_time'].append(packet.accel_raw_time)
			data['mag_raw_x'].append(packet.mag_raw_x)
			data['mag_raw_y'].append(packet.mag_raw_y)
			data['mag_raw_z'].append(packet.mag_raw_z)
			data['mag_raw_time'].append(packet.mag_raw_time)
			data['temperature'].append(packet.temperature)
			data['temperature_time'].append(packet.temperature_time)
			#print(f"packet: {packet}")
			#print(f'ITER ({iterations})' ) 
			iterations= iterations + 1
		return pd.DataFrame(data)