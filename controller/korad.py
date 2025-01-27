from ka3005p import PowerSupply

class KoradPower():
	def __init__(self):
		"""Loads the power supply object and disables it."""
		devices = PowerSupply.list_power_supplies()
		if devices:
			self.korad = PowerSupply(devices[0])
		else:
			raise Exception("KoradPower :: no devices found")
		self.powerInfo()
		self.powerOFF()

	def powerInfo(self):
		"""Prints power supply status"""
		print("\t << powerInfo() >> ")
		print(f"Current settings :: voltage = {self.korad.voltage} [V] \t current = {self.korad.current} [A]")
		print(f"Full status :: {self.korad.status} ")

	def powerON(self):
		"""Turns the power supply on."""
		self.korad.enable()

	def powerOFF(self):
		"""Turns the power supply off."""
		self.korad.disable()

	def changeVoltage(self,new_voltage: int):
		"""Changes the power supply voltage."""
		self.korad.voltage = new_voltage
		self.korad.enable()	