import libximc.highlevel as ximc
import time 
import pandas as pd
from typing import Tuple, List

class RotaryStage:
	def __init__(self, port: str, calib_coeff: float = None):
		self.port = port 						# COM port address
		self.stage = ximc.Axis(port) 			# ximc.Axis object
		self._stage_name = None 				# name of the rotary stage
		self.calib_coeff = calib_coeff			# step to degree coefficient in full step 

	@property
	def stage_name(self):
		"""Returns the stage name."""
		return self._stage_name
	
	@stage_name.setter 
	def stage_name(self, name: str):
		"""Sets the stage name and saves to flash memory"""
		self._stage_name = name
		self.stage.set_stage_name(ximc.stage_name_t(name))		# saves the stage name to flash 

	def open(self):
		"""Opens the stage, gets its name and applies units calibration if specified."""
		self.stage.open_device()
		self.stage_name = self.stage.get_stage_name().PositionerName
		if self.calib_coeff is not None:
			engine_settings = self.stage.get_engine_settings()
			self.stage.set_calb(self.calib_coeff, engine_settings.MicrostepMode)

	def close(self):
		"""Closes the stage, called automatically by garbage collector."""
		self.stage.close_device()	

	def getStatus(self):
		"""Returns the full stage status."""
		if self.calib_coeff is None:
			return self.stage.get_status() 
		else:
			return self.stage.get_status_calb()

	def getMoveState(self):
		"""Returns the motion status of the stage (ximc.MoveState)."""
		return self.stage.get_status().MoveSts 

	def getCmdMoveState(self):
		"""Returns the command motion state of the stage (ximc.MvcmdStatus)."""
		return self.stage.get_status().MvCmdSts

	def getPosition(self):
		"""Returns the stage current position."""
		if self.calib_coeff is None:
			return self.stage.get_position().Position 			# in steps
		else:
			return self.stage.get_position_calb().Position 	 	# in user units

	def getVelocity(self):
		"""Returns the stage current velocity."""
		if self.calib_coeff is None:
			return self.stage.get_status().CurSpeed 
		else:
			return self.stage.get_status_calb().CurSpeed
	
	def getInfo(self):
		"""Prints the current settings."""
		if self.calib_coeff is None: 
			engine_settings = self.stage.get_engine_settings()
			home_settings = self.stage.get_home_settings()
			move_settings = self.stage.get_move_settings()
		else :
			engine_settings = self.stage.get_engine_settings_calb()
			home_settings = self.stage.get_home_settings_calb()
			move_settings = self.stage.get_move_settings_calb()
		print(f"-------------------------------------------------------------------------------------------------------")
		print(f'\t\t << {self.stage_name} current settings >>')
		for settings in [engine_settings,home_settings,move_settings]:
			print(type(settings).__name__.upper(),':\n',settings)
		print(f"-------------------------------------------------------------------------------------------------------")

	def shiftPosition(self, move_degrees: float):
		""" Relative movement of the stage by a value in degrees.
			@param	move_degrees	-- value to shift by in degrees
		"""
		if self.calib_coeff is None:
			print(' * shiftPosition() :: Calibration coefficient must be specified.')
			return
		print(f"<> shiftPosition({move_degrees}) :: Current {self.stage_name} position: {self.getPosition():.3f} deg")	
		print(f"\tMoving {self.stage_name} by {move_degrees} deg ..." )
		self.stage.command_movr_calb(move_degrees)
		self.stage.command_wait_for_stop(10)
		print(f"<> shiftPosition({move_degrees}) :: New {self.stage_name} position: {self.getPosition():.3f} deg")	

	def setVelocityCalb(self,ang_vel):
		"""Set stage target velocity in user-units."""
		if self.calib_coeff is None:
			print(' * setVelocityCalb() :: Calibration coefficient must be specified.')
			return	
		move_settings = self.stage.get_move_settings_calb()
		move_settings.Speed = abs(ang_vel)					# only positive values are accepted 
		self.stage.set_move_settings_calb(move_settings)

	def moveVelocity(self, ang_vel):
		""" Moves rotary stage with constant angular velocity.
			@param	ang_vel -- target velocity in [deg/s]
		"""
		if self.calib_coeff is None:
			print(' * moveVelocity() :: Calibration coefficient must be specified.')
			return	
		self.setVelocityCalb(ang_vel)
		if (ang_vel > 0):					# assume command_right() as positive direction
			self.stage.command_right()
		else :
			self.stage.command_left()
		while (self.getMoveState() & ximc.MoveState.MOVE_STATE_TARGET_SPEED) == 0:
			print(f'<> moveVelocity({ang_vel}) :: Accelerating {self.stage_name} to {ang_vel} deg/s ...')
			time.sleep(1)
		time.sleep(0.5)						# give the stage a moment to update the move status (for accurate logging)
		print(f"<> moveVelocity({ang_vel}) :: Targer velocity reached, moving {self.stage_name} at {self.getVelocity():.3f} deg/s ..." )

	def stop(self):
		"""Stops the stage motion."""  
		print(f"<> stop({self.stage_name}) :: moveState = {self.getMoveState()}, \t CMDmoveState = {self.getCmdMoveState()} <>")
		if (self.getCmdMoveState() & ximc.MvcmdStatus.MVCMD_RUNNING):
			self.stage.command_sstp()
			self.stage.command_wait_for_stop(10)
		print(f"<> stop({self.stage_name}) :: moveState = {self.getMoveState()}, \t CMDmoveState = {self.getCmdMoveState()}, done! <>")

	def goZero(self):
		"""Moves the stage to the zero position."""
		print(f"\t<> goZero({self.stage_name}) called <>")
		if (self.getCmdMoveState() & ximc.MvcmdStatus.MVCMD_RUNNING):
			self.stop()
			self.stage.command_wait_for_stop(10)
		self.setVelocityCalb(300)
		self.stage.command_move(0,0)
		self.stage.command_wait_for_stop(10)
		print(f"\t<> goZero({self.stage_name}) done !\tCurrent {self.stage_name} position = {self.getPosition()} <>")

	def setHomePosition(self):
		"""Sets the home position of an opened axis."""
		h = self.stage.get_home_settings()
		h.HomeDelta = 0
		h.uHomeDelta = 0 
		saved_fast = h.FastHome
		saved_ufast = h.uFastHome
		if (h.HomeFlags & ximc.HomeFlags.HOME_MV_SEC_EN != 0):
			h.FastHome = 100
			h.uFastHome = 0
		self.stage.set_home_settings(h)
		old_pos = self.stage.get_status().CurPosition
		old_upos = self.stage.get_status().uCurPosition
		self.stage.command_home()
		while (self.stage.get_status().MvCmdSts is (ximc.MvcmdStatus.MVCMD_HOME | ximc.MvcmdStatus.MVCMD_RUNNING)):
			time.sleep(0.1)
		assert (self.stage.get_status().MvCmdSts is ximc.MvcmdStatus.MVCMD_HOME),'command_home() failed'
		new_pos = self.stage.get_status().CurPosition
		new_upos = self.stage.get_status().uCurPosition
		h.HomeDelta = old_pos-new_pos
		h.uHomeDelta = old_upos-new_upos
		h.FastHome = saved_fast
		h.uFastHome = saved_ufast
		self.stage.set_home_settings(h)
		self.stage.command_move(old_pos, old_upos)
		while (self.stage.get_status().MvCmdSts == (ximc.MvcmdStatus.MVCMD_MOVE | ximc.MvcmdStatus.MVCMD_RUNNING)):
			time.sleep(0.1)
		assert(self.stage.get_status().MvCmdSts is ximc.MvcmdStatus.MVCMD_MOVE),"Movement was not completed"
		self.stage.command_zero()
	
	def configureSettings(self, set_home: bool = False, save_flash: bool = False, new_stage_name: str = None):
		""" Sets the current position as home, configures the stage to default settings and saves to flash memory.
			@param	set_home 			-- set the current position as home
			@param	save_flash 			-- save the settings to flash memory
			@param	new_stage_name 		-- name for the stage 
		"""
		if (self.calib_coeff is None):
			print(' * configureSettings() :: Calibration coefficient not specified.')
			return
		if(set_home):
			print("\t <> Setting current position as home <>\n")
			self.setHomePosition()
			print(f"Current position for {self.stage_name} ::\n{self.getPosition()}")
			print(f"Current HomeDelta for {self.stage_name} ::\n{self.stage.get_home_settings().HomeDelta}")		 
		if(new_stage_name is not None):
			self.stage_name = new_stage_name

		brake_settings = self.stage.get_brake_settings()
		move_settings_calb = self.stage.get_move_settings_calb()
		home_settings_calb = self.stage.get_home_settings_calb()

		brake_settings.BrakeFlags = 0x01 				# BRAKE_ENABLED 	
		move_settings_calb.Speed = 200 					# degrees/s 
		move_settings_calb.Accel = 200 					
		move_settings_calb.Decel = 200 					
		move_settings_calb.AntiplaySpeed = 200 			# degrees/s 
		home_settings_calb.FastHome = 60
		home_settings_calb.SlowHome = 0.01

		self.stage.set_brake_settings(brake_settings)
		self.stage.set_move_settings_calb(move_settings_calb)
		self.stage.set_home_settings_calb(home_settings_calb)	
		self.getInfo()					# print current settings
		if(save_flash):
			print("\t <> Saving settings to flash <>\n")
			self.stage.command_save_settings()

	def measureStageData(self, stage_points, lock, func, *args) -> Tuple[pd.DataFrame,pd.DataFrame]:
		""" Records the stage data during execution of some other job.
			@param 	stage_points 	-- number of samples for stage data
			@param	lock 			-- a lock for synchronization control (or pass None)
			@param	func 			-- some function to execute while data is being recorded
			@param	*args 			-- arguments to pass to the job function
			returns the stage data df, as well as the output of the job function. 
		"""
		stage_status_data = {field : [] for field in ["Position","Velocity","Time"]}
		func_results = []
	
		start_time = time.time()
		while (len(stage_status_data["Time"]) < stage_points):
			status = self.getStatus()
			position = self.getPosition()
			velocity = self.getVelocity()
			stage_status_data["Position"].append(position)
			stage_status_data["Velocity"].append(velocity)
			stage_status_data["Time"].append(time.time() - start_time)
			print(f"{self.stage_name} status ::\tPosition = {position:.3f} , Velocity = {velocity:.3f}")
			if lock:
				with lock:
					func_results.append(func(*args))
			else:
				func_results.append(func(*args))

		return pd.concat(func_results, axis=0), pd.DataFrame(stage_status_data)

	@staticmethod
	def loadAxes(calib_coeffs: list = [0.6, 0.01]) -> List[ximc.Axis]:	# inside of List[RotaryStage]
		""" Opens all axes from the connected devices list.
			@param calib_coeffs -- step to degree conversion factors of connected stages (defaults to the current setup)
			returns a list of opened and user-units calibrated axis objects (inside RotaryStage class).
		"""
		devices = ximc.enumerate_devices(
			ximc.EnumerateFlags.ENUMERATE_NETWORK |
			ximc.EnumerateFlags.ENUMERATE_PROBE
		)
		if len(devices) == 0:
			print(" * loadAxes() :: No devices were found.")
		else:
			print("Found {} real device(s):".format(len(devices)))
			for device in devices:
				print("  {}".format(device))
		assert len(calib_coeffs) == len(devices), "Mismatched number of devices and calibration coefficients."
		axes = [] 					
		for device, calib_coeff in zip(devices,calib_coeffs):
			axes.append(RotaryStage(device['uri'], calib_coeff))
		for axis in axes:
			axis.open()						# RotaryStage method
			if(axis.stage._is_opened):
				print(f"<> loadAxes() {axis.stage_name} (calib_coeff = {axis.calib_coeff}) opened! <>")
		return axes

	# class MoveState:						actual state of stage
	# 	MOVE_STATE_MOVING          = 0x01			0001
	# 	MOVE_STATE_TARGET_SPEED    = 0x02			0010
	# 	MOVE_STATE_ANTIPLAY        = 0x04			0100

	# class MvcmdStatus:					controller command to stage
	# 	MVCMD_NAME_BITS    = 0x3F
	# 	MVCMD_UKNWN        = 0x00
	# 	MVCMD_MOVE         = 0x01
	# 	MVCMD_MOVR         = 0x02
	# 	MVCMD_LEFT         = 0x03
	# 	MVCMD_RIGHT        = 0x04
	# 	MVCMD_STOP         = 0x05
	# 	MVCMD_HOME         = 0x06
	# 	MVCMD_LOFT         = 0x07
	# 	MVCMD_SSTP         = 0x08
	# 	MVCMD_ERROR        = 0x40
	# 	MVCMD_RUNNING      = 0x80