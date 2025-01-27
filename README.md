# IMU-calibration
Controller and data processing for UM7 orientation sensor.

# Repo structure:
- controller
  - controller.py         (PID controller based on simple-pid library)
  - controller_example.py (example of running the controller)
  - korad.py              (based on ka3005p PowerSupply)
  - rotaryStages.py       (based on libximc library)
  - um7.py                (based on rsl_comm_py library)
- processing
    - accel_processing.py
    - allan.py
    - gyro_processing.py
    - run_results.py    (contains function calls to data processing)
    - utils.py
    - UM7_data  (contains the calibration data)

# Using processing
Sensor data archive: https://drive.google.com/file/d/1EK62wzkjGX4GBCBHg8snI6bThMVn3LAt/view?usp=sharing .
The archived sensor data should be unpacked to the processing folder (as shown above), the relative paths are noted in run_results.py.
GYROS_3 folder contains the same data as in GYRO_X,Y,Z, sorted by temperature. Each sensing axis recorded on separate day, so each temperature in GYROS_3 has 3 sets of biases. Bias on day X (i.e., 'X_vel_0_gyro_data.csv') is to be subtracted from the rest of data collected on day X. Some of the stage data in GYRO_X,Y was not recorded correctly, but the set velocity was in fact reached.
