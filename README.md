pySatlantic
===========

_Python package to decode raw binary frames from SeaBird Satlantic instruments._

The module is made to be integrated into scripts or other software, as it stands, it can also be used to convert raw files recorded with SatView to human readable CSV files.
pySatlantic was primarily tested with Satlantic HyperSAS data including Lt, Li, Lu, Es, and THS sensors.
The immersion coefficients can be used.
Definition of additional fit process are needed for other instruments.

## Installation
The package runs with python 3 only and can be installed from the [GitHub repository](https://github.com/OceanOptics/pySatlantic/) with:

    python setup.py install

## Convert binary files recorded with SatView to CSV
The package can be used as it stands from the command line to convert raw files from SatView 

    python -m pySatlantic [-h] [-v] [--version] [-i] cal src

or directly from the installed module

    python -m pySatlantic.__main__ [-h] [-v] [--version] [-i] cal src

positional arguments:
  cal             Calibration file.
  src             Raw file to decode and calibrate.

optional arguments:
  -h, --help      show this help message and exit
  -v, --verbose   Enable verbosity.
  --version       Prints version information.
  -i, --immersed  Apply immersion coefficients.
  
## Integrate in other software
The class `Instrument` provides key methods to handle the binary ACS data
* `read_calibration_file`/`read_calibration_dir`/`read_sip_file`: Parse calibration file(s) needed to unpack and calibrate binary frames.
* `parse_frame`: Decode binary frame and apply calibration fit to retrieve data in scientific units

The class `BinReader` helps to separate individual frames looking the registration bytes `b'SAT'`. An example of usage of that class is `SatViewRawToCSV` which converts a raw file recorded with SatView into a CSV file. 
