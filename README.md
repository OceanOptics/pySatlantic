pySatlantic
===========
[![Python 3](https://img.shields.io/badge/Python-3-blue.svg)](https://www.python.org/downloads/)
[![license MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/OceanOptics/pySatlantic/blob/master/LICENSE)

_Python package to unpack binary messages from Satlantic instruments._

This module provides a simple way to unpack binary messages from Satlantic instruments (e.g. HyperSAS, HyperPro, HTSRB, HyperNAV). This is likely not the fastest implementation of the Satlantic protocol parser. However, it is easy to use and fast enough to process hyperspectral spectrums from multiple sensors (e.g. HyperSAS & Es) on a Raspberry Pi 3 in real-time.


## Installation
The package runs with python 3 and can be installed from the setup file or directly with pip.

    # Installation through setup file
    python setup.py install
    
    # Installation through pip
    pip install pySatlantic


## Convert binary files recorded with SatView to CSV
The module can be used to convert raw files recorded with SatView to CSV files.

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
The class `Instrument` provides key methods to handle the binary frames from Satlantic instruments
* `read_calibration_file`/`read_calibration_dir`/`read_sip_file`: Parse calibration file(s) needed to unpack and calibrate binary frames.
* `find_frame`: Find known frame in a binary array
* `parse_frame`: Unpack binary frame and apply calibration fit to convert data in scientific units

The class `Instrument` support the immersed flag for each sensor independently. The immersion flag is accessible through `Instrument.cal[<frame_header>].immersed` with `frame_header` being the sensor to be immersed or not.


The class `BinReader` helps to separate individual frames looking for the registration bytes `b'SAT'`. An example of usage of that class is `SatViewRawToCSV` which converts a raw file recorded with SatView into a CSV file. 


## Sensors Tested
List of frames and associated calibration files tested:
  + Es:
    + HED
    + HSE
  + HyperSAS
    + HLD
    + HSL
  + HTSRB
    + HSD
    + HST
    + THS
  + HyperPro
    + HPE
    + HPL
    + MPR
    + PED
    + PLD
    + SATBB2F
    + SATFLCD
  + HyperNav
    + SATXDZ
    + SATXLZ

Some functionalities of the Satlantic protocol are not implemented. Please make a feature request if you would like to the support of specific sensors (through the 'Issues' section of the GitHub repository).
