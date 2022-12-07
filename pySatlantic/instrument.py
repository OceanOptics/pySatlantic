from __future__ import print_function
import os
from struct import unpack
import zipfile
import numpy as np
import csv
from operator import itemgetter
from datetime import datetime, timedelta, timezone
import warnings
from collections import namedtuple


# Error Management
class SatlanticInstrumentError(Exception):
    pass


class FrameError(SatlanticInstrumentError):
    pass


class FrameHeaderNotFoundError(FrameError):
    pass


class FrameLengthError(FrameError):
    pass


class ParserError(SatlanticInstrumentError):
    pass


class ParserDecodeError(ParserError):
    pass


class ParserTypeError(ParserError):
    pass


class ParserFitError(ParserError):
    pass


class CalibrationFileError(SatlanticInstrumentError):
    pass


class CalibrationFileExtensionError(CalibrationFileError):
    pass


class CalibrationFileEmptyError(CalibrationFileError):
    pass


FrameContainer = namedtuple('FrameContainer', ['header', 'data', 'timestamp', 'valid'])


def sat_dtype_to_np_dtype(sat_data_type, sat_field_length):
    if sat_data_type == 'AS':
        # ASCII string (text)
        if sat_field_length:
            return f'<{sat_field_length}U'
        else:
            # Assume maximum size of 12 characters
            return f'<12U'
    elif sat_data_type == 'AI':
        # ASCII integer number
        return np.int32
    elif sat_data_type == 'AF':
        # ASCII floating point number
        return np.float32
    elif sat_data_type == 'BS' and sat_field_length == 1:
        # signed short 1 byte
        return np.int8
    elif sat_data_type == 'BU' and sat_field_length == 1:
        # unsigned short 1 byte
        return np.uint8
    elif sat_data_type == 'BS' and sat_field_length == 2:
        # signed short 2 bytes
        return np.int16
    elif sat_data_type == 'BU' and sat_field_length == 2:
        # unsigned short 2 bytes
        return np.uint16
    elif sat_data_type == 'BS' and sat_field_length == 4:
        # signed integer 4 bytes
        return np.int32
    elif sat_data_type == 'BU' and sat_field_length == 4:
        # unsigned integer 4 bytes
        return np.uint32
    elif sat_data_type == 'BF' and sat_field_length == 4:
        # float
        return np.float32
    elif sat_data_type == 'BD' and sat_field_length == 8:
        # double float
        return np.float64
    else:
        raise ParserTypeError(f'Unknown format decoder {sat_data_type}{sat_field_length}')


class Parser:
    """
    The Parser class builds a parser from Satlantic Calibration or Telemetry Definition Files (cal or tdf).
        Parser contains information to unpack binary data or split ascii data into meaningful fields.
        The Parser also contains information to calibrate fields from engineering units to calibration units

    Follow Satlantic's Data Format Standard SAT-DN-00134, ver 6.1
    """

    CORE_VARIABLE_TYPES = ['LT', 'LI', 'LU', 'LD', 'LS', 'ED', 'ES', 'EU', 'EV', 'EF', 'UV']

    def __init__(self, cal_filename=None, immersed=False):
        # Metadata
        self.instrument = ''
        self.sn = -999
        self.frame_header = ''
        self.frame_header_length = 0
        self.immersed = immersed
        self.frame_rate = None
        self.baudrate = None
        self.calibration_time = {}
        self.calibration_temperature = None
        self.calibration_dark_average = None
        self.response_temperature = None
        self.response_dark_average = None
        self.thermal_response = []
        # Parsing variables
        self.key = []
        self.type = []
        self.id = []
        self.units = []
        self.variable_frame_length = False
        self.field_length = []
        self.field_separator = []
        self.data_type = []  # AS, AI, BS, BD, BU
        self.cal_nlines = []
        self.fit_type = []  # NONE, COUNT, POLYU, or OPTIC3
        self.cal_coefs = []
        self.frame_nfields = 0
        self.frame_length = 0
        self.frame_fmt = '!'  # byte order is MSB (network ! works)
        self.frame_terminator = None
        self.frame_terminator_bytes = None
        self.check_sum_index = None
        # Variable groups (only for variable_frame_length)
        self.core_variables = []
        self.auxiliary_variables = []
        self.unusable_variables = []
        self.core_groupname = ''
        self.unusable_groupname = ''
        # Numpy Calibration coefficients
        self.core_cal_coefs = None
        self.unusable_cal_coefs = None

        if cal_filename is not None:
            # load cal file defining the frames for the data file
            self.read(cal_filename)

    def read(self, filename):
        """
        Read calibration or telemetry definition file

        .cal file specifications:
              lines ignored:
                  lines starting with a #
                  blank lines
              all lines must have 7 elements separated by spaces:
                  type: <string> variable name or type
                  id: <string> variable type or wavelength
                  units: <string> physical units of field
                  field_length: <positive integer> number of bytes (1,2,4,6)
                  data_type: <key_word> type of bytes (AS, AI, AF, BS, BD, BU)
                  cal_lines: <positive integer> number of lines with calibration
                      coefficients (calibration lines) to immediately follow
                      the current sensor definition line
                  fit_type: <key_word> type of special processing needed to
                      convert the sensor value into physical units
                      (NONE, COUNT, POLYU, or OPTIC3)

        Update member variables:
          self.instrument <string> instrument name
          self.sn <string> instrument serial number
          self.frame_header <string> instrument frame header
          self.immersed <bool> instrument is immersed in water (True) or in the air (False)
          self.frame_rate <int> instrument expected frame rate
          self.baudrate <int> instrument expected serial baud rate
          self.calibration_time <dict> instrument calibration time
          self.calibration_temperature <float> instrument calibration temperature
          self.calibration_dark_average <float> instrument average calibration dark
          self.response_temperature <string> instrument thermal response type for correction
          self.response_dark_average <string> instrument average dark response for correction
          self.thermal_response <list> instrument thermal response coefficients
          self.key <list> concatenate variable type and variable id (if id is not None, otherwise same as type)
          self.type <list> variable names
          self.id <list> variable types
          self.units <list> variable units
          self.variable_frame_length <boolean> if frame has a variable of fields in frames
                  True (if FIELD is present) use field_separator to parse data
                  False (default) use field_length and frame_fmt to parse data
          self.field_length <list> number of bytes for each variable
          self.field_separator <list> symbol preceding field to separate them in case of variable length frame
          self.data_type <list> type of bytes for each variable
          self.cal_nlines <list> number of line of coefficients following
          self.fit_type <list> equation to process variables
          self.cal_coefs <list<list>> coefficients for the equations
          self.frame_nfields <integer> number of variable in a frame
          self.frame_length <integer> number of bytes in a frame
          self.frame_fmt <string> format string for struct.unpack
          self.core_variables <list> selection of variables part of CORE_VARIABLE_TYPE with fit_type != NONE
          self.auxiliary_variables <list> selection of variables absent from CORE_VARIABLE_TYPE
          self.unusable_variables <list> selection of variables part of CORE_VARIABLE_TYPE with fit_type == NONE
          self.core_groupname = '' <string> name of group of variable being part of core_variables
          self.unusable_groupname = '' <string> name of group of variable being part of unusable_variables
          self.core_cal_coefs = None <np.array> cal_coefs specific to core_variables in numpy array
          self.unusable_cal_coefs = None <np.array> cal_coefs specific to unusable_core_variables in numpy array
        """

        with open(filename, 'r') as f:
            for l in f:  # read file line by line
                lc = l.strip()  # remove leading and trailing characters
                if lc == '' or lc[0] == '#':
                    # Skip empty line and comments
                    continue
                # Frame header
                if lc[0:10] == 'INSTRUMENT':
                    # Special case instrument name
                    ls = lc.split()
                    self.instrument = ls[1]
                    self.frame_header += ls[1]
                    self.frame_header_length += int(ls[3])
                    self.frame_length += int(ls[3])
                    continue
                if lc[0:2] == 'SN':
                    # Special case serial number
                    ls = lc.split()
                    self.sn = int(ls[1])
                    self.frame_header += ls[1]
                    self.frame_header_length += int(ls[3])
                    self.frame_length += int(ls[3])
                    continue
                if lc[0:14] == 'VLF_INSTRUMENT':
                    # Special case used for variable length frame instruments
                    self.variable_frame_length = True
                    ls = lc.split()
                    self.frame_header = ls[1]
                    self.frame_header_length = int(ls[3])
                    self.frame_length += self.frame_header_length
                    if self.frame_header_length == 10:
                        # Most sensor frame headers are:
                        #       instrument model: 6 characters (can be char or digit, e.g. SATDI4)
                        #       serial number: 4 digits
                        idx = 6
                        if self.sn == -999:
                            try:
                                self.sn = int(ls[1][idx:])
                            except ValueError:
                                foo = [c.isdigit() for c in ls[1]]
                                idx = -foo[-1::-1].index(False)
                                if idx:
                                    self.sn = int(ls[1][idx:])
                        if self.instrument == '':
                            self.instrument = ls[1][0:idx]
                    continue
                # Variable frame length separator
                if lc[0:5] == 'FIELD':
                    # self.variable_frame_length = True
                    self.field_separator.append(bytes(lc.split()[2][1:-1], "ASCII").decode("unicode_escape"))
                    continue
                # Variable frame length terminator
                if lc[0:10] == 'TERMINATOR':
                    self.frame_terminator = bytes(lc.split()[2][1:-1], "ASCII").decode("unicode_escape")
                    self.frame_terminator_bytes = bytes(self.frame_terminator, "ASCII")  # need to go through string otherwise append \\
                    if not self.frame_terminator and int(l.split()[3]) == 2:
                        self.frame_terminator = '\r\n'
                        self.frame_terminator_bytes = b'\x0D\x0A'
                    if self.variable_frame_length:
                        self.field_separator.append(self.frame_terminator)
                if lc[0:4] == 'CRLF':
                    self.frame_terminator = '\r\n'
                    self.frame_terminator_bytes = b'\x0D\x0A'
                # Pseudo Sensors
                if lc[0:4] == 'RATE':
                    self.frame_rate = int(lc.split()[1])
                    continue
                if lc[0:8] == 'DATARATE':
                    self.baudrate = int(lc.split()[1])
                    continue
                if lc[0:7] == 'CALTIME':
                    ls = lc.split()
                    self.calibration_time[ls[1]] = float(ls[2][1:-1])
                    continue
                if lc[0:7] == 'CALTEMP' or lc[0:13] == 'CAL_SPEC_TEMP':
                    self.calibration_temperature = float(lc.split()[1])
                    continue
                if lc[0:13] == 'CAL_DARK_AVER':
                    self.calibration_dark_average = float(lc.split()[1])
                    continue
                if lc[0:13] == 'RSP_SPEC_TEMP':
                    self.response_temperature = lc.split()[6]
                    warnings.warn("RSP_SPEC_TEMP not supported")
                    continue
                if lc[0:13] == 'RSP_DARK_AVER':
                    self.response_dark_average = lc.split()[6]
                    warnings.warn("RSP_DARK_AVER not supported")
                    continue
                if lc[0:12] == 'THERMAL_RESP':
                    # Get lines of coefficients corresponding to thermal response
                    l = f.readline().strip().split()
                    for c in l:
                        self.thermal_response.append(float(c))
                    continue
                # Frame fields
                ls = lc.split('#')[0].split()
                if len(ls) == 7:
                    # Get values from each line
                    self.type.append(ls[0])
                    self.id.append(ls[1])
                    if ls[1] != 'NONE':
                        self.key.append(ls[0] + '_' + ls[1])
                    else:
                        self.key.append(ls[0])
                    self.units.append(ls[2][1:-1])  # rm initial and final '
                    try:
                        self.field_length.append(int(ls[3]))
                    except ValueError as e:
                        if self.variable_frame_length:
                            # Special case of variable field length is accepted for variable length frame
                            self.field_length.append(None)
                        else:
                            # Fixed length frame must have a valid field length
                            raise e
                    self.data_type.append(ls[4])
                    self.cal_nlines.append(int(ls[5]))
                    self.fit_type.append(ls[6])
                    if self.cal_nlines[-1] == 0:
                        # No lines of coefficient following
                        self.cal_coefs.append(-999)
                    else:
                        # Load all lines of coefficient following
                        foo = []
                        for i in range(self.cal_nlines[-1]):
                            l = f.readline().strip().split()
                            for c in l:
                                foo.append(float(c))
                        self.cal_coefs.append(foo)
                else:
                    raise CalibrationFileError('Cal Incomplete line')

            if not self.variable_frame_length:
                # Build frame parser for fixed length frames
                self.frame_nfields = len(self.type)
                for i in range(self.frame_nfields):
                    self.frame_length += self.field_length[i]
                    self.frame_fmt += self._sat_dtype_to_struct(self.data_type[i], self.field_length[i])

            # Check for variable frame length data type
            if self.variable_frame_length:
                for t in self.data_type:
                    if t not in ['AS', 'AI', 'AF']:
                        raise ParserTypeError(f"Invalid data type {t} for variable frame length.")

            # Special case of NMEA_CHECKSUM (not documented in SAT-DN-00134)
            #       data-type should be AI but it's an hexadecimal int instead of int.
            if 'NMEA_CHECKSUM' in self.key:
                i = self.key.index('NMEA_CHECKSUM')
                self.type[i] = 'CHECK'
                self.id[i] = 'SUM'
                self.key[i] = f'{self.type}_{self.id}'
                self.data_type[i] = 'AI16'

            # Specify end of check sum computation
            if not self.variable_frame_length and 'CHECK_SUM' in self.key:
                self.check_sum_index = -self.field_length[self.key.index('CHECK_SUM')]
                if 'TERMINATOR' in self.id:
                    self.check_sum_index -= self.field_length[self.id.index('TERMINATOR')]
                elif 'TERMINATOR' in self.type:
                    self.check_sum_index -= self.field_length[self.type.index('TERMINATOR')]

            # Group Variables
            self.core_variables = [i for i, (x, y) in enumerate(zip(self.type, self.fit_type))
                                   if x.upper() in self.CORE_VARIABLE_TYPES and y != 'NONE']
            if self.core_variables:
                self.core_groupname = f'{self.type[self.core_variables[0]]}'
            self.unusable_variables = [i for i, (x, y) in enumerate(zip(self.type, self.fit_type))
                                       if x.upper() in self.CORE_VARIABLE_TYPES and y == 'NONE']
            if self.unusable_variables:
                self.unusable_groupname = f'{self.type[self.core_variables[0]]}_RAW'
            self.auxiliary_variables = [i for i, (x, y) in enumerate(zip(self.type, self.fit_type)) if
                                        x.upper() not in self.CORE_VARIABLE_TYPES]

            # Convert calibration coefficients to numpy array for fast computation
            if self.core_variables:
                self.core_cal_coefs = np.array(itemgetter(*self.core_variables)(self.cal_coefs)).transpose()
            if self.unusable_variables:
                self.unusable_cal_coefs = np.array(itemgetter(*self.unusable_variables)(self.cal_coefs)).transpose()

            # Check if data_type is valid for fit type
            for data, fit in zip(self.data_type, self.fit_type):
                if fit in ['OPTIC1', 'DDMMYY']:
                    if data not in ['BU', 'BS', 'AI']:
                        raise ParserTypeError(f"Valid data types for OPTIC1 are BU, BS, and AI.")
                elif fit in ['GPSTIME', 'GPSPOS', 'DDMM', 'HHMMSS']:
                    if data not in ['BF', 'BD', 'AF']:
                        raise ParserTypeError(f"Valid data types for {fit} are BF, BD, and AF.")
                elif fit in ['OPTIC2', 'OPTIC3', 'THERM1', 'POW10', 'POLYU', 'POLYF', 'TIME2']:
                    if data == 'AS':
                        raise ParserTypeError(f"All data types are valid with {fit}, except AS.")
                elif fit in ['GPSHEMI', 'GPSMODE', 'GPSSTATUS']:
                    if data != 'AS':
                        raise ParserTypeError(f"The only valid data type for {fit} is AS.")
                elif fit not in ['COUNT', 'NONE', 'DELIMITER']:
                    raise ParserFitError(f"Fit type {fit} not supported.")

    @staticmethod
    def _sat_dtype_to_struct(sat_data_type, sat_field_length):
        if sat_field_length is None:
            raise CalibrationFileError('Field length must be an integer for fixed length frames.')
        if sat_data_type == 'AS':
            # ASCII string (text)
            return str(sat_field_length) + 's'
        elif sat_data_type == 'AI':
            # ASCII integer number
            return str(sat_field_length) + 's'
        elif sat_data_type == 'AF':
            # ASCII floating point number
            return str(sat_field_length) + 's'
        elif sat_data_type == 'BS' and sat_field_length == 1:
            # signed short 1 byte
            return 'b'
        elif sat_data_type == 'BU' and sat_field_length == 1:
            # unsigned short 1 byte
            return 'B'
        elif sat_data_type == 'BS' and sat_field_length == 2:
            # signed short 2 bytes
            return 'h'
        elif sat_data_type == 'BU' and sat_field_length == 2:
            # unsigned short 2 bytes
            return 'H'
        elif sat_data_type == 'BS' and sat_field_length == 4:
            # signed integer 4 bytes
            return 'i'
        elif sat_data_type == 'BU' and sat_field_length == 4:
            # unsigned integer 4 bytes
            return 'I'
        elif sat_data_type == 'BF' and sat_field_length == 4:
            # float
            return 'f'
        elif sat_data_type == 'BD' and sat_field_length == 8:
            # double float
            return 'd'
        else:
            raise ParserTypeError(f'Unknown byte decoder combination {sat_data_type} {sat_field_length}')

    def __str__(self):
        return f'{self.frame_header}\n' + \
               f'\tInstrument: {self.instrument}\n' + \
               f'\tSerial number: {self.sn}\n' + \
               f'\tNumber of fields: {self.frame_nfields}\n' + \
               f'\tVariable frame length: {self.variable_frame_length}\n' + \
               f'\tFrame length (in bytes): {self.frame_length}\n' + \
               f'\tVariables type: {self.type}\n' + \
               f'\tVariables id: {self.id}\n' + \
               f'\tVariables units: {self.units}\n' + \
               f'\tVariables fit type:{self.fit_type}\n' + \
               f'\tVariables field length: {self.field_length}\n' + \
               f'\tVariables data type: {self.data_type}\n' + \
               f'\tVariables frame format: {self.frame_fmt}\n'


class Instrument:
    """
    Instrument class parse raw data from Satlantic instruments (HyperSAS, HyperNAV) if calibration files are passed
    """

    ENCODING = 'utf-8'
    UNICODE_HANDLING = 'replace'
    VALID_SIP_EXTENSIONS = ['.sip', '.zip']
    VALID_CAL_EXTENSIONS = ['.cal', '.tdf']

    def __init__(self, filename=None, immersed=False):
        self.cal = dict()
        self.max_frame_header_length = 0
        if filename is not None:
            self.read_calibration(filename, immersed)

    def read_calibration(self, filename, immersed=False):
        """
        Automatically choose method to read calibration file depending on files extension.

        :param filename: list of calibration files or string of one calibration file
        :param immersed: False: sensor in air, True: sensor immersed in water
        :return:
        """

        if type(filename) is not list:
            filename = [filename]
        for f in filename:
            if os.path.isdir(f):
                self.read_calibration_dir(f, immersed)
            elif os.path.isfile(f):
                ext = os.path.splitext(f)[1]
                if ext in self.VALID_SIP_EXTENSIONS:
                    self.read_sip_file(f, immersed)
                elif ext in self.VALID_CAL_EXTENSIONS:
                    self.read_calibration_file(f, immersed)
                else:
                    raise CalibrationFileExtensionError(f'File extension incorrect: {f}')
            else:
                raise FileNotFoundError(f'No such file or directory: {f}')

    def read_calibration_file(self, filename, immersed=False):
        ext = os.path.splitext(filename)[1]
        if ext in self.VALID_CAL_EXTENSIONS:
            foo = Parser(filename, immersed)
            self.cal[foo.frame_header] = foo
            self.max_frame_header_length = max(self.max_frame_header_length, len(foo.frame_header))
        else:
            raise CalibrationFileExtensionError(f'File extension incorrect: {f}')

    def read_calibration_dir(self, dirname, immersed=False):
        empty_dir = True
        for fn in os.listdir(dirname):
            if os.path.isfile(os.path.join(dirname, fn)) and \
                    os.path.splitext(fn)[1] in self.VALID_CAL_EXTENSIONS and os.path.basename(fn)[0] != '.':
                # File exist, valide extension, and not hidden file
                empty_dir = False
                foo = Parser(os.path.join(dirname, fn), immersed)
                self.cal[foo.frame_header] = foo
                self.max_frame_header_length = max(self.max_frame_header_length, len(foo.frame_header))
        if empty_dir:
            raise CalibrationFileEmptyError('No calibration file found in directory')

    def read_sip_file(self, filename, immersed=False):
        empty_sip = True
        archive = zipfile.ZipFile(filename, 'r')
        dirsip = os.path.dirname(filename)
        archive.extractall(path=dirsip)
        for fn in archive.namelist():
            if os.path.splitext(fn)[1] in self.VALID_CAL_EXTENSIONS and os.path.basename(fn)[0] != '.':
                # Valide extension and not hidden file
                empty_sip = False
                foo = Parser(os.path.join(dirsip, fn), immersed)
                self.cal[foo.frame_header] = foo
                self.max_frame_header_length = max(self.max_frame_header_length, len(foo.frame_header))
        if empty_sip:
            raise CalibrationFileEmptyError('No calibration file found in sip')

    def parse_frame_v0(self, frame):
        warnings.warn("deprecated method", DeprecationWarning)
        # DEPRECATED (different output and slower as treat each wavelength individually)
        # get frame_header
        frame_header = frame[0:10].decode(self.ENCODING, self.UNICODE_HANDLING)
        try:
            parser = self.cal[frame_header]
        except KeyError:
            raise FrameHeaderNotFoundError('Unable to find frame header in loaded calibration files')
        if parser.variable_frame_length:
            # Variable length frame
            d = dict()
            # Get byte value from each field (type)
            frame = frame[11:].decode(self.ENCODING, self.UNICODE_HANDLING)  # skip first value separator (comma)
            for k, s, t in zip(parser.key[0:-1], parser.field_separator[1:], parser.data_type):
                index_sep = frame.find(s)
                # Convert from byte to proper d type
                d[k] = frame[0:index_sep]
                if t == 'AI':
                    d[k] = int(d[k])
                elif t == 'AF':
                    d[k] = float(d[k])
                elif t != 'AS':
                    raise ParserDecodeError("Parser data type not supported '" + t + "'")
                frame = frame[index_sep + 1:]
            # Last element
            d[parser.key[-1]] = frame
            if parser.data_type[-1] == 'AI':
                d[parser.key[-1]] = int(d[parser.key[-1]])
            elif parser.data_type[-1] == 'AF':
                d[parser.key[-1]] = float(d[parser.key[-1]])
            elif parser.data_type[-1] != 'AS':
                raise ParserDecodeError("Parser data type not supported '" + parser.data_type[-1] + "'")
        else:
            # Fixed length frame
            # Decode binary data
            if parser.frame_length != len(frame):
                # print('Fixed frame lenght wrong size.')
                # print(frame_header, parser.frame_length, len(frame[10:]))
                return {}
            d = unpack(parser.frame_fmt, frame[10:])
            # Convert from tuple to list and remove terminator (\r\n) DEPRECATED
            # if self.include_terminator:
            #     d = list(d[0:-1])
            # else:
            #     d = list(d)
            d = list(d)
            # Skip Terminator
            if 'TERMINATOR' in parser.id:
                del d[parser.id.index('TERMINATOR')]
            aint = None
            # Loop through all the fields
            for j in range(parser.frame_nfields):
                # Decode ASCII
                if parser.data_type[j] in ['AS', 'AI', 'AF']:
                    d[j] = d[j].decode(self.ENCODING, self.UNICODE_HANDLING)
                if parser.data_type[j] == 'AI':
                    d[j] = int(d[j])
                elif parser.data_type[j] == 'AF':
                    d[j] = float(d[j])
                # Apply special processing
                if parser.fit_type[j] == 'POLYU':
                    # Un-factored polynomial
                    foo = 0
                    for k in range(len(parser.cal_coefs[j])):
                        foo += parser.cal_coefs[j][k] * d[j] ** k
                    d[j] = foo
                elif parser.fit_type[j] == 'OPTIC2':
                    a0 = parser.cal_coefs[j][0]
                    a1 = parser.cal_coefs[j][1]
                    im = parser.cal_coefs[j][2]
                    d[j] = im * a1 * (d[j] - a0)
                elif parser.fit_type[j] == 'OPTIC3':
                    # Get INTTIME (need to be computed before)
                    # integration time of sensor sampling
                    if aint is None:
                        aint = d[parser.type.index('INTTIME')]
                    # Get other coefs
                    a0 = parser.cal_coefs[j][0]
                    a1 = parser.cal_coefs[j][1]
                    im = parser.cal_coefs[j][2]  # Incorrect if immersed
                    cint = parser.cal_coefs[j][3]
                    # Compute Lu in scientific units
                    d[j] = im * a1 * (d[j] - a0) * (cint / aint)
                elif parser.fit_type[j] not in ['NONE', 'COUNT']:
                    raise ParserFitError("Parser fit type not supported '" + parser.fit_type[j] + "'")
            # Create Dict
            d = dict(zip(parser.key, d))
        return d

    def check_frame(self, frame):
        """
        Check completeness of frame, recommended to run before parsing
        DEPRECATED prefer find_frame instead as it only supports frame header of size 10
        :param frame:
        :return: empty string when pass check, one word otherwise indicating the source of the problem
        """
        # Check frame length
        if len(frame) < 10:
            return 'short'
        # Check frame header
        frame_header = frame[0:10].decode(self.ENCODING, self.UNICODE_HANDLING)
        if frame_header not in self.cal.keys():
            if frame_header[:6] == 'SATHDR':
                # Exception for Header Frame Recorded by SatView
                return ''
            # Option 1: unknown frame header
            # Option 2: Poor parsing
            return 'unknown'
        # Check frame consistent with parser description
        parser = self.cal[frame_header]
        if parser.variable_frame_length:
            # Check Terminator
            if frame[-len(parser.frame_terminator):].decode("unicode_escape") == parser.frame_terminator:
                return ''
            else:
                return 'terminator'
        else:
            # Check Size & Frame Terminator if one
            if parser.frame_length == len(frame):
                if not parser.frame_terminator:
                    return ''
                elif frame[-len(parser.frame_terminator):].decode("unicode_escape") == parser.frame_terminator:
                    return ''
                else:
                    return 'terminator'
            elif parser.frame_length > len(frame):
                return 'short'
            elif parser.frame_length < len(frame):
                return 'long'

    def find_frame(self, buffer):
        """
        Find the first known and complete frame from the buffer
        :param buffer: byte array
        :return: frame: first known frame (frame header is loaded)
                 frame_type: frame header of frame found
                 buffer_post_frame: buffer left after the frame
                 buffer_pre_frame: buffer preceding the first frame returned (likely unknown frame header)
        """
        # Find first frame header
        end_search_index = len(buffer)
        if end_search_index < self.max_frame_header_length:
            # Buffer too short to fine frame header
            return bytearray(), None, buffer, bytearray()
        frame_header = None
        frame_header_index = -1
        for fh in self.cal.keys():
            fhi = buffer.find(bytes(fh, self.ENCODING), 0, end_search_index)
            if fhi == 0:
                frame_header = fh
                frame_header_index = fhi
                break
            elif fhi > 0:
                frame_header = fh
                frame_header_index = fhi
                end_search_index = fhi
        if frame_header:
            if self.cal[frame_header].variable_frame_length:
                # Look for frame terminator
                frame_end_index = buffer.find(self.cal[frame_header].frame_terminator_bytes,
                                              frame_header_index)
                if frame_end_index == -1:
                    # Buffer too short (need to get more data in buffer)
                    return bytearray(), None, buffer, bytearray()
                frame_end_index += len(self.cal[frame_header].frame_terminator_bytes)
                return buffer[frame_header_index:frame_end_index], frame_header,\
                       buffer[frame_end_index:], buffer[:frame_header_index]
            else:
                frame_end_index = frame_header_index + self.cal[frame_header].frame_length
                if len(buffer) - frame_end_index < 0:
                    # Buffer too short (need to get more data in buffer)
                    return bytearray(), None, buffer, bytearray()
                if self.cal[frame_header].frame_terminator:
                    if buffer[frame_header_index:frame_end_index][-len(self.cal[frame_header].frame_terminator_bytes):] \
                            != self.cal[frame_header].frame_terminator_bytes:
                        # Invalid frame terminator (skip frame)
                        return bytearray(), frame_header, buffer[frame_end_index:], buffer[:frame_end_index]
                return buffer[frame_header_index:frame_end_index], frame_header,\
                       buffer[frame_end_index:], buffer[:frame_header_index]
        else:
            # No frame found
            return bytearray(), None, bytearray(), buffer

    def parse_frame(self, frame, frame_header=None, *args, **kwargs):
        if not frame_header:
            # Attempt to guess frame header from Standard 10 characters SAT headers
            frame_header = frame[0:10].decode(self.ENCODING, self.UNICODE_HANDLING)
            if frame_header not in self.cal.keys():
                # Attempt to find one of the know frame header
                frame_header = None
                for fh in self.cal.keys():
                    fhi = frame.find(bytes(fh, self.ENCODING))
                    if fhi == 0:
                        frame_header = fh
                        break
                if not frame_header:
                    raise FrameHeaderNotFoundError('Frame header not found or parser missing.')
        parser = self.cal[frame_header]
        raw_decoded = self.decode(frame, parser)
        return self.calibrate(raw_decoded, parser, frame, *args, **kwargs)

    @staticmethod
    def decode(frame, parser):
        if parser.variable_frame_length:
            # Variable length frame
            rd = list()
            # Decode value of each field
            frame = frame[parser.frame_header_length + 1:].decode(Instrument.ENCODING,
                                                                  Instrument.UNICODE_HANDLING)  # skip first value separator (comma)
            for s, t in zip(parser.field_separator[1:], parser.data_type[:-1]):
                index_sep = frame.find(s)
                if index_sep == -1:
                    valid_frame = False
                    continue
                rd.append(Instrument._decode_ascii_data(frame[0:index_sep], t, force_ascii=True))
                frame = frame[index_sep + 1:]
        else:
            # Fixed length frame
            # Decode binary data
            if parser.frame_length != len(frame):
                raise FrameLengthError('Unexpected frame length: %s expected %d actual %d' %
                                       (parser.frame_header, parser.frame_length, len(frame)))
            rd = unpack(parser.frame_fmt, frame[10:])
            # Decode ASCII variables
            rd = [Instrument._decode_ascii_data(v, t) for v, t in zip(rd, parser.data_type)]
        return rd

    @staticmethod
    def calibrate(rd, parser, frame=None, flag_get_auxiliary_variables=None, flag_get_unusable_variables=False):
        # Get integration time if available as required from OPTIC3 fit
        if 'INTTIME' in parser.type:
            i = parser.type.index('INTTIME')
            aint = Instrument._fit_data(rd[i], parser.fit_type[i], parser.cal_coefs[i], immersed=parser.immersed)
        else:
            aint = None
        # Core Variables (same data and fit types, serialize process in numpy)
        if parser.core_variables:
            d = {parser.core_groupname: Instrument._fit_data(np.array(itemgetter(*parser.core_variables)(rd)),
                                                             parser.fit_type[parser.core_variables[0]],
                                                             parser.core_cal_coefs, aint, parser.immersed)}
        else:
            d = dict()
        # Unusable Variables (same data and fit types, serialize process in numpy)
        if flag_get_unusable_variables and parser.unusable_variables:
            d[parser.unusable_groupname] = Instrument._fit_data(np.array(itemgetter(*parser.unusable_variables)(rd)),
                                                                parser.fit_type[parser.unusable_variables[0]],
                                                                parser.unusable_cal_coefs,
                                                                aint, parser.immersed)
        # Auxiliary variables (default: off: if core_variables | on: if no core variables)
        if (flag_get_auxiliary_variables is None and not parser.core_variables) or flag_get_auxiliary_variables:
            valid_frame = True
            for j in parser.auxiliary_variables:
                # Special fields
                if parser.key[j] == 'CRLF_TERMINATOR':
                    if not parser.variable_frame_length:
                        if rd[j] != 3338:  # unpack('!H', b'\r\n') as data type in calibration file is BU
                            valid_frame = False
                    continue
                elif parser.type[j] == 'TERMINATOR':
                    if not parser.variable_frame_length:
                        if rd[j] != parser.frame_terminator:
                            valid_frame = False
                    continue
                elif parser.key[j] == 'CHECK_SUM':
                    if parser.variable_frame_length:
                        # if rd[j] != self.compute_nmea_check_sum(frame):
                        #     valid_frame = False
                        pass
                    elif frame is not None:
                        if rd[j] != Instrument.compute_check_sum(frame, parser.check_sum_index):
                            valid_frame = False
                d[parser.key[j]] = Instrument._fit_data(rd[j], parser.fit_type[j], parser.cal_coefs[j], aint,
                                                        parser.immersed)
        else:
            valid_frame = None
        return d, valid_frame

    @staticmethod
    def _decode_ascii_data(value, data_type, force_ascii=False):
        if data_type in ['AS', 'AI', 'AI16', 'AF']:
            try:
                # Convert from byte to string
                foo = value.decode(Instrument.ENCODING, Instrument.UNICODE_HANDLING)
            except (UnicodeDecodeError, AttributeError):
                foo = value
            if data_type == 'AI':
                return int(foo)
            elif data_type == 'AI16':
                return int(foo, 16)
            elif data_type == 'AF':
                return float(foo)
            return foo
        elif force_ascii:
            raise ParserDecodeError('Non ASCII data type ' + data_type)
        return value

    @staticmethod
    def _fit_data(value, fit_type, cal_coefs, aint=None, immersed=False):
        if fit_type == 'OPTIC2':
            # Special factored polynomial with one gain range. For optical sensors only.
            a0, a1, im = cal_coefs
            im = im if immersed else 1.0
            return im * a1 * (value - a0)
            # return (value - sc_coefs[0]) * sc_coefs[1]
        elif fit_type == 'OPTIC3':
            # Special factored polynomial with one linearly adaptive gain range. For optical sensors only.
            # aint = integration time (INTTIME)
            a0, a1, im, cint = cal_coefs
            im = im if immersed else 1.0
            return im * a1 * (value - a0) * (cint / aint)
            # return (value - sc_coefs[0]) * sc_coefs[1] / aint
        elif fit_type == 'POW10':
            # Special exponential equation for logarithmic sensors.
            a0, a1, im = cal_coefs
            im = im if immersed else 1.0
            return im * 10 ** ((value - a0)/a1)
        elif fit_type == 'POLYU':
            # Un-factored polynomial
            foo = 0
            for k in range(len(cal_coefs)):
                foo += cal_coefs[k] * value ** k
            return foo
        elif fit_type == 'POLYF':
            # Factored polynomial
            foo = cal_coefs[0]
            for k in range(1, len(cal_coefs)):
                foo *= value - cal_coefs[k]
            return foo
        elif fit_type == 'GPSTIME':
            # Universal Coordinated Time of GPS data.
            # value format is hhmmss.s
            # return decimal hours
            hh = int(value/10000)
            mm = int((value-hh*10000) / 100)
            ss = value - hh*10000 - mm*100
            return hh + mm / 60 + ss / 3600
        elif fit_type == 'GPSPOS':
            # GPS global position.
            # value format is dddmm.m
            # return decimal degrees
            d = int(value/100)
            m = value-d*100
            return d + m / 60
        elif fit_type == 'GPSHEMI':
            # GPS global position hemisphere.
            # value format is N, E, S, W
            return 1.0 if value in ['N', 'E'] else -1.0 if value in ['S', 'W'] else 0.0
        elif fit_type == 'GPSMODE':
            # GPS Positioning System Mode Indicator.
            if value == 'A':
                return 1.0
            elif value == 'D':
                return 2.0
            elif value == 'E':
                return 3.0
            elif value == 'M':
                return 4.0
            elif value == 'S':
                return 5.0
            elif value == 'N':
                return 6.0
            else:
                return 0.0
        elif fit_type == 'GPSSTATUS':
            # GPS system status.
            return True if value == 'A' else False
        elif fit_type == 'DDMM':
            # GPS global position in degrees, minutes, and seconds.
            d = int(value / 100)
            m = int(value - d * 100)
            s = int(((value - d * 100) - m) * 60)
            return f'{d} {m}\' {s}\'\''
        elif fit_type == 'HHMMSS':
            # Universal Coordinated Time of GPS data in hours, minutes, and seconds.
            # DIVERGE FROM SATLANTIC SPECIFICATIONS CONVERT TO PYTHON timedelta
            hh = int(value / 10000)
            mm = int((value - hh * 10000) / 100)
            ss = value - hh * 10000 - mm * 100
            return timedelta(hours=hh, minutes=mm, seconds=ss)
        elif fit_type == 'DDMMYY':
            # GPS date of global position.
            # DIVERGE FROM SATLANTIC SPECIFICATIONS CONVERT TO PYTHON datetime
            dd = int(value / 10000)
            mm = int((value - dd * 10000) / 100)
            yy = int(value - dd * 10000 - mm * 100)
            return datetime(yy, mm, dd)
        elif fit_type == 'TIME2':
            # Time tag for the frame.
            dt = datetime.utcfromtimestamp(value)
            return dt.strftime('%y-%m-%d %H.%M.%S')
        elif fit_type == 'COUNT':
            # Raw or un-calibrated information.
            return value
        elif fit_type == 'NONE':
            # Unusable data. (Used if frame data is to be ignored)
            # warnings('Fit type processing should not be applied to a NONE sensor.')
            return None
        else:
            raise ParserFitError(f"Fit type {fit_type} not supported.")

    @staticmethod
    def compute_check_sum(frame, check_sum_index=-3):
        # Last byte of the sum of all bytes substracted from 0
        #   from frame header (included) to checksum (excluded)
        return np.uint8(0 - sum(frame[0:check_sum_index]))

    def compute_nmea_check_sum(self, frame, start_index=1, end_index=-5):
        """
        Compute checksum for NMEA frames
        :param frame:
        :param start_index:
        :param end_index:
        :return:
        """
        checksum = 0
        for s in frame[1:-5].decode(self.ENCODING, self.UNICODE_HANDLING):
            checksum ^= ord(s)
        return checksum

    def read_satview(self, filename, **kwargs):
        """
        Read SatView raw file
        Assume frame is valid if parse_frame return valid=None

        :param filename: SatView file to read
        :param kwargs: key word arguments for parse_frame
        :param debug: display unknown bytes and unknown frames
        :return: list of FrameContainers, metadata dictionary
        """
        data = []
        ignored_bytes, invalid_timestamps = 0, 0
        valid_frames, invalid_frames = dict(), dict()
        sathdr_present = False
        with open(filename, 'rb') as f:
            buffer = bytearray(f.read(32768))
            while buffer:
                frame = True
                while frame or unknown_bytes:
                    # Get Frame
                    frame, frame_header, buffer, unknown_bytes = self.find_frame(buffer)
                    if unknown_bytes:
                        if unknown_bytes[:6] == b'SATHDR':
                            sathdr_present = True
                        else:
                            # print(unknown_bytes)
                            ignored_bytes += len(unknown_bytes)
                            if frame_header and not frame:
                                if frame_header not in invalid_frames.keys():
                                    invalid_frames[frame_header] = 1
                                else:
                                    invalid_frames[frame_header] += 1
                    if frame:
                        if len(buffer) >= 7:
                            # Get SatView timestamp
                            ts = buffer[:7]
                            buffer = buffer[7:]
                            # Parse timestamp
                            ts = unpack('!ii', b'\x00' + ts)
                            try:
                                timestamp = datetime.strptime(f'{ts[0]}{ts[1]:09d}000', '%Y%j%H%M%S%f')
                            except ValueError as e:
                                warnings.warn('Time Impossible, frame likely corrupted.')
                                timestamp = 'NaN'
                                invalid_timestamps += 1
                            # Calibrate frame
                            parsed_frame, valid = self.parse_frame(frame, frame_header, **kwargs)
                            if valid == True or valid is None:
                                data.append(FrameContainer(header=frame_header, data=parsed_frame,
                                                           timestamp=timestamp, valid=valid))
                                if frame_header not in valid_frames.keys():
                                    valid_frames[frame_header] = 1
                                else:
                                    valid_frames[frame_header] += 1
                            else:
                                if frame_header not in invalid_frames.keys():
                                    # print(frame)
                                    invalid_frames[frame_header] = 1
                                else:
                                    invalid_frames[frame_header] += 1
                        else:
                            # Missing data to read timestamp from SatView
                            # Refill buffer and go grab more data
                            buffer = frame + buffer
                            break
                tmp_buffer = f.read(32768)
                if tmp_buffer:
                    buffer.extend(tmp_buffer)
                else:
                    break
        # Read metadata from file
        stat = os.stat(filename)
        meta = dict(file_name=os.path.basename(filename), file_size=stat.st_size,
                    file_created=datetime.utcfromtimestamp(stat.st_ctime).replace(tzinfo=timezone.utc),
                    file_modified=datetime.utcfromtimestamp(stat.st_mtime).replace(tzinfo=timezone.utc),
                    sathdr_present=sathdr_present, valid_frames=valid_frames, invalid_frames=invalid_frames,
                    invalid_timestamps=invalid_timestamps, ignored_bytes=ignored_bytes)
        return data, meta

    def __str__(self):
        foo = ""
        for c in self.cal.values():
            foo += str(c)
        return foo


class CSVWriter:

    def __init__(self):
        self.f = None
        self.writer = None

    def open(self, filename, fieldnames):
        self.f = open(filename, 'w')
        self.writer = csv.writer(self.f)  # , fieldnames=fieldnames)
        # self.writer.writeheader()
        self.writer.writerow(fieldnames)

    def write(self, data):
        self.writer.writerow(data)

    def close(self):
        if self.f:
            self.f.close()


class SatViewRawToCSV:
    # REGISTRATION = b'SAT'
    READ_SIZE = 1024

    def __init__(self, calibration_filename, raw_filename=None, immersed=False):
        self.buffer = bytearray()
        self.frame_buffer = bytearray()
        self.w = dict()
        self.frame_received = 0
        self.frame_parsed = 0
        self.frame_unregistered = 0
        self.missing_frame_header = []
        self.instrument = Instrument(calibration_filename, immersed)
        if raw_filename:
            self.run(raw_filename)

    def run(self, filename):
        with open(filename, 'rb') as f:
            # Open output csv files
            [filename, _] = os.path.splitext(filename)
            for k, cal in self.instrument.cal.items():
                self.w[k] = CSVWriter()
                disp_key = cal.key.copy()
                disp_aux_var = cal.auxiliary_variables.copy()
                if 'TERMINATOR' in cal.key:
                    del disp_aux_var[disp_aux_var.index(disp_key.index('TERMINATOR'))]
                    del disp_key[disp_key.index('TERMINATOR')]
                elif 'CRLF_TERMINATOR' in cal.key:
                    del disp_aux_var[disp_aux_var.index(disp_key.index('CRLF_TERMINATOR'))]
                    del disp_key[disp_key.index('CRLF_TERMINATOR')]
                if cal.core_variables:
                    fieldnames = ['TIMESTAMP'] + list(itemgetter(*cal.core_variables)(disp_key)) + \
                                 list(itemgetter(*disp_aux_var)(disp_key))
                else:
                    fieldnames = ['TIMESTAMP'] + disp_key
                self.w[k].open(filename + '_' + k + '.csv', fieldnames)
            # Parse data
            data = f.read(self.READ_SIZE)
            while data:
                self.data_read(data)
                data = f.read(self.READ_SIZE)
            # Close files
            for k in self.instrument.cal.keys():
                self.w[k].close()

    # Method using check frame (less elegant)
    # def data_read(self, data):
    #     self.buffer.extend(data)
    #     while self.REGISTRATION in self.buffer:
    #         frame, tmp = self.buffer.split(self.REGISTRATION, 1)
    #         if self.frame_buffer:
    #             frame = self.frame_buffer + self.REGISTRATION + frame
    #             self.frame_buffer = None
    #         if len(frame) > 10:
    #             # Check frame without satview timestamp which would be the 7 last bytes
    #             fail = self.instrument.check_frame(self.REGISTRATION + frame[:-7])
    #             if not fail:
    #                 self.handle_frame(self.REGISTRATION + frame[:-7], frame[-7:])
    #                 self.buffer = tmp
    #             elif fail == 'short':
    #                 self.frame_buffer = frame
    #                 self.buffer = tmp
    #             else:
    #                 self.buffer = tmp
    #                 print('WARNING: Unable to register data.')
    #                 self.frame_unregistered += 1
    #         else:
    #             self.buffer = tmp
    #
    # def last_data_read(self):
    #     self.handle_frame(self.REGISTRATION + self.buffer[:-7], self.buffer[-7:])

    def data_read(self, data):
        self.buffer.extend(data)
        frame = True
        while frame:
            # Get Frame
            frame, frame_header, self.buffer, unknown_bytes = self.instrument.find_frame(self.buffer)
            if unknown_bytes:
                # print('WARNING: Skipped non-registered data.')
                # print(unknown_bytes)
                pass
            if frame:
                if len(self.buffer) >= 7:
                    # Get SatView timestamp
                    timestamp = self.buffer[:7]
                    self.buffer = self.buffer[7:]
                    # Handle frame
                    self.handle_frame(frame, timestamp, frame_header)
                else:
                    # Missing data to read timestamp from SatView
                    # Refill buffer and go grab more data
                    self.buffer = frame + self.buffer
                    break

    def handle_frame(self, frame, timestamp, frame_header=None):
        # Skip SatView Metadata Frames
        if frame[:6] == b'SATHDR':
            return
        self.frame_received += 1
        # Parse timestamp
        d = unpack('!ii', b'\x00' + timestamp)
        try:
            timestamp = datetime.strptime(str(d[0]) + str(d[1]) + '000', '%Y%j%H%M%S%f').strftime(
                '%Y/%m/%d %H:%M:%S.%f')[:-3]
        except ValueError:
            warnings.warn('Time Impossible, frame likely corrupted.')
            timestamp = 'NaN'
        # Parse frame data
        try:
            if not frame_header:
                frame_header = frame[0:10].decode(self.instrument.ENCODING, self.instrument.UNICODE_HANDLING)
            [parsed_frame, valid_frame] = self.instrument.parse_frame(frame, frame_header,
                                                                      flag_get_auxiliary_variables=True)
        except FrameHeaderNotFoundError:
            if frame_header not in self.missing_frame_header:
                self.missing_frame_header.append(frame_header)
                warnings.warn(f'Missing calibration file for: {frame_header}')
            return
        except FrameLengthError as e:
            print(e)
            return
        if valid_frame:
            self.frame_parsed += 1
        # Write data
        if self.instrument.cal[frame_header].core_variables:
            data = next(iter(parsed_frame.values())).tolist()
            data = ["%.10f" % v for v in data]
            for k in [k for i, k in enumerate(self.instrument.cal[frame_header].key) if
                      i in self.instrument.cal[frame_header].auxiliary_variables]:
                if k in parsed_frame.keys():
                    if isinstance(parsed_frame[k], float):
                        data.append('%.4f' % parsed_frame[k])
                    else:
                        data.append(str(parsed_frame[k]))
        else:
            data = []
            for k in self.instrument.cal[frame_header].key:
                if k in parsed_frame.keys():
                    if isinstance(parsed_frame[k], float):
                        data.append('%.4f' % parsed_frame[k])
                    else:
                        data.append(str(parsed_frame[k]))
        self.w[frame_header].write([timestamp] + data)

    def __del__(self):
        if hasattr(self, 'w'):
            for k in self.w.keys():
                self.w[k].close()
