from __future__ import print_function
from os import listdir
from os.path import isfile, join, dirname, splitext, isdir
from struct import unpack
import zipfile
import numpy as np
import csv
from operator import itemgetter
from datetime import datetime


# Error Management
class SatlanticInstrumentError(Exception):
    pass


class FrameError(SatlanticInstrumentError):
    pass


class FrameHeaderIncompleteError(FrameError):
    pass


class FrameHeaderNotFoundError(FrameError):
    pass


class FrameLengthError(FrameError):
    pass


class ParserError(SatlanticInstrumentError):
    pass


class ParserDecodeError(ParserError):
    pass


class ParserFitError(ParserError):
    pass


class CalibrationFileError(SatlanticInstrumentError):
    pass


class CalibrationFileExtensionError(CalibrationFileError):
    pass


class Calibration:
    """
    Calibration class for parsing single calibration files
        get the format of the raw data
        get the calibration equations and coefficients
    """

    CORE_VARIABLE_TYPES = ['LT', 'LI', 'LU', 'ED', 'ES', 'EU']

    def __init__(self, cal_filename=None, immersed=False):
        # Metadata
        self.instrument = ''
        self.sn = -999
        self.frame_header = ''
        self.immersed = immersed
        self.frame_rate = None
        self.baudrate = None
        self.calibration_time = {}
        self.calibration_temperature = None
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
        Read filename which is expected to be a .cal file

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
            # for l in f.readlines(): load entire file in memory
            for l in f:  # read file line by line
                lc = l.strip()  # remove leading and trailing characters
                if lc == '' or lc[0] == '#':
                    # Skip empty line and comments
                    continue
                # Frame header
                if lc[0:10] == 'INSTRUMENT':
                    # Special case instrument name
                    self.instrument = lc[11:17]
                    continue
                if lc[0:2] == 'SN':
                    # Special case serial number
                    self.sn = int(lc[3:7])
                    continue
                if lc[0:14] == 'VLF_INSTRUMENT':
                    # Special case used for variable length frame instruments
                    self.instrument = lc[15:25]
                    continue
                # Variable frame length
                if lc[0:5] == 'FIELD':
                    self.variable_frame_length = True
                    ls = l.split()
                    self.field_separator.append(bytes(ls[2][1:-1], "ASCII").decode("unicode_escape"))
                    continue
                # Terminator included in cal
                if lc[0:4] == 'CRLF' or lc[0:10] == 'TERMINATOR':
                    # Skip terminator
                    continue
                # Pseudo Sensors
                if lc[0:4] == 'RATE':
                    self.frame_rate = int(l.split()[1])
                    continue
                if lc[0:8] == 'DATARATE':
                    self.baudrate = int(l.split()[1])
                    continue
                if lc[0:7] == 'CALTIME':
                    ls = l.split()
                    self.calibration_time[ls[1]] = float(ls[2][1:-1])
                    continue
                if lc[0:7] == 'CALTEMP':
                    self.calibration_temperature = float(l.split()[1])
                    continue
                if lc[0:12] == 'THERMAL_RESP':
                    # Get lines of coefficients corresponding to thermal response
                    l = f.readline().strip().split()
                    for c in l:
                        self.thermal_response.append(float(c))
                    continue
                # Frame fields
                ls = l.split('#')[0].split()
                if len(ls) == 7:
                    # Get values from each line
                    self.type.append(ls[0])
                    self.id.append(ls[1])
                    if ls[1] != 'NONE':
                        self.key.append(ls[0] + '_' + ls[1])
                    else:
                        self.key.append(ls[0])
                    self.units.append(ls[2][1:-1])  # rm initial and final '
                    if not self.variable_frame_length:
                        self.field_length.append(int(ls[3]))
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
            # Build frame header
            if self.sn == -999:
                self.frame_header = self.instrument
            else:
                self.frame_header = self.instrument + '%0*d' % (4, self.sn)
            if not self.variable_frame_length:
                # Build frame parser
                self.frame_nfields = len(self.type)
                for i in range(self.frame_nfields):
                    self.frame_length += self.field_length[i]
                    if self.data_type[i] == 'AS':
                        # ASCII string (text)
                        self.frame_fmt += str(self.field_length[i]) + 's'
                    elif self.data_type[i] == 'AI':
                        # ASCII integer number
                        self.frame_fmt += str(self.field_length[i]) + 's'
                    elif self.data_type[i] == 'AF':
                        # ASCII floating point number
                        self.frame_fmt += str(self.field_length[i]) + 's'
                    elif self.data_type[i] == 'BS' and self.field_length[i] == 1:
                        # signed short 2 bytes
                        self.frame_fmt += 'b'
                    elif self.data_type[i] == 'BU' and self.field_length[i] == 1:
                        # unsigned short 2 bytes
                        self.frame_fmt += 'B'
                    elif self.data_type[i] == 'BS' and self.field_length[i] == 2:
                        # signed short 2 bytes
                        self.frame_fmt += 'h'
                    elif self.data_type[i] == 'BU' and self.field_length[i] == 2:
                        # unsigned short 2 bytes
                        self.frame_fmt += 'H'
                    elif self.data_type[i] == 'BS' and self.field_length[i] == 4:
                        # signed integer 4 bytes
                        self.frame_fmt += 'i'
                    elif self.data_type[i] == 'BU' and self.field_length[i] == 4:
                        # unsigned integer 4 bytes
                        self.frame_fmt += 'I'
                    elif self.data_type[i] == 'BF' and self.field_length[i] == 4:
                        # float
                        self.frame_fmt += 'f'
                    elif self.data_type[i] == 'BD' and self.field_length[i] == 8:
                        # double float
                        self.frame_fmt += 'd'
                    else:
                        raise CalibrationFileError('Missing byte decoder ' +
                              str(self.data_type[i]) + str(self.field_length[i]))
                # if force_terminator: DEPRECATED
                #     # Add terminator at the end
                #     # 2 bytes for carriage return (CR) and line feed (LF)
                #     # This was required for HyperNav files
                #     self.frame_fmt += 'H'  # 'H' -> 3338 or '2s' -> b'\r\n'
                #     self.frame_length += 2

            # Group Variables
            self.core_variables = [i for i, (x, y) in enumerate(zip(self.type, self.fit_type))
                                   if x.upper() in self.CORE_VARIABLE_TYPES and y != 'NONE']
            if self.core_variables:
                self.core_groupname = '%s_%s' % (self.type[self.core_variables[0]], self.instrument)
            self.unusable_variables = [i for i, (x, y) in enumerate(zip(self.type, self.fit_type))
                                       if x.upper() in self.CORE_VARIABLE_TYPES and y == 'NONE']
            if self.unusable_variables:
                self.unusable_groupname = '%s_%s_RAW' % (self.type[self.core_variables[0]], self.instrument)
            self.auxiliary_variables = [i for i, (x, y) in enumerate(zip(self.type, self.fit_type)) if
                                        x.upper() not in self.CORE_VARIABLE_TYPES]

            # Convert calibration coefficients to numpy array for fast computation
            if self.core_variables:
                self.core_cal_coefs = np.array(itemgetter(*self.core_variables)(self.cal_coefs)).transpose()
            if self.unusable_variables:
                self.unusable_cal_coefs = np.array(itemgetter(*self.unusable_variables)(self.cal_coefs)).transpose()

            # check for errors
            if self.variable_frame_length:
                for t in self.data_type:
                    if t not in ['AS', 'AI', 'AF']:
                        raise ValueError('Invalid data_type for Variable frame length.')

    def __str__(self):
        if self.variable_frame_length:
            return 'Instrument: ' + self.instrument + '\n' + \
                   'Serial number: ' + str(self.sn) + '\n' + \
                   'Number of fields: ' + str(self.frame_nfields) + '\n' + \
                   'Variable frame length: ' + str(self.variable_frame_length) + '\n' + \
                   'Frame length (in bytes): ' + str(self.frame_length) + '\n' + \
                   'Variables type: ' + str(self.type) + '\n' + \
                   'Variables id: ' + str(self.id) + '\n' + \
                   'Variables units: ' + str(self.units) + '\n' + \
                   'Variables fit type:' + str(self.fit_type) + '\n' + \
                   'Variables field separator: ' + str(self.field_separator) + '\n' + \
                   'Variables data type: ' + str(self.data_type) + '\n'
        else:
            return 'Instrument: ' + self.instrument + '\n' + \
                   'Serial number: ' + str(self.sn) + '\n' + \
                   'Number of fields: ' + str(self.frame_nfields) + '\n' + \
                   'Frame length (in bytes): ' + str(self.frame_length) + '\n' + \
                   'Variables type: ' + str(self.type) + '\n' + \
                   'Variables id: ' + str(self.id) + '\n' + \
                   'Variables units: ' + str(self.units) + '\n' + \
                   'Variables fit type:' + str(self.fit_type) + '\n' + \
                   'Variables field length: ' + str(self.field_length) + '\n' + \
                   'Variables data type: ' + str(self.data_type) + '\n' + \
                   'Variables frame format: ' + str(self.frame_fmt) + '\n'


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

        if filename is not None:
            if type(filename) is not list:
                filename = [filename]
            for f in filename:
                if isdir(f):
                    self.read_calibration_dir(f, immersed)
                else:
                    _, ext = splitext(f)
                    if ext in self.VALID_SIP_EXTENSIONS:
                        self.read_sip_file(f, immersed)
                    elif ext in self.VALID_CAL_EXTENSIONS:
                        self.read_calibration_file(f, immersed)

    def read_calibration_file(self, filename, immersed=False):
        _, ext = splitext(filename)
        if ext in self.VALID_CAL_EXTENSIONS:
            foo = Calibration(filename, immersed)
            self.cal[foo.frame_header] = foo
        else:
            raise CalibrationFileExtensionError('Calibration file extension incorrect.')

    def read_calibration_dir(self, dirname, immersed=False):
        for fn in listdir(dirname):
            if isfile(join(dirname, fn)):
                _, ext = splitext(fn)
                if ext in self.VALID_CAL_EXTENSIONS:
                    foo = Calibration(join(dirname, fn), immersed)
                    self.cal[foo.frame_header] = foo

    def read_sip_file(self, filename, immersed=False):
        archive = zipfile.ZipFile(filename, 'r')
        dirsip = dirname(filename)
        archive.extractall(path=dirsip)
        for fn in archive.namelist():
            _, ext = splitext(fn)
            if ext in self.VALID_CAL_EXTENSIONS:
                foo = Calibration(join(dirsip, fn), immersed)
                self.cal[foo.frame_header] = foo

    def parse_frame_v0(self, frame):
        # DEPRECATED (different output and slower as treat each wavelength individually)
        # get frame_header
        frame_header = frame[0:10].decode(self.ENCODING, self.UNICODE_HANDLING)
        if not frame_header:
            raise FrameHeaderIncompleteError('Unable to resolve frame header in ' + str(frame))
        if frame_header not in self.cal.keys():
            raise FrameHeaderNotFoundError('Unable to find frame header in loaded calibration files.')
        parser = self.cal[frame_header]
        if parser.variable_frame_length:
            # Variable length frame
            d = dict()
            # Get byte value from each field (type)
            frame = frame[11:].decode(self.ENCODING, self.UNICODE_HANDLING) # skip first value separator (comma)
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
            if parser.frame_length != len(frame[10:]):
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

    def parse_frame(self, frame, flag_get_auxiliary_variables=None, flag_get_unusable_variables=False):
        # get frame_header
        frame_header = frame[0:10].decode(self.ENCODING, self.UNICODE_HANDLING)
        if not frame_header:
            raise FrameHeaderIncompleteError('Unable to resolve frame header in ' + str(frame))
        if frame_header not in self.cal.keys():
            raise FrameHeaderNotFoundError('Unable to find frame header in loaded calibration files.')
        parser = self.cal[frame_header]
        if parser.variable_frame_length:
            # Variable length frame
            d = dict()
            # Decode value of each field
            frame = frame[11:].decode(self.ENCODING, self.UNICODE_HANDLING) # skip first value separator (comma)
            for k, s, t in zip(parser.key[0:-1], parser.field_separator[1:], parser.data_type):
                index_sep = frame.find(s)
                d[k] = self._decode_ascii_data(frame[0:index_sep], t, force_ascii=True)
                frame = frame[index_sep + 1:]
            # Decode last field
            d[parser.key[-1]] = self._decode_ascii_data(frame, parser.data_type[-1], force_ascii=True)
        else:
            # Fixed length frame
            # Decode binary data
            if parser.frame_length != len(frame[10:]):
                raise FrameLengthError('Unexpected frame length: %s is %d instead of %d.' %
                                       (frame_header, parser.frame_length, len(frame[10:])))
            rd = unpack(parser.frame_fmt, frame[10:])
            # Decode ASCII variables
            rd = [self._decode_ascii_data(v, t) for v, t in zip(rd, parser.data_type)]
            # if self.include_terminator:
            #     # Remove terminator that was included up to now(\r\n)
            #     rd = rd[0:-1]
            # Get integration time if available as required from OPTIC3 fit
            if 'INTTIME' in parser.type:
                i = parser.type.index('INTTIME')
                aint = self._fit_data(rd[i], parser.fit_type[i], parser.cal_coefs[i], immersed=parser.immersed)
            else:
                aint = None
            # Core Variables (same data and fit types, serialize process in numpy)
            if parser.core_variables:
                d = {parser.core_groupname: self._fit_data(np.array(itemgetter(*parser.core_variables)(rd)),
                                                           parser.fit_type[parser.core_variables[0]],
                                                           parser.core_cal_coefs, aint, parser.immersed)}
            else:
                d = dict()
            # Unusable Variables (same data and fit types, serialize process in numpy)
            if flag_get_unusable_variables:
                d[parser.unusable_groupname] = self._fit_data(np.array(itemgetter(*parser.unusable_variables)(rd)),
                                                              parser.fit_type[parser.unusable_variables[0]],
                                                              parser.unusable_cal_coefs,
                                                              aint, parser.immersed)
            # Auxiliary variables (default: off: if core_variables | on: if no core variables)
            if (flag_get_auxiliary_variables is None and not parser.core_variables) or flag_get_auxiliary_variables:
                for j in parser.auxiliary_variables:
                    d[parser.key[j]] = self._fit_data(rd[j], parser.fit_type[j], parser.cal_coefs[j], aint,
                                                      parser.immersed)

        return d, frame_header

    def _decode_ascii_data(self, value, data_type, force_ascii=False):
        if data_type in ['AS', 'AI', 'AF']:
            try:
                foo = value.decode(self.ENCODING, self.UNICODE_HANDLING)
            except (UnicodeDecodeError, AttributeError):
                foo = value
            if data_type == 'AI':
                return int(foo)
            elif data_type == 'AF':
                return float(foo)
            return foo
        elif force_ascii:
            raise ParserDecodeError('Non ASCII data type ' + data_type + '.')
        return value

    @staticmethod
    def _fit_data(value, fit_type, cal_coefs, aint=None, immersed=False):
        if fit_type == 'POLYU':
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
        elif fit_type == 'OPTIC2':
            a0 = cal_coefs[0]
            a1 = cal_coefs[1]
            im = cal_coefs[2] if immersed else 1.0
            return im * a1 * (value - a0)
            # return (value - sc_coefs[0]) * sc_coefs[1]
        elif fit_type == 'OPTIC3':
            # aint = integration time (INTTIME)
            a0 = cal_coefs[0]
            a1 = cal_coefs[1]
            im = cal_coefs[2] if immersed else 1.0
            cint = cal_coefs[3]
            return im * a1 * (value - a0) * (cint / aint)
            # return (value - sc_coefs[0]) * sc_coefs[1] / aint
        elif fit_type in ['NONE', 'COUNT']:
            return value
        else:
            raise ParserFitError("Parser fit type not supported '" + fit_type + "'")

    def __str__(self):
        foo = ""
        for c in self.cal.values():
            foo += str(c)
        return foo


class BinReader:
    REGISTRATION = b'SAT'
    READ_SIZE = 1024

    def __init__(self, filename=None):
        self.buffer = bytearray()
        if filename:
            self.run(filename)

    def run(self, filename):
        with open(filename, 'rb') as f:
            data = f.read(self.READ_SIZE)
            while data:
                self.data_read(data)
                data = f.read(self.READ_SIZE)
            self.handle_last_frame(self.REGISTRATION + self.buffer)

    def data_read(self, data):
        self.buffer.extend(data)
        while self.REGISTRATION in self.buffer:
            frame, self.buffer = self.buffer.split(self.REGISTRATION, 1)
            if frame:
                self.handle_frame(self.REGISTRATION + frame)

    def handle_frame(self, frame):
        raise NotImplementedError('Implement functionality in handle packet')

    def handle_last_frame(self, frame):
        return self.handle_frame(frame)


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


class SatViewRawToCSV(BinReader):

    FRAME_TERMINATOR = b'\r\n'

    def __init__(self, calibration_filename, raw_filename, immersed=False):
        # TODO handle instrument immersion coefficients
        # TODO handle auxiliary and unusable data
        self.w = dict()
        self.frame_parsed = 0
        self.missing_frame_header = []
        self.instrument = Instrument(calibration_filename, immersed)
        [filename, _] = splitext(raw_filename)
        for k, cal in self.instrument.cal.items():
            self.w[k] = CSVWriter()
            if cal.core_variables:
                fieldnames = ['TIMESTAMP'] + list(itemgetter(*cal.core_variables)(cal.key))
            else:
                fieldnames = ['TIMESTAMP'] + cal.key
            self.w[k].open(filename + '_' + k + '.csv', fieldnames)
        super(SatViewRawToCSV, self).__init__(raw_filename)

    def handle_frame(self, frame):
        try:
            [frame, timestamp] = frame.split(self.FRAME_TERMINATOR)
        except ValueError:
            return
        # Skip frame header
        if frame[:6] == b'SATHDR':
            return
        # Decode SatView timestamp
        d = unpack('!ii', b'\x00' + timestamp)
        timestamp = datetime.strptime(str(d[0]) + str(d[1]) + '000', '%Y%j%H%M%S%f').strftime('%Y/%m/%d %H:%M:%S.%f')[:-3]
        # Decode frame data
        try:
            [parsed_frame, frame_header] = self.instrument.parse_frame(frame)
        except FrameHeaderNotFoundError:
            frame_header = frame[0:10].decode(self.instrument.ENCODING, self.instrument.UNICODE_HANDLING)
            if frame_header not in self.missing_frame_header:
                self.missing_frame_header.append(frame_header)
                print('WARNING: Missing calibration file for: ' + frame_header)
            return
        if self.instrument.cal[frame_header].core_variables:
            data = next(iter(parsed_frame.values())).tolist()
            data = ["%.10f" % v for v in data]
        else:
            data = []
            for k in self.instrument.cal[frame_header].key:
                if k in parsed_frame.keys():
                    if isinstance(parsed_frame[k], float):
                        data.append('%.2f' % parsed_frame[k])
                    else:
                        data.append(str(parsed_frame[k]))
        # Write data
        self.w[frame_header].write([timestamp] + data)
        self.frame_parsed += 1

    def __del__(self):
        for k in self.w.keys():
            self.w[k].close()

