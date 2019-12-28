import argparse
from pySatlantic import __version__
from pySatlantic.instrument import SatViewRawToCSV

# Argument Parser
parser = argparse.ArgumentParser(prog="python -m pySatlantic")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Enable verbosity.")
parser.add_argument("--version", action="version", version='pySatlantic v' + __version__,
                    help="Prints version information.")
parser.add_argument("cal", type=str,
                    help="Calibration file.")
parser.add_argument("src", type=str,
                    help="Raw file to decode and calibrate.")
parser.add_argument("-i", "--immersed", action="store_false",
                    help="Apply immersion coefficients.")
args = parser.parse_args()

# Decode and Calibrate binary file
if args.verbose:
    print('Converting ' + args.src + ' ... ', end='', flush=True)
sat = SatViewRawToCSV(args.cal, args.src, args.immersed)
if args.verbose:
    print('Done')
    print('Frame extracted: ' + str(sat.frame_parsed))
