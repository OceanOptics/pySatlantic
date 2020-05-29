import unittest
import sqlite3
import numpy as np


# Data to test read frame fixed length
TEST_FRAME_VARIABLE_LENGTH = b'SATTHS0009,044,0002338.48,$R45.80P-48.06T26.8X-51.9Y-239.1Z-74.9C283.5*5D\r\n'
TEST_FRAME_FIXED_LENGTH = b'SATHSL0251\x08\x00\x00\x00\x08I\x08\x15\x08\xc0\x08H\x08;\x08\xb4\x08W\x08E\x08\x80\x08:\x08T\x08`\x08n\x08q\x08_\x08W\x08q\x08{\x08c\x08W\x08k\x08Z\x08J\x08Q\x08C\x08L\x08?\x084\x08:\x08B\x08m\x08D\x08i\x08i\x08E\x08W\x08S\x08E\x08K\x08K\x08C\x087\x08^\x08C\x085\x08\x81\x08A\x083\x08\x95\x08Q\x08-\x08i\x08;\x08\r\x08d\x08q\x08*\x08v\x08\xa2\x08C\x08\x96\x08m\x08I\x08\x8b\x08S\x08R\x08\x95\x081\x08>\x08\x8a\x08A\x08E\x08\x85\x08\'\x08G\x08T\x08\x1a\x084\x08Y\x07\xfd\x08:\x08G\x08\x05\x08\x8d\x089\x08\'\x08u\x083\x08T\x08>\x08-\x08R\x08\x17\x08+\x08m\x08\x1b\x08&\x08\x84\x080\x089\x08x\x08%\x080\x08\x8b\x08\x19\x085\x08y\x08\n\x08\x1d\x08q\x082\x08c\x08\x8c\x08J\x08^\x08}\x08X\x08$\x08t\x08N\x08\x1a\x08w\x08b\x08#\x08\x7f\x08Q\x081\x08x\x08I\x086\x08p\x08]\x08\x18\x08n\x08E\x08\x1e\x08\x91\x080\x085\x08\xaa\x08\x17\x08_\x08u\x08\x0c\x08[\x08-\x08\x15\x08Y\x08-\x08 \x08S\x08$\x08+\x08O\x08\x13\x08;\x089\x08"\x08^\x08@\x08\x0e\x08u\x08E\x08\x17\x08\xbf\x081\x082\x08\xbc\x08<\x08A\x08\x81\x08+\x087\x08c\x08\'\x08.\x08g\x089\x08,\x08a\x0f\x08X\x00\x00\x00\x00\x00\xc0S0009943.02\x88\x0D\x0A'
TEST_FRAME_FIXED_LENGTH_DICT_v0 = {'INTTIME_LT': 2.048, 'SAMPLE_DELAY': 0.0, 'LT_304.90': 2121, 'LT_308.23': 2069, 'LT_311.55': 2240, 'LT_314.88': 2120, 'LT_318.20': 2107, 'LT_321.53': 2228, 'LT_324.86': 2135, 'LT_328.19': 2117, 'LT_331.52': 2176, 'LT_334.85': 2106, 'LT_338.18': 2132, 'LT_341.51': 2144, 'LT_344.85': 2158, 'LT_348.18': 0.09223884461490882, 'LT_351.52': 0.08245744030266479, 'LT_354.85': 0.07594628162062825, 'LT_358.19': 0.07335500874564135, 'LT_361.52': 0.07191492352967144, 'LT_364.86': 0.0705451996840113, 'LT_368.20': 0.07133452127578, 'LT_371.54': 0.07531652001584827, 'LT_374.88': 0.0771416161563252, 'LT_378.22': 0.07814502985322579, 'LT_381.56': 0.08084086746670166, 'LT_384.90': 0.08068243419932675, 'LT_388.24': 0.08210230160568764, 'LT_391.58': 0.08049452410959368, 'LT_394.92': 0.07799620364297159, 'LT_398.27': 0.07639014880996027, 'LT_401.61': 0.07434297385204998, 'LT_404.95': 0.07448000608586834, 'LT_408.30': 0.06732293537173449, 'LT_411.64': 0.06589885304920992, 'LT_414.99': 0.06259362311781239, 'LT_418.34': 0.05676957740749963, 'LT_421.68': 0.05451373844350024, 'LT_425.03': 0.05173504862864702, 'LT_428.38': 0.04864964721674803, 'LT_431.72': 0.047396592855274186, 'LT_435.07': 0.04675953316824514, 'LT_438.42': 0.04613102593816792, 'LT_441.77': 0.04530622504052865, 'LT_445.12': 0.04776054349944275, 'LT_448.47': 0.047572171618255325, 'LT_451.82': 0.04768765101195385, 'LT_455.17': 0.05305012942206662, 'LT_458.52': 0.05140461973101742, 'LT_461.87': 0.05236168516965263, 'LT_465.22': 0.05923674158550265, 'LT_468.58': 0.05679673465267469, 'LT_471.93': 0.056481734998772896, 'LT_475.28': 0.06120890032084813, 'LT_478.63': 0.059621805436719584, 'LT_481.99': 0.05735384294673988, 'LT_485.34': 0.06290976529149013, 'LT_488.69': 0.06342640979572187, 'LT_492.05': 0.0587343164845626, 'LT_495.40': 0.06362259958677108, 'LT_498.76': 0.06493524808490107, 'LT_502.11': 0.05930529658347232, 'LT_505.47': 0.0638653262173376, 'LT_508.82': 0.06063797070259319, 'LT_512.17': 0.05771780699405085, 'LT_515.53': 0.0612864539641052, 'LT_518.88': 0.056662276696529144, 'LT_522.24': 0.05621188326287072, 'LT_525.60': 0.059198288388064645, 'LT_528.95': 0.05283622229111488, 'LT_532.31': 0.05330055067177218, 'LT_535.66': 0.05688299358522996, 'LT_539.02': 0.052116673563023026, 'LT_542.37': 0.05203472585117372, 'LT_545.73': 0.05516589670880056, 'LT_549.09': 0.04999151485487021, 'LT_552.44': 0.051819463267643404, 'LT_555.80': 0.05233150223659715, 'LT_559.15': 0.049000012863228866, 'LT_562.51': 0.050984371699511276, 'LT_565.87': 0.052986641098871336, 'LT_569.22': 0.04820709737638678, 'LT_572.58': 0.05234192142981946, 'LT_575.93': 0.05511476800202174, 'LT_579.29': 0.05222298862916912, 'LT_582.64': 0.06325928798335524, 'LT_586.00': 0.06139286802877165, 'LT_589.35': 0.06279724822999681, 'LT_592.71': 0.07214888327726418, 'LT_596.06': 0.07073857039220575, 'LT_599.42': 0.07703004845001445, 'LT_602.77': 0.07851860602560957, 'LT_606.13': 0.077079425650686, 'LT_609.48': 0.07899094433933973, 'LT_612.84': 0.07290572907816316, 'LT_616.19': 0.07200177732529228, 'LT_619.55': 0.07799753150065686, 'LT_622.90': 0.0727641769730306, 'LT_626.25': 0.07592696150173515, 'LT_629.61': 0.08665107285248393, 'LT_632.96': 0.0814876214846107, 'LT_636.31': 0.0836806597777972, 'LT_639.67': 0.09196437325143313, 'LT_643.02': 0.08481373727217835, 'LT_646.37': 0.08759816272023525, 'LT_649.72': 0.09608430186453953, 'LT_653.07': 0.08709609565670953, 'LT_656.42': 0.09030899071288946, 'LT_659.77': 0.09760437253367008, 'LT_663.12': 0.0862910895139595, 'LT_666.47': 0.08937113705772205, 'LT_669.82': 0.09836065386432644, 'LT_673.17': 0.09171734506857153, 'LT_676.52': 0.09771378242941248, 'LT_679.87': 0.1030060235854304, 'LT_683.22': 0.09575589722883404, 'LT_686.56': 0.09899991247620417, 'LT_689.91': 0.10362105326957394, 'LT_693.26': 0.09868881845944512, 'LT_696.60': 0.09399205438648833, 'LT_699.95': 0.10372356314189256, 'LT_703.30': 0.09949157065440788, 'LT_706.64': 0.093912784569945, 'LT_709.98': 0.10347599724391422, 'LT_713.33': 0.10135662881665028, 'LT_716.67': 0.09471508702268816, 'LT_720.01': 0.10521251795928399, 'LT_723.35': 0.10186959511048001, 'LT_726.70': 0.0990078033262018, 'LT_730.04': 0.10907652396071675, 'LT_733.38': 0.10631745882095234, 'LT_736.72': 0.10616453273669807, 'LT_740.06': 0.11572280501129763, 'LT_743.39': 0.11564225684040992, 'LT_746.73': 0.1099936840250215, 'LT_750.07': 0.12397240435459404, 'LT_753.41': 0.1204321169931552, 'LT_756.74': 0.11845849607068938, 'LT_760.08': 0.13566670194673747, 'LT_763.41': 0.12444880144305603, 'LT_766.74': 0.12810924788309308, 'LT_770.08': 0.14618084567631476, 'LT_773.41': 0.12962983641446202, 'LT_776.74': 0.14189897992263859, 'LT_780.07': 0.14903062099032027, 'LT_783.40': 0.13719760947787132, 'LT_786.73': 0.15356019441614002, 'LT_790.06': 0.1487268252513604, 'LT_793.39': 0.15005682483243837, 'LT_796.71': 0.16677805862874198, 'LT_800.04': 0.16394710809008192, 'LT_803.36': 0.16721832082649166, 'LT_806.69': 2131, 'LT_810.01': 2084, 'LT_813.33': 2091, 'LT_816.66': 2127, 'LT_819.98': 2067, 'LT_823.30': 2107, 'LT_826.62': 2105, 'LT_829.93': 2082, 'LT_833.25': 2142, 'LT_836.57': 2112, 'LT_839.88': 2062, 'LT_843.20': 2165, 'LT_846.51': 2117, 'LT_849.82': 2071, 'LT_853.13': 2239, 'LT_856.44': 2097, 'LT_859.75': 2098, 'LT_863.06': 2236, 'LT_866.37': 2108, 'LT_869.67': 2113, 'LT_872.98': 2177, 'LT_876.28': 2091, 'LT_879.59': 2103, 'LT_882.89': 2147, 'LT_886.19': 2087, 'LT_889.49': 2094, 'LT_892.79': 2151, 'LT_896.08': 2105, 'LT_899.38': 2092, 'LT_902.68': 2145, 'DARK_SAMP_LT': 15, 'DARK_AVE_LT': 2136, 'AUX': 0, 'TEMP_PCB': 46.0, 'FRAME_COUNTER': 83, 'TIMER': 9943.02, 'CHECK_SUM': 136}
TEST_FRAME_FIXED_LENGTH_DICT = {'INTTIME_LT': 2.048, 'SAMPLE_DELAY': 0.0, 'LT_SATHSL_RAW': np.array([2121, 2069, 2240, 2120, 2107, 2228, 2135, 2117, 2176, 2106, 2132, 2144, 2158, 2131, 2084, 2091, 2127, 2067, 2107, 2105, 2082, 2142, 2112, 2062, 2165, 2117, 2071, 2239, 2097, 2098, 2236, 2108, 2113, 2177, 2091, 2103, 2147, 2087, 2094, 2151, 2105, 2092, 2145]), 'LT_SATHSL': np.array([0.09223884461490882, 0.08245744030266479, 0.07594628162062825, 0.07335500874564135, 0.07191492352967144, 0.0705451996840113, 0.07133452127578, 0.07531652001584827, 0.0771416161563252, 0.07814502985322579, 0.08084086746670166, 0.08068243419932675, 0.08210230160568764, 0.08049452410959368, 0.07799620364297159, 0.07639014880996027, 0.07434297385204998, 0.07448000608586834, 0.06732293537173449, 0.06589885304920992, 0.06259362311781239, 0.05676957740749963, 0.05451373844350024, 0.05173504862864702, 0.04864964721674803, 0.047396592855274186, 0.04675953316824514, 0.04613102593816792, 0.04530622504052865, 0.04776054349944275, 0.047572171618255325, 0.04768765101195385, 0.05305012942206662, 0.05140461973101742, 0.05236168516965263, 0.05923674158550265, 0.05679673465267469, 0.056481734998772896, 0.06120890032084813, 0.059621805436719584, 0.05735384294673988, 0.06290976529149013, 0.06342640979572187, 0.0587343164845626, 0.06362259958677108, 0.06493524808490107, 0.05930529658347232, 0.0638653262173376, 0.06063797070259319, 0.05771780699405085, 0.0612864539641052, 0.056662276696529144, 0.05621188326287072, 0.059198288388064645, 0.05283622229111488, 0.05330055067177218, 0.05688299358522996, 0.052116673563023026, 0.05203472585117372, 0.05516589670880056, 0.04999151485487021, 0.051819463267643404, 0.05233150223659715, 0.049000012863228866, 0.050984371699511276, 0.052986641098871336, 0.04820709737638678, 0.05234192142981946, 0.05511476800202174, 0.05222298862916912, 0.06325928798335524, 0.06139286802877165, 0.06279724822999681, 0.07214888327726418, 0.07073857039220575, 0.07703004845001445, 0.07851860602560957, 0.077079425650686, 0.07899094433933973, 0.07290572907816316, 0.07200177732529228, 0.07799753150065686, 0.0727641769730306, 0.07592696150173515, 0.08665107285248393, 0.0814876214846107, 0.0836806597777972, 0.09196437325143313, 0.08481373727217835, 0.08759816272023525, 0.09608430186453953, 0.08709609565670953, 0.09030899071288946, 0.09760437253367008, 0.0862910895139595, 0.08937113705772205, 0.09836065386432644, 0.09171734506857153, 0.09771378242941248, 0.1030060235854304, 0.09575589722883404, 0.09899991247620417, 0.10362105326957394, 0.09868881845944512, 0.09399205438648833, 0.10372356314189256, 0.09949157065440788, 0.093912784569945, 0.10347599724391422, 0.10135662881665028, 0.09471508702268816, 0.10521251795928399, 0.10186959511048001, 0.0990078033262018, 0.10907652396071675, 0.10631745882095234, 0.10616453273669807, 0.11572280501129763, 0.11564225684040992, 0.1099936840250215, 0.12397240435459404, 0.1204321169931552, 0.11845849607068938, 0.13566670194673747, 0.12444880144305603, 0.12810924788309308, 0.14618084567631476, 0.12962983641446202, 0.14189897992263859, 0.14903062099032027, 0.13719760947787132, 0.15356019441614002, 0.1487268252513604, 0.15005682483243837, 0.16677805862874198, 0.16394710809008192, 0.16721832082649166]), 'DARK_SAMP_LT': 15, 'DARK_AVE_LT': 2136, 'AUX': 0, 'TEMP_PCB': 46.0, 'FRAME_COUNTER': 83, 'TIMER': 9943.02, 'CHECK_SUM': 136}
TEST_CALIBRATION_SIP_FILE = 'test_data/HyperSAS009_20150728/THS0009_13Jun06.sip'

# Data to test parsing of SatView Raw Files
TEST_DATA = ['test_data/HyperSAS009_20150728/HS_a0_t45_i45.raw',
             'test_data/ES187_20180814/ES187_20180814_2035.raw',
             'test_data/HTSRB007_20180814/HTSRB007_ES187_20180814_0824.raw',
             'test_data/HyperPro068_20180826/HyperPro068_ES187_20180826_2048.raw',
             'test_data/HyperNav001_20170809/HyperNav1_Multi1.raw',
             'test_data/HyperNav002_20170807/HyperNav2_MultiCast1.raw',
             'test_data/HyperPro012_20170808/HyperPro_MultiCast1.raw',
             'test_data/HyperSAS009_20200527/HyperSAS+Ref_LabTest_20200527_2.raw']
TEST_CAL = ['test_data/HyperSAS009_20150728/',                       # Optics Class 2015
            'test_data/ES187_20180814/HSE0187_23Aug17.sip',          # EXPORTS 1
            'test_data/HTSRB007_20180814/HST007_18June26.sip',     # EXPORTS 1
            ['test_data/HyperPro068_20180826/MPR068_18Jan05.sip', 'test_data/HyperPro068_20180826/HSE0187_23Aug17.sip'],  # EXPORTS 1
            'test_data/HyperNav001_20170809/',                       # Hawaii
            'test_data/HyperNav002_20170807/',                       # Hawaii
            ['test_data/HyperPro012_20170808/MPR0012_17Aug07.sip', 'test_data/HyperPro012_20170808/HSE0266_16Aug11.sip'],  # Hawaii
            'test_data/HyperSAS009_20200527/HyperSAS+ES_20200212.sip']   # Dev Test
TEST_FRAME_TO_PARSE = [563,
                       141,
                       329,
                       5837,
                       2180,
                       2319,
                       4900]

# Data to test pySAS POC parsing
TEST_PYSAS_DATABASE = 'test_data/pySAS001_20180315/pysas_20180315.db'
TEST_PYSAS_CAL = 'test_data/pySAS001_20180315/THS0009_15Aug20.sip'


class TestHyperSASFunctions(unittest.TestCase):

    def test_read_frame_variable_length(self):
        from pySatlantic import instrument
        i = instrument.Instrument(TEST_CALIBRATION_SIP_FILE)
        [d, valid] = i.parse_frame(TEST_FRAME_VARIABLE_LENGTH)
        # self.assertEqual(frame_header, 'SATTHS0009')
        self.assertEqual(d['FRAME_COUNTER'], 44)
        self.assertEqual(d['TIMER'], 2338.48)
        self.assertEqual(d['START'], '$')
        self.assertEqual(d['ROLL'], 45.80)
        self.assertEqual(d['PITCH'], -48.06)
        self.assertEqual(d['TEMP_PCB'], 26.8)
        self.assertEqual(d['MAG_X'], -51.9)
        self.assertEqual(d['MAG_Y'], -239.1)
        self.assertEqual(d['MAG_Z'], -74.9)
        self.assertEqual(d['COMP'], 283.5)
        self.assertEqual(d['EXTRA'], '5D')
        self.assertTrue(valid)

    def test_read_frame_fixed_length(self):
        from pySatlantic import instrument
        i = instrument.Instrument(TEST_CALIBRATION_SIP_FILE, immersed=True)
        # Test older version
        d_v0 = i.parse_frame_v0(TEST_FRAME_FIXED_LENGTH)
        self.assertCountEqual(d_v0, TEST_FRAME_FIXED_LENGTH_DICT_v0)
        for k in d_v0.keys():
            if isinstance(d_v0[k], np.ndarray):
                self.assertTrue(np.allclose(d_v0[k], TEST_FRAME_FIXED_LENGTH_DICT_v0[k]))
                # self.assertTrue(np.alltrue(d[k], TEST_FRAME_FIXED_LENGTH_DICT[k]))
            else:
                self.assertEqual(d_v0[k], TEST_FRAME_FIXED_LENGTH_DICT_v0[k])
        # Test current version
        d, valid = i.parse_frame(TEST_FRAME_FIXED_LENGTH,\
                                 flag_get_auxiliary_variables=True, flag_get_unusable_variables=True)
        # self.assertEqual(frame_header, 'SATHSL0251')
        self.assertCountEqual(d, TEST_FRAME_FIXED_LENGTH_DICT)
        for k in d.keys():
            if isinstance(d[k], np.ndarray):
                self.assertTrue(np.allclose(d[k], TEST_FRAME_FIXED_LENGTH_DICT[k]))
                # self.assertTrue(np.alltrue(d[k], TEST_FRAME_FIXED_LENGTH_DICT[k]))
            else:
                self.assertEqual(d[k], TEST_FRAME_FIXED_LENGTH_DICT[k])

    def test_pysas_database(self):
        # NAAMES IV dataset
        from pySatlantic import instrument
        i = instrument.Instrument(TEST_PYSAS_CAL)
        db = sqlite3.connect(TEST_PYSAS_DATABASE)
        with db:
            cur = db.cursor()
            cur.execute("SELECT id, THS, Lt, Li FROM hypersas LIMIT 10000")
            frames = cur.fetchall()
        db.close()
        valid_frame_count, no_frame_count = 0, 0
        for f in frames:
            if f[1]:
                frame = b'SATTHS0009' + f[1]
                data = i.parse_frame(frame + b'\x0D\x0A', 'SATTHS0009', True, True)
            elif f[2]:
                frame = b'SATHSL0251' + f[2]
                data = i.parse_frame(frame + b'\x0D\x0A', 'SATHSL0251', True, True)
            elif f[3]:
                frame = b'SATHSL0250' + f[3]
                data = i.parse_frame(frame + b'\x0D\x0A', 'SATHSL0250', True, True)
            else:
                no_frame_count += 1
            if data:
                valid_frame_count += 1
        self.assertEqual(valid_frame_count, 10000)
        self.assertEqual(no_frame_count, 0)

    def test_conversion_to_csv(self):
        from pySatlantic import instrument
        for data, cal, n in zip(TEST_DATA, TEST_CAL, TEST_FRAME_TO_PARSE):
            foo = instrument.SatViewRawToCSV(cal, data)
            print(data, foo.frame_parsed, foo.frame_received)
            self.assertEqual(foo.frame_parsed, n)
            self.assertEqual(foo.frame_received, n)
            self.assertEqual(foo.frame_unregistered, 0)


if __name__ == "__main__":
    unittest.main()

