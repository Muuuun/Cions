import xml.etree.cElementTree as ET
import cion.XML_to_dict as XMLdic

def pulse_durations(sequence_dict):
    seq_string = sequence_dict['sequence'].replace('\t','')
    seq_string = seq_string.split('\n')
    seq_new = []
    for i in range(len(seq_string)):
        if seq_string[i]!='':
            seq_new = seq_new + [seq_string[i]]

    duration_list = []
    for element in seq_new:
        duration_list = duration_list +[element.split(' ')]
    return duration_list

def timestamp(duration, clock_period):
    # convert duration to a 40 bit binary string
    N = int(duration/clock_period-1)
    return timestamp_generator(N)

def timestamp_generator(N):
    W = 40 
    # convert number N to a W-bit binary timestap
    T = (W-N)&(2**W-1)
    T = '{0:040b}'.format(T)
    Res = ''
    for i in range(W):
        j = (int(T[W-1-i:W],2) < i)|(int(T[W-1-i:W],2) >= (1<<i) + i)
        Res = str(int(j)) + Res
    return Res

def chapter_padding(chapter):
    return chapter[:18]+'{0:08b}'.format(0) + ' '+chapter[18:]

def chapter_durations(pulses, dict):
    #convert [['chapter name',duration],...] array to [[00000,100],...] array
    duration_list = [] 
    for element in pulses:
        duration_list = duration_list + [[dict[element[0]],float(element[1])]]
    return duration_list

def pre_binary(pulses, clock_period):
    # change [chapter, duration] to binary strings
    duration_list = [] 
    for element in pulses:
        duration_list = duration_list + [[chapter_padding(element[0]),timestamp(element[1], clock_period)]]
    return duration_list

def packets_generator(pulses, ext_trig, repeat):
    start_packet = '100' + str(int(ext_trig)) + '{0:052b}'.format(0) + '{0:024b}'.format(int(repeat))
    pulse_packates = [start_packet]
    i = 0
    # convert pulses to timestamp
    for element in pulses:
        packet = '010'
        packet = packet + '{0:05b}'.format(0) + element[1] + element[0].replace(' ','')
        pulse_packates = pulse_packates + [packet]
        i += 1
    # padding the time packet
    if len(pulses) < 15:
        for i in range(15 - len(pulses)):
            packet = '010'
            packet = packet + '{0:05b}'.format(0) + timestamp_generator(1-1) + '{0:032b}'.format(0)
            pulse_packates = pulse_packates + [packet]

    if len(pulses)%5 != 0&len(pulses) >= 15:
        for i in range(5 - len(pulses)%5):
            packet = '010'
            packet = packet + '{0:05b}'.format(0) + timestamp_generator(1-1) + '{0:032b}'.format(0)
            pulse_packates = pulse_packates + [packet]
    # generate end packets
    end_packets_3 = '010' + '{0:05b}'.format(0) + timestamp_generator(8-1) + '{0:032b}'.format(0)
    end_packets_2 = '010' + '{0:05b}'.format(0) + timestamp_generator(2-1) + '{0:032b}'.format(0)
    end_packets_1 = '011' + '{0:05b}'.format(0) + timestamp_generator(1-1) + '{0:032b}'.format(0)
    pulse_packates = pulse_packates + [end_packets_3, end_packets_2, end_packets_1]
    pulse_packates_hex = ''
    for packet in pulse_packates:
        pulse_packates_hex = pulse_packates_hex + hex(int(packet,2))[2:] + ' '
    # end_packet = 
    return pulse_packates_hex

def xml_to_packets(file, repeat=100, clock_period=5E-3, ext_trig=1):

    channel_return = 24 #how many channel return from FPGA

    tree = ET.parse(file) # read the xml file
    root = tree.getroot()
    sequence_dict = XMLdic.XmlDictConfig(root)
    
    pulses  = pulse_durations(sequence_dict)
    pulses  = chapter_durations(pulses,sequence_dict['chapter'])

    duration_total = 0
    for i in pulses:
        duration_total = duration_total + i[1]

    pulses  = pre_binary(pulses, clock_period)
    packets = packets_generator(pulses, ext_trig, repeat)

    return [packets, int(repeat*channel_return), duration_total*repeat*1E-6]