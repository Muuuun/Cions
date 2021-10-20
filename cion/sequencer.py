from typing import Optional
from matplotlib import pyplot as plt
import numpy as np
import serial
from cion.gates import *
from cion.serial_comm import *
from cion.string_converter import *
from cion.data import *

all = []

ser = serial.Serial(
            port='COM3',
            baudrate=4000000,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_TWO,
            bytesize=serial.EIGHTBITS
            )

class Experiment:
    def __init__(self, ion_number = -1, rf_pick = 854, melt_status = False, eta = 0.098):
        self.ion_number = ion_number
        self.rf_pick = rf_pick
        self.mode_frequency = np.random.random(ion_number)
        self.melt_status = melt_status
        self.Lamb_Dicke_parameter = eta
        self.channel = [2,4,0,6]
        self.sequence = Sequence(ion_number=ion_number)
        self.clock_period = 5E-3
        [all.append(i) for i in range(ion_number)]

        assert(isinstance(ion_number, int) and ion_number >= 1)
        assert(isinstance(melt_status, bool))

        self.path_prefix = ''

    def set_path_prefix(self, prefix):
        self.path_prefix = prefix

    def set_ion_number(self, ion_number):
        self.ion_number = ion_number
        all.clear()
        [all.append(i) for i in range(ion_number)]
        if ion_number > len(self.channel):
            print("Alert! The ion number in current setup is larger than the predefined channel numbers.\n Try using\
            member function \'set_channel\' to reset channel number.")

    def set_channel(self, channel):
        '''
        reset channel and convert matrix
        '''
        assert(self.ion_number == len(channel))
        self.channel = channel

        m = 2*self.ion_number-1
        self.convert_matrix = np.zeros((self.ion_number, m), dtype=int)
        for i in range(self.ion_number):
            self.convert_matrix[i][channel[i]] = 1

    #def Class pulse

    def run_once(self, repeat=100, ext_trig=1):
        global channel_return
        pulses = self.sequence.perform_sequence()
        pulses = pre_binary(pulses, self.clock_period)
        packets = [packets_generator(pulses, ext_trig, repeat), int(repeat*channel_return)]
        result = seq_send(ser,packets)
        return result

    def print_info_all():
        #print(global_parameters)
        #print(sequences)
        pass

    def print_info_global():
        pass

    def print_sequence(self, size=(20, 4)):
        figure = plt.figure(figsize=size)
        plot_sequence(self.sequence.gate_sequence, remap_time=False)
        plt.tight_layout()
        plt.show()
        pass

    def save_info_to_database():
        pass


class Sequence:
    def __init__(self, ion_number = 5):
        self.gate_sequence = []
        self.ion_number = ion_number # total ions
        self.time_stamp = [0 for i in range(self.ion_number)]
        self.gate_time = []

    def reset_sequence(self, make_sure = False):
        if make_sure == False:
            print("This operation will delete the sequence in current object, if you confirm \
                    to do that please set the argument \'make_sure\' to be True.")
        else:
            self.gate_sequence = []
            #del(self.gate_time)
            self.gate_time = []

    #continuous_seq = [RSB(paras,10.0),opitical_pumping(102,13),...]

    #seq.add_sequence(EIT(100...),continuous_seq,optical_pumping)

    def add_gate(self, gate):
        ion = gate.ion_index
        assert ion != None
        tn = gate.duration
        tl = gate.latency
        if isinstance(ion, int):
            self.gate_sequence.append((ion, self.time_stamp[ion]+tl, tn, gate.pulse_type))
            self.time_stamp[ion] += tn + tl
        else:
            tmax = -1
            for single_ion in ion:
                tmax = max(tmax, self.time_stamp[single_ion])
            self.gate_sequence.append((ion, tmax+tl, tn, gate.pulse_type))
            for single_ion in ion:
                self.time_stamp[single_ion] = tmax + tn +tl

    def add_gates(self, *args):
        gate_num = len(args)
        for i in range(gate_num):
            if isinstance(args[i], BaseGate):
                self.add_gate(args[i])
            elif isinstance(args[i], list):
                sub_sequence = args[i]
                for pulse in sub_sequence:
                    self.add_gate(pulse)
            elif isinstance(args[i], tuple):
                pulse_info = args[i]
                tmax = -1
                if pulse_info[0] == 'sync':
                    for single_ion in pulse_info[1]:
                        tmax = max(tmax, self.time_stamp[single_ion])
                    for single_ion in pulse_info[1]:
                        self.time_stamp[single_ion] = tmax
                else:
                    pass

    def new_sequence(self, *args):
        self.clear()
        gate_num = len(args)
        for i in range(gate_num):
            if isinstance(args[i], BaseGate):
                self.add_gate(args[i])
            elif isinstance(args[i], list):
                sub_sequence = args[i]
                for pulse in sub_sequence:
                    self.add_gate(pulse)
            elif isinstance(args[i], tuple):
                pulse_info = args[i]
                tmax = -1
                if pulse_info[0] == 'sync':
                    for single_ion in pulse_info[1]:
                        tmax = max(tmax, self.time_stamp[single_ion])
                    for single_ion in pulse_info[1]:
                        self.time_stamp[single_ion] = tmax
                else:
                    pass

    def print_info(self):
        tmax = 0
        for pulse in self.gate_sequence:
            tmax = max(tmax, pulse[1]+pulse[2])
        print('----sequence info----')
        print('total pulses: {}'.format(len(self.gate_sequence)))
        print('sequence time: {} microseconds'.format(tmax))
        print(self.gate_sequence)
        print('---------------------')

    def show_sequence(self):
        #draw a graph demonstrating the layout of sequences like Cirq does
        pass

    #def convert_sequence(self):
        # convert sequence to xml format

    def reformat_sequence(self):
        seq = self.gate_sequence
        time_index = set()

        for pulse in seq:
            time_index.add(pulse[1])
            time_index.add(pulse[1]+pulse[2])
        time_index = list(time_index)
        time_index.sort()
        #print(time_index)
        start_point = {}
        end_point = {}
        for idx,pulse in enumerate(seq):
            lt = pulse[1]
            rt = pulse[1]+pulse[2]
            if start_point.get(lt) != None:
                start_point[lt].append((idx,pulse[3],pulse[0]))
            else:
                start_point[lt] = [(idx,pulse[3],pulse[0])]
            if end_point.get(rt) != None:
                end_point[rt].append((idx,pulse[3],pulse[0]))
            else:
                end_point[rt] = [(idx,pulse[3],pulse[0])]

        return time_index, start_point, end_point

    def get_chapter_data(self, pulse_type, ion_index):
        '''
        pulse_type is a string, such as 'EIT', 'optical_pumping', 'RSB' ...
        ion_index is an int or tuple, such as: 0,2,3 or (3), (0,2), (1,3,4)
        '''
        #TODO
        #this function is left for modification to get real chapter data according to the experiment configurations.
        return real_chapter_dict[pulse_type]

    def perform_sequence(self):
        # generate chapter data according to the datasheet and invoke the serial module
        # need the explict rules of parallel pulses
        time_index, start_point, end_point = self.reformat_sequence()
        active_set = set()
        ll = len(time_index)
        result_list = []
        #print(start_point)
        #print(end_point)
        for idx, t in enumerate(time_index):
            if end_point.get(t) != None:
                for item in end_point.get(t):
                    #active_set.remove(item[1])
                    if isinstance(item[2], int):
                        active_set.remove((item[1], item[2]))
                    else:
                        active_set.remove((item[1], tuple(item[2])))
            if start_point.get(t) != None:
                for item in start_point.get(t):
                    if isinstance(item[2], int):
                        active_set.add((item[1], item[2]))
                    else:
                        active_set.add((item[1], tuple(item[2])))
            if idx < ll-1:
                chapter_data = 0
                for active_pulse in active_set:
                    temp_str = self.get_chapter_data(active_pulse[0], active_pulse[1]).replace(' ','')
                    temp_data = int(temp_str, 2)
                    chapter_data = chapter_data | temp_data
                chapter_str = str(bin(chapter_data))
                chapter_str = chapter_str[2:32]
                chapter_str = '0'*(24-len(chapter_str)) + chapter_str
                chapter_str = chapter_str[0:8] + ' ' + chapter_str[8:16] + ' ' + chapter_str[16:24]
                result_list.append([chapter_str, time_index[idx+1]-t])

        result_list.append(['100000000000000000000000', 0.1])
        result_list.append(['100000000000000000110000', 0.1])

        return result_list

    def clear(self):
        self.gate_sequence = []
        self.time_stamp = [0 for i in range(self.ion_number)]
        self.gate_time = []


def plot_sequence(sequence, remap_time=True, ax: Optional[plt.Axes] = None):
    sequence = tuple(sequence)
    ax = plt.gca() if ax is None else ax

    moments_org2rmp, moments_rmp2org = remap_to_indices(
        moment
        for (_, start_time, duration, _) in sequence
        for moment in (start_time, start_time + duration)
    )

    ion_org2rmp, ion_rmp2org = remap_to_indices(
        ion
        for (ions, start_time, duration, _) in sequence
        for ion in iter_int_or_tuple(ions)
    )

    for ions, start_time, duration, pulse_type in sequence:
        ions = sorted(list(iter_int_or_tuple(ions)))

        if remap_time:
            remapped_start_moment = moments_org2rmp[start_time]
            remapped_end_moment = moments_org2rmp[start_time + duration]
        else:
            remapped_start_moment = start_time
            remapped_end_moment = start_time + duration

        polygon_xy = []
        for ion in ions:
            remapped_ion = ion_org2rmp[ion]

            l = remapped_start_moment
            r = remapped_end_moment
            w = r - l
            h = 4 / 5
            cx = (l + r) / 2
            cy = remapped_ion

            rect = plt.Rectangle((l, cy - h / 2), w, h,
                facecolor='white', edgecolor='black', linewidth=3,
                zorder=2.5)
            ax.add_patch(rect)

            text = plt.Text(cx, cy, pulse_type,
                verticalalignment='center', horizontalalignment='center',
                wrap=True, color='black',
                zorder=2.5)
            ax.add_artist(text)

            polygon_xy.append((cx, cy))

        if len(ions) > 1:
            polygon = plt.Polygon(polygon_xy, False,
                edgecolor='black', linewidth=3,
                zorder=2.4)
            ax.add_patch(polygon)

    ax.set_xlabel("time")

    if remap_time:
        padding_x = 3 / 5
        ax.set_xlim(0 - padding_x, len(moments_rmp2org) - 1 + padding_x)
    else:
        padding_x = (max(moments_rmp2org) - min(moments_rmp2org)) / 20
        ax.set_xlim(min(moments_rmp2org) - padding_x, max(moments_rmp2org) + padding_x)

    ax.grid(axis='x', color='grey', ls='--')
    if remap_time:
        ax.set_xticks(range(len(moments_rmp2org)))
    else:
        ax.set_xticks(moments_rmp2org)
    ax.set_xticklabels(moments_rmp2org)

    ax.set_ylabel("ion")

    padding_y = 3 / 5
    ax.set_ylim(len(ion_rmp2org) - 1 + padding_y, -padding_y)
    ax.grid(axis='y', color='black')
    ax.set_yticks(range(len(ion_rmp2org)))
    ax.set_yticklabels(ion_rmp2org)

    ax.set_frame_on(False)


def remap_to_indices(items, key=None):
    items_set = set(items)
    items_sorted = sorted(items_set, key=key)
    moments_dict = {t: i for i, t in enumerate(items_sorted)}

    org2rmp = moments_dict
    rmp2org = items_sorted
    return org2rmp, rmp2org


def iter_int_or_tuple(item):
    try:
        for sub_item in item:
            yield sub_item
    except TypeError:
        yield item
