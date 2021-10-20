real_chapter_dict = {
    ('EIT',5):'00000001 00000001 00000001', 
    'EIT':'00000001 00000001 00000001', 
    'Doppler':'00000010 00000010 00000010', 
    'Detection':'10000001 00000000 00001111',
    'optical_pumping':'00000100 00000100 00000100',
    'RSB':'00001000 00001000 00001000', 
    'Idle': '00000000 00000000 00000000', 
    'Rx': '00000000 00000000 00000000',
    'parallel_xx':'00110000 00000000 00000000',
    'parallel_yy':'00000000 00110000 00000000', 
    'parallel_zz':'00000000 00000000 00110000', 
    'global_xx':'11000000 00000000 00000000',
    'global_xx':'00000000 11000000 00000000',
    'global_xx':'00000000 00000000 11000000'
              }

class BaseGate:
    def __init__(self, duration, latency=0, pulse_type = None, amp = None, freq = None, offset = None):
        # f(t) = amp * cos(2pi*freq*t+offset)
        self.duration = duration
        self.latency = 0
        self.pulse_type = pulse_type
        self.amp = amp
        self.freq = freq
        self.offset = offset
        self.ion_index = None
        self.para_set = False
        self.on_flag = False

    def on(self, ion_index):
        assert isinstance(ion_index, int) or isinstance(ion_index, list)
        #self.ion_index = ion_index
        if self.on_flag == False:
            self.ion_index = ion_index
            self.on_flag = True
            return self
        else:
            return BaseGate(self.duration, self.latency, self.pulse_type).on(ion_index)

    def set_parameters(self, amp = None, freq = None, offset = None, duration = None, pulse_type = None):
        if amp != None:
            self.amp = amp
        if omega != None:
            self.omega = omega
        if offset != None:
            self.offset = offset
        if duration != None:
            self.duration = duration
        if pulse_type != None:
            self.pulse_type = pulse_type
        self.para_set = True


class EIT(BaseGate):
    def __init__(self, duration, latency = 0):
        self.duration = duration
        self.pulse_type = 'EIT'
        self.latency = latency
        self.on_flag = False
    '''
    def __init__(self, duration, amp = 20, freq = 100, offset = 0, awg = None, fpga = None):
        self.duration = duration
        super().__init__(duration, amp, freq, offset)
        self.pulse_type = 'EIT'
    '''
class Doppler(BaseGate):
    def __init__(self, duration, latency = 0):
        self.duration = duration
        self.pulse_type = 'Doppler'
        self.latency = latency
        self.on_flag = False

class Detection(BaseGate):
    def __init__(self, duration, latency = 0):
        self.duration = duration
        self.pulse_type = 'Detection'
        self.latency = latency
        self.on_flag = False
        
class Optical_pumping(BaseGate):
    def __init__(self, duration, latency = 0):
        self.duration = duration
        self.pulse_type = 'optical_pumping'
        self.latency = latency
        self.on_flag = False

class RSB(BaseGate):
    def __init__(self, duration, latency = 0, duration_head = None, duration_tail = None):
        self.duration = duration
        self.pulse_type = 'RSB'
        self.latency = latency
        self.duration_head = duration_head
        self.duration_tail = duration_tail
        self.on_flag = False

class Idle(BaseGate):
    def __init__(self, duration, latency = 0):
        self.duration = duration
        self.pulse_type = 'Idle'
        self.latency = latency
        self.on_flag = False

# the following contents are for self_defined pulses/gates
# we give a example here:
def example_pulse(ion, amp, freq, offset, duration, pulse_type = None):
    return BaseGate(ion, amp, freq, offset, duration, pulse_type)
# When we call this function 'example(2, 1, 2, np.pi/4, 100)', a BaseGate object will be generated
# Performed on 2-th ion, pulse amplitude = 1, angular frequency is 2(MHZ), offset is pi/4, duration is 100 microseconds.


def Rx(duration, latency=0,pulse_type=None):
    # specify the following arguments
    # amp, omega, duration
    amp = None
    omega = None
    offset = 1
    pulse_type = 'Rx'
    return BaseGate(duration = duration, pulse_type = pulse_type, latency=latency)

def parallel_entanglement(duration, pulse_type = 'xx'):
    amp = None
    freq = None
    offset = None
    pulse_type = 'parallel_' + pulse_type
    return BaseGate(duration = duration, pulse_type = pulse_type)

def global_gate(duration, pulse_type = 'xx'):
    amp = None
    freq = None
    offset = None
    pulse_type = 'global_' + pulse_type
    return BaseGate(amp, omega, offset, duration, pulse_type)

def sync(ion_index):
    if type(ion_index) == int:
        ion_index = [ion_index]
    return ('sync', ion_index)


#def pending_all(latency = 0):
def pending(latency = 0):
    # synchoronize the timestamp of all ions, which means the following pulses start at the time of the lastest pulse
    # for example, if the maxmimal ending time of previous pulses is 'tm', then after a pending \
        #  operation the following pulses will start at 'tm+lantency'
    # default pending latency is 0
    return ('pending', latency)