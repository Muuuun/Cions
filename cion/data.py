import numpy as np
import time
import os
import scipy.io
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

channel_return = 24 
data_dirt = r'D:\Data'

path_prefix = data_dirt + time.strftime("\%Y\%Y%m\%Y%m%d")
if not os.path.exists(path_prefix):
    os.makedirs(path_prefix)



def new_data_file():
    file_name = time.strftime("\%Y%m%d%H%M%S.npy")
    return file_name

def save_data(result):
    file_to_save = new_data_file()
    data_path    = path_prefix + file_to_save
    data_to_save = list(result)
    data_to_save = np.array(data_to_save)
    np.save(data_path, data_to_save)
    return file_to_save[1:]


class scanParameterType():
    time = 1
    frequency = 2

def raw_count(path):
    '''
    return two lists: the first is the list of scan parameter(time or frequency), the second is the related list of matrices
    '''

    try:
        mat = scipy.io.loadmat(path)
    except Exception as e:
        if type(e) == FileNotFoundError:
            path = path_prefix + path
            mat = scipy.io.loadmat(path)

    #print(mat)

    list_time = []
    list_values = []
    for key in mat:
        timeStr = key.lstrip('{').rstrip('}')
        list_time.append(float(timeStr))
        list_values.append(mat[key])
    return list_time, list_values

def average(filename, state=False, threshold=None):
    if state:
        list_time, list_values = raw_count(filename)
        list_values = np.transpose(list_values)
        list_values = np.array([np.where(list_values[i] > threshold[i], 1, 0) for i in np.arange(len(list_values))])
        list_values = np.transpose(list_values)
        list_values = [np.mean(value,axis=0) for value in list_values]
    else:
        list_time, list_values = raw_count(filename)
        list_values = [np.mean(value,axis=0) for value in list_values]
    return [np.array(list_time), np.array(list_values)]

def average_plot(filename, state=False, threshold=None):
    data = average(filename, state=state, threshold=threshold)
    plt.plot(data[0],data[1])
    plt.show()
    return
def pre_process(list_matrices, convert_matrix):
    results = []
    num,dim = list_matrices[0].shape

    results = []
    for index, raw_matrix in enumerate(list_matrices):
        new_matrix = np.transpose(raw_matrix)
        new_matrix = convert_matrix @ new_matrix
        avrg = np.sum(new_matrix, axis = 1)/num
        results.append(avrg)

    return results

def average_fit(file_name, convert_matrix = None, threshold = None, para_type = scanParameterType.time):
    assert(type(convert_matrix) != type(None))

    list_para, list_matrices = raw_count(file_name)
    avrg_data = pre_process(list_matrices, convert_matrix)
    results = avrg_data
    if threshold != None:
        results = []
        for avrg in avrg_data:
            avrg = (avrg > threshold).astype(int)
            results.append(avrg)
    return results

def correlation(file_name, convert_matrix = None, threshold = None, para_type = scanParameterType.time):
    pass


def cosine_func(x,a0,a1,a2,a3):
    return a0 * np.sin(a1*x+a2) + a3

def gaussian_func(x,a,mu,sigma):
    return a*np.exp(-(x-mu)**2/(2*sigma**2))

def gaussian_func2(x, a, mu, sigma_reverse):
    return a*np.exp(-(x-mu)**2 * sigma_reverse**2)

def thermal_single_func(x, p0, gamma, omega):
    return 1/2*p0*(1 - np.exp(-gamma*x)*np.cos(omega*x))

def combinatorial_number(n,m):
    return math.factorial(n) // (math.factorial(m)*math.factorial(n-m))

def Laguerre(n,x):
    sum = 0
    for k in range(n+1):
        sum += (-1)**k*combinatorial_number(n+1,n-k)*(x**k / math.factorial(k))
    return sum

def thermal_func(x, *args):
    '''
    pn, gamma, omega are all lists
    Laguerre function:
        L_n^a(x) = \sum_{k=0}^{n} (-1)^k C_{n+a}^{n-k} x^k/k!
    '''
    eta = 0.098 #eta is a pre-given constant

    n = len(args) // 2

    assert (len(args) == 2*n+1)

    pn = np.array(args[0:n])
    gamma = np.array(args[n:2*n])
    omega = args[-1]


    omega_l = np.array([omega]+[0 for i in range(n-1)])
    for i in range(1,n):
        omega_l[i] = omega * np.exp(-eta*eta/2) * eta * np.sqrt(1/(i+1)) * Laguerre(i,eta*eta)
    sum_p = 0
    for i in range(n):
        sum_p += 1/2*pn[i]*(1 - np.exp(-gamma[i]*x) * np.cos(omega_l[i]*x))
    return sum_p

def automatic_find_initial_omega(xdata, ydata):
    pass

def check_fitting_quality(ion, xdata, ydata, y_fit):
    pass

def gaussian_fit(fileName, convert_matrix = None, threshold = None, para_type = scanParameterType.frequency, plot_figure = False):
    list_frequency, list_matrices = raw_count(fileName)
    avrg_data_all = pre_process(list_matrices, convert_matrix)

    ion_number = convert_matrix.shape[0]
    fit_paras = []

    for ion_index in range(ion_number):
        avrg_single_ion = [avrg[ion_index] for avrg in avrg_data_all]
        xdata = np.array(list_frequency)
        ydata = np.array(avrg_single_ion)

        #mean,std=scipy.stats.norm.fit(ydata)
        a0 = max(ydata)
        a1 = xdata[np.argmax(ydata)]
        #a2 = np.std(ydata)
        a2 = np.std(ydata) * (xdata[1] - xdata[0]) / a0
        p0 = [a0, a1, a2]

        #a2 = sum(y * (x - a1)**2)
        #sigma_reverse = 1/(a2 * np.sqrt(2)
        #p0 = [a0, a1, sigma_reverse]
        #p_l = [a0/2, xdata[0], a2/2]
        #p_h = [a0*2, xdata[-1], a2*2]

        print(p0)
        popt, pcov = curve_fit(gaussian_func, xdata, ydata, p0=p0)
        #popt, pcov = curve_fit(gaussian_func2, xdata, ydata, p0=p0)
        fit_paras.append(popt)

        fit_data = gaussian_func(xdata, *popt)
        check_fitting_quality(ion_index, xdata, ydata, fit_data)
        #print('fit_paras', popt)

    if plot_figure:
        plt.figure(figsize=(8,8))
        for ion_index in range(ion_number):
            avrg_single_ion = [avrg[ion_index] for avrg in avrg_data_all]
            x_fit = np.linspace(min(list_frequency),max(list_frequency), 100)
            avrg_fit = [gaussian_func(x, *fit_paras[ion_index]) for x in x_fit]

            plt.subplot(ion_number,1,ion_index+1)
            plt.plot(list_frequency, avrg_single_ion)

            xdata = np.array(list_frequency)
            ydata = np.array(avrg_single_ion)
            #a0 = max(ydata)
            #a1 = xdata[np.argmax(ydata)]
            #a2 = sum(y * (x - a1)**2)
            #ydata2 = gaussian_func(xdata, a0, a1, a2)
            #plt.plot(xdata, ydata2)

            plt.plot(x_fit, avrg_fit)
            #print(fit_paras[ion_index])
            plt.title(('This is ion {}, '+r'$\mu $'+'= {:.4f}, '+r'$\sigma = {:.4f}$').format(ion_index, fit_paras[ion_index][1], fit_paras[ion_index][2]))
            #plt.title(('This is ion {}, '+r'$\mu $'+'= {:.4f}, '+r'$\sigma = {:.4f}$').format(ion_index, fit_paras[ion_index][1], np.sqrt(2)/fit_paras[ion_index][2]))
            plt.xlabel('frequency '+' (MHz)')
            plt.ylabel('average count')

        plt.tight_layout()

    return fit_paras


def rabi_fit(fileName, convert_matrix = None, threshold = None, para_type = scanParameterType.frequency, plot_figure = False):
    list_time, list_matrices = raw_count(fileName)
    avrg_data_all = pre_process(list_matrices, convert_matrix)
    print(list_matrices)
    print(avrg_data_all)

    ion_number = convert_matrix.shape[0]

    fit_paras = []

    for ion_index in range(ion_number):
        avrg_single_ion = [avrg[ion_index] for avrg in avrg_data_all]
        fs = np.fft.fftfreq(len(list_time))
        fs = np.fft.fftfreq(len(list_time), list_time[1]-list_time[0])
        Y = abs(np.fft.fft(avrg_single_ion))
        freq = abs(fs[np.argmax(Y[1:])+1])
        #print(freq)
        xdata = list_time
        ydata = avrg_single_ion
        a0 = max(avrg_single_ion) - min(avrg_single_ion)
        a1 = 2 * np.pi * freq
        a2 = 0
        a3 = np.mean(avrg_single_ion)
        p0 = [a0,a1,a2,a3]
        popt, pcov = curve_fit(cosine_func, list_time, avrg_single_ion, p0=p0)
        fit_paras.append(popt)

        fit_data = gaussian_func(xdata, *popt)
        check_fitting_quality(ion_index, xdata, ydata, fit_data)

    if plot_figure:
        plt.figure(figsize=(8,8))
        for ion_index in range(ion_number):
            avrg_single_ion = [avrg[ion_index] for avrg in avrg_data_all]
            x_fit = np.linspace(min(list_time),max(list_time), 100)
            avrg_fit = [cosine_func(x, *fit_paras[ion_index]) for x in x_fit]

            plt.subplot(ion_number,1,ion_index+1)
            plt.plot(list_time, avrg_single_ion)
            plt.plot(x_fit, avrg_fit)
            plt.title('This is ion {}, the pi-pulse period is {:.4f} '.format(ion_index, (np.pi)/popt[1])+r'$\mu s$')
            plt.xlabel('time '+r'$(\mu s)$')
            plt.ylabel('average count')
        plt.tight_layout()

    return ([a[1]/(2*np.pi) for a in fit_paras])

def thermal_fit(fileName, convert_matrix = None, threshold = None, para_type = scanParameterType.frequency, plot_figure = False):
    list_time, list_matrices = raw_count(fileName)
    avrg_data_all = pre_process(list_matrices, convert_matrix)

    ion_number = convert_matrix.shape[0]

    fit_paras = []

    for ion_index in range(ion_number):
        avrg_single_ion = [avrg[ion_index] for avrg in avrg_data_all]
        fs = np.fft.fftfreq(len(list_time))
        fs = np.fft.fftfreq(len(list_time), list_time[1]-list_time[0])
        Y = abs(np.fft.fft(avrg_single_ion))
        freq = abs(fs[np.argmax(Y[1:])+1])
        print(freq)

