
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import struct
from scipy.fftpack import fft
import sys
import time


class AudioStream(object):
    def __init__(self):

        # stream constants
        self.CHUNK = 1024 * 2
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.pause = False

        # stream object
        self.p = pyaudio.PyAudio()
        for i in range(self.p.get_device_count()):
            if "stereo" in self.p.get_device_info_by_index(i)['name'].lower():
                self.device = i
                break
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            output=True,
            frames_per_buffer=self.CHUNK,
            input_device_index=self.device
        )
        self.init_plots()
        self.start_plot()

    def init_plots(self):

        # x variables for plotting
        x = np.arange(0, 2 * self.CHUNK, 2)
        xf = np.linspace(0, self.RATE, self.CHUNK)

        # create matplotlib figure and axes
        self.fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(15, 7))
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)

        # create a line object with random data
        self.line, = ax1.plot(x, np.random.rand(self.CHUNK), '-', lw=2)

        # create semilogx line for spectrum
        self.line_fft, = ax2.plot(x, np.random.rand(self.CHUNK), '-', lw=2)

        # create line for volatility
        self.line_vol, = ax3.plot(x, np.random.rand(self.CHUNK), '-', lw=2)

        # format waveform axes
        ax1.set_title('AUDIO WAVEFORM')
        ax1.set_xlabel('samples')
        ax1.set_ylabel('volume')
        ax1.set_ylim(0, 255)
        ax1.set_xlim(0, 2 * self.CHUNK)
        plt.setp(
            ax1, yticks=[0, 128, 255],
            xticks=[0, self.CHUNK, 2 * self.CHUNK],
        )
        plt.setp(ax2, yticks=[0, 1],)
        plt.setp(ax3, yticks=[0, 1],)

        # format spectrum axes
        ax2.set_xlim(20, 2048)

        # format volatility axes
        ax3.set_title('VOLATILITY')
        ax3.set_xlabel('samples')
        ax3.set_ylabel('volatility')

        plt.show(block=False)

    def start_plot(self):
        vol_size = 50
        combine_vol_chunks = 2**6
        print('stream started')
        frame_count = 0
        start_time = time.time()
        vol_aplifier = 15
        voL_scale = 1
        last_frames = np.zeros((vol_size,self.CHUNK))
        while not self.pause:

            data = self.stream.read(self.CHUNK)
            data_int = struct.unpack(str(2 * self.CHUNK) + 'B', data)
            data_np = np.array(data_int).astype('b')[::2] + 128

            self.line.set_ydata(data_np)

            # compute FFT and update line
            yf = fft(data_int)

            self.line_fft.set_ydata(
                np.abs(yf[0:self.CHUNK]) / (128 * self.CHUNK))

            # compute volatility
            last_frames[frame_count % vol_size] = data_np[:]
            #volatility = [np.std(last_frames[:,i//combine_vol_chunks]) / np.mean(last_frames[:,i//combine_vol_chunks]) for i in range(self.CHUNK)]
            # make a fast version of volatility calculation
            volatility = last_frames.std(axis=0) / np.mean(last_frames, axis=0)


            for i in range(len(volatility) // combine_vol_chunks):
                volatility[i*combine_vol_chunks:(i+1) * combine_vol_chunks] = [max(volatility[i*combine_vol_chunks:(i+1) * combine_vol_chunks])] * combine_vol_chunks


            volatility /= np.max(volatility)
            #print(volatility)

            volatility = np.array(volatility) ** vol_aplifier
            volatility = np.array(volatility) * voL_scale


            volatility.resize(self.CHUNK)


            self.line_vol.set_ydata(volatility)
            ## Amplify


            # update figure canvas
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            if (frame_count % vol_size == 0):
                print('volatility = {:.2f} Sek'.format((time.time() - start_time)))
                start_time = time.time()
            frame_count += 1

        else:
            self.fr = frame_count / (time.time() - start_time)
            print('average frame rate = {:.0f} FPS'.format(self.fr))
            self.exit_app()

    def exit_app(self):
        print('stream closed')
        self.p.close(self.stream)

    def onClick(self, event):
        #self.pause = True
        pass


if __name__ == '__main__':
    AudioStream()
