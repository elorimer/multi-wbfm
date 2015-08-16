#!/usr/bin/env python

import sys
import numpy as np
import scipy.signal as spsig
import socket
import struct
import pickle
import time
import math


SAMPLE_RATE = 9.6e5
#CENTER_FREQ = 95.3e6
#CENTER_FREQ = 94.9e6
CENTER_FREQ = 95.1e6

# Low pass filter params
FILTER_ORDER = 7
DEVIATION_FREQ = (192e3 * 2) / SAMPLE_RATE
CHANNEL_DECIMATION = 5         ## 960 Khz => 192 Khz


def rtl_connect(host="127.0.0.1", port=1234):
    s = socket.socket()
    s.connect((host, port))
    s.recv(12)              ## discard the header
    return s

def rtl_receive(s, buf, nbytes):
    return s.recv_into(buf, nbytes, socket.MSG_WAITALL)

def rtl_set_freq(s, freq):
    s.send(struct.pack("!BI", 0x01, freq))

def rtl_close(s):
    s.close()

def read_capture_rtl(data):
    data = data.astype(np.float)
    data = data - 127

    I = data[range(0, len(data), 2)]
    Q = data[range(1, len(data), 2)]

    return (I, Q)



def main(outFname, captureTime, tuningOffset):
    nsamples = CHANNEL_DECIMATION * int(captureTime * SAMPLE_RATE / CHANNEL_DECIMATION)

    channelB, channelA = spsig.butter(FILTER_ORDER, DEVIATION_FREQ)

    ## generate the sine wave at tuningOffset Hz for the down-mixer
    omegaDelta = (2.0 * math.pi) / (SAMPLE_RATE / tuningOffset)
    LO_sine = np.array(range(nsamples), dtype = float)
    LO_cosine = np.array(range(nsamples), dtype = float)
    LO_sine = np.sin(LO_sine * omegaDelta)
    LO_cosine = np.cos(LO_cosine * omegaDelta)
#    LO = LO_cosine + 1j * LO_sine
    ## We want the difference spectrum ...
    LO = LO_sine + 1j * LO_cosine

    rtl = rtl_connect()
    rtl_set_freq(rtl, CENTER_FREQ)
    discard = np.empty(SAMPLE_RATE * 2, dtype=np.uint8)
    rtl_receive(rtl, discard, len(discard))

    buf = np.empty(nsamples * 2, dtype=np.uint8)

    print "Reading %d samples ..." % nsamples
    start = time.time()
    rtl_receive(rtl, buf, len(buf))
    end = time.time()
    print "Read in %f ms." % ((end - start) * 1000.0)
    start = time.time()
    I, Q = read_capture_rtl(buf)
    end = time.time()
    print "read_capture_rtl = %f ms" % ((end - start) * 1000.0)

    ## Channelization.
    print "Channelizing ..."

    start = time.time()
    print "LPF ..."
    Ii = spsig.lfilter(channelB, channelA, I)
    Qq = spsig.lfilter(channelB, channelA, Q)
    baseband_I = Ii.reshape(nsamples / CHANNEL_DECIMATION, CHANNEL_DECIMATION)[:,0]
    baseband_Q = Qq.reshape(nsamples / CHANNEL_DECIMATION, CHANNEL_DECIMATION)[:,0]
    end = time.time()
    print "Channelization took %f ms." % ((end - start) * 1000.0)

    pickle.dump(np.array(baseband_I + 1j * baseband_Q, 'complex'), open(outFname % 0, "w"))

    start = time.time()
    print "Down-mixing ..."
    complexSignal = I + 1j * Q
    signal = complexSignal * LO
    I, Q = signal.real, signal.imag
    print "LPF ..."
    Ii = spsig.lfilter(channelB, channelA, I)
    Qq = spsig.lfilter(channelB, channelA, Q)
    baseband_I = Ii.reshape(nsamples / CHANNEL_DECIMATION, CHANNEL_DECIMATION)[:,0]
    baseband_Q = Qq.reshape(nsamples / CHANNEL_DECIMATION, CHANNEL_DECIMATION)[:,0]
    end = time.time()
    print "Channelization took %f ms." % ((end - start) * 1000.0)

    pickle.dump(np.array(baseband_I + 1j * baseband_Q, 'complex'), open(outFname % 1, "w"))


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print "Usage: %s <output raw baseband template> <capture time> <tuning>" % sys.argv[0]
        sys.exit(0)

    main(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]))
