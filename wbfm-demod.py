#!/usr/bin/env python

import sys
import numpy as np
import scipy.signal as spsig
import struct
import pickle
import time


SAMPLE_RATE = 192e3

# Low pass filter params
FILTER_ORDER = 7
CUTOFF_FREQ = (15e3 * 2) / SAMPLE_RATE

AUDIO_DECIMATION_FACTOR = 6         ## 192 Khz / 6 = 32 Khz


def main(basebandFname, outFname):
    baseband = pickle.load(open(basebandFname))
    audioFile = open(outFname, "w")

    monoB, monoA = spsig.butter(FILTER_ORDER, CUTOFF_FREQ)
    start = time.time()
    deEmphasisB, deEmphasisA = spsig.butter(1, (2.1e3 * 2.0) / SAMPLE_RATE)
    nsamples = AUDIO_DECIMATION_FACTOR * int(len(baseband) / AUDIO_DECIMATION_FACTOR)

    I = baseband.imag
    Q = baseband.real

    ## Add an extra sample since diff deletes one
    phase = np.insert(np.arctan2(Q, I), 0, 0)
    signal = np.diff(np.unwrap(phase))

    ## De-emphasis filter
    deEmphasized = spsig.lfilter(deEmphasisB, deEmphasisA, signal)

    ## WBFM mono is up to 15 Khz.  LPF to 15 Khz
    filtered = spsig.lfilter(monoB, monoA, deEmphasized)

    ## But we're still at SAMPLE_RATE frequency.
    ## Downsample to 16 Khz
    pcm = filtered.reshape(nsamples / AUDIO_DECIMATION_FACTOR, AUDIO_DECIMATION_FACTOR)[:,0]

    ## Normalize it?
    m = np.mean(pcm)
    pcm -= m
    scale = 65536.0 / (np.max(pcm) - np.min(pcm))
    pcm *= scale

    wav = pcm.astype('<i2')
    audioFile.write(wav)

    audioFile.close()
    end = time.time()
    print "Time: %f ms." % ((end - start) * 1000.0)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print "Usage: %s <baseband signal path> <PCM out>" % sys.argv[0]
        sys.exit(0)

    main(sys.argv[1], sys.argv[2])
