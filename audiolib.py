import numpy as np
import matplotlib.pyplot as plt

import librosa


def get_audio_from_dataset(filename : str, root : str = './dataset', fold : int = 1, graph : bool = True) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(f'{root}/audio/fold{fold}/{filename}.wav', sr=16000)
    if graph:
        plt.figure(figsize=(15, 3))
        plt.plot(y, lw=0.1, alpha=1)
        plt.show()
    return y, sr

def rect(wi : np.ndarray) -> np.ndarray:
    y = np.ones_like(wi)
    y[(wi < 0)] = 0
    y[(wi > 1)] = 0
    return y

def hann(wi : np.ndarray) -> np.ndarray:
    y = np.sin(wi * np.pi) ** 2
    y[(wi < 0)] = 0
    y[(wi > 1)] = 0
    return y

def hamm(wi : np.ndarray, a0 : float = 0.54) -> np.ndarray:
    y = a0 - ((1 - a0) * np.cos(2 * np.pi * wi))
    y[(wi < 0)] = 0
    y[(wi > 1)] = 0
    return y


def show_window(window_fn, n : int = 200):
    plt.figure(figsize=(8, 5))
    xs = np.linspace(0, 1, n, True)
    plt.ylim(0, 1)
    plt.plot(xs, window_fn(xs), lw=1)
    plt.show()




def stft(
        audio : np.ndarray, sampling_rate : int, 
        wsize : int = 512, window_fn = hann, hsize : int = None, fixed : bool = True
    ) -> np.ndarray:
    
    spect = []
    
    # wsize = int(wsize_ms * 1e-3 * sampling_rate)
    # wsize_ms = 1e3 * wsize / sampling_rate
    # if hsize_ms is None: hsize_ms = wsize_ms / 2
    # hsize = int(hsize_ms * 1e-3 * sampling_rate)
    if hsize is None: hsize = wsize // 2
    
    window = window_fn(np.linspace(0, 1, wsize, True))
    
    wtemp = np.zeros_like(audio)
    
    i = 0
    while i < len(audio):
        # print(f'\r{i}/{len(audio)}...', end='')        
        
        if fixed:
            sliced = audio[i : i + wsize] + 0
            if len(sliced) < wsize:
                sliced = np.concat((sliced, np.zeros(( wsize - len(sliced), ))))
            sliced *= window
        else:
            sliced = wtemp + 0
            sliced[i : i + wsize] = window[:len(sliced[i : i + wsize])]
            sliced *= audio
        
        
        spect.append(np.fft.rfft(sliced))
        
        i += hsize
    
    spect = np.array(spect, dtype=np.complex64)
    spect = np.transpose(spect)
    spect = spect[::-1, :-1]
    
    return spect






def analyse_audio(filename : str = None, audio = None, **kwargs):
    
    sr = 16000
    if filename is not None:
        audio, sr = get_audio_from_dataset(filename, graph=False, **kwargs,)
    
    rect_spectrogram = stft(audio, sr, wsize=1024, hsize=None, fixed=True, window_fn=rect)
    hann_spectrogram = stft(audio, sr, wsize=1024, hsize=None, fixed=True, window_fn=hann)
    hamm_spectrogram = stft(audio, sr, wsize=1024, hsize=None, fixed=True, window_fn=hamm)
    
    rect_spectrogram = np.log(np.absolute(rect_spectrogram)) * 20
    hann_spectrogram = np.log(np.absolute(hann_spectrogram)) * 20
    hamm_spectrogram = np.log(np.absolute(hamm_spectrogram)) * 20
    
    fig, ax = plt.subplots(4, 1, figsize=(15, 8))
    
    timeax = np.array(list(range(len(audio)))) / sr
    ax[0].plot(
        timeax, 
        audio, 
        alpha=1, lw=0.2
    )
    ax[0].spines['bottom'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].xaxis.set_visible(False)
    ax[0].yaxis.set_visible(False)
    ax[0].set_xlim((timeax[0], timeax[-1]))
    ax[0].title.set_text('audio')
    
    
    ax[1].set_yticks(np.linspace(0, rect_spectrogram.shape[0] - 1, 5, True))
    ax[1].xaxis.set_visible(False)
    ax[1].yaxis.set_visible(False)
    img = ax[1].imshow(
        rect_spectrogram, cmap='inferno',
        aspect=(1e-1 * rect_spectrogram.shape[1]/rect_spectrogram.shape[0])
    )
    ax[1].title.set_text('rectangular windowing')
    
    ax[2].set_yticks(np.linspace(0, hann_spectrogram.shape[0] - 1, 5, True))
    ax[2].xaxis.set_visible(False)
    ax[2].yaxis.set_visible(False)
    img = ax[2].imshow(
        hann_spectrogram, cmap='inferno',
        aspect=(1e-1 * hann_spectrogram.shape[1]/hann_spectrogram.shape[0])
    )
    ax[2].title.set_text('hanning windowing')
    
    ax[3].set_yticks(np.linspace(0, hamm_spectrogram.shape[0] - 1, 5, True))
    ax[3].xaxis.set_visible(False)
    ax[3].yaxis.set_visible(False)
    img = ax[3].imshow(
        hamm_spectrogram, cmap='inferno',
        aspect=(1e-1 * hamm_spectrogram.shape[1]/hamm_spectrogram.shape[0])
    )
    ax[3].title.set_text('hamming windowing')
    
    fig.tight_layout()
    plt.show()






