import librosa
import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy.signal import correlate

#=================================
#load data
def cross_correlation(data):
    shift = 980
    begin = 0
    end = 9600 - 1
    j=1
    for w in range(1, 88):
        Window = data[begin:end, :]
        begin = begin + shift
        end = end + shift
        Sig1 = Window[:, 0]
        Sig2 = Window[:, 1]
        feature = np.correlate(Sig1, Sig2, 'same')
        feature2 = len(feature) // 2
        feature3 = feature2 + 25
        feature4 = feature2 - 25
        Sig3 = feature[feature4:feature3 + 1, ]
        if j == 1:
            C = np.reshape(Sig3, (51, 1))
        else:
            C = np.hstack([C, np.reshape(Sig3, (51, 1))])
        j=j+1
    return C


file_name = "C:/Users/Lenovo/PycharmProjects/AI_Project/idmt_traffic_all.txt"


file = open(file_name, "r")
list = file.readlines()


for i in list:
    #Melspectrogra_Feature
    data, sr = librosa.load(i.strip())
    d = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=2048, win_length=1024, hop_length=512)
    d = np.array(d)

    #Cross_Correlate
    sr, data = wavfile.read(i.strip())
    k = cross_correlation(data)
    F = np.vstack([d, k])

    Fname = i.split('.')
    FeatureDir = 'C:/Users/Ladan_Gh/PycharmProjects/AI_Project/data/features/'


    csvFileName = FeatureDir + Fname[0] + '.csv'
    np.savetxt(csvFileName, F, delimiter=',')
    print(csvFileName+' is Saved.')