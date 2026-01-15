#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch, hilbert, convolve, medfilt
from scipy import signal
from scipy.signal.windows import hann, boxcar
from scipy.stats import zscore


# %% initiate object
class dataset:
    '''
        Tạo object dataset để preprocessing dữ liệu EEG.

        Cung cấp:
            1. một đối tượng dưới dạng tập dữ liệu, với định dạng EEG của Brainclincis Diagnostics.
            2. Các chức năng hiệu chỉnh và loại bỏ nhiễu EEG được mô tả bên dưới.
            3. Các tùy chọn để lưu bộ dữ liệu EEG đã được xử lý trước dưới dạng .csv, .npy (pickle), hoặc .mat

        INPUT :
        - filename: đường dẫn tới file EEG (CSV/EDF). Trong pipeline đang dùng CSV.
        - Fs: tần số lấy mẫu (mặc định 500 Hz)

        OUTPUT của object (sau khi chạy preprocessing):
        - self.data: dữ liệu EEG (và kênh phụ) sau lọc/sửa artifact/đánh dấu artifact
        - self.labels: nhãn kênh tương ứng với self.data
        - self.artifacts: dict chứa các artifact đã phát hiện (EMG/JUMP/KURT/SWING/EB/EOG...)
        - self.info: dict meta thông tin (lọc, chất lượng dữ liệu, v.v.)
        - self.trl, self.arttrl, self.artidata: phục vụ segment và lưu artifact segments

        Ghi chú:
        - Dữ liệu EEG chuẩn Brainclinics: 26 kênh EEG + kênh EOG và kênh phụ khác.
    '''

    def __init__(self, filename, Fs = 500): 
        self.artifacts = {}     # lưu artifact detections (trl, samps, bad channels, bridging, ...)
        self.info = {}          # lưu metadata của quá trình preprocessing
        self.data = []          # dữ liệu tín hiệu (kênh x mẫu) hoặc (trial x kênh x mẫu) tuỳ stage
        self.trl = []           # danh sách đoạn (begin,end) theo sample sau khi segment
        self.artidata = []      # dữ liệu cắt ra chỉ phần artifact (nếu remove_artifact='yes')
        self.arttrl = []        # trl của artifact tổng hợp
        self.info['fileID'] = filename # lưu đường dẫn file input
        self.Fs = Fs            # sampling rate (mặc định 500 Hz)
        

        # Danh sách nhãn kênh theo chuẩn Brainclinics Diagnostics
        # 0..25: EEG (26 kênh)
        # VPVA/VNVB/HPHL/HNHR: EOG đơn cực để tạo VEOG/HEOG bipolar
        # Erbs/OrbOcc/Mass: kênh phụ khác (EMG/điểm đo khác)
        self.labels = ['Fp1','Fp2',
               'F7','F3','Fz','F4','F8',
               'FC3','FCz','FC4',
               'T7','C3','Cz','C4','T8',
               'CP3','CPz','CP4',
               'P7','P3','Pz','P4','P8',
               'O1','Oz','O2',
               'VPVA','VNVB','HPHL','HNHR', 'Erbs', 'OrbOcc','Mass']

        # Ngoài ra, cần khởi tạo một từ điển định nghĩa các kênh lân cận cho mỗi kênh EEG, để sửa chữa kênh nếu nó quá nhiễu, bị nối tắt hoặc bị hỏng.
        # Dictionary: hàng xóm của mỗi kênh EEG (dùng cho nội suy kênh xấu / bridging)
        # key: label kênh EEG, value: list label kênh lân cận
        self.neighblabels = {'Fp1': ['Fp2','F7', 'F3'],
                             'Fp2': ['Fp1', 'F8','F4'],
                             'F7': ['Fp1','F3','F7'],
                             'F3': ['Fp1','Fz','FC3','F7'],
                             'Fz': ['F4','FCz','F3'],
                             'F4': ['Fp2','F8','FC4','Fz'],
                             'F8': ['Fp2','F4','T8'],
                             'FC3':['F3', 'C3','FCz'],
                             'FCz':['Fz', 'FC3','FC4','Cz'],
                             'FC4':['F4','FCz','C4'],
                             'T7': ['F7', 'P7', 'C3'],
                             'C3': ['FC3','Cz','CP3'],
                             'Cz': ['FCz','CPz','C3','C4'],
                             'C4': ['Cz', 'CP4', 'FC4'],
                             'T8': ['F8', 'P8', 'C4'],
                             'CP3': ['C3','CPz','P3'],
                             'CPz': ['Cz','CP4','CP3','Pz'],
                             'CP4': ['C4','P4','CPz'],
                             'P7': ['F7','P3','O1'],
                             'P3': ['P7','CP3','Pz','O1'],
                             'Pz': ['P3','CPz','P4','Oz'],
                             'P4': ['Pz','CP4','P8','O2'],
                             'P8': ['T8','P4','O2'],
                             'O1': ['P7','P3','Oz'],
                             'Oz': ['O1','Pz','O2'],
                             'O2': ['Oz','P4','P8']}

    def loaddata(self):
        '''
            Load dữ liệu từ fileID (hiện code chỉ xử lý CSV nếu filename kết thúc .csv).
            - Đọc CSV theo đúng thứ tự cột trong self.labels
            - Chuyển thành matrix dạng (n_channels, n_samples)
            - labels chuyển thành np.array để where/compare nhanh
            INPUT: khởi tạo
            OUTPUT: dataset object bao gồm dữ liệu.
        '''
        # Nếu file là CSV
        if self.info['fileID'][-4:] =='.csv':
            # Đọc các cột đúng bằng self.labels (tránh sai thứ tự hoặc thiếu cột)
            tmp = pd.read_csv(self.info['fileID'],low_memory=False,sep = ',',
                              header = 0, usecols=self.labels, float_precision = 'high')
            # tmp.values: (n_samples, n_channels) -> transpose thành (n_channels, n_samples)
            self.data = tmp.values.T.astype(float)
            self.labels=np.array(self.labels)

    def bipolarEOG(self):
        '''
            Tính toán điện nhãn cầu lưỡng cực (bipolar EOG) từ các bản ghi điện nhãn cầu ['VPVA','VNVB','HPHL','HNHR']
            (chỉ áp dụng cho dữ liệu thô (.csv)
            Tạo kênh EOG bipolar:
            - VEOG = VPVA - VNVB
            - HEOG = HPHL - HNHR

            Sau đó:
            - Giữ 26 kênh EEG đầu (0..25)
            - Thêm 2 kênh VEOG, HEOG
            - Thêm các kênh phụ từ index 30 trở đi (Erbs/OrbOcc/Mass), bỏ 4 kênh EOG đơn cực.
            - Cập nhật labels tương ứng.

            INPUT: khởi tạo
            OUTPUT: Đối tượng tập dữ liệu bao gồm dữ liệu, ['VPVA','VNVB','HPHL','HNHR'] được thay thế bằng ['VEOG','HEOG'].
        '''
        # Lấy index của các kênh EOG đơn cực trong labels
        VPVA = np.where(self.labels=='VPVA')[0][0];VNVB = np.where(self.labels=='VNVB')[0][0]
        HPHL = np.where(self.labels=='HPHL')[0][0];HNHR = np.where(self.labels=='HNHR')[0][0]
        
        # Tạo data mới với VEOG, HEOG thay cho 4 kênh EOG đơn cực
        self.data = np.vstack((self.data[0:26],[(self.data[VPVA] - self.data[VNVB]), (self.data[HPHL] - self.data[HNHR])], self.data[30:]))
        
        # Tạo labels mới 
        self.labels = np.append(self.labels[0:26],np.append(['VEOG','HEOG'], self.labels[30:]))

    def demean(self):
        '''
            Baseline correction đơn giản:
            - Trừ mean theo thời gian cho mỗi kênh trong 30 kênh đầu (EEG+VEOG+HEOG+... tuỳ cấu trúc).
            INPUT: khởi tạo
            OUTPUT: Đối tượng tập dữ liệu với dữ liệu đã được loại bỏ giá trị trung bình.
        '''
        self.data[:30,:] = self.data[:30,:]-(np.nanmean(self.data[:30,:],axis=1).reshape((self.data[:30,:].shape[0],1)))
        self.info['demeaned']= 'all channels'

    def apply_filters(self, trlpadding=10, hpfreq=0.5, lpfreq=100, notchfilt = 'yes', notchfreq=50, Q=100):
        '''
            Lọc tín hiệu theo thứ tự:
            1) Notch (mặc định 50 Hz, Q=100) bằng iirnotch + filtfilt
            2) High-pass Butterworth bậc 4 (mặc định 0.5 Hz) + filtfilt
            3) Low-pass Butterworth bậc 4 (mặc định 100 Hz) + filtfilt

            INPUT: khởi tạo
            - notchfreq: tần số cắt, số thực
            - hpfreq: tần số thông cao, số thực > 0.5
            - lpfreq: tần số thông thấp, số thực
            OUTPUT: Đối tượng tập dữ liệu với dữ liệu đã được lọc

        '''
        # Nyquist frequency
        nyq = 0.5 * self.Fs
        n_rows = self.data.shape[0] # số kênh hiện có

        chans = n_rows
        for r in range(chans):
            # Notch filter nếu bật
            if notchfilt=='yes':
                b, a = iirnotch(notchfreq, Q, fs=self.Fs)
                data = filtfilt(b, a, self.data[r,:])
            else:
                data = self.data[r,:]
            
            # High-pass
            normal_cutoff = hpfreq / nyq
            b, a = butter(4, normal_cutoff, btype='highpass', analog=False)
            hpdata = filtfilt(b, a, data)
            
            # Low-pass
            normal_cutoff = lpfreq / nyq
            b, a = butter(4, normal_cutoff, btype='lowpass', analog=False)
            self.data[r,:] = filtfilt(b, a, hpdata)

        # Dọn biến tạm
        del hpdata, data, b, a

        # Lưu log filter setting
        self.info['filtered']= ['hp: '+str(hpfreq) +' ,lp: '+ str(lpfreq) + ' ,notch: '+str(notchfreq)]

    def apply_bpfilter(self,freqrange):
        '''
        Band-pass filter (Butterworth bậc 4) theo freqrange = [low, high]
        - Thiết kế ở dạng SOS
        - Áp dụng zero-phase bằng sosfiltfilt cho toàn bộ self.data
        '''
        nyq = 0.5 * self.Fs #250
        high_pass = freqrange[0] / nyq
        low_pass = freqrange[1] / nyq

        # bandpassfilter
        sos = butter(4, [high_pass, low_pass], btype='bandpass', analog=False, output = 'sos')
        self.data = sosfiltfilt(sos, self.data)

    def correct_EOG(self, lpfreq = 15, vthreshold = 0.2, vpadding = 0.3, hthreshold = 0.2, hpadding =0.3):
        '''
            Phát hiện artifact mắt và hồi quy (regression) EOG lên EEG để trừ (Gratton-style).

            Quy trình (cho từng kênh EOG: VEOG rồi HEOG):
            - Low-pass EOG ở lpfreq
            - Tính envelope qua Hilbert
            - Làm mượt envelope bằng boxcar (0.2s)
            - Detect đoạn vượt threshold bằng zscore (_detect_artifact)
            - Mở rộng đoạn theo padding (vpadding/hpadding)
            - Với từng EEG channel và từng đoạn artifact:
                + Ước lượng hệ số k bằng least squares (np.linalg.lstsq)
                + Áp taper Tukey để giảm nhiễu broadband do discontinuity
                + Trừ k*EOG khỏi EEG trong đoạn đó
            - Lưu ARTtrl vào self.artifacts['VEOG'] hoặc self.artifacts['HEOG']

            INPUT:
            -------------------------------------------------------------------
            - lpfreq:       tần số cắt thấp, kiểu float, mặc định = 15
            - threshold:    ngưỡng, giá trị z, mặc định = 0.5
            - padding:      khoảng đệm xung quanh vị trí đạt ngưỡng, tương đối so với độ dài của nhiễu, 
                            mặc định = 2 (theo kinh nghiệm, khoảng đệm gấp 2 lần độ dài của nhiễu trước
                            và sau khi nhiễu đạt đến điểm tối ưu cho kết quả tốt nhất)

            Returns
            -------------------------------------------------------------------
            Đối tượng tập dữ liệu với dữ liệu đã được hiệu chỉnh cho chuyển động mắt theo chiều dọc và 
            chiều ngang, và mẫu bắt đầu và kết thúc của mỗi nhiễu được phát hiện có thể được tìm thấy 
            trong trường artifact.
        '''

        eye_channel = ['VEOG','HEOG']   # 2 kênh EOG bipolar
        trlpadding = 1                  # padding thêm 1 giây để xử lý biên
        n_data_rows = 26                # chỉ sửa 26 kênh EEG

        # Aweight: lưu trọng số hồi quy theo (EOG_type, EEG_channel, time)
        Aweight = np.zeros((len(eye_channel),n_data_rows,(2*trlpadding*self.Fs)+self.data.shape[1]))
        
        # datapaddedEOG: EOG đã padding đầu-cuối
        datapaddedEOG = np.zeros((len(eye_channel),(2*trlpadding*self.Fs)+self.data.shape[1]))
#        import matplotlib.pyplot as plt
#        plt.plot(self.data[8,:])
#        plt.show()

        for n in range(len(eye_channel)):
            # Chọn padding/threshold theo VEOG hay HEOG
            if n == 0:
                padding = vpadding
                threshold = vthreshold
            elif n == 1:
                padding = hpadding
                threshold = hthreshold
            #Atrl = []

            # Lấy signal của kênh EOG hiện tại
            EOG = self.data[np.where(self.labels==eye_channel[n])[0]][0]

            # Khởi tạo biến tạm
            hilEOG = np.zeros((1,self.data.shape[1]))
            hilbEOG = hilEOG.copy()
            filtEOG = hilEOG.copy()

            # Low-pass EOG để lấy thành phần chậm (mắt)
            nyq = 0.5 * self.Fs
            normal_cutoff = lpfreq / nyq
            b, a = butter(4, normal_cutoff, btype='lowpass', analog=False)
            filtEOG = filtfilt(b, a, EOG)

            # Hilbert analytic signal để lấy envelope # zero-padding 20% độ dài
            hilEOG  = hilbert(filtEOG.copy(), N=int(len(filtEOG)+
                                             len(filtEOG)*0.20), axis = -1)
            hilbEOG = hilEOG[:filtEOG.shape[0]]
            amplenv = np.abs(hilbEOG)

            # Làm mượt envelope bằng boxcar 0.2s
            boxdata = convolve(amplenv, boxcar(int(0.2*self.Fs)), mode ='same', method ='direct')

            # Padding dữ liệu EOG để tránh vấn đề biên khi detect artifact
            datapaddedEOG[n,:] = np.hstack((filtEOG[:trlpadding*self.Fs],filtEOG,filtEOG[len(filtEOG)-trlpadding*self.Fs:]))
            datapaddedboxdata = np.hstack((boxdata[:trlpadding*self.Fs],boxdata,boxdata[len(filtEOG)-trlpadding*self.Fs:]))
            
            # Detect các đoạn artifact trên envelope đã smooth
            Atrl, Asamps = self._detect_artifact(datapaddedboxdata,threshold)

            # Padding dữ liệu EEG để đồng bộ với datapaddedEOG
            datapaddeddata = np.zeros((n_data_rows,datapaddedEOG.shape[1]))
            for r in range(n_data_rows):
                datapaddeddata[r,:] = np.hstack((self.data[r,:trlpadding*self.Fs],self.data[r,:],self.data[r,len(filtEOG)-trlpadding*self.Fs:]))

            # Tạo vector đánh dấu sample thuộc artifact (đã mở rộng padding theo độ dài artifact)
            artsamples = np.zeros(datapaddeddata.shape[1],dtype=int)
            if len(Atrl) > 0:
                for i in range(Atrl.shape[0]):
                    if Atrl[i,0]==0:
                        artsamples[0:Atrl[0,1]+int((Atrl[0,1]-0)*padding)]=1
                    elif Atrl[i,1]==datapaddeddata.shape[1]:
                        artsamples[Atrl[i,0]-int((Atrl[i,1]-Atrl[i,0])*padding):datapaddeddata.shape[1]]=1
                    else:
                        artsamples[Atrl[i,0]-int((Atrl[i,1]-Atrl[i,0])*padding):Atrl[i,1]+int((Atrl[i,1]-Atrl[i,0])*padding)]=1

            # Từ artsamples, gom lại thành các đoạn ARTtrl (start/end)
            p = np.where(artsamples==1)[0]
            startidxs=0
            if len(p) > 1:
                if p[0]==0:
                    startidxs = np.append(startidxs,0)

                startidxs = np.append(startidxs,[np.where(np.diff(artsamples)==1)[0]+1])# diff =1
                startidxs = startidxs[1:]

                endidxs = np.hstack([np.where(np.diff(artsamples)==-1)[0]+1])#diff = -1
                if len(endidxs)<len(startidxs):
                    endidxs = np.append(endidxs,datapaddeddata.shape[1])

                ARTtrl = np.array([0,0],dtype=int)
                for i in range(len(startidxs)):
                    ARTtrl = np.vstack((ARTtrl,[startidxs[i],endidxs[i]]))
                ARTtrl = ARTtrl[1:]

                # Log + lưu ARTtrl
                print('Eye artifact correction: correcting '+str(ARTtrl.shape[0])+ ' '+ eye_channel[n] + ' eye artifact(s)')
                self.artifacts[eye_channel[n]] =  ARTtrl

                # EOGweight: trọng số theo (artifact_segment, EEG_channel, time)
                EOGweight = np.zeros((len(ARTtrl),n_data_rows,datapaddeddata.shape[1]))
                Atmpweight = EOGweight.copy()

                # EOG dạng vector cột để lstsq
                EOG_row_vec = datapaddedEOG[n,:].reshape((datapaddedEOG.shape[1], 1))

                newdata = np.zeros((datapaddeddata.shape))
                for r in range(n_data_rows):
                    for k in range(ARTtrl.shape[0]):
                            # Tukey taper để giảm biên gãy gây phổ rộng
                            Attaper = signal.windows.tukey(len(np.arange(ARTtrl[k,0],ARTtrl[k,1])),alpha = 0.025)

                            # Ước lượng hệ số hồi quy k (least squares) trên đoạn artifact
                            Atmpweight[k,r,ARTtrl[k,0]:ARTtrl[k,1]] = np.linalg.lstsq(
                                    EOG_row_vec[ARTtrl[k,0]:ARTtrl[k,1]],
                                    datapaddeddata[r,ARTtrl[k,0]:ARTtrl[k,1]],rcond=None)[0]
                            
                            # Nhân taper vào hệ số theo thời gian
                            EOGweight[k,r,ARTtrl[k,0]:ARTtrl[k,1]] = Attaper*Atmpweight[k,r,ARTtrl[k,0]:ARTtrl[k,1]]

                    # Cộng các segment artifact lại thành trọng số tổng
                    Aweight[n,r,:] = np.sum(EOGweight[:,r,:], axis=0)

                    # Trừ thành phần EOG đã mô hình hoá khỏi EEG
                    newdata[r,:] = datapaddeddata[r,:]-((Aweight[n,r,:])*datapaddedEOG[n,:])

                    # Bỏ padding, ghi ngược vào self.data
                    self.data[r,:] = newdata[r,(trlpadding*self.Fs):datapaddedEOG.shape[1]-(trlpadding*self.Fs)]
            else:
                # Không phát hiện artifact -> log
                ARTtrl = []
                print('Eye artifact correction: correcting 0 '+ eye_channel[n] + ' eye artifact(s)')
                self.info[eye_channel[n]] = '0 artifacts detected @ threshold: '+str(threshold)+' and corrected'


    def detect_emg(self, hpfreq = 75, lpfreq = 95, threshold = 4, padding=0.1):
        ''' 
            Detect EMG:
            - Band-pass 75-95 Hz cho toàn bộ kênh
            - Hilbert envelope
            - Smooth envelope bằng Hann window 0.5s
            - Z-score theo thời gian cho từng kênh
            - Chọn điểm vượt mean(Z)+threshold và envelope>3
            - Boxcar 0.5s để làm dày vùng artifact
            - _artifact_samps_trl để gom thành trl và padding theo giây

            INPUT:
            - hpfreq: highpass frequency
            - lpfreq: lowpass frequency
            - threshold: số độ lệch chuẩn
            - padding: padding the artifact (in seconds)

            OUTPUT: Đối tượng tập dữ liệu với mẫu bắt đầu và kết thúc của mỗi artifact được phát hiện có thể được tìm thấy trong trường artifact 
            Mẫu bắt đầu và kết thúc của mỗi artifact được lưu giữ trong trường artifact  để loại bỏ sau này.
            Lưu:
            - self.artifacts['EMGsamps'], self.artifacts['EMGtrl']
            - self.info['EMG detection']
        '''

        nyq = 0.5 * self.Fs
        high_pass = hpfreq / nyq
        low_pass = lpfreq / nyq

        # Band-pass theo SOS
        sos = butter(4, [high_pass, low_pass], btype='bandpass', analog=False, output = 'sos')
        filtEMG = sosfiltfilt(sos, self.data)

        # Chọn N cho Hilbert (đảm bảo chẵn)
        N=int(filtEMG.shape[1])#+filtEMG.shape[1]*0.10)
        if N % 2 == 0:
            N=N
        else:
            N=N+1

         # Envelope Hilbert
        hilbEMG  = hilbert(filtEMG.copy(), N=N, axis = -1)
        amplenv = np.abs(hilbEMG[:,:filtEMG.shape[1]])

        n_data_rows=26 
        EMGsamps = np.zeros((n_data_rows,self.data.shape[1]))

        # Smooth bằng Hann 0.5s
        hanndata = np.zeros((n_data_rows,self.data.shape[1]))
        for r in range(n_data_rows):
            hanndata[r,:] = convolve(amplenv[r,:], hann(int(0.5*self.Fs),sym=True), mode ='same')#, method ='direct')
        
        # Z-score theo kênh
        Zdata = zscore(hanndata,axis=1)
        tmpEMGsamps = np.zeros((hanndata.shape[0],hanndata.shape[1]))
        inpEMGsamps = np.zeros((hanndata.shape[0],hanndata.shape[1]))

        # Thresholding
        for r in range(hanndata.shape[0]):
            if ~np.isnan(Zdata[r,0]):
                sidx = np.where(Zdata[r,:] > np.nanmean(Zdata)+threshold)[0]
                #introduce an absolute threshold to extract only EMG data that is evident?
                didx = np.where(amplenv[r,sidx]>3)
                tmpEMGsamps[r,sidx[didx]]=1

                # Boxcar 0.5s để làm dày vùng EMG
                boxdata = convolve(tmpEMGsamps[r,:], boxcar(int(0.5*self.Fs)), mode ='same', method ='direct')
                inpEMGsamps[r,np.where(boxdata>0)]=1

        # Gom artifact samples thành trl + padding
        EMGtrl, EMGsamps = self._artifact_samps_trl(inpEMGsamps, padding,self.Fs, self.data[-1].shape[0])

        print('EMG detection: detected '+str(len(EMGtrl))+' artifact(s)')

        self.info['EMG detection'] = str(len(EMGtrl))+' artifacts detected @ threshold: Z = '+str(threshold)
        self.artifacts['EMGsamps'] = EMGsamps
        self.artifacts['EMGtrl'] = EMGtrl


    def detect_jumps(self, padding=0.01, threshold = 5):
        '''
            Phát hiện các sự thay đổi hoặc nhảy vọt đột ngột bất thường trên đường cơ sở dữ liệu, cho mỗi kênh. 
            Điểm bắt đầu và thời gian lấy mẫu của mỗi sự thay đổi bất thường được lưu giữ trong trường sự thay đổi 
            bất thường để loại bỏ sau này.
            - Median filter (kernel 9) để giảm nhiễu
            - Lấy abs(diff) theo thời gian
            - Z-score của diff
            - Mark các điểm có Z > mean(Z)+threshold và diff>30
            - Gom thành trl bằng _artifact_samps_trl

            INPUT:
            - threshold: z-value, default = 2
            - padding: Khoảng đệm xung quanh artifact tính bằng giây, mặc định = 0,05, default = 0.05

            OUTPUT:
            - self.artifacts['JUMPsamps'], self.artifacts['JUMPtrl']
        '''

        n_data_rows = 26 

        inpJUMPsamps = np.zeros((n_data_rows,self.data.shape[1]))
        filtdata = np.zeros(self.data.shape)
        diffdata = np.zeros(self.data.shape)
        Zdata = np.zeros(self.data.shape)
        for r in range(n_data_rows):
            if ~np.isnan(self.data[r,0]):
                filtdata[r,:] = medfilt(self.data[r,:],kernel_size=(9))
                diffdata[r,1:] = np.abs(np.diff(filtdata[r,:],n=1))
                Zdata[r,:] = zscore(diffdata[r,:])
                sidx = (np.where(Zdata[r,:] > np.nanmean(Zdata)+threshold)[0])
                didx = np.where(diffdata[r,sidx]>30)[0]
                inpJUMPsamps[r,sidx[didx]]=1

        JUMPtrl, JUMPsamps = self._artifact_samps_trl(inpJUMPsamps, padding, self.Fs, self.data[-1].shape[0])

        print('Jump/ baseline shift : '+str(len(JUMPtrl))+ ' jumps/baselineshifts detected')

        self.info['jump detection'] = str(len(JUMPtrl))+' jumps/baseline shifts detected @ threshold: Z = '+str(threshold)
        self.artifacts['JUMPsamps'] = JUMPsamps
        self.artifacts['JUMPtrl'] = JUMPtrl


    def detect_kurtosis(self, threshold=8, padding=0.1, overlap=0.1, winlen = 4):
        '''
            Phát hiện các đoạn dữ liệu có độ nhọn cực đại cho mỗi kênh. Quá trình này được thực hiện trên cửa sổ trượt có chồng lấp,
            do đó chỉ những phần dữ liệu có độ nhọn cực đại mới được đánh dấu để loại bỏ sau này, chứ không phải toàn bộ các lần 
            thử nghiệm đều phải bị loại bỏ.
            Detect kurtosis cao theo cửa sổ trượt:
            - Tạo cửa sổ độ dài winlen giây, bước overlap giây
            - Tính kurtosis trên từng cửa sổ, gán giá trị cho đoạn cửa sổ đó
            - Mark sample nào kurtosis > threshold
            - Gom thành trl bằng _artifact_samps_trl
            INPUT:
            - threshold:    z-value, default = 4
            - padding:      Khoảng đệm xung quanh hiện vật tính bằng giây, default = 0.1
            - overlap:      lượng chồng chéo của các cửa sổ di chuyển, xác định độ phân giải,
                            tính bằng giây, default = 0.05
            - winlen:       độ dài của các cửa sổ trượt để đánh giá độ nhọn
            OUTPUT:
            - self.artifacts['KURTsamps'], self.artifacts['KURTtrl'] (nếu có)
        '''

        from scipy.stats import kurtosis

        if winlen == 'all':
            winlen = self.data.shape[-1]/self.Fs

        winstarts = np.arange(0,self.data.shape[1]-(winlen*self.Fs),overlap*self.Fs)
        winends = winstarts+winlen*self.Fs

        n_data_rows = 26 # number of EEG channels

        kurt = np.zeros((n_data_rows,self.data.shape[-1]))
        inpKURTsamps = kurt.copy()
        for r in range(n_data_rows):
            if ~np.isnan(self.data[r,0]):
                for w in range(len(winstarts)):
                    kurt[r,int(winstarts[w]):int(winends[w])] = kurtosis(self.data[r,int(winstarts[w]):int(winends[w])],fisher = True)

                if len(np.where(kurt[r,:]>threshold)[0]) > 0:
                    inpKURTsamps[r,np.where(kurt[r,:]>threshold)[0]]=1

        del kurt

        KURTtrl, KURTsamps = self._artifact_samps_trl(inpKURTsamps, padding, self.Fs, self.data[-1].shape[0])

        if len(KURTtrl)>0:
            self.artifacts['KURTsamps'] = KURTsamps
            print('kurtosis: '+str(KURTtrl.shape[0])+ ' samples with kurtosis detected')
            self.info['kurtosis detection'] = str(len(KURTtrl))+' samples with kurtosis detected @ threshold: Z = '+str(threshold)
        else:
            KURTtrl = np.array([])
            print('kurtosis: 0 samples with kurtosis detected')
            self.info['kurtosis detection'] = '0 samples with kurtosis detected @ threshold: Z = '+str(threshold)
        self.artifacts['KURTtrl'] = KURTtrl


    def detect_extremevoltswing(self, threshold = 200, padding = 0.05, overlap = 0.05, winlen = 0.5):
        '''
        Phát hiện các đoạn dữ liệu có sự dao động điện áp cực đoan cho mỗi kênh. Quá trình này được thực hiện trên cửa sổ trượt có chồng lấp, 
        do đó chỉ những phần dữ liệu có điện áp cực đoan mới được đánh dấu để loại bỏ sau này, chứ không phải toàn bộ các lần thử nghiệm đều phải bị loại bỏ.
        
        INPUT:
        - threshold:    z-value, default = 120
        - padding:      Khoảng đệm xung quanh hiện vật tính bằng giây, default = 0.1
        - overlap:      lượng chồng chéo của các cửa sổ di chuyển, xác định độ phân giải,
                        tính bằng giây, default = 0.05
        - winlen:       chiều dài của các cửa sổ trượt để tìm điện áp cực đại và cực tiểu
                        và đánh giá biên độ dao động

        OUTPUT:
        - self.artifacts['SWINGsamps'], self.artifacts['SWINGtrl']
        '''
        if winlen == 'all':
            winlen = self.data.shape[-1]/self.Fs

        winstarts = np.arange(0,self.data.shape[1]-(winlen*self.Fs),overlap*self.Fs)
        winends = winstarts+winlen*self.Fs

        n_data_rows = 26 

        swing = np.zeros((n_data_rows,self.data.shape[-1]))
        inpSWINGsamps = np.zeros((n_data_rows,self.data.shape[-1]))
        for r in range(n_data_rows):
            if ~np.isnan(self.data[r,0]):
                for w in range(len(winstarts)):
                    swing[r,int(winstarts[w]):int(winends[w])] = np.nanmax(self.data[r,int(winstarts[w]):int(winends[w])])-np.nanmin(self.data[r,int(winstarts[w]):int(winends[w])])
                if len(np.where(np.abs(swing[r,:])>threshold)[0]) > 0:
                    inpSWINGsamps[r,np.where(swing[r,:]>threshold)[0]]=1

        del swing

        SWINGtrl, SWINGsamps = self._artifact_samps_trl(inpSWINGsamps, padding, self.Fs, self.data[-1].shape[0])

        if len(SWINGtrl)>0:
            self.artifacts['SWINGsamps'] = SWINGsamps
            print('swing-detection: '+str(SWINGtrl.shape[0])+ ' samples with extreme voltage swing detected')
            self.info['swing-detection'] = str(len(SWINGtrl))+' samples with extreme voltage swing detected @ threshold: Z = '+str(threshold)
        else:
            SWINGtrl = np.array([])
            print('swing-detection: 0 samples with swing-detection')
            self.info['swing-detection'] = '0 samples with extreme voltage swing @ threshold: Z = '+str(threshold)

        self.artifacts['SWINGtrl'] = SWINGtrl

    def residual_eyeblinks(self, threshold = 0.5, padding = 0.1):
        '''
        Detect eyeblinks còn sót:
        - Band-pass 0.5-6 Hz
        - Hilbert envelope
        - Smooth bằng Hann 1s
        - Z-score, threshold
        - Envelope absolute > 60 (ngưỡng biên độ)
        - Gom thành trl bằng _artifact_samps_trl

        Lưu:
        - self.artifacts['EBsamps'], self.artifacts['EBtrl']
        '''
        nyq = 0.5 * self.Fs
        high_pass = 0.5 / nyq
        low_pass = 6/ nyq

        ''' bandpassfilter '''
        sos = butter(4, [high_pass, low_pass], btype='bandpass', analog=False, output = 'sos')
        filtEB = sosfiltfilt(sos, self.data)
#        filtpadding = filtEB.shape[1]*0.10

        N=int(filtEB.shape[1]+filtEB.shape[1]*0.10)
        if N % 2 == 0:
            N=N
        else:
            N=N+1

        hilbEB  = hilbert(filtEB.copy(), N=N, axis = -1)
        amplenv = np.abs(hilbEB[:,:filtEB.shape[1]])

        n_data_rows=26

        EBsamps = np.zeros((n_data_rows,self.data.shape[1]))

        hanndata = np.zeros((n_data_rows,self.data.shape[1]))
        ''' hanning smooth '''
        for r in range(n_data_rows):
            hanndata[r,:] = convolve(amplenv[r,:], hann(int(1*self.Fs),sym=True), mode ='same')#, method ='direct')

        ''' zvalue threshold '''
        Zdata = zscore(hanndata, axis=1)

        inpEBsamps = np.zeros((self.data.shape[0],hanndata.shape[1]))

        for r in range(hanndata.shape[0]):
            if ~np.isnan(self.data[r,0]):
                sidx = np.where(Zdata[r,:] > np.nanmean(Zdata)+threshold)[0]
                didx = np.where(amplenv[r,sidx]>60)
                inpEBsamps[r,sidx[didx]]=1

        EBtrl, EBsamps = self._artifact_samps_trl(inpEBsamps, padding, self.Fs, self.data[-1].shape[0])

        print('EB detection: detected '+str(len(EBtrl))+' artifact(s)')

        self.info['EB detection'] = str(len(EBtrl))+' artifacts detected @ threshold: Z = '+str(threshold)
        self.artifacts['EBsamps'] = EBsamps
        self.artifacts['EBtrl'] = EBtrl

    def define_artifacts(self, time_threshold = 1/3, z_threshold = 1.96):
        '''
            Xác định các artifact ảnh đã được phát hiện, lưu ý đến khả năng chồng chéo giữa các artifact này.

            Kênh lỗi: Nếu số lượng artifact ảnh trong một kênh vượt quá một khoảng thời gian tương đối nhất định, 
            kênh đó sẽ được đánh dấu là lỗi và được sửa chữa bằng phương pháp nội suy thông qua trung bình có trọng số 
            (dựa trên khoảng cách Euclid) của các kênh lân cận được chọn.

            Các kênh nối cũng được xác định trong hàm này và được sửa chữa.

            INPUT:
            - time_threshold:   tỷ lệ thời gian cho phép bị loại bỏ do dữ liệu nhiễu, nếu vượt quá tỷ lệ này, 
                                kênh sẽ được đánh dấu là bị lỗi và được sửa chữa, mặc định = 1/3
            - ztheshold:        số lượng độ lệch chuẩn cho phép đối với tín hiệu băng thông rộng thông thường
            OUTPUT (ghi vào object):
            - self.artifacts: thêm/ghi các key:
                * 'bad channels', 'bridging channels', 'empty channels'
            - self.info: thêm/ghi các key:
                * 'bad channels', 'bridging channel check', 'bridging channels',
                'empty channels', 'repairing channels', 'repaired channels', 'data quality',
                'no. segments'
            - self.data / self.labels: chèn thêm 1 kênh 'artifacts' (0/1 theo sample) vào giữa
            sau kênh 'O2' để dùng về sau khi segment/remove artifact.
            - self.trl: set thành [0, total_samples] (dữ liệu chưa segment, 1 đoạn duy nhất)
        '''
        # --- 1) Lấy các artifact đã detect trước đó
        if 'EMGtrl' in self.artifacts and len(self.artifacts['EMGtrl'])>0:
            emgtrl = self.artifacts['EMGtrl']       # các đoạn EMG (start/end sample)
            emgsamps = self.artifacts['EMGsamps']   # mask EMG theo kênh (channels x samples)
        else:
            emgtrl = []
            emgsamps = []
        if 'JUMPtrl' in self.artifacts and len(self.artifacts['JUMPtrl'])>0:
            jmptrl = self.artifacts['JUMPtrl']      # các đoạn jump/baseline shift
            jmpsamps = self.artifacts['JUMPsamps']
        else:
            jmptrl = []
            jmpsamps = []

        if 'KURTtrl' in self.artifacts and len(self.artifacts['KURTtrl'])>0:
            kurttrl = self.artifacts['KURTtrl']     # các đoạn kurtosis cao
            kurtsamps = self.artifacts['KURTsamps']
        else:
            kurttrl = []
            kurtsamps = []

        if 'SWINGtrl' in self.artifacts and len(self.artifacts['SWINGtrl'])>0:
            swingtrl = self.artifacts['SWINGtrl']     # các đoạn swing voltage lớn
            swingsamps = self.artifacts['SWINGsamps']
        else:
            swingtrl = []
            swingsamps = []

        if 'EBtrl' in self.artifacts and len(self.artifacts['EBtrl'])>0:
            ebtrl = self.artifacts['EBtrl']        # các đoạn swing voltage lớn
            ebsamps = self.artifacts['EBsamps']
        else:
            ebtrl = []
            ebsamps = []

        # --- 2) Collapse: hợp nhất tất cả artifact masks thành 1 ma trận artsamps ---
        # artsamps: (n_channels x n_samples) chỉ dùng 26 kênh EEG, dtype=int (0/1)
        artsamps = np.zeros((self.data.shape[0],self.data.shape[1]),dtype=int)
        n_data_rows = 26 

        for r in range(n_data_rows):
            if len(emgtrl) > 0:
                artsamps[r,np.where(emgsamps[r,:]==1)[0]]=1
            if len(jmptrl) > 0:
                artsamps[r, np.where(jmpsamps[r,:]==1)[0]]=1
            if len(kurttrl) > 0:
                artsamps[r, np.where(kurtsamps[r,:]==1)[0]]=1
            if len(swingtrl) >0:
                artsamps[r, np.where(swingsamps[r,:]==1)[0]]=1
            if len(ebtrl) >0:
                artsamps[r, np.where(ebsamps[r,:]==1)[0]]=1


        # --- 3) Đánh dấu bad channel theo tỷ lệ thời gian bị artifact ---
        # Nếu số sample bị artifact vượt total_samples * time_threshold -> badchan[r] = 1
        badchan = np.zeros((n_data_rows),dtype = int)
        for r in range(n_data_rows):
            if len(np.where(artsamps[r,:]==1)[0]) > self.data.shape[1]*time_threshold:
                badchan[r]=1

        # --- 4) Đánh dấu bad channel theo broadband power (55-95 Hz) ---
        # Ý tưởng: kênh có năng lượng cao bất thường trong dải 55-95Hz thường là nhiễu/EMG.
        # Tính FFT (scipy.fftpack.fft), nhân cửa sổ Hann, lấy power, lấy dải tần 55-95,
        # z-score theo frequency rồi lấy mean -> zdat theo kênh.
        from scipy.fftpack import fft
        from scipy.signal.windows import hann

        hannwin = hann(int(self.data.shape[-1]))                        # cửa sổ Hann theo độ dài recording
        power = np.abs(fft(self.data[:n_data_rows,:])*hannwin)**2       # power spectrum
        freqs = np.linspace(0, self.Fs/2,int(len(power[0,:])/2))        # vector tần số (nửa phổ)
        fid = [(np.where((freqs > 55) & (freqs < 95)))][0][0]           # index dải 55-95Hz
        overallpower = power[:n_data_rows,fid]                          # power trong dải 55-95
        zdat = np.nanmean(zscore(overallpower,axis=0),axis=1)           # z-score & lấy mean theo kênh
        badchan[np.where(zdat> np.nanmean(zdat)+ z_threshold)[0]] = 1   # set thêm bad channels
        idxbadchan = np.where(badchan ==1)[0] # index các kênh bad

        # Lưu danh sách bad channels theo label
        self.artifacts['bad channels']=[]
        if len(idxbadchan)>0:
            self.info['bad channels'] = str(len(idxbadchan)) + ' detected @ threshold: '+str(time_threshold)
            for b in range(len(idxbadchan)):
                self.artifacts['bad channels'] = np.append(self.artifacts['bad channels'],self.labels[idxbadchan[b]])
        else:
            self.info['bad channels'] = '0 bad channels @ threshold: '+ str(time_threshold)

        # --- 5) Phát hiện bridging channels
        # _bridging_check trả về index kênh bridging; nếu có thì sẽ sửa bằng nội suy ở bước sau.
        bridgeidx = self._bridging_check(self.data)[0]
        self.artifacts['bridging channels'] = []
        if len(bridgeidx)>0:
            #print('reparing bridging channels')
            self.info['bridging channel check'] = 'reparing bridging channels'
            for b in range(len(bridgeidx)):
                self.artifacts['bridging channels'] = np.append(self.artifacts['bridging channels'],self.labels[bridgeidx[b]])
        else:
            self.info['bridging channels'] = str(0)+' briding channels'

        # --- 6) Detect empty channels (NaN) ---
        # Kiểm tra sample thứ 2 (index 1) của 26 kênh EEG xem có NaN không.
        idxemptychan = np.where(np.isnan(self.data[:n_data_rows,1]))[0]
        self.artifacts['empty channels'] = []
        if len(idxemptychan)>0:
            self.info['empty channels'] = str(len(idxemptychan)) + 'empty channels detected'
            for b in range(len(idxemptychan)):
                self.artifacts['empty channels'] = np.append(self.artifacts['empty channels'],self.labels[idxemptychan[b]])
        else:
            self.info['empty channels'] = str(0)+' empty channels'

        # --- 7) Gom tất cả kênh cần sửa: bad + bridging + empty ---
        combidx=np.array((np.unique(np.hstack((idxbadchan,bridgeidx,idxemptychan))))).astype(int)

        # --- 8) Nội suy (repair) các kênh lỗi nếu có ---
        # _interpolate_data dùng neighbours + khoảng cách Euclid để tạo weighted average.
        # Nếu repair thành công: cập nhật self.data; reset artsamps ở kênh vừa sửa về 0 (coi như đã “clean”).
        if len(combidx)>=1:
            repaireddata =np.zeros((self.data.shape))
            print('Remove artifacts: repairing/ interpolating bad, empty and bridging channel(s) \n')
            repaireddata, self.info['repairing channels'], repaired, intchan = self._interpolate_data(self.data, self.labels, self.neighblabels, combidx)
            if repaired == 'yes':
                self.info['repaired channels'] = []
                self.data = repaireddata
                for b in range(len(intchan)):
                    self.info['repaired channels'] = np.append(self.info['repaired channels'],self.labels[intchan[b]])
                    artsamps[intchan[b],:] = 0
            elif repaired == 'no':
                self.info['data quality'] = 'bad'

        # --- 9) Tạo artifact vector chung theo sample (gộp theo kênh) ---
        artsamples = np.nanmax(artsamps,axis=0)

        # --- 10) Đánh giá chất lượng dữ liệu ---
        # bad nếu:
        # - tổng thời gian artifact > (1 - time_threshold) của toàn recording
        # - hoặc số kênh cần sửa quá 3 (combidx>3)
        if  len(np.where(artsamples==1)[0]) > self.data.shape[-1]*(1-time_threshold) or len(combidx)>3:# if 2/3 of the data is artifacts or there are 6 bad channels...
            self.info['data quality'] = 'bad'
        else:
            self.info['data quality'] = 'OK'

        # --- 11) Chèn kênh 'artifacts' vào sau kênh 'O2' ---
        # Mục tiêu: giữ thứ tự kênh gần tự nhiên, đồng thời có 1 kênh mask để segment/remove artifacts về sau.  
        Och = np.squeeze(np.where(np.array(self.labels)=='O2')[0])
        self.trl = np.array([0,self.data.shape[-1]],dtype=int)
        self.data = np.vstack((self.data[:Och+1,:],artsamples, self.data[Och+1:,:]))
        self.labels = np.hstack((self.labels[:Och+1], 'artifacts', self.labels[Och+1:]))
        self.info['no. segments']=0 # chưa thực hiện segment

    def segment(self, marking = 'no', trllength = 2, remove_artifact = 'no'):
        '''
        Cắt (segment) dữ liệu liên tục thành các epoch/trial có độ dài cố định.

        Có 2 chế độ:
        - remove_artifact='no': vẫn segment bình thường, giữ nguyên các epoch có chứa artifact.
        - remove_artifact='yes': loại bỏ các đoạn artifact; chỉ lấy các đoạn “sạch” đủ dài để tạo epoch.

        INPUT
        - marking: 'yes'/'no' (nếu có self.marking thì cũng segment tương tự)
        - trllength: độ dài epoch (giây); nếu 'all' thì epochlength = toàn bộ recording
        - remove_artifact: 'yes'/'no'

        OUTPUT:(ghi vào object)
        - self.data: dữ liệu sau segment (n_trials, n_channels, n_samples_per_epoch)
        - self.trl: mảng (n_trials,2) start/end sample cho mỗi epoch trong recording gốc
        - self.arttrl: mảng (n_artifacts,2) start/end sample của các đoạn artifact (nếu có kênh artifacts)
        - self.artidata: lưu lại dữ liệu trong các đoạn artifact (chỉ khi remove_artifact='yes')
        - self.info: cập nhật trạng thái artifact removal, số epoch, chất lượng dữ liệu
        '''
        # Tổng số sample của recording hiện tại (dữ liệu liên tục trước khi segment)
        totallength = self.data.shape[-1]

        # Xác định epochlength theo giây:
        # - nếu trllength='all' => 1 epoch dài bằng toàn bộ recording
        # - ngược lại => epochlength = trllength
        if trllength == 'all':
            epochlength = totallength/self.Fs
        else:
            epochlength = trllength

        # Nếu trong labels có kênh 'artifacts' (đã được chèn từ define_artifacts)
        if 'artifacts' in self.labels:
            # artidx: index của kênh artifacts trong self.labels
            artidx = np.where(self.labels=='artifacts')[0]

            # artsamples: vector 0/1 theo sample, 1 nghĩa là artifact
            # self.data[artidx,:] có shape (1, n_samples) nên lấy [0] để thành (n_samples,)
            artsamples = self.data[artidx,:][0]
            
            # p: danh sách index các sample có artifact (=1)
            p = np.where(artsamples==1)[0]

            # Nếu có ít nhất 1 sample artifact
            if len(p)>0:
                startidxs = np.hstack([np.where(np.diff(artsamples)==1)[0]+1])# diff =1
                endidxs = np.hstack([np.where(np.diff(artsamples)==-1)[0]+1])#diff = -1

                if len(endidxs)==0:
                    endidxs = np.hstack([endidxs,self.data.shape[-1]])
                if len(startidxs)==0:
                    startidxs = np.hstack([startidxs,0])

                if startidxs[-1] > endidxs[-1]:
                    endidxs = np.hstack([endidxs,self.data.shape[-1]])

                # Bảo vệ trường hợp artifact bắt đầu từ 0 nhưng diff không bắt được
                # (đảm bảo startidxs và endidxs khớp theo thứ tự)   
                if type(endidxs)==int:
                    if endidxs < startidxs:
                        startidxs = np.hstack([0,startidxs])
                elif endidxs[0] < startidxs[0]:
                        startidxs = np.hstack([0,startidxs])

                # ARTtrl: mảng (n_artifacts,2) lưu các đoạn artifact [start,end]
                ARTtrl = np.array([0,0],dtype=int)
                for i in range(len(startidxs)):
                    ARTtrl = np.vstack((ARTtrl,[startidxs[i], endidxs[i]]))
                ARTtrl = ARTtrl[1:] # bỏ dòng [0,0] ban đầu

                # ---- CASE 1: remove_artifact='yes' và có nhiều hơn 1 sample artifact ----
                if remove_artifact == 'yes' and len(p) > 1:
                    # Ý tưởng:
                    # - Duyệt từng đoạn artifact, lấy các khoảng “sạch” trước mỗi artifact đủ dài để cắt thành epoch, rồi nối lại.
                    # - Sau cùng lấy phần sạch sau artifact cuối.
                    # - Đồng thời lưu lại dữ liệu artifact trong self.artidata.
                    t = 0
                    trials=np.zeros((1,self.data.shape[1],int(self.Fs*epochlength)));marktrials = trials.copy();
                    trl = np.array([0,0],dtype=int)
                    # Duyệt từng đoạn artifact
                    for i in range(ARTtrl.shape[0]):
                        # Nếu vùng sạch từ t tới đầu artifact dài hơn 1 epoch => segment được
                        if (ARTtrl[i,0]-t)>(int(epochlength*self.Fs)):
                            tmp = self.data[:,t:ARTtrl[i,0]]
                            segs,segstrl = self._EEGsegmenting(np.asarray(tmp),epochlength)
                            trials = np.concatenate([trials,segs],axis=0)
                            trl = np.vstack([trl,segstrl+t])
                            if marking=='yes':
                                tmpmarks = self.marking[:,t:ARTtrl[i,0]]
                                markedsegs = self._EEGsegmenting(np.asarray(tmpmarks),epochlength)
                                marktrials = np.concatenate([marktrials,markedsegs],axis=0)
                        t = ARTtrl[i,1]

                    # Xử lý vùng sạch sau artifact cuối cùng đến hết recording
                    if ARTtrl[-1,1] < self.data.shape[-1]-epochlength*self.Fs:
                        tmp = self.data[:,t:self.data.shape[-1]]
                        segs, segstrl = self._EEGsegmenting(np.asarray(tmp),epochlength)
                        trials = np.concatenate([trials,segs],axis=0)
                        trl = np.vstack([trl,segstrl+t])
                        if marking=='yes':
                            tmpmarks = self.marking[:,t:ARTtrl[i,0]]
                            markedsegs = self._EEGsegmenting(np.asarray(tmpmarks),epochlength)
                            marktrials = np.concatenate([marktrials,markedsegs],axis=0)

                    # Lưu dữ liệu trong các đoạn artifact vào self.artidata
                    # Shape: (n_artifacts, n_channels, max_artifact_length)
                    self.artidata=np.zeros((ARTtrl.shape[0],self.data.shape[1],np.nanmax(np.diff(ARTtrl))))
                    for i in range(ARTtrl.shape[0]):
                        self.artidata[i,:,:np.diff(ARTtrl[i,:])[0]] = self.data[:,ARTtrl[i,0]:ARTtrl[i,1]]

                    # Gán kết quả sau khi đã loại artifact:
                    self.trl = trl[1:]      # bỏ dòng [0,0]
                    self.data = trials[1:]  # bỏ epoch rỗng đầu
                    self.arttrl = ARTtrl    # lưu trl của artifact
                    self.info['artifact removal'] = 'detected artifacts removed'
                    self.info['no. segments'] = len(trl)-1
                    
                    # Nếu số epoch còn lại quá ít so với kỳ vọng => đánh dấu chất lượng xấu
                    if self.info['no. segments'] < ((1/3)* (totallength/(epochlength*self.Fs))):
                        self.info['data quality'] = 'bad'

                # ---- CASE 2: remove_artifact='no' ----
                elif remove_artifact == 'no':
                    # Segment toàn bộ recording, không loại bỏ artifact
                    #print('no artifact removal')
                    self.data,self.trl = self._EEGsegmenting(self.data, epochlength)
                    
                    # Nếu có marking thì segment tương tự
                    if marking == 'yes':
                        self.marking = self._EEGsegmenting(self.marking, epochlength)[0]
                    
                    # Lưu trl của artifact (các đoạn start/end)
                    self.arttrl=ARTtrl
                    self.info['artifact removal'] = 'none removed'
                    self.info['no. segments'] = len(self.trl)
                    
                    # Nếu chọn trllength='all' (1 epoch) thì đánh giá theo tỷ lệ artifact trong recording
                    if trllength == 'all':
                        if  len(p) > ((2/3) * totallength):
                            self.info['data quality'] = 'bad'
                        else:
                            self.info['data quality'] = 'OK'

            # ---- CASE: có kênh artifacts nhưng không có sample artifact nào ----
            else:
                # Segment bình thường
                self.data,self.trl = self._EEGsegmenting(self.data, epochlength)
                
                # Segment marking nếu có
                if marking == 'yes':
                    self.marking = self._EEGsegmenting(self.marking, epochlength)[0]
                self.info['artifact removal'] = 'no artifacts detected'
                self.info['no. segments'] = len(self.trl)-1
                self.arttrl = [0]
                
                # Nếu số epoch ít hơn kỳ vọng (ngưỡng 1.3*...) => đánh dấu chất lượng xấu
                if self.info['no. segments'] < ((1.3) * (totallength/(epochlength*self.Fs))):
                    self.info['data quality'] = 'bad'

        # ---- CASE: không có kênh 'artifacts' trong labels ----
        else:
            # Segment bình thường
            self.data,self.trl = self._EEGsegmenting(self.data, epochlength)
            
            # Segment marking nếu có
            if marking=='yes':
                self.marking = self._EEGsegmenting(self.marking, epochlength)[0]

            self.info['artifact removal'] = 'no artifacts detected'
            self.info['no. segments'] = len(self.trl)-1
            self.arttrl = [0]

            if trllength == 'all':
               if  len(p) > (0.33 * totallength):
                   self.info['data quality'] = 'bad'
               else:
                   self.info['data quality'] = 'OK'
            elif self.info['no. segments'] < (0.33 * (totallength/(epochlength*self.Fs))):
                self.info['data quality'] = 'bad'


    def save_pdfs(self, savepath, inp='data', scaling =[-70,70]):
        """
        Lưu báo cáo PDF dạng đồ thị EEG theo từng segment (thường 10s nếu trước đó đã segment 10s),
        dùng để kiểm tra trực quan sau preprocessing. Có thể bao gồm kênh 'artifacts' nếu tồn tại.

        INPUT
        - savepath: thư mục output (sẽ tạo thêm thư mục con /pdf/)
        - inp: tên thuộc tính dữ liệu cần vẽ, thường 'data' hoặc 'artidata'
            + 'data': dữ liệu segment bình thường
            + 'artidata': dữ liệu riêng của các đoạn artifact (nếu đã tạo)
        - scaling: [ymin, ymax] giới hạn biên độ hiển thị (microvolt) cho mỗi kênh

        OUTPUT
        - Tạo file PDF tại: savepath + '/pdf/' + outname + '.pdf'
        """
        import numpy as np
        import matplotlib.pyplot as plt

        from matplotlib.collections import LineCollection
        #from matplotlib.ticker import MultipleLocator
        from matplotlib.backends.backend_pdf import PdfPages

        # Tắt interactive mode để không hiện figure khi chạy batch
        plt.ioff()

        # Lấy idcode và condition từ tên file đầu vào (theo format Brainclinics)
        idcode = self.info['fileID'].rsplit('/')[-1].split('.')[0]
        cond = self.info['fileID'].rsplit('/')[-1].split('.')[1]

        # Tính độ dài (giây) của dữ liệu hiện tại (n_samples / Fs)
        trllength = str(self.data.shape[-1]/self.Fs)

        # Tạo tên file output theo chất lượng dữ liệu
        if self.info['data quality'] == 'OK':
            outname = idcode + '_' + cond + '_' + trllength + 's'
        elif self.info['data quality'] == 'bad':
            outname = 'BAD_'+ idcode + '_' + cond + '_' + trllength + 's'
            print('saving: data has been marked as BAD')
        
        # Nếu không remove artifact (raw report) thì đổi prefix
        if self.info['artifact removal'] == 'none removed':
            outname = 'RawReport_' + idcode + '_' + cond + '_' + trllength + 's'
        elif self.info['artifact removal'] == 'no artifact detected':
            outname = idcode + '_' + cond + '_' + trllength + 's'

        # Tạo thư mục pdf nếu chưa có
        if not os.path.exists(savepath+'/pdf/'):
            os.mkdir(savepath+'/pdf/')
        pdfpath = savepath+ '/pdf/'

        # Lấy dữ liệu cần vẽ từ thuộc tính theo tên inp ('data' hoặc 'artidata')
        odata = getattr(self, inp)

        if inp =='artidata':
            trl = self.arttrl
        else:
            trl = self.trl

        # Nếu có kênh 'artifacts' thì:
        # - scale kênh artifacts lên (x50) để nhìn rõ khi plot chung với EEG
        # - chỉ lấy 27 kênh đầu (26 EEG + artifacts) để plot
        if 'artifacts' in self.labels:
            odata[:,26,:]=odata[:,26,:]*50
            data = odata[:,:27,:]
            self.labels = self.labels[:27]

        # Nếu có kênh 'Events' thì:
        # - scale nhỏ xuống (x0.001)
        # - ghép thêm vào data để plot
        if 'Events' in self.labels:
            events = np.where(self.labels == 'Events')[0]
            evdat = odata[:,events,:]*0.001
            data = np.vstack((data,evdat))
            self.labels= np.vstack((self.labels,'events'))
        
        # Nếu có kênh 'ECG' thì:
        # - scale nhỏ xuống (x0.001)
        # - ghép thêm vào data để plot
        if 'ECG' in self.labels:
            ecg = np.where(self.labels == 'ECG')[0]
            ecgdat = odata[:,ecg,:]*0.001
            data = np.vstack((data,ecgdat))
            self.labels= np.vstack((self.labels,'ECG'))

        # Số trial/segment, số kênh, số sample trong mỗi segment
        n_trials, n_rows,n_samples = data.shape[0],data.shape[1], data.shape[2]

        import datetime
        # Mở PdfPages để ghi nhiều trang vào 1 file pdf
        with PdfPages(pdfpath+outname+'.pdf') as pp:
            # --- Trang bìa ---
            firstPage = plt.figure(figsize=(11.69,8.27))
            firstPage.clf()
            t =  datetime.datetime.now()
            txt = 'Raw Data Report \n \n' + idcode + ' ' + cond + '\n \n' + ' Report created on ' + str(t)[:16] + '\n by \n \n Research Institute Brainclinics \n Brainclinics Foundation \n Nijmegen, the Netherlands'
            firstPage.text(0.5,0.5,txt, transform=firstPage.transFigure, size=22, ha="center")
            pp.savefig()

            # --- Mỗi segment 1 trang ---
            for seg in range(n_trials):
                # Tạo figure (đoạn này tạo rồi close ngay, sau đó tạo lại; giữ nguyên theo code gốc)
                fig = plt.figure(num = seg, figsize = (20,12), tight_layout=True)
                plt.close()

                # Trục thời gian theo giây cho segment hiện tại
                t = np.arange(0,n_samples/self.Fs, (n_samples/self.Fs)/n_samples)

                fig = plt.figure(num = seg, figsize = (20,12), tight_layout=True)
                ax1 = fig.add_subplot(1,1,1)
                plt.subplots_adjust(bottom = 0.2)

                # Title: id + condition + số thứ tự segment
                ax1.set_title(idcode + ' ' + cond +'\n Segment: '+ str(seg+1) +' of '+str(n_trials))

                # Thiết lập trục y: “xếp chồng” các kênh lên nhau theo offset (dr)
                dmin = scaling[0]#data.min()
                dmax = scaling[1]#data.max()
                dr = (dmax - dmin) * 0.7  # khoảng cách giữa các kênh trên trục y
                y0 = dmin
                y1 = (n_rows-1) * dr + dmax
                ax1.set_ylim(y0, y1)

                # segments: list các đường (t, signal) cho từng kênh
                # ticklocs: vị trí y-offset cho từng kênh
                segments = []
                ticklocs = []
                for i in range(n_rows):
                    segments.append(np.column_stack((t, data[seg,i,:])))
                    ticklocs.append(i * dr)

                # X-ticks: chia đều 10 phần theo thời gian segment
                ticks = np.arange(0,(data.shape[-1]/self.Fs)+((data.shape[-1]/self.Fs)/10),(data.shape[-1]/self.Fs)/10)
                ax1.set_xticks(ticks,minor=False)

                # Nhãn trục x: thời gian tuyệt đối theo recording gốc
                # (bắt đầu tại trl[seg,0]/Fs, tăng đến hết segment)
                ticksl = np.arange(np.around(trl[seg,0]/self.Fs,decimals=2),np.around((trl[seg,0]/self.Fs)+(n_samples/self.Fs),decimals=2)+1,np.around((n_samples/self.Fs)/10,decimals=2))

                ticklabels = list(ticksl)#np.arange(ticks)
                xlabels = [ '%.1f' % elem for elem in ticklabels]
                xlabels = np.array(xlabels,dtype=str)
                ax1.set_xticklabels(xlabels)

                # offsets: vector offset y cho từng kênh để LineCollection “xếp chồng”
                offsets = np.zeros((n_rows, 2), dtype=float)
                offsets[:,1] = ticklocs

                # LineCollection: vẽ tất cả kênh như nhiều line, đảo ngược để label hiển thị đúng thứ tự
                lines = LineCollection(np.flipud(segments), linewidths=(0.6), offsets=offsets, transOffset=None, colors = 'k')
                ax1.add_collection(lines)

                ax1.set_yticks(ticklocs)

                ax1.set_yticklabels(self.labels[::-1])

                ax1.set_xlabel('Time (s)')

                pp.savefig()
                plt.close()

    def save(self, savepath, matfile='no', csv = 'no', npy = 'yes'):
        """
        Lưu dữ liệu EEG sau preprocessing/segment.

        INPUT
        - savepath: thư mục lưu
        - matfile: 'yes'/'no' lưu thêm .mat
        - csv: 'yes'/'no' lưu dạng csv theo từng epoch
        - npy: 'yes'/'no' lưu pickle của toàn bộ object (vars(self)) thành file .npy

        OUTPUT
        - Nếu csv='yes': tạo thư mục csv_data_<cond>_<len>s/ và lưu mỗi epoch 1 file csv
        - Nếu npy='yes': lưu <outname>.npy (pickle dump vars(self))
        - Nếu matfile='yes': lưu <outname>.mat (dict theo cấu trúc Matlab)
        """
        import pandas as pd
        import scipy.io as sio

        print('saving data \n')
        
        # Lấy idcode và condition từ tên file đầu vào
        idcode = self.info['fileID'].rsplit('/')[-1].split('.')[0]
        cond = self.info['fileID'].rsplit('/')[-1].split('.')[1]

        # Tạo tên file theo đánh giá chất lượng
        trllength = str(self.data.shape[-1]/self.Fs)
        if self.info['data quality'] == 'OK':
            outname = idcode + '_' + cond + '_' + trllength + 's'
        else:
            outname = 'BAD_'+ idcode + '_' + cond + '_' + trllength + 's'
            print('saving: data has been marked as BAD')

        # --- Lưu CSV ---
        if csv == 'yes':
            if os.path.isdir(savepath + '/csv_data_' + cond + '_' + trllength + 's/'):
                csvpath = savepath + '/csv_data_' + cond + '_' + trllength + 's/'
            else:
                os.mkdir(savepath + '/csv_data_' + cond + '_' + trllength + 's/')
                csvpath = savepath + '/csv_data_' + cond + '_' + trllength + 's/'

            # Nếu self.data là 3D (n_trials, n_channels, n_samples)
            # => lưu mỗi trial thành 1 file csv (time x channels)
            for i in range(self.data.shape[0]):
                if len(self.data.shape) == 3:
                    df = pd.DataFrame(self.data[i,:,:].T)
                    df.to_csv(csvpath + str((self.trl[i,0]/self.Fs)*1000) + '.csv',sep=',',header = list(self.labels),compression = None)
                else:
                    df = pd.DataFrame(self.data[:,:].T)
                    df.to_csv(csvpath + str(0)+'.csv',sep=',',header = list(self.labels),compression = None)

            #'''======== save info in txt format (per condition) ========'''
            #df = pd.DataFrame(self.info)
            #df.T.to_csv(csvpath + outname + '_info.txt',header=None, sep=' ', mode='a')

        # --- Lưu pickle .npy ---
        if npy == 'yes':
            import pickle
            npypath=os.path.join(savepath,outname +'.npy')
            # Lưu dict các thuộc tính của object (vars(self))
            with open(npypath, 'wb') as output:  
                pickle.dump(vars(self), output, -1)

        # --- Lưu Matlab .mat ---
        if matfile == 'yes':
            mat_dataset = {'labels': self.labels,
                           'trials': self.data,
                           'dimord' :'rpt_chan_time',
                           'artifacts': self.arttrl,
                           'Fs':500,
                           'time': np.arange(0,(self.data.shape[-1]/self.Fs),1/self.Fs),
                           'info': self.info}
            sio.savemat(savepath + '/' + outname +'.mat', mat_dataset)

    def plot_EEG(self, inp='data' , scaling=[-70,70], title=None):
        """
        Vẽ EEG dạng “stacked traces” và có nút bấm để chuyển qua lại giữa các segment (trial).

        INPUT
        - inp: 'data' hoặc 'artidata'
        - scaling: [ymin, ymax] cho mỗi kênh
        - title: nếu None dùng tên fileID, nếu có thì dùng title làm tên figure

        OUTPUT
        - Hiển thị figure; trả về 2 button objects (next, prev)
        """

        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button
        from matplotlib.collections import LineCollection

        # Class nội bộ để xử lý callback cho nút next/prev
        class GUIButtons(object):

            def __init__(self, tmpdata, axs, t, Fs, trls):
                self.tmpdata = tmpdata
                self.axs = axs
                self.t = t
                self.Fs = Fs
                self.trls = trls

            trl = 0
            end = 0

            def nextb(self, event):
                """Chuyển sang segment kế tiếp và cập nhật LineCollection"""
                self.trl += 1
                i = self.trl

                n_trials = self.tmpdata.shape[0]
                n_rows = self.tmpdata.shape[1]

                if i >= n_trials:
                    i = n_trials
                    self.axs['ax1'].set_title('Last sample reached. Cannot go forwards')
                else:
                    segments=[];
                    for r in range(n_rows):
                        segments.append(np.column_stack((self.t, self.tmpdata[i,r,:])))

                    # Cập nhật dữ liệu cho LineCollection hiện tại
                    linesn = self.axs['ax1'].collections[0]
                    linesn.set_segments(np.flipud(segments))

                    # Cập nhật tiêu đề và nhãn thời gian tuyệt đối
                    self.axs['ax1'].set_title('Segment: '+str(i+1) + ' of ' + str(n_trials))
                    #self.axs['ax1'].set_xticks(ticks,minor=False)
                    ticksl = np.arange(np.around(self.trls[i,0]/self.Fs,decimals=2),np.around((self.trls[i,0]/self.Fs)+(self.tmpdata.shape[-1]/self.Fs),decimals=2)+((data.shape[-1]/self.Fs)/10),np.around((self.tmpdata.shape[-1]/self.Fs)/10,decimals=2))

                    ticklabels = list(ticksl)#np.arange(ticks)
                    xlabels = [ '%.1f' % elem for elem in ticklabels]
                    xlabels = np.array(xlabels,dtype=str)

                    self.axs['ax1'].set_xticklabels(xlabels)
                    plt.show()


            def prevb(self, event):
                """Chuyển về segment trước đó và cập nhật LineCollection"""
                self.trl -= 1
                i = self.trl

                n_trials = self.tmpdata.shape[0]
                n_rows = self.tmpdata.shape[1]

                if i < 0:
                    i = 0
                    self.axs['ax1'].set_title('First sample reached. Cannot go backwards')
                else:
                    segments=[];
                    for r in range(n_rows):
                        segments.append(np.column_stack((self.t, self.tmpdata[i,r,:])))

                    linesn = self.axs['ax1'].collections[0]
                    linesn.set_segments(np.flipud(segments))
                    self.axs['ax1'].set_title('Segment: '+str(i+1) + ' of ' + str(n_trials))
                    ticksl = np.arange(np.around(self.trls[i,0]/self.Fs,decimals=2),np.around((self.trls[i,0]/self.Fs)+(self.tmpdata.shape[-1]/self.Fs),decimals=2)+((data.shape[-1]/self.Fs)/10),np.around((self.tmpdata.shape[-1]/self.Fs)/10,decimals=2))

                    ticklabels = list(ticksl)#np.arange(ticks)
                    xlabels = [ '%.2f' % elem for elem in ticklabels]
                    xlabels = np.array(xlabels,dtype=str)
                    self.axs['ax1'].set_xticklabels(xlabels)
                    plt.show()

         # Lấy dữ liệu cần plot từ object
        data = getattr(self, inp)

        # Chọn trl tương ứng
        if inp =='artidata':
            trl = self.arttrl
        else:
            trl = self.trl

        # Nếu data 3D: (n_trials, n_rows, n_samples)
        if len(data.shape) == 3:
            n_samples, n_rows, n_trials = data.shape[2], data.shape[1], data.shape[0]
            # Nếu có nhiều hơn 26 kênh => scale một số kênh phụ để dễ nhìn
            if n_rows >26:
                n_samples, n_rows, n_trials = data.shape[2], data.shape[1], data.shape[0]
                if 'Erbs' in self.labels:
                    Erbs = np.where(self.labels== 'Erbs')[0]
                    data[:,Erbs,:]=data[:,Erbs,:]*0.15  #downscale ECG
                if 'artifacts' in self.labels:
                    artifacts = np.where(self.labels == 'artifacts')[0]
                    data[:,artifacts,:]=data[:,artifacts,:]*50 #upscale artifacts
                if 'Mass' in self.labels:
                    mass = np.where(self.labels == 'Mass')[0]
                    data[:,mass,:]= data[:,mass,:]*0.01
                if 'OrbOcc' in self.labels:
                    orbocc = np.where(self.labels == 'OrbOcc')[0]
                    data[:,orbocc,:]= data[:,orbocc,:]*0.01

        # Nếu data 2D: (n_rows, n_samples) => coi như 1 trial
        elif len(data.shape) == 2:
            n_samples, n_rows = data.shape[1], data.shape[0]
            if n_rows >26:
                n_samples, n_rows = data.shape[1], data.shape[0]
                if 'ECG' in self.labels:
                    ECG = np.where(self.labels== 'ECG')[0]
                    data[ECG,:]=data[ECG,:]*0.15 #downscale ECG
                if 'artifacts' in self.labels:
                    artifacts = np.where(self.labels == 'artifacts')[0]
                    data[artifacts,:]=data[artifacts,:]*50 #upscale artifacts
                if 'Events' in self.labels:
                    events = np.where(self.labels == 'Events')[0]
                    data[events,:]= data[events,:]*0.01

            n_trials = 1
            trl = np.array([0,0],dtype=int)

        # Trục thời gian theo giây
        t = np.arange(0,n_samples/self.Fs, (n_samples/self.Fs)/n_samples)

        # Tạo figure
        if title == None:
            fig = plt.figure(self.info['fileID'].rsplit('/')[-1], figsize = (6,9))
        else:
            fig = plt.figure(title, figsize = (6,9))

        ax1 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(bottom = 0.2)
        ax1.set_title('Segment: '+ str(1) +' of '+str(n_trials))

        # Set y-limits và khoảng cách giữa các kênh
        dmin = scaling[0]#data.min()
        dmax = scaling[1]#data.max()
        dr = (dmax - dmin) * 0.7  
        y0 = dmin
        y1 = (n_rows-1) * dr + dmax
        ax1.set_ylim(y0, y1)

        # Chuẩn bị segments (t, signal) và vị trí y-offset
        segments = []
        ticklocs = []
        for i in range(n_rows):
            if len(data.shape) == 3:
                segments.append(np.column_stack((t, data[0,i,:])))
            elif len(data.shape) == 2:
                segments.append(np.column_stack((t, data[i,:])))

            ticklocs.append(i * dr)

        # X-ticks chia 10 phần
        ticks = np.arange(0,(data.shape[-1]/self.Fs)+((data.shape[-1]/self.Fs)/10),(data.shape[-1]/self.Fs)/10)
        ax1.set_xticks(ticks,minor=False)

        # X-label hiển thị thời gian tuyệt đối theo trl
        ticksl = np.arange(np.around(trl.flat[0]/self.Fs,decimals=2),np.around((trl.flat[0]/self.Fs)+(n_samples/self.Fs),decimals=2)+1,np.around((n_samples/self.Fs)/10,decimals=2))

        ticklabels = list(ticksl)#np.arange(ticks)
        xlabels = [ '%.1f' % elem for elem in ticklabels]
        xlabels = np.array(xlabels,dtype=str)
        ax1.set_xticklabels(xlabels)

        # offsets để xếp chồng kênh
        offsets = np.zeros((n_rows, 2), dtype=float)
        offsets[:,1] = ticklocs

        # Vẽ LineCollection
        lines = LineCollection(np.flipud(segments), linewidths=(0.8), offsets=offsets, transOffset=None, colors = 'k')
        ax1.add_collection(lines)

        ax1.set_yticks(ticklocs)

        ax1.set_yticklabels(self.labels[::-1])

        ax1.set_xlabel('Time (s)')


        # Tạo nút next/prev
        axs = {}
        axs['ax1'] = ax1
        axs['axnext'] = plt.axes([0.84, 0.10, 0.10, 0.04]) #next button
        axs['axprev'] = plt.axes([0.72, 0.10, 0.10, 0.04]) #previous button

        callback = GUIButtons(data,axs,t,self.Fs,trl)

        bnext = Button(axs['axnext'], '>')
        bnext.on_clicked(callback.nextb)
        axs['axnext']._button = bnext

        bprev = Button(axs['axprev'], '<')
        bprev.on_clicked(callback.prevb)
        axs['axprev']._button = bprev

        plt.show()
        plt.axis('tight')

        return bnext, bprev

    def rereference(self, newrefchan = None):
        """
        Rereference dữ liệu EEG (chỉ 26 kênh EEG đầu), trừ đi tín hiệu reference.

        INPUT
        - newrefchan:
            + 'avgref': reference = trung bình các kênh EEG (0..25) theo từng trial
            + tên kênh cụ thể: reference = kênh đó (trung bình theo chiều trial nếu cần)

        OUTPUT
        - self.data được cập nhật sau khi trừ reference
        - self.info['rereferenced'] lưu loại reference đã dùng
        """
        # ref: placeholder (shape giống 1 kênh theo thời gian trên từng trial)
        ref = np.empty(self.data[:,1,:].shape);ref[:]=np.nan

        # avg reference: trung bình các kênh EEG cho từng trial, theo trục channel
        if newrefchan == 'avgref':
            ref = np.nanmean(self.data[:,:26,:],axis =1)
        else:
            idx = np.where(self.labels==newrefchan)
            ref = np.nanmean(self.data[:,idx,:])

        # Trừ reference khỏi từng trial và từng kênh EEG (0..25)
        for tr in range(self.data.shape[0]):
            for r in range(26): #only the EEG channels!
                self.data[tr,r,:] = self.data[tr,r,:] - ref[tr,:]

        # Lưu metadata
        self.info['rereferenced'] =  newrefchan

    '''========================================================================='''
    '''===========================   SUBFUNCTIONS   ============================'''
    '''========================================================================='''

    def _detect_artifact(self,inp,threshold):
        """
        Phát hiện các đoạn artifact trên 1 chuỗi 1D (inp) bằng ngưỡng z-score.

        INPUT
        - inp: vector 1 chiều (ví dụ: envelope của EOG sau lọc/smooth)
        - threshold: ngưỡng z-score; đánh dấu artifact nếu |z| > threshold

        OUTPUT
        - Atrl: mảng (n_artifacts x 2) chứa [begin, end] sample index cho từng đoạn artifact liên tiếp
        - Asamps: vector index các sample thỏa điều kiện |z| > threshold
        """
        from scipy.stats import zscore
        
        # Tính z-score để chuẩn hóa và threshold theo độ lệch chuẩn
        zdata = zscore(inp)
        #print(inp)

        # Lấy tất cả sample vượt ngưỡng theo hai phía (+/-)
        Asamps = [np.where((zdata > threshold) | (zdata < -1*threshold))][0][0]
        
        # Khởi tạo danh sách đoạn artifact (begin-end)
        Atrl = np.array([0,0],dtype=int)
        
        # Gom các sample liên tiếp thành các đoạn [begin, end]
        begin = Asamps[0] 
        for e in range(len(Asamps)):
            if e >= len(Asamps)-1:
                # sample cuối: đóng đoạn
                end = Asamps[-1]
                Atrl = np.vstack((Atrl,[begin,end]))
            elif Asamps[e+1] == Asamps[e]+1:
                # còn liên tiếp => tiếp tục
                continue
            else:
                # bị đứt đoạn => đóng đoạn hiện tại và mở đoạn mới
                end = Asamps[e]
                Atrl = np.vstack((Atrl,[begin,end]))
                begin = Asamps[e+1]
        # bỏ hàng khởi tạo [0,0]
        Atrl = Atrl[1:] 

        return Atrl, Asamps

    def _EEGsegmenting(self,inp, trllength, fs=500, overlap=0):
        """
        Cắt dữ liệu EEG thành các epoch cố định theo thời lượng (giây).

        INPUT
        - inp: mảng (n_channels x n_totalsamples)
        - trllength: độ dài epoch (giây)
        - fs: sampling rate (Hz)
        - overlap: tỉ lệ overlap giữa các epoch (0 => không overlap)

        OUTPUT
        - data: mảng (n_trials x n_channels x n_samples_per_epoch)
        - trl: mảng (n_trials x 2) chứa [start, end] sample index (theo inp) cho từng epoch
        """
        # Số sample mỗi epoch
        epochlength = int(trllength*fs)

        # Bước nhảy giữa các epoch (giảm khi overlap > 0)
        stepsize = (1-overlap)*epochlength

        # Kích thước dữ liệu đầu vào
        n_totalsamples, n_samples, n_rows = inp.shape[1],epochlength, inp.shape[0]
        
        # Số epoch có thể tạo (làm tròn xuống theo stepsize)
        n_trials = int(n_totalsamples/stepsize)

        # Tạo danh sách [start, end] cho từng epoch
        trl = np.array([0,0],dtype=int)
        t = 0
        for i in range(n_trials):
            trl = np.vstack((trl,[t,t+n_samples]))
            t += stepsize

        trl = trl[1:]

        # Cắt dữ liệu theo trl
        data = np.zeros((n_trials, n_rows, int(n_samples)))
        for i in range(n_trials):
            if trl[i,0] <= n_totalsamples-n_samples:
                data[i,:,:]= inp[:,trl[i,0]:trl[i,1]]

        return data, trl

    def _interpolate_data(self,inp, labels, neighbours, intchan):
        """
        Nội suy (repair) các kênh bị lỗi dựa trên trung bình có trọng số theo khoảng cách Euclid
        tới các kênh lân cận.

        INPUT
        - inp: mảng (n_channels x n_samples) (dữ liệu continuous, chưa segment)
        - labels: list/array tên kênh tương ứng với inp
        - neighbours: dict {label: [label_lan_can_1, ...]} danh sách kênh lân cận theo layout
        - intchan: list/array chỉ số (index) các kênh cần repair

        OUTPUT
        - inp: dữ liệu đã được cập nhật (các kênh trong intchan bị thay bằng nội suy)
        - feedback: string mô tả kết quả repair
        - repair: 'yes' nếu repair được tất cả, 'no' nếu có kênh không repair được
        - intchan: trả lại danh sách index kênh đã yêu cầu
        """
        from scipy.spatial import distance
        
        # Tọa độ 3D cố định cho 26 kênh EEG (theo layout định nghĩa sẵn)
        channellocations = np.array([[84.06,-26.81,-10.56],[83.74,29.41,-10.04],[41.69,-66.99,-15.96],[51.87,-48.05,39.87],[57.01,0.9,66.36],[51.84,50.38,41.33],[41.16,68.71,-15.31],[21.02,-58.83,54.82],[24.63,0.57,87.63],[21.16,60.29,55.58],[-16.52,-83.36,-12.65],[-13.25,-65.57,64.98],[-11.28,0.23,99.81],[-12.8,66.5,65.11],[-16.65,84.44,-11.79],[-48.48,-65.51,68.57],[-48.77,-0.42,98.37],[-48.35,65.03,68.57],[-75.17,-71.46,-3.7],[-80.11,-55.07,59.44],[-82.23,-0.87,82.43],[-80.13,53.51,59.4],[-75.17,71.1,-3.69],[-114.52,-28.98,9.67],[-117.79,-1.41,15.84],[-114.68,26.89,9.45]])
        labelarray = np.array(labels)
        repaired = [];repair = []

        # Lặp qua từng kênh cần repair
        for b in range(len(intchan)):
            # Lấy nhãn các kênh lân cận của kênh đang xét
            interpneighbs = []
            neighblabels = np.array(neighbours[labelarray[intchan[b]]])

            # Map nhãn lân cận -> index trong labelarray
            neighbidx = np.zeros((len(neighblabels)),dtype='int')
            for nb in range(len(neighblabels)):
                neighbidx[nb] = np.squeeze(np.squeeze(np.where(labelarray==neighblabels[nb])[0]))

            # Chỉ dùng các kênh lân cận không nằm trong danh sách kênh xấu (intchan)
            interpneighbs = neighbidx[np.where(np.in1d(neighbidx,intchan, invert=True))[0]]
            # Tọa độ kênh cần nội suy
            intchancoords = channellocations[intchan[b]]

            # Điều kiện: cần ít nhất 2 kênh lân cận tốt để nội suy
            if len(interpneighbs) >= 2:
                neighbcoords = channellocations[interpneighbs]

                # Tính khoảng cách Euclid từ kênh xấu tới từng kênh lân cận
                weights = np.zeros((len(interpneighbs)))
                wghtneighbs = np.zeros((len(interpneighbs),inp.shape[1]))

                for nb in range(len(interpneighbs)):
                    weights[nb] = distance.euclidean(intchancoords,neighbcoords[nb])
                sumweights = np.sum(weights)

                # Trọng số: (sum - dist) rồi chuẩn hóa (kênh gần hơn => trọng số lớn hơn)
                W = (sumweights-weights)#/np.nanmax(sumweights-weights)
                wghts = W/np.sum(W)

                # Tạo tín hiệu nội suy = tổng (tín hiệu kênh lân cận * trọng số)
                for nb in range(len(interpneighbs)):
                    wghtneighbs[nb,:] = (inp[interpneighbs[nb],:]*wghts[nb])#/sumweights

                inp[intchan[b],:] = np.sum(wghtneighbs,axis=0)

                feedback = 'repaired '+str(len(intchan))+' bad, empty and/or bridging channels'
                repaired = np.append(repaired,'yes')
            else:
                # Không đủ lân cận tốt để nội suy
                print('to many bad neighbours, not possible to repair channel: '+ labelarray[intchan[b]] +' ('+str(intchan[b])+')')
                feedback = 'not repaired the '+str(len(intchan))+ ' channels, there were to many bad neighbouring channels'
                repaired = np.append(repaired,'no')

            # Tổng hợp trạng thái repair toàn cục
            if 'no' in repaired:
                repair = 'no'
            else:
                repair ='yes'

        return inp, feedback, repair, intchan

    def _bridging_check(self,inp):
        """
        Phát hiện hiện tượng bridging (2 kênh bị “nối gel”, tín hiệu gần như giống nhau)
        bằng “electrical distance” (ED) dựa trên Tenke & Kayser (2001).

        INPUT
        - inp: mảng (n_channels x n_samples) (thường chỉ 26 kênh EEG)

        OUTPUT
        - bridgechanidx: mảng các index kênh nằm trong ít nhất 1 cặp bridging
        - bridgepairs: list các cặp (i, j) được phát hiện bridging
        """
        n_data_rows = 26
        ED = np.zeros((n_data_rows,n_data_rows))

        # ED(r1,r2) = mean( ((x1-x2) - mean(x1-x2))^2 )
        # Nếu ED == 0: hai kênh hoàn toàn trùng nhau theo metric này
        for r1 in range(n_data_rows):
            for r2 in range(n_data_rows):
                ED[r1,r2] = np.squeeze(np.nanmean(np.square((inp[r1,:]-inp[r2,:])-(np.nanmean(inp[r1,:]-inp[r2,:])))))

        # Tìm các vị trí ED == 0
        tmpidx = np.where(ED == 0)

        # Loại bỏ trường hợp r1==r2 (đường chéo)
        bridgeidx = np.where(np.not_equal(tmpidx[0],tmpidx[1]) == True)[0]
        tmpidx = np.asarray(tmpidx).T

        # Gom các cặp (x,y) sao cho không lặp (x,y) và (y,x)
        bridgepairs = []
        for x,y in tmpidx[bridgeidx,:]:
            if (x, y) not in bridgepairs and (y, x ) not in bridgepairs:
                bridgepairs.append((x, y))

        # Các kênh xuất hiện trong bridgepairs
        bridgechanidx = np.unique(bridgepairs)

        return bridgechanidx, bridgepairs

    def _artifact_samps_trl(self,ARTsamps, artpadding, Fs, totalsamps):
        """
        Chuyển ma trận đánh dấu artifact theo kênh (0/1) thành:
        - vector artifact tổng hợp (theo thời gian, gộp qua kênh)
        - danh sách đoạn artifact [start, end]
        Đồng thời áp dụng padding (mở rộng) quanh artifact.

        INPUT
        - ARTsamps: mảng (n_channels x totalsamps) với 1 tại sample bị đánh dấu artifact
        - artpadding: padding theo giây (mở rộng trước/sau đoạn artifact)
        - Fs: sampling rate
        - totalsamps: tổng số sample của recording

        OUTPUT
        - ARTtrl: mảng (n_artifacts x 2) các đoạn artifact sau khi gộp qua kênh (không padding trong bước gộp cuối)
        - paddedARTsamps: mảng (n_channels x totalsamps) sau khi áp dụng padding theo từng kênh
        """
        def find_artifacts(inpdata, totalsamps, artpadding = 0):
            """
            Subfunction: từ vector 0/1 theo thời gian -> padding và trả về:
            - tmpARTsamps: vector 0/1 sau padding
            - tmpARTtrl: danh sách đoạn [start, end] sau padding
            """
            tmpARTsamps=np.zeros((inpdata.shape))
            p = np.where(inpdata==1)[0]

            # Xác định điểm bắt đầu (diff==1) và kết thúc (diff==-1)
            if p[0] == 0:
                upidxs = np.append(0,np.where(np.diff(inpdata)==1)[0])# diff =1
            else:
                upidxs = np.where(np.diff(inpdata)==1)[0]
            if p[-1] == totalsamps:
                downidxs = np.append(np.where(np.diff(inpdata)==-1)[0],totalsamps)# diff =1
            else:
                downidxs = np.where(np.diff(inpdata)==-1)[0]

            # Nếu số start > số end thì bổ sung end cuối
            if len(upidxs)>len(downidxs):
                downidxs = np.append(np.where(np.diff(inpdata)==-1)[0],totalsamps)

            # Áp dụng padding theo sample
            startidxs = upidxs-int(artpadding*Fs)
            endidxs = downidxs+int(artpadding*Fs)

            # Tạo danh sách đoạn và vector padded
            tmpARTtrl = np.array([0,0],dtype=int)
            for k in range(len(startidxs)):
                if startidxs[k] <= 0:
                    startidxs[k]=0
                if endidxs[k] >= totalsamps:
                    endidxs[k] = totalsamps

                tmpARTsamps[startidxs[k]:endidxs[k]]=1
                tmpARTtrl = np.vstack((tmpARTtrl,[startidxs[k],endidxs[k]]))
            tmpARTtrl = tmpARTtrl[1:]

            return tmpARTsamps, tmpARTtrl

        # --- Áp dụng padding theo từng kênh ---
        n_data_rows = 26
        paddedARTsamps=np.zeros((ARTsamps.shape))

        for r in range(n_data_rows):
            cart = ARTsamps[r,:]
            p = np.where(cart==1)[0]
            if len(p) > 1:
                paddedARTsamps[r,:] = find_artifacts(cart, totalsamps, artpadding = artpadding)[0]
            else:
                paddedARTsamps[r,:]= np.zeros((cart.shape))

        # --- Gộp artifact qua kênh: lấy max theo trục channel ---
        maxARTsamps = np.nanmax(paddedARTsamps,axis=0)

        # Tạo danh sách đoạn artifact tổng hợp (không padding thêm ở bước này)
        p = np.where(maxARTsamps==1)[0]
        if len(p) > 1:
            ARTtrl = find_artifacts(maxARTsamps, totalsamps, artpadding=0)[1]
        else:
            ARTtrl = []

        return ARTtrl, paddedARTsamps

