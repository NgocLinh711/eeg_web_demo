#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interprocessing.py:
- Wrapper class interdataset để thao tác trên object đã preprocess (pickle dict)
- Các tiện ích: copy, downsample, rereference, bandpass, segment, save/load, plot
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io as sio

class interdataset(object):
    def __init__(self, d):
        self.__dict__ = d


    def copy(self):
        """
        Tạo bản sao độc lập (deep copy) của object, để sửa mà không ảnh hưởng bản gốc.
        Output: new_obj (interdataset)
        """
        import copy

        new_obj = self.__class__(self.__dict__)

        if hasattr(self, '__dict__'):
            for k in self.__dict__ :
                try:
                    attr_copy = copy.deepcopy(getattr(self, k))
                except Exception as e:
                    attr_copy = object_copy(getattr(self, k))
                setattr(new_obj, k, attr_copy)

            new_attrs = list(new_obj.__dict__.keys())
            for k in new_attrs:
                if not hasattr(self, k):
                    delattr(new_obj, k)
            return new_obj
        else:
            print('was not able to copy')
            return self

    def downsample(self, downsample = None):
        """
        Downsample theo bước mẫu.
        Input: downsample (int) -> lấy mỗi downsample mẫu.
        Yêu cầu data dạng (n_trials, n_channels, n_samples).
        """
        self.data = self.data[:,:,np.arange(0,self.data.shape[-1],downsample)]

    def rereference(self, newrefchan = None):
        """
        Rereference dữ liệu EEG theo nhiều kiểu:
        - 'avgref': trừ trung bình 26 kênh EEG
        - 'hjort': local average reference theo danh sách neighblabels
        - 'longitudinalBipolar': chuyển sang montage bipolar dọc (18 kênh) + giữ ECG/artifacts/Events nếu có
        Output: cập nhật self.data, self.labels, self.info['rereferenced'] (một số mode)
        """
        ref = np.empty(self.data[:,1,:].shape);ref[:]=np.nan
        if newrefchan == 'avgref':
            ref = np.nanmean(self.data[:,:26,:],axis =1)
            for tr in range(self.data.shape[0]):
                for r in range(26): #only the EEG channels!
                    self.data[tr,r,:] = self.data[tr,r,:] - ref[tr,:]

        elif newrefchan == None:
            print('NO REREFERENCING!: for rereferencing a newrefchan is needed')

        elif newrefchan == 'hjort':
            labelarray = np.array(self.labels[:26])
            neighbours = self.neighblabels
            for r in range(len(labelarray)):
                neighblabels = neighbours[labelarray[r]]
                neighbidx = np.zeros((len(neighblabels)), dtype='int')
                for n in range(len(neighblabels)):
                    neighbidx[n] = np.squeeze(np.where(labelarray==neighblabels[n])[0])
                ref = np.nanmean(self.data[:,neighbidx,:], axis = 1)
                for tr in range(self.data.shape[0]):
                    self.data[tr,r,:]= self.data[tr,r,:]- ref[tr,:]

        elif newrefchan == 'longitudinalBipolar':
            # map cặp kênh A-B => A-B
            labelarray = np.array(self.labels[:26])
            channels ={1:['Fp1','F3'], 2:['F3','C3'], 3:['C3','P3'],
                       4:['P3','O1'], 5:['Fp2','F4'], 6:['F4','C4'],
                       7:['C4','P4'], 8:['P4','O2'], 9:['Fp1','F7'],
                       10:['F7','T3'], 11:['T3','P7'], 12:['P7','O1'],
                       13:['Fp1','F8'], 14:['F8','T4'], 15:['T4','P8'],
                       16:['P8','O2'], 17:['Fz','Cz'], 18:['Cz','Pz']}
            newdat = np.zeros((self.data.shape[0],len(channels),self.data.shape[-1]))
            labels = []
            for r in range(len(channels)):
                chan = np.where(self.labels == channels[r+1][0])[0]
                refchan = np.where(self.labels == channels[r+1][1])[0]
                newdat[:,r,:] = self.data[:,chan,:] - self.data[:,refchan,:]
                labels.append(channels[r+1][0]+'-'+channels[r+1][1])

            # giữ thêm kênh phụ nếu có
            if 'ECG' in self.labels:
                newdat= np.concatenate((newdat , self.data[:,np.where(self.labels == 'ECG')[0],:]),axis =1)
                labels.append('ECG')
            if 'artifacts' in self.labels:
                newdat = np.concatenate((newdat, self.data[:,np.where(self.labels == 'artifacts')[0],:]),axis =1)
                labels.append('artifacts')
            if 'Events' in self.labels:
                newdat = np.concatenate((newdat, self.data[:,np.where(self.labels == 'Events')[0],:]),axis =1)
                labels.append('Events')
            self.data = newdat
            self.labels = labels
            self.info['rereferenced'] =  newrefchan

    def apply_bpfilter(self,freqrange):
        """
        Bandpass Butterworth bậc 4 (sosfiltfilt, zero-phase).
        Input: freqrange = [low, high] (Hz)
        Output: self.data đã lọc.
        """
        from scipy.signal import butter, sosfiltfilt

        nyq = 0.5 * self.Fs
        high_pass = freqrange[0] / nyq
        low_pass = freqrange[1] / nyq

        ''' bandpassfilter '''
        sos = butter(4, [high_pass, low_pass], btype='bandpass', analog=False, output = 'sos')
        self.data = sosfiltfilt(sos, self.data)


    def segment(self, marking = 'no', trllength = 2, remove_artifact = 'no'):
        '''
        Chia dữ liệu EEG thành các epoch (đoạn) theo độ dài trllength (giây).
        Có thể giữ nguyên artifact hoặc loại bỏ các đoạn chứa artifact.

        INPUT:
        - marking: 'yes'/'no' (nếu có self.marking thì cũng segment tương ứng khi 'yes')
        - trllength: số giây mỗi epoch hoặc 'all' (lấy toàn bộ bản ghi làm 1 epoch)
        - remove_artifact: 'yes'/'no' (có loại bỏ vùng artifact hay không)

        OUPUT (cập nhật thuộc tính object):
        trl:    n x 2 array; mảng (n_trials, 2) chỉ số begin/end sample của mỗi epoch
        data:   dữ liệu đã segment (n_trials, n_channels, n_samples_epoch)
        info:   cập nhật mô tả quá trình (artifact removal, no. segments, data quality)
        arttrl: n X 2 array; danh sách đoạn artifact (begin/end) nếu có kênh artifacts
        artidata: dữ liệu nằm trong artifact (chỉ khi remove_artifact='yes')
        '''
        # Tổng số mẫu (sample) của bản ghi hiện tại
        totallength = self.data.shape[-1]

        # Xác định độ dài epoch (giây): 'all' -> toàn bộ, ngược lại dùng trllength
        if trllength == 'all':
            epochlength = totallength/self.Fs
        else:
            epochlength = trllength

        # Nếu có kênh đánh dấu artifact trong labels
        if 'artifacts' in self.labels:
            artidx = np.where(self.labels=='artifacts')[0]

            artsamples = self.data[0,artidx,:][0]
            #print('segmenting into trials of: '+str(epochlength)+' seconds')

            p = np.where(artsamples==1)[0]

            if len(p)>0:
                startidxs = np.hstack([np.where(np.diff(artsamples)==1)[0]+1])# diff =1
                endidxs = np.hstack([np.where(np.diff(artsamples)==-1)[0]+1])#diff = -1

                # Xử lý các trường hợp artifact bắt đầu/đến cuối file
                if len(endidxs)==0:
                    endidxs = np.hstack([endidxs,self.data.shape[-1]])
                if len(startidxs)==0:
                    startidxs = np.hstack([startidxs,0])

                # Đảm bảo có đủ end cho start cuối
                if startidxs[-1] > endidxs[-1]:
                    endidxs = np.hstack([endidxs,self.data.shape[-1]])

                # Đảm bảo đoạn đầu bắt từ 0 nếu cần (artifact bắt đầu từ đầu)
                if type(endidxs)==int:
                    if endidxs < startidxs:
                        startidxs = np.hstack([0,startidxs])
                elif endidxs[0] < startidxs[0]:
                        startidxs = np.hstack([0,startidxs])

                # Tạo ARTtrl: mỗi hàng là [start, end] của 1 đoạn artifact
                ARTtrl = np.array([0,0],dtype=int)
                for i in range(len(startidxs)):
                    ARTtrl = np.vstack((ARTtrl,[startidxs[i], endidxs[i]]))
                ARTtrl = ARTtrl[1:]

                # Nếu chọn loại bỏ artifact và thật sự có nhiều hơn 1 sample artifact
                if remove_artifact == 'yes' and len(p) > 1:
                    # Ghép các đoạn “sạch” nằm giữa các artifact, rồi segment các đoạn sạch đó
                    t = 0
                    trials=np.zeros((1,self.data.shape[1],int(self.Fs*epochlength)));marktrials = trials.copy();
                    trl = np.array([0,0],dtype=int)
                    for i in range(ARTtrl.shape[0]):
                        # Nếu đoạn sạch đủ dài để cắt epoch
                        if (ARTtrl[i,0]-t)>(int(epochlength*self.Fs)):
                            tmp = self.data[:,:,t:ARTtrl[i,0]]
                            segs,segstrl = EEGsegmenting(np.asarray(tmp),epochlength)
                            trials = np.concatenate([trials,segs],axis=0)
                            trl = np.vstack([trl,segstrl+t])
                            # Segment cả marking nếu có yêu cầu
                            if marking=='yes':
                                tmpmarks = self.marking[:,:,t:ARTtrl[i,0]]
                                markedsegs = EEGsegmenting(np.asarray(tmpmarks),epochlength)
                                marktrials = np.concatenate([marktrials,markedsegs],axis=0)
                        t = ARTtrl[i,1]

                    # Xử lý phần sạch sau artifact cuối cùng đến hết bản ghi
                    if ARTtrl[-1,1] < self.data.shape[-1]-epochlength*self.Fs:
                        tmp = self.data[:,:,t:self.data.shape[-1]]
                        segs, segstrl = EEGsegmenting(np.asarray(tmp),epochlength)
                        trials = np.concatenate([trials,segs],axis=0)
                        trl = np.vstack([trl,segstrl+t])
                        if marking=='yes':
                            tmpmarks = self.marking[:,:,t:ARTtrl[i,0]]
                            markedsegs = EEGsegmenting(np.asarray(tmpmarks),epochlength)
                            marktrials = np.concatenate([marktrials,markedsegs],axis=0)

                    # Lưu riêng dữ liệu nằm trong vùng artifact (để kiểm tra)
                    self.artidata=np.zeros((ARTtrl.shape[0],self.data.shape[1],np.nanmax(np.diff(ARTtrl))))
                    for i in range(ARTtrl.shape[0]):
                        self.artidata[i,:,:np.diff(ARTtrl[i,:])[0]] = self.data[0,:,ARTtrl[i,0]:ARTtrl[i,1]]

                    # Gán kết quả segment sau khi loại artifact
                    self.trl = trl[1:]
                    self.data = trials[1:]
                    self.arttrl = ARTtrl
                    self.info['artifact removal'] = 'detected artifacts removed'
                    self.info['no. segments'] = len(trl)-1

                    # Nếu số epoch sạch quá ít so với kỳ vọng => dữ liệu “bad”
                    if self.info['no. segments'] < ((1/3)* (totallength/(epochlength*self.Fs))):
                        self.info['data quality'] = 'bad'

                # Nếu không loại artifact: segment toàn bộ (artifact vẫn tồn tại trong kênh artifacts)
                elif remove_artifact == 'no':
                    #print('no artifact removal')
                    self.data,self.trl = EEGsegmenting(self.data, epochlength)
                    
                    # Segment marking nếu có
                    if marking == 'yes':
                        self.marking = EEGsegmenting(self.marking, epochlength)[0]
                    self.arttrl=ARTtrl
                    self.info['artifact removal'] = 'none removed'
                    self.info['no. segments'] = len(self.trl)

                    # Nếu lấy 'all' mà artifact chiếm quá nhiều thời gian => dữ liệu “bad”
                    if trllength == 'all':
                        if  len(p) > ((2/3) * totallength):
                            self.info['data quality'] = 'bad'
                        else:
                            self.info['data quality'] = 'OK'

            # Trường hợp có kênh artifacts nhưng không có sample nào bị đánh dấu
            else:
                self.data,self.trl = EEGsegmenting(self.data, epochlength)
                if marking == 'yes':
                    self.marking = EEGsegmenting(self.marking, epochlength)[0]
                self.info['artifact removal'] = 'no artifacts detected'
                self.info['no. segments'] = len(self.trl)-1
                self.arttrl = [0]

                # Nếu số epoch thu được thấp hơn nhiều so với kỳ vọng => “bad”
                if self.info['no. segments'] < ((1.3) * (totallength/(epochlength*self.Fs))):
                    self.info['data quality'] = 'bad'

        # Không có kênh artifacts: segment trực tiếp
        else:
            self.data,self.trl = EEGsegmenting(self.data, epochlength)
            if marking=='yes':
                self.marking = EEGsegmenting(self.marking, epochlength)[0]

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


    def save_pdfs(self, preprocpath, inp='data', scaling =[-70,70]):
        """
        Xuất PDF báo cáo dạng EEG plot theo từng segment (mỗi trang 1 segment).
        Input:
          - preprocpath: folder lưu
          - inp: 'data' hoặc 'artidata'
          - scaling: giới hạn y-axis
        Output: file PDF trong <preprocpath>/<idcode>/pdf/
        """
        import numpy as np

        from matplotlib.collections import LineCollection
        #from matplotlib.ticker import MultipleLocator
        from matplotlib.backends.backend_pdf import PdfPages

        plt.ioff()

        if '.csv' in self.info['fileID']:
            idcode = self.info['fileID'].rsplit('/')[-1].split('.')[0]
            cond = self.info['fileID'].rsplit('/')[-1].split('.')[1]
        else:
            idcode = self.info['fileID'].rsplit('/')[-1].split('.')[0][:10]
            cond = self.info['fileID'].rsplit('/')[-1].split('-1')[1][-6:-4]

        trllength = str(self.data.shape[-1]/self.Fs)
        if self.info['data quality'] == 'OK':
            outname = idcode + '_' + cond + '_' + trllength + 's'
        elif self.info['data quality'] == 'bad':
            outname = 'BAD_'+ idcode + '_' + cond + '_' + trllength + 's'
            print('saving: data has been marked as BAD')
        if self.info['artifact removal'] == 'none removed':
            outname = 'RawReport_' + idcode + '_' + cond + '_' + trllength + 's'
        elif self.info['artifact removal'] == 'no artifact detected':
            outname = idcode + '_' + cond + '_' + trllength + 's'

        '''======== tạo folder mỗi idcode ========'''


        if os.path.isdir(preprocpath+idcode):
            pass
        else:
            os.mkdir(preprocpath+idcode)

        if os.path.isdir(preprocpath + idcode + '/pdf/'):
            savepath = preprocpath + idcode + '/pdf/'
        else:
            os.mkdir(preprocpath + idcode + '/pdf/')
            savepath = preprocpath + idcode + '/pdf/'

        odata = getattr(self, inp)

        if inp =='artidata':
            trl = self.arttrl
        else:
            trl = self.trl

        if 'artifacts' in self.labels:
            odata[:,26,:]=odata[:,26,:]*50
            data = odata[:,:27,:]
            self.labels = self.labels[:27]
        else:
            data = odata[:,:26,:]
            self.labels = self.labels[:26]
        if 'Events' in self.labels:
            events = np.where(self.labels == 'Events')[0]
            evdat = odata[:,events,:]*0.001
            data = np.vstack((data,evdat))
            self.labels= np.vstack((self.labels,'events'))


        n_trials, n_rows,n_samples = data.shape[0],data.shape[1], data.shape[2]
#        if n_rows > 26:
##            data = data[:,[:26,artifacts],:]
#            n_trials, n_rows, n_samples = data.shape[0],data.shape[1], data.shape[2]
#            #data[:,-4,:]=data[:,-4,:]*0.4 #downscale ECG
#            if 'Events' in self.labels:
#                data[:,events,:]=data[:,events,:]*0.001 #downscale events
#            if 'ECG' in self.labels:
#                data[:,ECG,:]=data[:,ECG,:]*0.005 #downscale ECG
#            if 'artifacts' in self.labels:
#                data[:,artifacts,:]=data[:,artifacts,:]*50#upscale artifacts

        import datetime
        with PdfPages(savepath+outname+'.pdf') as pp:
            #pp = PdfPages(savepath+outname+'test.pdf')
            firstPage = plt.figure(figsize=(11.69,8.27))
            firstPage.clf()
            t =  datetime.datetime.now()
            txt = 'Raw Data Report \n \n' + idcode + ' ' + cond + '\n \n' + ' Report created on ' + str(t)[:16] + '\n by \n \n Research Institute Brainclinics \n Brainclinics Foundation \n Nijmegen, the Netherlands'
            firstPage.text(0.5,0.5,txt, transform=firstPage.transFigure, size=22, ha="center")
            pp.savefig()

            for seg in range(n_trials):
                fig = plt.figure(num = seg, figsize = (20,12), tight_layout=True)

                plt.close()
                t = np.arange(0,n_samples/self.Fs, (n_samples/self.Fs)/n_samples)

                fig = plt.figure(num = seg, figsize = (20,12), tight_layout=True)
                ax1 = fig.add_subplot(1,1,1)
                plt.subplots_adjust(bottom = 0.2)
                ax1.set_title(idcode + ' ' + cond +'\n Segment: '+ str(seg+1) +' of '+str(n_trials))

                dmin = scaling[0]#data.min()
                dmax = scaling[1]#data.max()
                dr = (dmax - dmin) * 0.7  # Crowd them a bit.
                y0 = dmin
                y1 = (n_rows-1) * dr + dmax
                ax1.set_ylim(y0, y1)

                segments = []
                ticklocs = []
                #ticks = np.arange(0,int(n_samples/self.Fs),np.around((int((n_samples/self.Fs))/10),decimals=1))
                for i in range(n_rows):
                    segments.append(np.column_stack((t, data[seg,i,:])))
                    ticklocs.append(i * dr)

                ticks = np.arange(0,(data.shape[-1]/self.Fs)+((data.shape[-1]/self.Fs)/10),(data.shape[-1]/self.Fs)/10)
                ax1.set_xticks(ticks,minor=False)

                ticksl = np.arange(np.around(trl[seg,0]/self.Fs,decimals=2),np.around((trl[seg,0]/self.Fs)+(n_samples/self.Fs),decimals=2)+1,np.around((n_samples/self.Fs)/10,decimals=2))

                ticklabels = list(ticksl)#np.arange(ticks)
                xlabels = [ '%.1f' % elem for elem in ticklabels]
                xlabels = np.array(xlabels,dtype=str)
                ax1.set_xticklabels(xlabels)

                offsets = np.zeros((n_rows, 2), dtype=float)
                offsets[:,1] = ticklocs

                lines = LineCollection(np.flipud(segments), linewidths=(0.6), offsets=offsets, transOffset=None, colors = 'k')
                ax1.add_collection(lines)

                ax1.set_yticks(ticklocs)

                ax1.set_yticklabels(self.labels[::-1])

                ax1.set_xlabel('Time (s)')

                pp.savefig()
                plt.close()

        #plt.ion()
    def save(self, preprocpath, npy = 'yes', matfile = None, csv = None):
        """
        Lưu dữ liệu sau interprocessing theo các định dạng được chọn.

        Input:
        - preprocpath: thư mục gốc để lưu (sẽ tự tạo nếu chưa có)
        - npy: 'yes' -> pickle toàn bộ object (self) ra file
        - matfile: 'yes' -> lưu thêm file .mat
        - csv: 'yes' -> lưu từng segment ra .csv

        Output:
        - Tạo thư mục: <preprocpath>/<idcode>/
        - (csv='yes') Tạo thư mục: <preprocpath>/<idcode>/csv_data_<cond>_<trllength>s/
            và sinh các file .csv theo từng segment (hoặc 1 file nếu data 2D)
        - (npy='yes') Sinh file pickle: <preprocpath>/<idcode>/<outname>
        - (matfile='yes') Sinh file Matlab: <preprocpath>/<idcode>/<outname>.mat
        """
        import pandas as pd

        print('saving data \n')
        '''======== thu thập thông tin về data ========'''
        idcode = self.info['fileID'].rsplit('/')[-1].split('.')[0]
        cond = self.info['fileID'].rsplit('/')[-1].split('.')[1]
        trllength = str(self.data.shape[-1]/self.Fs)
        if self.info['data quality'] == 'OK':
            outname = idcode + '_' + cond + '_' + trllength + 's'
        else:
            outname = 'BAD_'+ idcode + '_' + cond + '_' + trllength + 's'
            print('saving: data has been marked as BAD')

        '''======== tạo folder mỗi idcode ========'''
        if not os.path.isdir(preprocpath):
            os.mkdir(preprocpath)
        if not os.path.isdir(preprocpath + idcode + '/'):
            os.mkdir(preprocpath + idcode + '/')

        #savepath = preprocpath + idcode + '/'
        # --- Lưu CSV (tùy chọn) ---
        if csv == 'yes':
            if os.path.isdir(preprocpath + idcode + '/csv_data_' + cond + '_' + trllength + 's/'):
                csvpath = preprocpath + idcode + '/csv_data_' + cond + '_' + trllength + 's/'
            else:
                os.mkdir(preprocpath + idcode + '/csv_data_' + cond + '_' + trllength + 's/')
                csvpath = preprocpath + idcode + '/csv_data_' + cond + '_' + trllength + 's/'

            for i in range(self.data.shape[0]):
                if len(self.data.shape) == 3:
                    df = pd.DataFrame(self.data[i,:,:].T)
                    df.to_csv(csvpath + str((self.trl[i,0]/self.Fs)*1000) + '.csv',sep=',',header = list(self.labels),compression = None)
                else:
                    df = pd.DataFrame(self.data[:,:].T)
                    df.to_csv(csvpath + str(0)+'.csv',sep=',',header = list(self.labels),compression = None)

        # --- Lưu pickle (tùy chọn) ---
        if npy == 'yes':
            import _pickle as pickle

            with open(preprocpath+'/'+ idcode + '/' + outname, 'wb') as output:  # Overwrites any existing file.
                pickle.dump(self, output, -1)

        # --- Lưu .mat (tùy chọn) ---
        if matfile == 'yes':
            print('saving .mat file')
            mat_dataset = {'labels': self.labels,
                           'trials': self.data,
                           'dimord' :'rpt_chan_time',
                           'artitrials': self.artidata,
                           'Fs': 500,
                           'time': np.arange(0,(self.data.shape[-1]/self.Fs),1/self.Fs),
                           'info': self.info,
                           'trl': self.trl,
                           'arttrl': self.arttrl}
            sio.savemat(preprocpath+'/'+ idcode + '/' + outname + '.mat', mat_dataset)

    def plot_EEG(self, inp='data', marking='no', channels = 'EEG', scaling=[-70,70], title=None):
        """
        Vẽ EEG theo kiểu “stacked traces” cho từng segment (trial) và cho phép duyệt qua các segment bằng nút < >.

        Input:
        - inp: 'data' hoặc 'artidata' (dữ liệu EEG bình thường hoặc dữ liệu đoạn artifact nếu có)
        - marking: 'yes' để chồng thêm tín hiệu marking (màu đỏ) nếu self.marking tồn tại
        - channels: 'EEG' (chỉ 26 kênh EEG) hoặc 'all' (giữ toàn bộ kênh trong labels)
        - scaling: [ymin, ymax] (microvolts) để đặt thang hiển thị từng kênh
        - title: tiêu đề figure (None -> dùng fileID)

        Output:
        - Hiển thị figure matplotlib + 2 nút điều hướng (Next/Prev)
        - Return: (bnext, bprev) là 2 đối tượng Button
        """

        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button
        from matplotlib.collections import LineCollection
#        from matplotlib.colors import ListedColormap, BoundaryNorm

        data = getattr(self, inp)

        if marking == 'yes':
            markingdata = getattr(self, 'marking')

        if channels == 'EEG':
            data = data[:,:26,:]
            labels = self.labels[:26]
            if marking == 'yes':
                markingdata = markingdata[:,:26,:]
        else:
            labels = self.labels

        if inp =='artidata':
            trl = self.arttrl
        else:
            trl = self.trl

        n_samples, n_rows, n_trials = data.shape[-1], data.shape[-2], data.shape[-3]
        
        # Nếu có kênh phụ
        if n_rows >26:
            if 'ECG' in labels:
                ECG = np.where(labels== 'ECG')[0]
                data[:,ECG,:]=data[:,ECG,:]*0.5  #downscale ECG
            if 'artifacts' in labels:
                artifacts = np.where(labels == 'artifacts')[0][0]
                artmarks = np.empty((data.shape[-1]));artmarks[:]=np.nan
                for tr in range(n_trials):
                    artmarks[np.where(data[tr,artifacts,:]==1)]=0
                    data[tr,artifacts,:]=artmarks#data[artifacts,:]*50 #upscale artifacts
            if 'Events' in labels:
                events = np.where(labels == 'Events')[0]

        t = np.arange(0,n_samples/self.Fs, (n_samples/self.Fs)/n_samples)

        if title == None:
            fig = plt.figure(self.info['fileID'].rsplit('/')[-1], figsize = (6,9))
        else:
            fig = plt.figure(title, figsize = (6,9))

        ax1 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(bottom = 0.2)
        #ax1.set_title('Segment: '+ str(1) +' of '+str(n_trials))

        dmin = scaling[0]#data.min()
        dmax = scaling[1]#data.max()
        dr = (dmax - dmin) * 0.7  
        y0 = dmin
        y1 = (n_rows-1) * dr + dmax
        ax1.set_ylim(y0, y1)

        # Chuẩn bị segments cho LineCollection (vẽ stacked traces)
        segments = [];markings = []
        ticklocs = []
        for r in range(n_rows):
            if len(data.shape) == 3:
                segments.append(np.column_stack((t, data[0,r,:])))
                if marking == 'yes':
                    filler = np.zeros((markingdata[0,0,:].shape));filler[:]=np.nan
                    if r <= 26:
                        markings.append(np.column_stack((t, markingdata[0,r,:])))
                    else:
                        markings.append(np.column_stack((t, filler)))
            ticklocs.append(r * dr)

        offsets = np.zeros((n_rows, 2), dtype=float)
        offsets[:,1] = ticklocs
        
        lines = LineCollection(np.flipud(segments), offsets=offsets, transOffset=None, colors = 'k', linewidth=np.flipud(np.hstack([np.ones(26),2])))
        ax1.add_collection(lines)
        if marking == 'yes':
            lines2 = LineCollection(np.flipud(markings), linewidth=(2), offsets=offsets, transOffset=None, colors = 'r')
            ax1.add_collection(lines2)

        ticks = np.arange(0,(data.shape[-1]/self.Fs)+((data.shape[-1]/self.Fs)/10),(data.shape[-1]/self.Fs)/10)
        ax1.set_xticks(ticks,minor=False)
        ticksl = np.arange(np.around(trl.flat[0]/self.Fs,decimals=3),np.around((trl.flat[0]/self.Fs)+(n_samples/self.Fs),decimals=3)+1,np.around((n_samples/self.Fs)/10,decimals=3))
        ticklabels = list(ticksl)#np.arange(ticks)
        xlabels = [ '%.2f' % elem for elem in ticklabels]
        xlabels = np.array(xlabels,dtype=str)
        ax1.set_xticklabels(xlabels)
        ax1.set_xlabel('Time (s)')

        ax1.set_yticks(ticklocs)
        ax1.set_yticklabels(labels[::-1])
        ax1.set_ylabel('EEG labels')

        axs = {}
        axs['ax1'] = ax1
        axs['axnext'] = plt.axes([0.84, 0.10, 0.10, 0.04])#next button
        axs['axprev'] = plt.axes([0.72, 0.10, 0.10, 0.04])#previous button
        axs['offsets'] = offsets

        class GUIButtons(object):
            """
            Handler cho 2 nút Next/Prev:
            - Cập nhật segments của LineCollection để hiển thị trial kế tiếp/trước đó
            - Cập nhật nhãn trục X theo trl của trial hiện tại
            """
            def __init__(self, tmpdata, axs, t, Fs, trls, marking = None):
                self.tmpdata = tmpdata
                if len(axs['ax1'].collections) == 2:
                    self.tmpmarking = marking
                else:
                    self.tmpmarking = np.empty(self.tmpdata.shape);self.tmpmarking[:]=np.nan
                self.axs = axs
                self.t = t
                self.Fs = Fs
                self.trls = trls
                self.index = 0
                self.offsets = axs['offsets']

            def nextb(self, event):

                n_trials, n_rows = self.tmpdata.shape[0], self.tmpdata.shape[1]

                self.index += 1
                i = self.index

                if i >= n_trials:
                    i = n_trials
                    self.axs['ax1'].set_title('Last sample reached. Cannot go forwards')
                else:
                    segments=[];markings=[]
                    for r in range(n_rows):
                        segments.append(np.column_stack((self.t, self.tmpdata[i,r,:])))
                        if len(axs['ax1'].collections) ==2:
                            filler = np.zeros((markingdata[0,0,:].shape));filler[:]=np.nan
                            if r <= 26:
                                markings.append(np.column_stack((self.t, self.tmpmarking[i,r,:])))
                            else:
                                markings.append(np.column_stack((self.t, filler)))

                    linesn = self.axs['ax1'].collections[0]
                    linesn.set_segments(np.flipud(segments))

                    if len(axs['ax1'].collections) ==2:
                        linesn2=self.axs['ax1'].collections[1]
                        linesn2.set_segments(np.flipud(markings))
#                        linesn2.set_offsets(self.offsets)

                    self.axs['ax1'].set_title('Segment: '+str(i+1) + ' of ' + str(n_trials))
                    #self.axs['ax1'].set_xticks(ticks,minor=False)
                    ticksl = np.arange(np.around(self.trls[i,0]/self.Fs,decimals=3),np.around((self.trls[i,0]/self.Fs)+(self.tmpdata.shape[-1]/self.Fs),decimals=3)+((data.shape[-1]/self.Fs)/10),np.around((self.tmpdata.shape[-1]/self.Fs)/10,decimals=3))

                    ticklabels = list(ticksl)#np.arange(ticks)
                    xlabels = [ '%.2f' % elem for elem in ticklabels]
                    xlabels = np.array(xlabels,dtype=str)

                    self.axs['ax1'].set_xticklabels(xlabels)
#                    self.axs['ax1'].axis('tight')
                    #plt.show()


            def prevb(self, event):
                self.index -= 1
                i = self.index

                n_trials = self.tmpdata.shape[0]
                n_rows = self.tmpdata.shape[1]
                #n_samples = self.tmpdata.shape[2] #amount of timepoint measured

                if i < 0:
                    i = 0
                    self.axs['ax1'].set_title('First sample reached. Cannot go backwards')
                else:
                    segments=[];markings = []
                    for r in range(n_rows):
                        segments.append(np.column_stack((self.t, self.tmpdata[i,r,:])))
                        markings.append(np.column_stack((self.t, self.tmpmarking[i,r,:])))

                    linesn = self.axs['ax1'].collections[0]
                    linesn.set_segments(np.flipud(segments))
                    if len(axs['ax1'].collections) ==2:
                        linesn2=self.axs['ax1'].collections[1]
                        linesn2.set_segments(np.flipud(markings))

                    self.axs['ax1'].set_title('Segment: '+str(i+1) + ' of ' + str(n_trials))
                    #start = self.trls[i,0]/self.Fs
                    ticksl = np.arange(np.around(self.trls[i,0]/self.Fs,decimals=3),np.around((self.trls[i,0]/self.Fs)+(self.tmpdata.shape[-1]/self.Fs),decimals=3)+((data.shape[-1]/self.Fs)/10),np.around((self.tmpdata.shape[-1]/self.Fs)/10,decimals=3))

                    #ticks = np.arange(start,start+(self.tmpdata.shape[-1]/self.Fs)+1,np.around((self.tmpdata.shape[-1]/self.Fs)+1,decimals=1))
                    ticklabels = list(ticksl)#np.arange(ticks)
                    xlabels = [ '%.2f' % elem for elem in ticklabels]
                    xlabels = np.array(xlabels,dtype=str)
                    self.axs['ax1'].set_xticklabels(xlabels)
                    # plt.show()
                    # plt.axis('tight')

        if marking == 'yes':
            callback = GUIButtons(data,axs,t,self.Fs,trl, marking=markingdata)
        else:
            callback = GUIButtons(data,axs,t,self.Fs,trl)

        ''' buttons '''
        bnext = Button(axs['axnext'], '>')
        bnext.on_clicked(callback.nextb)
        axs['axnext']._button = bnext

        bprev = Button(axs['axprev'], '<')
        bprev.on_clicked(callback.prevb)
        axs['axprev']._button = bprev

        plt.show()
        plt.axis('tight')

        return bnext, bprev

def EEGsegmenting(inp, trllength, fs=500, overlap=0):
    """
    Mục đích:
      Cắt tín hiệu EEG liên tục thành các epoch/segment có độ dài cố định (trllength giây),
      trả về:
        - data: mảng (n_trials, n_rows, n_samples) chứa từng segment
        - trl:  mảng (n_trials, 2) chứa [start, end] theo chỉ số sample cho mỗi segment

    Lưu ý:
      - Hàm đang giả định inp có dạng 3D và chỉ lấy trial đầu tiên: inp[0, :, :]
        (tức inp ~ (n_trials?, n_rows, n_totalsamples); nếu inp chỉ là 2D sẽ lỗi)
      - overlap là tỉ lệ chồng lấp giữa các segment (0 = không chồng lấp).
    """
    n_samples = int(trllength*fs)
    stepsize = (1-overlap)*n_samples

    n_totalsamples, n_rows = inp.shape[-1], inp.shape[1]
    n_trials = int(n_totalsamples/stepsize)

    trl = np.array([0,0],dtype=int)
    t = 0
    for i in range(n_trials):
        trl = np.vstack((trl,[t,t+n_samples]))
        t = t+stepsize

    trl = trl[1:]

    data = np.zeros((n_trials, n_rows, int(n_samples)))
    for i in range(n_trials):
        if trl[i,0] <= n_totalsamples-n_samples:
            data[i,:,:]= inp[0,:,trl[i,0]:trl[i,1]]

    return data, trl

