#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils.autopreprocessing import dataset as ds
from utils.inout import FilepathFinder as FF
import os
from pathlib import Path
import numpy as np
import copy

def autopreprocess_standard(varargsin, subject = None, startsubj =0):
    """
    Pipeline preprocessing chuẩn.

    Parameters
    ----------
    varargsin : dict (bắt buộc)
        - 'sourcepath': đường dẫn tới folder chứa dữ liệu raw/derivatives (có sub-*/ses-*/eeg/*.csv)
        - 'preprocpath': đường dẫn output để lưu dữ liệu đã preprocess
        - 'condition': (optional) danh sách điều kiện cần xử lý, ví dụ ['EO','EC'] hoặc 'all'
        - 'exclude': (optional) danh sách pattern cần loại trừ (ở code này chỉ set default, không dùng tiếp)
    subject : optional
        - None: xử lý toàn bộ subject từ startsubj trở đi
        - int: coi như index của subject trong danh sách subs
        - str: ID của subject (tên folder, ví dụ 'sub-19681349')
    startsubj : int
        - chỉ số bắt đầu trong danh sách subject (dùng để chạy theo batch)

    Output
    ------
    - Lưu file preprocessed theo cấu trúc: preprocpath/sub-XXXX/ses-*/eeg/<saved files>
    - Lưu PDF report raw (theo từng segment 10s) nếu rawreport == 'yes'
    """

    # ==========================
    # 1) Validate input arguments
    # ==========================
    # Defining the reading path
    if not 'sourcepath' in varargsin:
        raise ValueError('sourcepath not defined, where is your data?')

    if not 'preprocpath' in varargsin:
        raise ValueError('preprocpath not defined')

    # Lấy input/output path
    sourcepath = varargsin['sourcepath']
    preprocpath = varargsin['preprocpath']

    # In ra để log
    print(sourcepath)
    print(preprocpath)

    # ==========================================
    # 2) Sanity check: có file EEG CSV hay không
    # ==========================================
    # Liệt kê tất cả file có chứa pattern 'eeg.csv' dưới sourcepath
    csv = FF('eeg.csv',sourcepath)
    csv.get_filenames()

    # Nếu không tìm thấy file nào -> dừng
    if len(csv.files)<1:
        raise ValueError('no csv files found in this specified path, please check your sourcepath '+sourcepath)

    # ==================================
    # 3) Read optional configuration keys
    # ==================================
    # Nếu không có 'condition' thì mặc định xử lý EO + EC
    if not 'condition' in varargsin:
        reqconds = ['EO','EC']

    # Nếu không có 'exclude' thì set default []
    if not 'exclude' in varargsin:
        varargsin['exclude'] = []

    # Bật/tắt xuất PDF raw report
    rawreport = 'yes'

    # ============================================
    # 4) Build subject inventory (list subfolders)
    # ============================================
    # Lấy các folder con trong sourcepath, loại trừ các folder có chuỗi:
    # 'preprocessed', 'results', 'DS'
    subs = [s for s in os.listdir(sourcepath) if os.path.isdir(os.path.join(sourcepath,s)) if not any([e in s for e in ['preprocessed','results','DS']])]
    subs = np.sort(subs)  # sort theo tên
    print(str(len(subs))+' subjects')
#    subs = [s for s in os.listdir(sourcepath+'/') if os.path.isdir(os.path.join(sourcepath,s)) and not '.' in s]
#    subs = np.sort(subs)

    # ============================================
    # 5) Decide which subjects to process (subarray)
    # ============================================
    k=startsubj
    # subject=None: chạy từ startsubj tới hết
    # subject=int: coi như index trong subs
    # subject=str: tìm index của subject name trong subs
    if subject == None:
        subarray = range(k,len(subs))
        subarray = [subject]
    elif type(subject) == str:
        subarray = np.array([np.where(subs==subject)[0]][0])

    sp = k # counter để in progress theo thứ tự batch

    # ==========================
    # 6) Main loop: subjects
    # ==========================
    for s in subarray:
        print('[INFO]: processing subject: '+str(sp) +' of '+str(len(subs)))
        
        # Liệt kê session folders trong mỗi subject
        sessions = [session for session in os.listdir(os.path.join(sourcepath,subs[s])) if not any([e in session for e in ['preprocessed','results','DS']])]
        subs = np.sort(subs)

        # ==========================
        # 7) Loop: sessions
        # ==========================
        for sess in range(len(sessions)):
            conditions = []

            # Liệt kê các file/entry trong folder eeg/ của session
            allconds = np.array([conds for conds in os.listdir(os.path.join(sourcepath,subs[s],sessions[sess]+'/eeg/')) if not any([e in conds for e in ['preprocessed','results','DS']])])
            
            # Nếu reqconds == 'all' thì lấy toàn bộ, ngược lại lọc theo EO/EC
            if reqconds == 'all':
                conditions = allconds
            else:
                # Giữ lại file nào có chứa chuỗi trong reqconds (EO/EC)
                conditions = np.array([conds for conds in allconds if any([a.upper() in conds for a in reqconds])])

            # ==========================
            # 8) Loop: conditions/files
            # ==========================
            for c in range(len(conditions)):
                print(conditions[c])

                # Nếu có file điều kiện
                if len(conditions)>0:
                    # Đường dẫn đầy đủ tới file EEG csv của điều kiện đó
                    inname = os.path.join(sourcepath, subs[s], sessions[sess]+'/eeg/',conditions[c])
                    
                    # ==========================
                    # 9) Run preprocessing steps
                    # ==========================
                    tmpdat = ds(inname)             # tạo dataset object với filename
                    tmpdat.loaddata()               # đọc csv vào tmpdat.data và tmpdat.labels
                    tmpdat.bipolarEOG()             # tạo VEOG/HEOG bipolar từ VPVA/VNVB/HPHL/HNHR
                    tmpdat.apply_filters()          # lọc notch/hp/lp theo default trong class
                    tmpdat.correct_EOG()            # phát hiện và hồi quy artifact mắt ra khỏi EEG
                    tmpdat.detect_emg()             # phát hiện EMG
                    tmpdat.detect_jumps()           # phát hiện jump/baseline shift
                    tmpdat.detect_kurtosis()        # phát hiện kurtosis cao
                    tmpdat.detect_extremevoltswing()# phát hiện voltage swing lớn
                    tmpdat.residual_eyeblinks()     # phát hiện voltage swing lớn
                    tmpdat.define_artifacts()       # tổng hợp artifact + đánh dấu bad/bridging + interpolate
                    
                    # ==========================
                    # 10) Save preprocessed object
                    # ==========================
                    trllength = 'all'
                    npy = copy.deepcopy(tmpdat)
                    npy.segment(trllength = trllength, remove_artifact='no')
                    # subpath = os.path.join(preprocpath,subs[s])
                    # Path(subpath).mkdir(parents=True, exist_ok=True)
                    sesspath = os.path.join(preprocpath,subs[s],sessions[sess]+'/eeg/')
                    Path(sesspath).mkdir(parents=True, exist_ok=True)
                    npy.save(sesspath)

                    # ==========================
                    # 11) Optional: save raw report PDF
                    # ==========================
                    if rawreport == 'yes':
                        lengthtrl = 10 # segment 10s để vẽ report
                        pdf = copy.deepcopy(tmpdat)
                        pdf.segment(trllength = lengthtrl, remove_artifact='no')
                        pdf.save_pdfs(sesspath)
                   # except:
                   #     print('processing of '+inname+ ' went wrong')
        sp=sp+1 # tăng counter subject

