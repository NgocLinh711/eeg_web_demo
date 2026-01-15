#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

def loadbysubID(root_dir, subID, condition):
    """
    *---------------------------------------------------------------------*
    Load dữ liệu đã preprocess của 1 subject theo ID và condition.

    Cách dùng:
        data = loadbysubID(root_dir, subID, condition)

    Input:
        root_dir  : path gốc chứa dữ liệu đã preprocess (có thể ở level cao,
                    nhưng càng gần nơi chứa file thực tế thì tìm càng nhanh)
        subID     : ID subject (chuỗi 8 chữ số)
        condition : điều kiện cần load, ví dụ 'EO' hoặc 'EC'

    Output:
        Trả về 1 object interdataset (ids(preproc)) chứa:
          - EEG data
          - EEG labels
          - info về preprocessing trước đó
          - các hàm xử lý/tiện ích đi kèm (tuỳ implement của interdataset)

    Ghi chú:
        - Hàm này sẽ tìm tất cả file *.npy chứa subID trong tên file
        - Lọc tiếp theo condition và loại file có chứa 'BAD'
        - Lấy phần tử đầu tiên [0] trong danh sách phù hợp
    *-----------------------------------------------------------------------*
    """
    import fnmatch
    from BCD_gitprojects.data_processing.processing.interprocessing import interdataset as ids
    
    def find(IDcode, pattern, path): 
        # Duyệt toàn bộ thư mục con dưới path để tìm các file:
        #   - tên file chứa IDcode
        #   - khớp pattern ('*.npy')
        # Output: list các full-path phù hợp
        result = []
        for root, dirs, files in os.walk(path):
            for name in files:
                if IDcode in name and fnmatch.fnmatch(name, pattern):
                    result.append(os.path.join(root, name))
        return result

    filepaths = find(subID,'*.npy', root_dir)

    # Chọn file theo condition (EO/EC) và không chứa 'BAD'
    inname = [f for f in filepaths if condition in f and not 'BAD' in f][0]

    print(inname)

    # Load nội dung pickle từ file .npy
    with open(inname,'rb') as input: preproc = pickle.load(input)

    # Bọc dữ liệu preproc vào interdataset và trả về
    return ids(preproc)

class FilepathFinder(object):
    """
    *---------------------------------------------------------------------*
    Class để thu thập danh sách đường dẫn file theo pattern trong root_dir.

    Mục đích:
        - Quét toàn bộ cây thư mục dưới root_dir
        - Lấy ra danh sách file có tên chứa pattern (vd '.npy', 'eeg.csv', ...)
        - Dùng cho DataLoader / fit_generator (theo mô tả comment gốc)

    Cách dùng:
        finder = FilepathFinder(pattern, root_dir)
        finder.get_filenames()
        files = finder.files

    Input khi khởi tạo:
        pattern  : chuỗi cần xuất hiện trong tên file (không phải regex)
        root_dir : thư mục gốc để quét

    Output:
        - Sau get_filenames(): self.files là list filepaths tìm được

    Ghi chú:
        - Docstring gốc có nói exclude/test_size nhưng trong code hiện tại:
          + __init__ chỉ nhận (pattern, root_dir)
          + không có tham số exclude/test_size và không có GroupShuffleSplit
        - Comment gốc cũng nói “only takes in first sessions” nhưng code không lọc session
    *-----------------------------------------------------------------------*
    """
    import os, pickle
    def __init__ (self, pattern, root_dir):
        self.pattern = pattern
        self.root_dir = root_dir

    def get_filenames(self, sessions='all'):
        """
        Public method: lưu self.files bằng kết quả tìm kiếm
        Output:
          set self.files; không return
        """
        self.files = self.__find()

    def __find(self):
        """
        Private method: walk toàn bộ root_dir và lấy file có tên chứa self.pattern
        Output:
          result: list full-path các file phù hợp 
        """
        import  os
        result = []
        for root, dirs, files in os.walk(self.root_dir):
            for name in files:
                if self.pattern in name:
                    #print('yes!')
                    #print(self.pattern)
                    result.append(os.path.join(root, name))
        print(str(len(result)) + ' files listed')
        return result

#%% fileloader template
"""
# TEMPLATE (không chạy trực tiếp): minh hoạ cách duyệt sourcepath để tạo danh sách file theo subject/session/condition

# Danh sách từ khoá cần loại trừ khi duyệt folder/file (vd folder hệ thống Mac, folder DS, ...)
exclude = ['Apple','DS','._']; s=[]

# Lấy danh sách subject folder trong sourcepath, loại các folder có chứa chuỗi trong exclude
subs = [s for s in os.listdir(sourcepath)
        if os.path.isdir(os.path.join(sourcepath,s))
        and not any([excl in s for excl in exclude])]
subs = np.sort(subs)

# Nếu varargs['condition'] có thì dùng, nếu không mặc định xử lý EO/EC
if varargs['condition']:
    reqconds = varargs['condition']
else:
    reqconds = ['EO','EC']

# Xác định subarray (danh sách subject sẽ xử lý):
# - subject == None: xử lý từ k đến hết
# - subject là int: coi như index trong subs
# - subject là str: tìm vị trí subject trong subs
k = startsubj
if subject == None:
    subarray = np.arange(k,len(subs))
elif type(subject) == int:
    subarray = [subject]
elif type(subject) == str:
    subarray = np.array([np.where(subs==subject)[0]][0])

files = []
for s in subarray:
    # Lấy danh sách session folder của subject, loại exclude và chỉ lấy folder thật
    sessions = [session for session in os.listdir(os.path.join(sourcepath,subs[s]))
                if not any([excl in session for excl in exclude])
                and os.path.isdir(os.path.join(sourcepath,subs[s],session))]

    for sess in range(len(sessions)):
        conditions = []

        # Lấy tất cả file .csv trong session folder, loại các file có chứa exclude
        allconds = np.array([conds for conds in os.listdir(os.path.join(sourcepath,subs[s],sessions[sess]))
                             if ('.csv' in conds) and not any([excl in conds for excl in exclude])])

        # Nếu yêu cầu 'all' thì lấy hết, nếu không thì lọc theo EO/EC (substring match)
        if reqconds == 'all':
            conditions = allconds
        else:
            conditions = np.array([conds for conds in allconds
                                   if any([a.upper() in conds for a in reqconds])])

            # Vòng lặp c ở đây là nơi bạn thường build file list / chạy xử lý cho từng file
            for c in range(len(conditions)):
                pass
"""

