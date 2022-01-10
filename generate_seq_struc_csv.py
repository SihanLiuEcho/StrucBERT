import os
import re
import pandas as pd
#data_path = '/mnt/data/huangbin/sc_wangsheng'
struc_data_path = '/mnt/data/huangbin/sc_wangsheng'
seq_data_path = '/mnt/data/huangbin/fasta'
def get_structure_files(data_path, format): 
    #获得data_path路径下的结构码文件名，并将其划分成train、val、test三个数据集的文件名
    res = []
    files = os.listdir(data_path)
    files = sorted(files)
    for file in files:
        if file.endswith(format): #.cle或.fasta
            res.append(file)
    #print(res)
    total_len = len(res)
    train = []
    val = []
    test = []
    train = res[:int(total_len*0.7)]
    val = res[int(total_len*0.7):int(total_len*0.9)]
    test = res[int(total_len*0.9):]
    return train, val, test
def get_seq_files(struc_file_names, format): 
    #根据结构码文件名生成对应序列文件名
    res = []
    for file in struc_file_names:
        res.append(file.replace('.cle', '.fasta'))
    return res

def get_structure_sequences(data_path, struc_files):
    #提取data_path文件夹下struc_files的序列/结构码 生成器
    for struc_file in struc_files:
        #print(struc_file)
        dir = os.path.join(data_path, struc_file)
        #print(dir)
        with open(dir, 'r') as f:
            #print(struc_file)
            lines = f.readlines()
            sequence_name = lines[0].replace(">","")
            sequence_data = "".join(lines[1:]).replace("\n","")
            sequence_list = re.findall(".{1}", sequence_data)
            #sequence_data = ' '.join(sequence_list)
            yield sequence_data

# 生成csv文件
# sequence, structure
def generate_csv(seq_data_path, struc_data_path, seq_files, struc_files, filename):
    sequences = []
    structures = []
    for sequence_data,structure_data in zip(get_structure_sequences(seq_data_path, seq_files), get_structure_sequences(struc_data_path, struc_files)):
        seq_len = len(sequence_data)
        new_seq = ' '.join(sequence_data)
        new_struc = ' '.join(structure_data.replace('R','').rjust(seq_len, 'R'))
        
        #assert len(new_seq) == len(new_struc)
        if len(new_seq) != len(new_struc):
            continue 
        sequences.append(new_seq)
        structures.append(new_struc)
        
    dataframe = pd.DataFrame({'sequence':sequences,'structure':structures})

    #将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv(filename,index=False,sep=',')


struc_train_files, struc_val_files, struc_test_files = get_structure_files(struc_data_path, '.cle')
seq_train_files = get_seq_files(struc_train_files, '.fasta')
seq_val_files = get_seq_files(struc_val_files, '.fasta')
seq_test_files = get_seq_files(struc_test_files, '.fasta')
generate_csv(seq_data_path, struc_data_path, seq_train_files, struc_train_files, './seq_struc_data/train.csv')
generate_csv(seq_data_path, struc_data_path, seq_val_files, struc_val_files, './seq_struc_data/val.csv')
generate_csv(seq_data_path, struc_data_path, seq_test_files, struc_test_files, './seq_struc_data/test.csv')



