import xlrd
import numpy as np

VEC_DIM = 128

class SQ:

    def __init__(self,sens1,sens2,labels):
        self.sens1 = sens1
        self.sens2 = sens2
        self.labels = labels



def set_train(path):

    book = xlrd.open_workbook(path)
    sheet = book.sheet_by_name('Sheet1')
    rows = sheet.nrows #获取行数
    # cols = sheet.ncols #获取列数
    # for c in range(cols): #读取每一列的数据
    #     c_values = sheet.col_values(c)
    #     print(c_values)
    sq = SQ([], [], [])
    for r in range(rows):#读取每一行的数据
        r_values = sheet.row_values(r)
        for j in range(len(r_values)):
            j_values = str2list(r_values[j])
            for k in range(len(r_values)):
                k_values = str2list(r_values[k])
                sq.sens1.append(j_values)
                sq.sens2.append(k_values)
                sq.labels.append([1.0,0.0])
        for m in range(rows):
            if m!=r:
                m_values = sheet.row_values(m)
                for j in range(len(m_values)):
                    j_values =str2list(m_values[j])
                    for k in range(len(r_values)):
                        k_values = str2list(r_values[k])
                        sq.sens1.append(j_values)
                        sq.sens2.append(k_values)
                        sq.labels.append([0.0, 1.0])
            else:
                pass
    return sq
    # print(sheet.cell(1,1)) #读取指定单元格的数据


def str2list(seq):

    list = []
    slist = []
    seq = seq.replace('[',' ').replace(']', ' ').replace(',', '').replace("'", '').split()
    for i in range(len(seq)):
        slist.append(float(seq[i]))
        if (i+1) % VEC_DIM == 0:
            list.append(slist)
            slist = []
    return list


train_sqsvec = set_train('s_questions_vec.xls')
print(len(train_sqsvec.sens1))


def padding(sqsvec):
    sens_num = len(sqsvec.sens1)
    max_length = 20
    lengthList1 = []
    lengthList2  = []

    padding_dataset1 = np.zeros([sens_num, max_length, VEC_DIM], dtype=np.float32)
    padding_dataset2 = np.zeros([sens_num, max_length, VEC_DIM], dtype=np.float32)

    for idx, seq in enumerate(sqsvec.sens1):
        padding_dataset1[idx,:len(seq),:] = seq
        lengthList1.append(len(seq))
    for idx, seq in enumerate(sqsvec.sens2):
        padding_dataset2[idx,:len(seq),:] = seq
        lengthList2.append(len(seq))

    return padding_dataset1,padding_dataset2,lengthList1,lengthList2


train_sqsvec = train_sqsvec
train_sqsvec.sens1, train_sqsvec.sens2, sens1_length, sens2_length = padding(train_sqsvec)


def set_test(path1,path2):

    book = xlrd.open_workbook(path1)
    sheet = book.sheet_by_name('Sheet1')
    test_list = []
    r_values = sheet.row_values(0)
    value = r_values[0]
    test_data = str2list(value)
    test_list.append(test_data)
    book = xlrd.open_workbook(path2)
    sheet = book.sheet_by_name('Sheet1')
    for r in range(sheet.nrows):
        r_values = sheet.row_values(r)
        for value in r_values:
            test_data = str2list(value)
            test_list.append(test_data)

    return test_list


test_sqsvec = set_test("n_questions_vec.xls","s_questions_vec.xls")
