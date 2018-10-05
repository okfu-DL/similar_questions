import re
import xlrd, xlwt


VEC_DIM = 128

def read_worddict(path):

    with open(path, 'r', encoding="utf-8") as f:
        g = f.read().replace(':[', ' ').replace(']', ' ').split()
    vec = []
    WordDict = {}
    word = None
    for i in range(len(g)):
        if i % (VEC_DIM+1) != 0:
            vec.append(g[i])
        else:
            word = g[i]
        if (i + 1) % (VEC_DIM+1) == 0:
            WordDict[word] = vec
            vec = []

    return WordDict


#StopWordtmp = ['。', '，', '！', '？', '：', '“', '”', '‘', '’', '（', '）', '【', '】', '｛', '｝', '-', '－', '～', '［', '］', '〔', '〕', '．', '＠', '￥', '•', '.']
StopWordtmp = [' ', u'\u3000', u'\x30fb', u'\u3002', u'\uff0c', u'\uff01', u'\uff1f', u'\uff1a', u'\u201c', u'\u201d', u'\u2018', u'\u2019', u'\uff08', u'\uff09', u'\u3010', u'\u3011', u'\uff5b', u'\uff5d', u'-', u'\uff0d', u'\uff5e', u'\uff3b', u'\uff3d', u'\u3014', u'\u3015', u'\uff0e', u'\uff20', u'\uffe5', u'\u2022', u'.']

WordDic = read_worddict(path='vec_fd.txt')
StopWord = []
span = 16


def InitStopword():
    for key in StopWordtmp:
        StopWord.append(key)


def WordSeg(Inputpath, Outputpath):

    book = xlrd.open_workbook(Inputpath)
    sheet = book.sheet_by_name('Sheet1')
    rows = sheet.nrows

    rb = xlwt.Workbook(Outputpath)
    sheet1 = rb.add_sheet(u'Sheet1', cell_overwrite_ok=True)

    for r in range(rows):
        row_values = sheet.row_values(r)
        for j in range(len(row_values)):
            sen = row_values[j]
            tmpstr = row_values[j]
            for k in range(len(tmpstr)):
                if tmpstr[k] in StopWord:
                    sen = tmpstr.replace(tmpstr[k],"")
            # senList = []
            # tmpword = ''
            # for k in range(len(col_values)) :
            #     if col_values[k] in StopWord:
            #         senList.append(tmpword)
            #         tmpword = ''
            #     else:
            #         tmpword += col_values[k]
            #         if k == len(col_values) - 1:
            #             senList.append(tmpword)
            #
            # tmplist = ''
            # for key in senList:
            #     tmplist += key
            tmplist = PreSenSeg(sen, span)
            sheet1.write(r,j,str(tmplist))
    rb.save(Outputpath)
    print("successful transfer")


def PreSenSeg(sen, span):
    post = span
    if len(sen) < span:
        post = len(sen)
    cur = 0
    revlist = []
    while 1:
        if cur >= len(sen):
            break
        # s = re.search(u"^[0|1|2|3|4|5|6|7|8|9|\uff11|\uff12|\uff13|\uff14|\uff15|\uff16|\uff17|\uff18|\uff19|\uff10|\u4e00|\u4e8c|\u4e09|\u56db|\u4e94|\u516d|\u4e03|\u516b|\u4e5d|\u96f6|\u5341|\u767e|\u5343|\u4e07|\u4ebf|\u5146|\uff2f]+", sen[cur:])
        # if s:
        #     if s.group() != '':
        #         revlist.append(s.group())
        #     cur = cur + len(s.group())
        #     post = cur + span
        #     if post > len(sen):
        #         post = len(sen)
        # s = re.search(u"^[a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z|A|B|C|D|E|F|G|H|I|J|K|L|M|N|O|P|Q|R|S|T|U|V|W|X|Y|Z|\uff41|\uff42|\uff43|\uff44|\uff45|\uff46|\uff47|\uff48|\uff49|\uff47|\uff4b|\uff4c|\uff4d|\uff4e|\uff4f|\uff50|\uff51|\uff52|\uff53|\uff54|\uff55|\uff56|\uff57|\uff58|\uff59|\uff5a|\uff21|\uff22|\uff23|\uff24|\uff25|\uff26|\uff27|\uff28|\uff29|\uff2a|\uff2b|\uff2c|\uff2d|\uff2e|\uff2f|\uff30|\uff31|\uff32|\uff33|\uff35|\uff36|\uff37|\uff38|\uff39|\uff3a]+", sen[cur:])
        # if s:
        #     if s.group() != '':
        #         revlist.append(s.group())
        #     cur = cur + len(s.group())
        #     post = cur + span
        #     if post > len(sen):
        #         post = len(sen)
        if (sen[cur:post] in WordDic.keys()) or (cur + 1 == post):
            if sen[cur:post] != '':
                revlist.append(WordDic[sen[cur:post]])
            cur = post
            post = post + span
            if post > len(sen):
                post = len(sen)
        else:
            post -= 1
    return revlist

if __name__ == "__main__":

    InitStopword()
    WordSeg('s_questions.xls','s_questions_vec.xls')
    WordSeg('n_questions.xls','n_questions_vec.xls')





