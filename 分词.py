
string = "005496274521"
word_library = ['005','4','496','21','5','45','7']
word_list = []
def isaword(i,j):
    if i < j:
        # print(string[i:j])
        if string[i:j] in word_library:
            word_list.append(string[i:j])
            isaword(j, len(string))
        else:
            isaword(i, j - 1)
    else:
        if i < len(string):
            print(string[i]+'不存在')
            isaword(i+1,len(string))



i = 0
j = len(string)

isaword(i,j)
print(word_list)

