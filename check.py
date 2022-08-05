import pydicom
import pandas as pd



def to_df(fname):

    dco = pydicom.dcmread(fname)

    element = []
    for elem in dco:
        element.append(elem.keyword)

    element = [v for v in element if v]
    element.remove('PixelData')

    # print (element)

    dic = {}
    for i in range(len(element)):
        dic.update({element[i]: dco[element[i]].value})

    # print (dic)

    a = pd.DataFrame(list(dic.items()), columns=['key','values'])

    return a

i0005548 = to_df("/media/compu/ssd2t/cxr_all/cxr/cac0/I0005548.dcm")
i0005547 = to_df("/media/compu/ssd2t/cxr_all/cxr/cac0/I0005547.dcm")

c = pd.merge(i0005548,i0005547,left_on='key',right_on='key', how='outer')

c.to_csv('./comparison.csv')

