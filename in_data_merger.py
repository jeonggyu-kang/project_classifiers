import os
import pandas as pd 

np_path = "./data/cxr/cxr4"
list_file = "./cac_2020_3q4q.csv"

all_files = []

for (path, dirs, files) in os.walk(np_path):

    for file in files:
        path_file = path[len(np_path):] + '/' + file

        if path_file[-3:] == "npy":
            all_files.append(path_file)

print (len(all_files))

df = pd.read_csv(list_file, encoding="euc-kr")


def string(x):

    len1 = len(str(x))
    full_name = "./cxr/cac4/" + '0' * (8-len1)  + str(x) + ".dcm"

    return full_name

def topath(x):

    full_name = "./cxr/cac4/" + x[1:9] + ".dcm"

    return full_name



df['file_name'] = df['file_name'].apply(lambda x: string(x))
all_files2 = list(map(topath, all_files))


all_files3 = pd.DataFrame(zip(all_files2, all_files2), columns = ['file_name', 'file_name2'])
print (type(all_files3))

df2 = pd.merge(df, all_files3, how='outer', on = 'file_name' )

print (df2)

index1 = df2[df2['file_name2'] == None].index

df2.drop(index1)

df2.to_csv("cac_merged.csv")