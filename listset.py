import os
import pandas as pd


source_file = "./cxr_clinical_new.csv"
dest_dir = "./data"
input_file = "file_name"
outcome = "age"
outcome2 = "sex"
even_class = False
train_ratio = 80
nclass = 4



df = pd.read_csv(source_file)

print ("lenth1:", len(df))

df = df.loc[:, [input_file, outcome, outcome2]]
df.dropna(inplace = True)

print ("lenth2:", len(df))

df.rename(columns = {input_file : 'dcm_path' , outcome : 'age', outcome2 : 'sex' }, inplace = True)

if even_class:
    classes = df['score'].unique()
    
    var_counts = []

    for i in classes:
        var_counts.append(len(df.loc[df['score'] == i]))
    
    min_num = min(var_counts)
    index = var_counts.index(min_num)

    train_mini = pd.DataFrame(columns=['dcm_path','score'])
    test_mini  = pd.DataFrame(columns=['dcm_path','score'])

    for i in classes:
        df1 = df[df['score'] == i]
        df1 = df1.sample(frac=1).reset_index(drop=True)
        df1 = df1.iloc[:min_num,:]

        row_cut = int(len(df1)*train_ratio/100) 

        train_mini = pd.concat([train_mini, df1.iloc[:row_cut,:]], ignore_index=True)
        test_mini  = pd.concat([test_mini,  df1.iloc[row_cut:,:]], ignore_index=True)

    train_mini.to_csv("./train.csv")
    test_mini.to_csv("./test.csv")

    train_mini.to_parquet(dest_dir+'/train_dataset_cac.parquet')
    test_mini.to_parquet(dest_dir+'/test_dataset_cac.parquet')


else: 
    row_cut = int(len(df)*train_ratio/100)

    df = df.sample(frac=1).reset_index(drop=True)

    df_training = df.iloc[:row_cut,:]
    df_test     = df.iloc[row_cut:,:]

    print (df_training)

    df_training.to_parquet(dest_dir+'/train_dataset_cac.parquet')
    df_test.to_parquet(dest_dir+'/test_dataset_cac.parquet')