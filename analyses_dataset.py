import pandas as pd
import numpy as np


class Info:
    
    def __init__(self, csv_file):
        self.csv_file = csv_file

    def analizer(self):    

        info_col = ["name", "type", "non_missing", "n_class", "classes", "counts","ratios", "distribution"]

        data_csv = "./cxr_clinical.csv"

        df = pd.read_csv(data_csv)

        variable = df.columns
        
        var_name        = var
        var_type        = df[var].dtypes
        var_non_missing = df[var].notna().sum()
        var_n_class     = df[var].nunique()
        var_distribution = None

        if var_n_class < 10:

            var_classes = df[var].dropna().unique().tolist()
            var_classes.sort()
            var_counts = []
            var_ratios = []

            if var_type == int:
                
                var_classes = list(map(int, df[var].dropna().unique().tolist()))
                var_classes.sort()

            
            for i in var_classes:
                var_counts.append(len(df.loc[df[var] == i]))
                var_ratios.append(round(len(df.loc[df[var] == i])/var_non_missing*100,2))
            
            
        else:

            var_n_class = None
            var_classes = None
            var_counts = None
            var_ratios = None

            if var_type != object:

                values = df[var].dropna()
                values = np.sort(values)
                var_distribution = [round(min(values),2), 
                                    round(np.percentile(values, 5),2), 
                                    round(np.percentile(values, 10),2), 
                                    round(np.percentile(values, 25),2), 
                                    round(np.percentile(values, 50),2),
                                    round(np.percentile(values, 75),2),
                                    round(np.percentile(values, 90),2),
                                    round(np.percentile(values, 95),2),
                                    round(max(values),2)]

        return ([var_name, var_type, var_non_missing, var_n_class, var_classes, var_counts, var_ratios, var_distribution])

    var_character = []



    if __name__ == '__main__':

        for i in variable:
        var_character.append(info(i))

        a = pd.DataFrame(var_character, columns = info_col)

        print (a)

        a.to_csv('./a.csv')