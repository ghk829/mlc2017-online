import pandas as pd
import numpy as np
sample = pd.read_csv("./sampleSubmission.csv")
pred = pd.read_csv("./predictions.csv")
df=pd.merge(sample,pred,on="Id")
print "Precision : " + str(len(np.where(df['Category_x']==df['Category_y'])[0])/np.float(len(sample)))