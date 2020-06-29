import pandas as pd
import random
from sklearn.model_selection import train_test_split
def getCreditData(split = 0.3):
    attributeNames = ['Status', 'Duration','History','Purpose','Amount','Savings','Employment','Installment rate','Personal status','Other debtors','Residence since','Property','Age','Plans', 'Housing','Credit num','Job','Maintenance','Telephone', 'foreign','class']
    data = pd.read_csv('german.data', names=attributeNames, index_col=False, delimiter=' ')
    data = data.append(data.loc[data['class']==2,:], ignore_index=True)
    data = data.iloc[:,:]
    data['Duration'] = pd.cut(data['Duration'], [x*6 for x in range(13)])
    data['Amount'] = pd.cut(data['Amount'], [x*1000 for x in range(20)])
    # data['Installment rate'] = pd.cut(data['Installment rate'], [x*1 for x in range(0,5)], include_lowest=False)
    # data['Residence since'] = pd.cut(data['Residence since'], [x*1 for x in range(0,5)], include_lowest=False)
    data['Age'] = pd.cut(data['Age'], [x*10 for x in range(1,9,1)])
    lab = data.pop('class')
    lab[lab==1] = "Yes"
    lab[lab==2] = "No"
    return train_test_split(data, lab, test_size=split, random_state=1, shuffle=True)

