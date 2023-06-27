# SLRM
import pandas as pd
import numpy as np
import seaborn as sns
data = pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx")

#Rename columns
column_names = {'X1':'Relative_Compactness',
                'X2': 'Surface_Area', 
                'X3':  'Wall_Area', 'X4': 'Roof_Area',
                'X5': 'Overall_Height',
                'X6': 'Orientation',
                'X7': 'Glazing_Area', 
                'X8': 'Glazing_Area_Distribution', 
                'Y1': 'Heating_Load',
                'Y2': 'Cooling_Load'}

data = data.rename(columns = column_names)

#Select a sample of the dataset
slr = data[['Relative_Compactness', 'Cooling_Load']].sample(15, random_state=2)
sns.regplot(x="Relative_Compactness", y="Cooling_Load", data=slr)
