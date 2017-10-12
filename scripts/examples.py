
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

import os
os.chdir('/Users/abhijit/ARAASTAT/Teaching/FreddieMacFinal/data')

lupus = pd.read_csv('lupus.csv')
lupus.head()
lupus['lupus'].value_counts()
lupus.columns
lupus.shape
dat = lupus.loc[:,['age','male','dead','lupus','ventilator']]
dat_vent= dat[dat['ventilator']==1]
dat_novent = dat[dat['ventilator']==0]

rf = RandomForestRegressor(n_estimators=500)
rf_vent = rf.fit(dat_vent.drop('dead', axis=1),dat_vent['dead'])
rf_novent = rf.fit(dat_novent.drop('dead', axis=1), dat_novent['dead'])

xb = xgb.XGBRegressor(n_estimators = 20)
xb_vent = xb.fit(dat_vent.drop('dead', axis=1), dat_vent['dead'])
xb_novent = xb.fit(dat_novent.drop('dead', axis=1), dat_novent['dead'])

from sklearn.model_selection import cross_val_predict
rng = np.random.RandomState(35)
p_vent_dvent = cross_val_predict(xb_vent, dat_vent.drop('dead', axis=1),
    dat_vent['dead'], cv=3)
p_vent_dnovent= xb_vent.predict(dat_novent.drop('dead',axis=1))
p_novent_dnovent = cross_val_predict(xb_novent, dat_novent.drop('dead',axis=1),
    dat_novent['dead'], cv=3)
p_novent_dvent = xb_novent.predict(dat_vent.drop('dead', axis=1))

p_vent = np.concatenate([p_vent_dvent,p_vent_dnovent])
p_novent= np.concatenate([p_novent_dvent, p_novent_dnovent])

eff_vent = p_vent - p_novent
pd.Series(eff_vent).describe()

dat1 = dat_vent.append(dat_novent)
plt.scatter(dat1['age'], eff_vent)
