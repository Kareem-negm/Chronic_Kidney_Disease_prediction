
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import webbrowser
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st
import pickle



st.write("""
# Kidney disease prediction application
 Detect if someone has Kidney Disease using Artificial intelligence !
""")


#Get the data
df = pd.read_csv("https://github.com/Kareem-negm/Chronic_Kidney_Disease_prediction/blob/main/Chronic_KIdney_Disease_data.csv")
#Show the data as a table (you can also use st.write(df))

st.image("https://github.com/Kareem-negm/Chronic_Kidney_Disease_prediction/blob/main/images/img.jpg")

st.subheader('Data Information:')

st.dataframe(df)
#Get statistics on the data
st.write(df.describe())
# Show the data as a chart.
st.set_option('deprecation.showPyplotGlobalUse', False)
f,ax = plt.subplots(figsize=(20, 20))
st.write(sns.heatmap(df.corr(),annot=True))
st.pyplot()


#Get the feature input from the user
def get_user_input():
    age = st.sidebar.slider('age in years', 1, 90, 29)
    bp = st.sidebar.slider("Blood Pressure(numerical) bp in mm/Hg)",50,180,100)
    sg = st.sidebar.selectbox("Specific Gravity(nominal) sg - (1.005,1.010,1.015,1.020,1.025)",(1.005,1.010,1.015,1.020,1.025))
    al = st.sidebar.selectbox('Albumin(nominal)al - (0,1,2,3,4,5)',(0,1,2,3,4,5))
    su = st.sidebar.selectbox(' Sugar(nominal) su - (0,1,2,3,4,5) ',(0,1,2,3,4,5))
    rbc= st.sidebar.selectbox('Red Blood Cells(nominal) rbc - (normal=0,abnormal=1)',(0,1))
    pc = st.sidebar.selectbox("Pus Cell (nominal)pc - (normal=0,abnormal=1)",(0,1))
    pcc = st.sidebar.selectbox('Pus Cell clumps(nominal)pcc - (notpresent =0,present =1)',(0,1))
    ba = st.sidebar.selectbox('Bacteria(nominal) ba - (notpresent =0,present =1)', (0, 1))
    bgr = st.sidebar.slider('Blood Glucose Random(numerical) bgr in mgs/d',22,490,200)
    bu  = st.sidebar.slider('Blood Urea(numerical) bu in mgs/dl',1,391,200)
    sc  = st.sidebar.slider('Serum Creatinine(numerical) sc in mgs/dl',0,76,40)
    sod  = st.sidebar.slider('Sodium(numerical) sod in mEq/L',4,163,100)
    pot  = st.sidebar.slider('Potassium(numerical) pot in mEq/L',2,47,30)
    hemo = st.sidebar.slider('Haemoglobin(numerical) hemo in gms', 3,17,10)
    pcv = st.sidebar.slider("Packed Cell Volume(numerical)",9,54,35)
    wc = st.sidebar.slider('White Blood Cell Count(numerical) wc in cells/cumm',2200,26400,3300)
    rc  = st.sidebar.slider('Red Blood Cell Count(numerical) rc in millions/cmm', 2,8)
    htn = st.sidebar.selectbox('Hypertension(nominal) htn - (yes=1,no=0)',(0,1))
    dm  = st.sidebar.selectbox("Diabetes Mellitus(nominal) dm - (yes=1,no=0)",(0,1))
    cad  = st.sidebar.selectbox('Coronary Artery Disease(nominal) cad - (good=1,poor=0)',(0,1))
    appet = st.sidebar.selectbox('Appetite(nominal) ppet - (good=1,poor=0)', (0, 1))
    pe = st.sidebar.selectbox('Pedal Edema(nominal) pe - (yes=1,no=0)',(0,1))
    ane  = st.sidebar.selectbox('Anemia(nominal)ane - (yes=1,no=0)',(0,1))
   
    user_data = {'age ': age ,
              'bp ': bp ,
                 'sg ': sg ,
                 'al ': al ,
                 'su ': su ,
              'rbc ': rbc ,
              'pc ': pc ,
                 'pcc ': pcc ,
                 'ba  ': ba  ,
                 'bgr  ': bgr  ,
                 'bu  ': bu  ,
                 'sc  ': sc  ,
                 'sod  ': sod  ,
                 'pot ': pot ,
              'hemo ': hemo ,
                 'pcv ': pcv ,
                 'wc  ': wc  ,
                 'rc  ': rc  ,
                 'htn  ': htn  ,
                 'dm  ': dm  ,
                 'cad  ': cad  ,
                 'appet  ': appet  ,
                 'pe  ': pe  ,
                 'ane  ': ane  ,
                 }
    features = pd.DataFrame(user_data, index=[0])
    return features

user_input = get_user_input()
st.subheader('User Input :')
st.write(user_input)


load_clf = pickle.load(open("https://github.com/Kareem-negm/Chronic_Kidney_Disease_prediction/blob/main/klney_clf.pkl", 'rb'))

prediction = load_clf.predict(user_input)
st.subheader('Classification: ')
st.write(prediction)

st.subheader('predicted probabilities: ')
prediction_proba = load_clf.predict_proba(user_input)
st.write(prediction_proba)


if prediction==0:
    
    st.subheader('you dont have Kidney disease , Enjoy and preserve your life')
else:
    st.subheader('you have Kidney disease , please Click on the next button to go to the tips page and go to the doctor as soon as possible')
    
    url = 'https://www.niddk.nih.gov/health-information/kidney-disease/chronic-kidney-disease-ckd/managing'

    if st.button('Open browser'):
        webbrowser.open_new_tab(url)
 
    
