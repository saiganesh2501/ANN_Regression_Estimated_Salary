import streamlit as st
import pandas as pd
import tensorflow as tf
from keras.models import load_model
import pickle
st.title('ANN_Regression_ExpectedSalary')

# load the model
model=load_model('regression_model.keras')

#load the pickle files
with open('Gender_label_encoder.pkl','rb') as file:
    gender_encode=pickle.load(file)
with open('Geography_one_hot.pkl','rb') as file:
    geo_encode=pickle.load(file)
with open('Standard_scaler.pkl','rb') as file:
    standard_scaler=pickle.load(file)

## take the input from user
geography=st.selectbox('Geography',geo_encode.categories_[0])
gender=st.selectbox('Gender',gender_encode.classes_)
age=st.slider('Age',18,92)
creditscore=st.number_input('CreditScore')
tenure=st.slider('Tenure',1,10)
balance=st.number_input('Balance')
num_of_products=st.slider('NumOfProducts',1,3)
has_cr_card=st.selectbox('HasCrCard',[0,1])
is_active_member=st.selectbox('IsActiveMember',[0,1])
exited=st.selectbox('Exited',[1,0])

#Convert this data into DataFrame type

data=pd.DataFrame({
    'CreditScore': [creditscore],
    'Gender' : [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'Exited': [exited],
    'Geography' :[geography],
})

#label encoding gender column
data['Gender']=gender_encode.transform(data['Gender'])

# one hot encoding geography column

geo_encoded=geo_encode.transform(data[['Geography']]).toarray()
geo_encoded=pd.DataFrame(geo_encoded,columns=geo_encode.get_feature_names_out())

#now concat the whole data frame by dropping Geography column

final_data=pd.concat([data.drop(['Geography'],axis=1),geo_encoded],axis=1)
final_scaled_data=standard_scaler.transform(final_data)
#predict the output
prediction=model.predict(final_scaled_data)

st.write(f'ExpectedSalary of the customer is {prediction[0][0]}')