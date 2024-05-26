import streamlit as st 
from streamlit_option_menu import option_menu
import pickle
import numpy as np
import pandas as pd

# Loading the saved model 
# def predict_model(x):
#     with open(r"C:\Users\barathy\rfr.pkl", 'rb') as f:
#         model = pickle.load(f)

#         return model
#     predict=predict_model



# page configurations

st.set_page_config(
    page_title="Singapore Resale Flat Prices",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
        
    )
with st.sidebar:   
    selected = option_menu("Main Menu", ["Home", 'Predict Prices'], 
        icons=['house', 'gear'], menu_icon="cast", default_index=1)
        
        


if selected=="Home":
    def col():

        st.markdown("# :blue[Singapore Resale Flat Prices Prediction]")
        st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
        st.markdown("### :blue[Technologies :] Python, Pandas, Numpy, Scikit-Learn, Streamlit, Python scripting, "
                    "Machine Learning, Data Preprocessing, Visualization, EDA, Model Building, Data Wrangling, "
                    "Model Deployment")
        st.markdown("### :blue[Overview :] This project aims to construct a machine learning model and implement "
                    "it as a user-friendly online application in order to provide accurate predictions about the "
                    "resale values of apartments in Singapore. This prediction model will be based on past transactions "
                    "involving resale flats, and its goal is to aid both future buyers and sellers in evaluating the "
                    "worth of a flat after it has been previously resold. Resale prices are influenced by a wide variety "
                    "of criteria, including location, the kind of apartment, the total square footage, and the length "
                    "of the lease. The provision of customers with an expected resale price based on these criteria is "
                    "one of the ways in which a predictive model may assist in the overcoming of these obstacles.")
        st.markdown("### :blue[Domain :] Real Estate")


    col()

# reading the csv file 
data =pd.read_csv(r"C:\Users\barathy\singapore Dataset.csv")

    # listing out the features from the dataset
# def Predict_prices():
if selected=="Predict Prices":
    st.markdown("# :blue[Predicting Results based on Trained Models]")
    
    with st.form('form1'):
        town = st.selectbox("Town" ,options =data.town.unique())
        flat_type =st.selectbox("Flat Type" , options=data.flat_type.unique())
        storeys =st.selectbox("Storey Range" , options=data.storey_range.unique())
        flat_model =st.selectbox("Flat Model", options=data.flat_model.unique())
        yr =st.selectbox("Built Year" ,options=data.year.unique())
        floor_area =st.number_input("Enter Floor Area (EX : 3..3456)")

        
        submit_button = st.form_submit_button(label="PREDICT RESALE PRICE")

        if submit_button:
                with open(r"C:\Users\barathy\dtr_model.pkl", 'rb') as file:
                    model = pickle.load(file)
                with open(r"C:\Users\barathy\scalerModel_fit.pkl",'rb') as file:
                    scaler =pickle.load(file)



                    user_data = np.array([[town,flat_type,storeys,flat_model,yr,floor_area]])
                    
                    scl_trans =scaler.transform(user_data)
                    prediction = model.predict(scl_trans)
                    predicted_price = prediction[0]
                    predicted_price=np.exp(predicted_price)
                    st.write(f"The Predicted Price is :  {round(predicted_price)}")







   




