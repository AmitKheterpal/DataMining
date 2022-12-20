import streamlit as st
import pickle
import numpy as np

pickle_in = open('classifier1', 'rb')
pkl_in = pickle.load(pickle_in)

lr_model = pkl_in[0] 
knn = pkl_in[1]
means = pkl_in[2]
std = pkl_in[3]

# pickle_in2 = open('classifier2', 'rb')
# knn = pickle.load(pickle_in2)


@st.cache()

# Define the function which will make the prediction using data
# inputs from users
def prediction(input_list, prediction_probability, model_type ='LR'):
    # Make predictions
    if model_type =='LR':
        prediction = (lr_model.predict_proba([input_list])[::,1]>= prediction_probability ).astype(int)
    elif model_type =='KNN':
        prediction = (knn.predict_proba([input_list])[::,1]>= prediction_probability ).astype(int)
    
    if prediction == 0:
        pred = 'Task is incomplete'
    else:
        pred = ' Congrats! This task is already completed'
    return pred
def main():
    st.markdown("<h1 style='text-align: center; color: red;'>HR Task Indicator</h1>", unsafe_allow_html=True)

    
    # Create input fields
    employee_experience = st.number_input("Experience of employee(0-20 yrs)",
                                  min_value=0,
                                  max_value=20,
                                  value=2,
                                  step=1,
                                 )
    training_level4 = st.number_input("Select training level(4) (0-1)",
                              min_value=0,
                              max_value=1,
                              value=1,
                              step=1
                             )

    training_level6 = st.number_input("Select training level(6) (0-1)",
                              min_value=0,
                              max_value=1,
                              value=1,
                              step=1
                             )
    training_level8 = st.number_input("Select training level(8)",
                          min_value=0,
                              max_value=1,
                              value=1,
                              step=1
                         )
    model_type = st.sidebar.selectbox("Model type", ['KNN','LR'])
    prediction_probability = [st.sidebar.slider("Probablity Threshold",0.0, 1.0, 0.3, 0.01)]
    
    
    #scaling data with test stats used for training
    
    
    input_data = [employee_experience,training_level4,training_level6,training_level8]
    input_list1 = np.divide(np.subtract(input_data,means),std).tolist()
    
    
    
    result = ""
#     with st.sidebar:
#         result = prediction(employee_experience,training_level4,training_level6,training_level8, prediction_probability, model_type)
#         st.success(result)
        

    
    # When 'Predict' is clicked, make the prediction and store it
    if st.button("Predict"):
        result = prediction(input_list1, prediction_probability, model_type)
        if result =='Congrats! This task is already completed':
            st.balloons()
        st.success(result)
        
    if st.button('Show ROC curve'):
        if model_type =='KNN':
            st.image('KNN_ROC.png',caption='KNN ROC image')
        elif model_type =='LR':
            st.image('LR_ROC.png',caption='LR ROC image')
       
if __name__=='__main__':
    main()
