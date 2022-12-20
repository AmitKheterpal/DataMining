import streamlit as st
import pickle
import numpy as np

pickle_in = open('classifier1', 'rb')
pkl_in = pickle.load(pickle_in)

lr_model = pkl_in[0] 
knn = pkl_in[1]
means = pkl_in[2]
std = pkl_in[3]


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
        pred = 'Sorry! Your Loan is likely to be rejected'
    else:
        pred = 'Congrats! Your Loan is likely to be approved'
    return pred



def set_citizen_type(citizen_type):
    citizen_bybirth = 0
    citizen_temporary = 0
    citizen_other = 0
    
    if citizen_type =='bybirth':
        citizen_bybirth = 1
    elif citizen_type =='temporary':
        citizen_temporary = 1
    else: citizen_other = 1
    
    return citizen_bybirth,citizen_temporary,citizen_other

def set_employment_type(emptype):
    emp_industrial=0
    emp_materials=0
    emp_consumer_services=0
    emp_healthcare=0
    emp_financials=0
    emp_utilities=0
    emp_education=0
    
    if emptype =='industrial':
        emp_industrial = 1
    elif emptype =='materials':
        emp_materials =1
    elif emptype =='consumer_services':
        emp_consumer_services =1
    elif emptype =='healthcare':
        emp_healthcare =1
    elif emptype =='financials':
        emp_financials =1
    elif emptype =='utilities':
        emp_utilities = 1
    elif emptype =='education':
        emp_education = 1
        
    return emp_industrial,emp_materials,emp_consumer_services,emp_healthcare,emp_financials,emp_utilities,emp_education
        
    

def main():
    
    # Create input fields
    st.markdown("<h1 style='text-align: center; color: red;'>Loan Approval Indicator</h1>", unsafe_allow_html=True)
    
    debt = st.number_input("How much debt you have (in 000's of Dollars)?",
                                  min_value=0.0,
                                  max_value=30.0,
                                  value=20.5,
                                  step=.1,
                                 )
    
    married = st.number_input("Are you married?(0-1)",
                                  min_value=0,
                                  max_value=1,
                                  value=1,
                                  step=1,
                                 )
    
    bank_customer = st.number_input("Are you a bank customer ?(0-1)",
                                  min_value=0,
                                  max_value=1,
                                  value=1,
                                  step=1,
                                 )
    employed = st.number_input("Are you Employed ?(0-1)",
                                  min_value=0,
                                  max_value=1,
                                  value=1,
                                  step=1,
                                 )
    years_employed = Income = st.number_input("How many years of Experience you got? ",
                          min_value=0,
                              max_value=30,
                              value=20,
                              step=10
                         )
    Income = st.number_input("Annual Income(in $) ",
                          min_value=0,
                              max_value=100000,
                              value=50000,
                              step=1000
                         )
    emptype = st.selectbox("Select Employee Type",['industrial','materials','consumer_services','healthcare','financials','utilities','education'])
    emp_industrial,emp_materials,emp_consumer_services,emp_healthcare,emp_financials,emp_utilities,emp_education = set_employment_type(emptype)
    
    prior_default = st.number_input("Do you have any prior Defaults ?(0-1)",
                                  min_value=0,
                                  max_value=1,
                                  value=1,
                                  step=1,
                                 )
    citizen_type = st.selectbox("Select Citizen type",['bybirth','temporary', 'other'])
    citizen_bybirth,citizen_temporary,citizen_other = set_citizen_type(citizen_type)
    
    credit_score = st.number_input("Enter your Credit Score",
                                  min_value=0,
                                  max_value=100,
                                  value=50,
                                  step=10,
                                 )
    
    drivers_license = st.number_input("Do you have a Driver License ?(0-1)",
                                  min_value=0,
                                  max_value=1,
                                  value=1,
                                  step=1,
                                 )
  
    model_type = st.sidebar.selectbox("Model type", ['KNN','LR'])
    prediction_probability = [st.sidebar.slider("Probablity Threshold",0.0, 1.0, 0.3, 0.01)]
    
    #scaling data with test stats used for training
    
    debt_norm = (debt - means[0])/std[0]
    married_norm = (married -means[0])/std[0]
    bank_customer_norm = (bank_customer -means[0])/std[0]
    emp_industrial_norm = (emp_industrial -means[0])/std[0]
    emp_materials_norm = (emp_materials -means[0])/std[0]
    emp_consumer_services_norm = (emp_consumer_services -means[0])/std[0]
    emp_healthcare_norm = (emp_healthcare -means[0])/std[0]
    emp_financials_norm = (emp_financials -means[0])/std[0]
    emp_utilities_norm = (emp_utilities -means[0])/std[0]
    emp_education_norm = (emp_education -means[0])/std[0]
    years_employed_norm = (years_employed -means[0])/std[0]
    prior_default_norm = (prior_default -means[0])/std[0]
    employed_norm = (employed -means[0])/std[0]
    credit_score_norm = (credit_score -means[0])/std[0]
    drivers_license_norm = (drivers_license -means[0])/std[0]
    citizen_bybirth_norm = (citizen_bybirth -means[0])/std[0]
    citizen_other_norm = (citizen_other -means[0])/std[0]
    citizen_temporary_norm = (citizen_temporary -means[0])/std[0]
    Income_norm = np.log1p(Income)
    
    
#     input_data = [debt,married,bank_customer,
#        emp_industrial,emp_materials,emp_consumer_services,
#        emp_healthcare,emp_financials,emp_utilities,emp_education,
#        years_employed,prior_default,
#        employed,credit_score,drivers_license,citizen_bybirth,
#        citizen_other,citizen_temporary]
    
    
#     input_list1 = np.divide(np.subtract(input_data,means),std).tolist()

#     debt_norm = input_list1[0]
#     married_norm =input_list1[1]
#     bank_customer_norm = input_list1[2]
#     emp_industrial_norm = input_list1[3]
#     emp_materials_norm = input_list1[4]
#     emp_consumer_services_norm = input_list1[5]
#     emp_healthcare_norm = input_list1[6]
#     emp_financials_norm = input_list1[7]
#     emp_utilities_norm = input_list1[8]
#     emp_education_norm = input_list1[9]
#     years_employed_norm = input_list1[10]
#     prior_default_norm = input_list1[11]
#     employed_norm = input_list1[12]
#     credit_score_norm = input_list1[13]
#     drivers_license_norm = input_list1[14]
#     citizen_bybirth_norm = input_list1[15]
#     citizen_other_norm = input_list1[16]
#     citizen_temporary_norm = input_list1[17]
#     Income_norm = np.log1p(Income)
    result = ""
#     with st.sidebar:
#         result = prediction(sch_dep_time,carrier_delta,carrier_us,
#        carrier_envoy,carrier_continental,carrier_discovery,
#        carrier_other,dest_jfk,dest_ewr,dest_lga,distance,
#        origin_dca,origin_iad,origin_bwi,bad_weather,Monday,
#        Tuesday,Wednesday,Thursday,Friday,Saturday,Sunday, prediction_probability, model_type)
#         st.success(result)
        

    
    # When 'Predict' is clicked, make the prediction and store it
    if st.button("Predict"):
        result = prediction([debt_norm,married_norm,bank_customer_norm,emp_industrial_norm,
                             emp_materials_norm,emp_consumer_services_norm,emp_healthcare_norm,
                             emp_financials_norm,emp_utilities_norm,emp_education_norm,
                            years_employed_norm,prior_default_norm,employed_norm,credit_score_norm,drivers_license_norm,
                            citizen_bybirth_norm,citizen_other_norm,citizen_temporary_norm,Income_norm], prediction_probability, model_type)
        if result =='Congrats! Your Loan is likely to be approved':
            st.balloons()
        st.success(result)
        
    if st.button('Show ROC curve'):
        if model_type =='KNN':
            st.image('KNN_ROC_p3.png',caption='KNN ROC image')
        elif model_type =='LR':
            st.image('LR_ROC_p3.png',caption='LR ROC image')
       
if __name__=='__main__':
    main()
