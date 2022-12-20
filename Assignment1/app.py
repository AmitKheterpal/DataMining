
import pickle
import streamlit as st



pickle_in = open('classifier', 'rb')
classifier = pickle.load(pickle_in)

@st.cache()

# Define the function which will make the prediction using data
# inputs from users
def prediction(chlorides, total_sulfur_dioxide,
               density, pH, alcohol):
    
    # Make predictions
    prediction = classifier.predict(
        [[chlorides, total_sulfur_dioxide,
               density, pH, alcohol]])
    
    if prediction == 0:
        pred = 'Wine is of Bad Quality'
    else:
        pred = ' Congrats! This wine is of best Quality'
    return pred

# This is the main function in which we define our webpage
def main():
    
    # Create input fields
    chlorides = st.number_input("Number of chlorides(0-1)",
                                  min_value=0.000,
                                  max_value=1.000,
                                  value=0.01,
                                  step=0.001,
                                 )
    total_sulfur_dioxide = st.number_input("total sulfur dioxide Level (0-200)",
                              min_value=0,
                              max_value=200,
                              value=120,
                              step=10
                             )

    density = st.number_input("density Level (0-1)",
                              min_value=0.00,
                              max_value=1.00,
                              value=0.01,
                              step=0.001
                             )
    pH = st.number_input("PH Index (0-10)",
                          min_value=0.00,
                          max_value=10.00,
                          value=5.3,
                          step=0.01
                         )
    alcohol = st.number_input("alcohol in Years(max. 20)",
                          min_value=0,
                          max_value=20,
                          value=5,
                          step=1
                         )

    result = ""
    
    # When 'Predict' is clicked, make the prediction and store it
    if st.button("Predict"):
        result = prediction(chlorides, total_sulfur_dioxide,
               density, pH, alcohol)
        st.success(result)
       
if __name__=='__main__':
    main()
    
