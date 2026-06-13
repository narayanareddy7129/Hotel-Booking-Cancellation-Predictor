import streamlit as st
import pandas as pd
import joblib
import sys
import os

# app.py lives in <project_root>/app/, while src/ and models/ live in <project_root>/
# Make '<project_root>/src' importable so joblib can find RawPreprocessor,
# FeatureEngineer, FeatureSelector, and ToDataFrame when unpickling full_pipeline.pkl
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

# Importing these makes the classes available under the 'feature_engineering'
# module name, which is what full_pipeline.pkl was pickled with.
from feature_engineering import FeatureEngineer, FeatureSelector, RawPreprocessor, ToDataFrame

PIPELINE_PATH = os.path.join(PROJECT_ROOT, 'models', 'full_pipeline.pkl')

st.set_page_config(page_title='Hotel Booking Cancellation Predictor', page_icon='H', layout='wide')

@st.cache_resource
def load_pipeline():
    return joblib.load(PIPELINE_PATH)

pipeline = load_pipeline()

st.title('Hotel Booking Cancellation Predictor')
st.markdown('Enter raw booking details below to predict whether the booking is likely to be **cancelled**.')
st.divider()

st.sidebar.header('Booking Details')

hotel               = st.sidebar.selectbox('Hotel Type', ['Resort Hotel', 'City Hotel'])
lead_time           = st.sidebar.slider('Lead Time (days)', 0, 737, 90)
arrival_month       = st.sidebar.selectbox('Arrival Month', [
    'January','February','March','April','May','June',
    'July','August','September','October','November','December'
])
stays_weekend       = st.sidebar.slider('Weekend Nights', 0, 20, 1)
stays_week          = st.sidebar.slider('Weekday Nights', 0, 50, 2)
adults              = st.sidebar.slider('Adults', 1, 10, 2)
children            = st.sidebar.selectbox('Children', [0, 1, 2, 3])
country             = st.sidebar.text_input('Country Code (e.g. PRT, GBR, USA)', 'PRT')
market_segment      = st.sidebar.selectbox('Market Segment', [
    'Direct','Corporate','Online TA','Offline TA/TO','Complementary','Groups','Aviation'
])
reserved_room_type  = st.sidebar.selectbox('Reserved Room Type', ['A','B','C','D','E','F','G','H','L'])
assigned_room_type  = st.sidebar.selectbox('Assigned Room Type', ['A','B','C','D','E','F','G','H','I','K','L'])
deposit_type        = st.sidebar.selectbox('Deposit Type', ['No Deposit','Refundable','Non Refund'])
customer_type       = st.sidebar.selectbox('Customer Type', ['Transient','Contract','Transient-Party','Group'])
adr                 = st.sidebar.number_input('Average Daily Rate (ADR)', 0.0, 600.0, 100.0)
prev_cancellations  = st.sidebar.slider('Previous Cancellations', 0, 26, 0)
prev_not_cancelled  = st.sidebar.slider('Previous Bookings Not Cancelled', 0, 72, 0)
days_waiting        = st.sidebar.slider('Days in Waiting List', 0, 391, 0)
is_repeated_guest   = st.sidebar.selectbox('Repeated Guest?', [0, 1], format_func=lambda x: 'Yes' if x else 'No')

# Build a single-row raw DataFrame. full_pipeline.predict() handles ALL preprocessing
# (lead_time_log, adr_capped, OHE/scaling, feature engineering, feature selection).
input_dict = {
    'hotel':                          hotel,
    'lead_time':                      lead_time,
    'arrival_date_month':             arrival_month,
    'stays_in_weekend_nights':        stays_weekend,
    'stays_in_week_nights':           stays_week,
    'adults':                         adults,
    'children':                       children,
    'country':                        country,
    'market_segment':                 market_segment,
    'reserved_room_type':             reserved_room_type,
    'assigned_room_type':             assigned_room_type,
    'deposit_type':                   deposit_type,
    'customer_type':                  customer_type,
    'adr':                            adr,
    'previous_cancellations':         prev_cancellations,
    'previous_bookings_not_canceled': prev_not_cancelled,
    'days_in_waiting_list':           days_waiting,
    'is_repeated_guest':              is_repeated_guest,
}

input_df = pd.DataFrame([input_dict])

if st.sidebar.button('Predict', type='primary'):
    pred  = pipeline.predict(input_df)[0]
    proba = pipeline.predict_proba(input_df)[0][1]

    st.subheader('Prediction Result')
    col1, col2, col3 = st.columns(3)
    with col1:
        if pred == 1:
            st.error('Likely to CANCEL')
        else:
            st.success('Likely NOT to Cancel')
    with col2:
        st.metric('Cancellation Probability', f'{proba*100:.1f}%')
    with col3:
        st.metric('Confidence', 'High' if abs(proba - 0.5) > 0.3 else 'Medium')

    st.divider()
    st.subheader('Booking Summary')
    st.dataframe(input_df.T.rename(columns={0: 'Value'}), use_container_width=True)
else:
    st.info('Fill in the booking details on the left and click Predict.')
