import streamlit as st
st.title('Ecommerce Customer Intelligence - AI')
functionality=st.sidebar.selectbox("Choose a function",("-","Recommendation","Forecasting","Customer Segmentation","CLTV Predictions","Dynamic Pricing"))



if functionality=="-":
  original_title = '<p style="font-family:Arial; color:Black; font-size: 25px;">This is an app built with the help of streamlit having various functionalities</p>'
  st.markdown(original_title, unsafe_allow_html=True)
  original_title = '<p style="font-family:Arial; color:Black; font-size: 25px;">Select a functionality fron the menu on the left </p>'
  st.markdown(original_title, unsafe_allow_html=True)
  

if functionality=='Recommendation':
  recommend()

elif functionality== 'Forecasting':
  forecast()

elif functionality=="Customer Segmentation":
  original_title = '<p style="font-family:Arial; color:Black; font-size: 30px;">Customer Segmentation</p>'
  st.markdown(original_title, unsafe_allow_html=True)  
  st.write("This function gives an option to view the dataframes generated using the RFM strategy along with dataframes about visitors belonging to a particular segment")
  cust_seg()

elif functionality=="CLTV Predictions":
  original_title = '<p style="font-family:Arial; color:Black; font-size: 30px;">CLTV Predictions</p>'
  st.markdown(original_title, unsafe_allow_html=True)
  st.write("This function generates the CLTV of the visitors based on a historical and predictive approach.")
  cltv()
elif functionality=="Dynamic Pricing":

  original_title = '<p style="font-family:Arial; color:Black; font-size: 30px;">Product Dynamic Pricing</p>'
  st.markdown(original_title, unsafe_allow_html=True)
  prodid=st.text_input("Provide Product ID",25)
  dynamic_pricing(int(prodid))
