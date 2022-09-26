def cltv():
    import streamlit as st
    import time
    import numpy as np
    import pandas as pd
    import lifetimes

    st.write("## Historial Approach")
    st.write('We are using a cohort model which groups customers based on the month of their purchase')

    @st.cache
    def load_data(path):
        df = pd.read_csv(path)
        return df


    df_1 = load_data('hist.csv')
    df_1 = df_1.iloc[: , 1:]
    if st.checkbox('Show Values'):
        st.write(df_1.style.background_gradient())

    st.write("## Predictive Approach")
    st.write("We are using a **Beta Geometric/Negative Binomial Distribution** on our recency and frequency values")

    df_2 = load_data('./events.csv')
    df_2 = df_2[df_2['event']=='transaction']

    import datetime
    times=[]
    for i in df_2['timestamp']:
        times.append(datetime.datetime.fromtimestamp(i//1000.0))
    df_2['timestamp']=times

    summary = lifetimes.utils.summary_data_from_transaction_data(df_2, 'visitorid', 'timestamp' )
    summary = summary.reset_index()

    one_time_buyers = round(sum(summary['frequency'] == 0)/float(len(summary))*(100),2)
    st.write("Percentage of customers purchase the item only once:", one_time_buyers ,"%")
    st.write('Histogram of the same')
    hist = np.histogram(summary['frequency'], bins=10, range=(0,10))[0]
    if st.checkbox('Show Histogram'):
        st.bar_chart(hist)

    st.write("### Expected Probabilities")
    st.write("This is the expected probability matrix of a visitor being alive (Visitor is worth pursuing)")
    bgf = lifetimes.BetaGeoFitter(penalizer_coef=0.0)
    bgf.fit(summary['frequency'], summary['recency'], summary['T'])

    from lifetimes.plotting import plot_probability_alive_matrix
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12,8))
    matrix = plot_probability_alive_matrix(bgf,cmap='plasma')
    if st.checkbox("Show Matrix"):
            st.pyplot(fig)



    st.write("### Final CLTV Values")
    t = st.text_input("Enter the time (in days) upto which predictions are required",30)
    t = int(t)
    summary['pred_num_txn'] = round(bgf.conditional_expected_number_of_purchases_up_to_time(t, summary['frequency'], summary['recency'], summary['T']),2)
    summary.sort_values(by='pred_num_txn', ascending=False).head(10).reset_index()
    summary.rename(columns={'pred_num_txn': 'CLTV'}, inplace=True)
    if st.checkbox("Show Final Values"):
        st.write(summary.sort_values('CLTV',ascending=False))

    x = st.text_input("Enter the visitorid for which CLTV Predictions are to be generated",1150086)
    x=int(x)

    st.write(" **CLTV for Visitor with id**",x, "is",summary[summary['visitorid']==x]['CLTV'].to_string(index=False))


    st.button("Re-run")