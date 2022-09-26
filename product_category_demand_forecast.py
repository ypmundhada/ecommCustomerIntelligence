def forecast():
    #!/usr/bin/env python
    # coding: utf-8

    # In[10]:


    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import streamlit as st




    # cat=pd.read_hdf("cat.hf5",key="cat")
    # df_cat_merge=pd.read_hdf("df_cat_merge.hf5",key="df_cat_merge")
    df_cat_merge_group=pd.read_hdf("df_cat_merge_group.hf5",key="df_cat_merge_group")
    # prod=pd.read_hdf("prod.hf5",key="prod")
    prod_group=pd.read_hdf("prod_group.hf5",key="prod_group")





    def category_demand_forecast(category_id):
      import datetime
      from statsmodels.tsa.stattools import adfuller
      from statsmodels.tsa.seasonal import seasonal_decompose
      from sklearn.metrics import mean_absolute_error, mean_squared_error
      from statsmodels.tsa.holtwinters import ExponentialSmoothing
      def test_stationarity(df, ts):
        rolmean = df[ts].rolling(window = 12, center = False).mean()
        rolstd = df[ts].rolling(window = 12, center = False).std()

        # Plot rolling statistics:
        # orig = plt.plot(df[ts], 
        #                 color = 'blue', 
        #                 label = 'Original')
        # mean = plt.plot(rolmean, 
        #                 color = 'red', 
        #                 label = 'Rolling Mean')
        # std = plt.plot(rolstd, 
        #                color = 'black', 
        #                label = 'Rolling Std')
        # plt.legend(loc = 'best')
        # plt.title('Rolling Mean & Standard Deviation for %s' %(ts))
        # plt.xticks(rotation = 45)
        # plt.show(block = False)
        # plt.close()

        # Perform Dickey-Fuller test:
        # Null Hypothesis (H_0): time series is not stationary
        # Alternate Hypothesis (H_1): time series is stationary
        # print('Results of Dickey-Fuller Test:')
        dftest = adfuller(df[ts], 
                          autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], 
                             index = ['Test Statistic',
                                      'p-value',
                                      '# Lags Used',
                                      'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        # print(dfoutput)
        return dfoutput[1]

      def ts_set(df,ts,cat):
        df_temp = df[df.categoryid==cat].reset_index()
        if df[df.categoryid==cat].index[0] != datetime.datetime.strptime('2015-05-03','%Y-%m-%d').date():
          df_temp = pd.DataFrame(np.insert(df_temp.values,0,values=[datetime.datetime.strptime('2015-05-03','%Y-%m-%d').date(),cat,0],axis=0))
        if df[df.categoryid==cat].index[-1] != datetime.datetime.strptime('2015-09-18','%Y-%m-%d').date():
          df_temp = pd.DataFrame(np.insert(df_temp.values,df_temp.shape[0],values=[datetime.datetime.strptime('2015-09-18','%Y-%m-%d').date(),cat,0],axis=0))

        df_temp.rename(columns={0:'timestamp',1:'catid',2:'wt_act'},inplace=True)
        df_temp.set_index('timestamp',inplace=True)
        df_temp = df_temp.asfreq(freq='1D')
        df_temp.catid = cat

        df_temp.wt_act.fillna(0.0,inplace=True)
        df_temp['wt_act'].replace({0:(np.round(df_temp.wt_act.mean(),2))},inplace=True)
        # print(df_temp)
        return df_temp
      def plot_transformed_data(df, ts, ts_transform):
      # Plot time series data
        f, ax = plt.subplots(1,1)
        ax.plot(df[ts])
        ax.plot(df[ts_transform], color = 'red')

      # Add title
        ax.set_title('%s and %s time-series graph' %(ts, ts_transform))

      # Rotate x-labels
        ax.tick_params(axis = 'x', rotation = 45)

      # Add legend
        ax.legend([ts, ts_transform])

        plt.show()
        plt.close()

        return

      def transf_plot_sta(df,ts):
        pval = []
        transf = []
        df['ts_log'] = df[ts].apply(lambda x: np.log(x))

    # Transformation - 7-day moving averages of log ts
        df['ts_log_moving_avg'] = df['ts_log'].rolling(window = 7,
                                                                    center = False).mean()

    # Transformation - 7-day moving average ts
        df['ts_moving_avg'] = df['ts'].rolling(window = 7,
                                                            center = False).mean()

    # Transformation - Difference between logged ts and first-order difference logged ts
    # df_example['ts_log_diff'] = df_example['ts_log'] - df_example['ts_log'].shift()
        df['ts_log_diff'] = df['ts_log'].diff()

    # Transformation - Difference between ts and moving average ts
        df['ts_moving_avg_diff'] = df['ts'] - df['ts_moving_avg']

    # Transformation - Difference between logged ts and logged moving average ts
        df['ts_log_moving_avg_diff'] = df['ts_log'] - df['ts_log_moving_avg']
        df = df.dropna()
        df.head()
    #     plot_transformed_data(df = df, 
    #                           ts = ts, 
    #                           ts_transform = 'ts_log')
    #     plot_transformed_data(df = df, 
    #                           ts = 'ts_log', 
    #                           ts_transform = 'ts_log_moving_avg')

    # # Plot data
    #     plot_transformed_data(df = df, 
    #                           ts = 'ts', 
    #                           ts_transform = 'ts_moving_avg')

    # # Plot data
    #     plot_transformed_data(df = df, 
    #                           ts = 'ts_log', 
    #                           ts_transform = 'ts_log_diff')

    # # Plot data
    #     plot_transformed_data(df = df, 
    #                           ts = 'ts', 
    #                           ts_transform = 'ts_moving_avg_diff')

    # # Plot data
    #     plot_transformed_data(df = df, 
    #                           ts = 'ts_log', 
    #                           ts_transform = 'ts_log_moving_avg_diff')
        t1 = test_stationarity(df = df, 
                          ts = 'ts_log')

        pval.append(t1)
      # transf.append(ts)
        # print(pval)
        t2 = test_stationarity(df = df, 
                          ts = 'ts_log_moving_avg')
        pval.append(t2)

        t3 = test_stationarity(df = df, 
                         ts = 'ts_moving_avg')
        pval.append(t3)
      # print(pval)
      # transf.append(ts)

    # Perform stationarity test
        t4 = test_stationarity(df = df,
                          ts = 'ts_log_diff')
        pval.append(t4)
      # print(pval)
      # transf.append(ts)
    # Perform stationarity test
        t5 = test_stationarity(df = df,
                          ts = 'ts_moving_avg_diff')
        pval.append(t5)
      # print(pval)
      # transf.append(ts)
    # Perform stationarity test
        t6 = test_stationarity(df = df,
                        ts = 'ts_log_moving_avg_diff')
        pval.append(t6)
      # print(pval)
      # transf.append(ts)
        return pval
      def plot_decomposition(df, ts, trend, seasonal, residual):
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize = (15, 5), sharex = True)

        ax1.plot(df[ts], label = 'Original')
        ax1.legend(loc = 'best')
        ax1.tick_params(axis = 'x', rotation = 45)

        ax2.plot(df[trend], label = 'Trend')
        ax2.legend(loc = 'best')
        ax2.tick_params(axis = 'x', rotation = 45)

        ax3.plot(df[seasonal],label = 'Seasonality')
        ax3.legend(loc = 'best')
        ax3.tick_params(axis = 'x', rotation = 45)

        ax4.plot(df[residual], label = 'Residuals')
        ax4.legend(loc = 'best')
        ax4.tick_params(axis = 'x', rotation = 45)
        plt.tight_layout()

      # Show graph
        plt.suptitle('Trend, Seasonal, and Residual Decomposition of %s' %(ts), 
                     x = 0.5, 
                     y = 1.05, 
                     fontsize = 18)
        plt.show()
        plt.close()

        return
      # print(category_id)

      if df_cat_merge_group[df_cat_merge_group.categoryid==category_id].empty:
        st.write(category_id)
        st.write("Forecasting is not availible for this category")


      else:
        df_t = ts_set(df_cat_merge_group,'wt_act',category_id)
        df_t = df_t.rename(columns={'wt_act':'ts'})
        # plt.plot(df_t.ts)
        # plt.xticks(rotation=45)
        # plt.show()
        test_stationarity(df_t,'ts')
        p = transf_plot_sta(df_t,'ts')
        eval = df_t.columns[np.argmin(p)+2]
        df_t.dropna(inplace=True)
        # from statsmodels.tsa.seasonal import seasonal_decompose
        decomposition = seasonal_decompose(df_t['ts'])
      # with monthly data and yearly seasonal cycle, m=12

        df_t.loc[:,'trend'] = decomposition.trend
        df_t.loc[:,'seasonal'] = decomposition.seasonal
        df_t.loc[:,'residual'] = decomposition.resid

        # plot_decomposition(df = df_t, 
        #                    ts = 'ts', 
        #                    trend = 'trend',
        #                    seasonal = 'seasonal', 
        #                    residual = 'residual')

        p_resid = test_stationarity(df = df_t.dropna(), ts = 'residual')
      # print(p_resid)
        if p_resid < p[np.argmin(p)]:
          eval = 'residual'
          df_t = df_t.dropna()
        # print(eval)

        tr_size = int(df_t.shape[0]*0.8)
        train = df_t.iloc[:tr_size]
        test = df_t.iloc[tr_size-1:]
        # fit_model = ExponentialSmoothing(train[eval],trend='add',seasonal='add',seasonal_periods=41).fit()
        # prediction = fit_model.forecast(28)
      # print(mean_squared_error(test[eval],prediction))
        # test[eval].plot(legend=True,figsize=(10,6))
        # prediction.plot(legend=True,figsize=(10,6),label='pred')
        fit_model = ExponentialSmoothing(df_t[eval],trend='add',seasonal='add',seasonal_periods=41).fit()
        prediction = fit_model.forecast(30)
      # train[eval].plot(legend=True,figsize=(10,6))

        
        # df_t[eval].plot(legend=True,figsize=(10,6))
        # prediction.plot(legend=True,figsize=(10,6),label='pred')


        def makegraph(x,y):
          import matplotlib.pyplot as plt
          from matplotlib.widgets import Cursor
          fig = plt.figure()
          fig.set_size_inches(8.5, 5)
          ax = fig.subplots()
          ax.plot(x,y, color = 'b')
          ax.grid()
          st.plotly_chart(fig)

          

        st.write(category_id)  
        st.write("Actual Data")  
        makegraph(df_t.index,df_t[eval])
        st.write("Predicted Data")
        makegraph(prediction.index,prediction)











    def product_demand_forecasts(*args):
      import datetime
      from statsmodels.tsa.stattools import adfuller
      from statsmodels.tsa.seasonal import seasonal_decompose
      from sklearn.metrics import mean_absolute_error, mean_squared_error
      from statsmodels.tsa.holtwinters import ExponentialSmoothing
      def test_stationarity(df, ts):
        rolmean = df[ts].rolling(window = 12, center = False).mean()
        rolstd = df[ts].rolling(window = 12, center = False).std()

        # Plot rolling statistics:
        # orig = plt.plot(df[ts], 
        #                 color = 'blue', 
        #                 label = 'Original')
        # mean = plt.plot(rolmean, 
        #                 color = 'red', 
        #                 label = 'Rolling Mean')
        # std = plt.plot(rolstd, 
        #                color = 'black', 
        #                label = 'Rolling Std')
        # plt.legend(loc = 'best')
        # plt.title('Rolling Mean & Standard Deviation for %s' %(ts))
        # plt.xticks(rotation = 45)
        # plt.show(block = False)
        # plt.close()
        dftest = adfuller(df[ts], 
                          autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], 
                             index = ['Test Statistic',
                                      'p-value',
                                      '# Lags Used',
                                      'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        # print(dfoutput)
        return dfoutput[1]

      def ts_set(df,ts,prod):
        df_temp = df[df.itemid==prod].reset_index()
        if df[df.itemid==prod].index[0] != datetime.datetime.strptime('2015-05-03','%Y-%m-%d').date():
          df_temp = pd.DataFrame(np.insert(df_temp.values,0,values=[datetime.datetime.strptime('2015-05-03','%Y-%m-%d').date(),prod,0],axis=0))
        if df[df.itemid==prod].index[-1] != datetime.datetime.strptime('2015-09-18','%Y-%m-%d').date():
          df_temp = pd.DataFrame(np.insert(df_temp.values,df_temp.shape[0],values=[datetime.datetime.strptime('2015-09-18','%Y-%m-%d').date(),prod,0],axis=0))

        df_temp.rename(columns={0:'timestamp',1:'itemid',2:'wt_act'},inplace=True)
        df_temp.set_index('timestamp',inplace=True)
        df_temp = df_temp.asfreq(freq='1D')
        df_temp.itemid = prod

        df_temp.wt_act.fillna(0.0,inplace=True)
        df_temp['wt_act'].replace({0:(np.round(df_temp.wt_act.mean(),2))},inplace=True)
        # print(df_temp)
        return df_temp
      def plot_transformed_data(df, ts, ts_transform):
      # Plot time series data
        f, ax = plt.subplots(1,1)
        ax.plot(df[ts])
        ax.plot(df[ts_transform], color = 'red')

      # Add title
        ax.set_title('%s and %s time-series graph' %(ts, ts_transform))

      # Rotate x-labels
        ax.tick_params(axis = 'x', rotation = 45)

      # Add legend
        ax.legend([ts, ts_transform])

        plt.show()
        plt.close()

        return

      def transf_plot_sta(df,ts):
        pval = []
        transf = []
        df['ts_log'] = df[ts].apply(lambda x: np.log(x))

    # Transformation - 7-day moving averages of log ts
        df['ts_log_moving_avg'] = df['ts_log'].rolling(window = 7,
                                                                    center = False).mean()

    # Transformation - 7-day moving average ts
        df['ts_moving_avg'] = df['ts'].rolling(window = 7,
                                                            center = False).mean()

    # Transformation - Difference between logged ts and first-order difference logged ts
    # df_example['ts_log_diff'] = df_example['ts_log'] - df_example['ts_log'].shift()
        df['ts_log_diff'] = df['ts_log'].diff()

    # Transformation - Difference between ts and moving average ts
        df['ts_moving_avg_diff'] = df['ts'] - df['ts_moving_avg']

    # Transformation - Difference between logged ts and logged moving average ts
        df['ts_log_moving_avg_diff'] = df['ts_log'] - df['ts_log_moving_avg']
        df = df.dropna()
        df.head()
    #     plot_transformed_data(df = df, 
    #                           ts = ts, 
    #                           ts_transform = 'ts_log')
    #     plot_transformed_data(df = df, 
    #                           ts = 'ts_log', 
    #                           ts_transform = 'ts_log_moving_avg')

    # # Plot data
    #     plot_transformed_data(df = df, 
    #                           ts = 'ts', 
    #                           ts_transform = 'ts_moving_avg')

    # # Plot data
    #     plot_transformed_data(df = df, 
    #                           ts = 'ts_log', 
    #                           ts_transform = 'ts_log_diff')

    # # # Plot data
    #     plot_transformed_data(df = df, 
    #                           ts = 'ts', 
    #                           ts_transform = 'ts_moving_avg_diff')

    # # Plot data
    #     plot_transformed_data(df = df, 
    #                           ts = 'ts_log', 
    #                           ts_transform = 'ts_log_moving_avg_diff')
        t1 = test_stationarity(df = df, 
                          ts = 'ts_log')

        pval.append(t1)
      # transf.append(ts)
        # print(pval)
        t2 = test_stationarity(df = df, 
                          ts = 'ts_log_moving_avg')
        pval.append(t2)

        t3 = test_stationarity(df = df, 
                         ts = 'ts_moving_avg')
        pval.append(t3)
      # print(pval)
      # transf.append(ts)

    # Perform stationarity test
        t4 = test_stationarity(df = df,
                          ts = 'ts_log_diff')
        pval.append(t4)
      # print(pval)
      # transf.append(ts)
    # Perform stationarity test
        t5 = test_stationarity(df = df,
                          ts = 'ts_moving_avg_diff')
        pval.append(t5)
      # print(pval)
      # transf.append(ts)
    # Perform stationarity test
        t6 = test_stationarity(df = df,
                        ts = 'ts_log_moving_avg_diff')
        pval.append(t6)
      # print(pval)
      # transf.append(ts)
        return pval
      def plot_decomposition(df, ts, trend, seasonal, residual):
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize = (15, 5), sharex = True)

        ax1.plot(df[ts], label = 'Original')
        ax1.legend(loc = 'best')
        ax1.tick_params(axis = 'x', rotation = 45)

        ax2.plot(df[trend], label = 'Trend')
        ax2.legend(loc = 'best')
        ax2.tick_params(axis = 'x', rotation = 45)

        ax3.plot(df[seasonal],label = 'Seasonality')
        ax3.legend(loc = 'best')
        ax3.tick_params(axis = 'x', rotation = 45)

        ax4.plot(df[residual], label = 'Residuals')
        ax4.legend(loc = 'best')
        ax4.tick_params(axis = 'x', rotation = 45)
        plt.tight_layout()

      # Show graph
        plt.suptitle('Trend, Seasonal, and Residual Decomposition of %s' %(ts), 
                     x = 0.5, 
                     y = 1.05, 
                     fontsize = 18)
        plt.show()
        plt.close()

        return
      # print(category_id)
      # print(product_id)
      for item in args:
        if prod_group[prod_group.itemid==item].empty:
          st.write(item)
          st.write("Forecasting is not availible for this Product ID")
          

        else:
          df_t = ts_set(prod_group,'wt_act',item)
          df_t = df_t.rename(columns={'wt_act':'ts'})
          # plt.plot(df_t.ts)
          # plt.xticks(rotation=45)
          # plt.show()
          test_stationarity(df_t,'ts')
          p = transf_plot_sta(df_t,'ts')
          eval = df_t.columns[np.argmin(p)+2]
          df_t.dropna(inplace=True)
          # from statsmodels.tsa.seasonal import seasonal_decompose
          decomposition = seasonal_decompose(df_t['ts'])
        # with monthly data and yearly seasonal cycle, m=12

          df_t.loc[:,'trend'] = decomposition.trend
          df_t.loc[:,'seasonal'] = decomposition.seasonal
          df_t.loc[:,'residual'] = decomposition.resid

          # plot_decomposition(df = df_t, 
          #                    ts = 'ts', 
          #                    trend = 'trend',
          #                    seasonal = 'seasonal', 
          #                    residual = 'residual')

          p_resid = test_stationarity(df = df_t.dropna(), ts = 'residual')
        # print(p_resid)
          if p_resid < p[np.argmin(p)]:
            eval = 'residual'
            df_t = df_t.dropna()
          # print(eval)

          tr_size = int(df_t.shape[0]*0.8)
          train = df_t.iloc[:tr_size]
          test = df_t.iloc[tr_size-1:]
          # fit_model = ExponentialSmoothing(train[eval],trend='add',seasonal='add',seasonal_periods=41).fit()
          # prediction = fit_model.forecast(28)
        # print(mean_squared_error(test[eval],prediction))
          # test[eval].plot(legend=True,figsize=(10,6))
          # prediction.plot(legend=True,figsize=(10,6),label='pred')
          fit_model = ExponentialSmoothing(df_t[eval],trend='add',seasonal='add',seasonal_periods=41).fit()
          prediction = fit_model.forecast(30)
        # train[eval].plot(legend=True,figsize=(10,6))

          def makegraph(x,y):
            import matplotlib.pyplot as plt
            from matplotlib.widgets import Cursor
            fig = plt.figure()
            fig.set_size_inches(8.5, 5)
            ax = fig.subplots()
            ax.plot(x,y, color = 'b')
            ax.grid()
            st.plotly_chart(fig)

          
          st.write(item)  
          st.write("Actual Data")  
          makegraph(df_t.index,df_t[eval])
          st.write("Predicted Data")
          makegraph(prediction.index,prediction)




    original_title = '<p style="font-family:Arial; color:Black; font-size: 30px;">Product and Category Forecasting</p>'
    st.markdown(original_title, unsafe_allow_html=True)

    parameter=st.selectbox("Select the Appropriate Parameter",("Category","Product"))

    if parameter=='Category':
        c_ids=st.text_input("Enter the Category_IDs seperated by a commas",959)
        li = []
        listx = c_ids.split (",")
        for z in listx:
            li.append(int(z))

        for i in li:
            category_demand_forecast(i)




    elif parameter=='Product':
        p_ids=st.text_input("Enter the Item_IDs seperated by a commas",25)
        li = []
        listx = p_ids.split (",")
        for z in listx:
            li.append(int(z))

        product_demand_forecasts(*li)