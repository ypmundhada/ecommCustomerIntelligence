#!/usr/bin/env python
# coding: utf-8

# In[2]:


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





    # In[ ]:





    # In[ ]:





    # In[17]:





    # In[ ]:





    # In[ ]:





    # In[ ]:





    # In[16]:





    # In[ ]:





    # In[ ]:





    # In[ ]:





    # In[ ]:





    # In[ ]:





    # In[ ]:





    # In[ ]:





    # In[ ]:





    # In[11]:





    # In[ ]:





    # In[14]:





    # In[18]:





    # In[ ]:

# In[3]:


def recommend():
    import streamlit as st
    # To make things easier later, we're also importing numpy and pandas for
    # working with sample data.
    import numpy as np
    import pandas as pd
    original_title = '<p style="font-family:Arial; color:Black; font-size: 30px;">Product and Category Recommendation</p>'
    st.markdown(original_title, unsafe_allow_html=True)
    option = st.selectbox("Select",( 'Category ID','Product ID'))
    st.write('You selected:', option)

    df_y=pd.read_hdf("df_y.hf5")
    df_z=pd.read_hdf("df_z.hf5")
    rules=pd.read_hdf("rules.hf5")



    if option=="Product ID":
        select=0
    else:
        select=1



    def common(list1,list2):
        intersection=(set(list1).intersection(list2))
        x=list(intersection)
        return x

    def commoncategory(li):
        c_category=[] 
        c_category=c_category+(df_y[li[0]]).tolist()

        for i in li:
            apr=df_y[i]
            intersection=common(c_category,apr)
            c_category.clear()
            c_category=c_category+intersection
        return c_category


    def takecatinput():
        catinput=st.text_input("Enter the Category_IDs Seperated by a comma",959)
        li = []
        listx = catinput.split (",")
        for z in listx:
            li.append(int(z))

        return li        

    def takeiteminput():
        item_input=st.text_input("Enter the Item_IDs seperated by a commas",29)
        li = []
        listx = item_input.split (",")
        for z in listx:
            li.append(int(z))

        cat_items=[]
        for j in li:
            cat_items=cat_items+list(df_z[df_z.itemid==j].categoryid.unique())    
        cat_items_unique=set(cat_items)
        r=[]
        r=list(set(cat_items))
        return r

    def takeinput():

        y=int
        if select==0:
            y=takeiteminput()
            return y
        elif select==1:
            y=takecatinput()
            return y
        else:
            st.write("Wrong Input. Please Try Again","\n")
            takeinput()



    def finalprocessing(commonapriori):
        final_items=[]    
        for i in commonapriori:
            final_items=final_items+df_z[df_z["categoryid"]==i].itemid.tolist()
        return(final_items)


    def all_in_each_cat(catlist):
        final_list=[]
        for i in catlist:
            final_list=final_list+df_z[df_z["categoryid"]==i].itemid.unique().tolist()
        return final_list



    def top5(commonapriori,categorylist):


        df_temp=pd.DataFrame(None)
        for j in categorylist:
            for i in rules.index:
                if j in rules.iloc[i]["antecedents"]:
                    df_temp=df_temp.append(rules.iloc[i])


        df_temp1=pd.DataFrame(None)
        df_temp.reset_index(drop=True,inplace=True)

        for i in pd.Series(df_temp.index):
            for j in df_temp.iloc[i]["consequents"]:  
                if j in commonapriori:
                    df_temp1=df_temp1.append(df_temp.iloc[i])
                    break

        df_temp1.reset_index(inplace=True)
        df_deal=df_temp1
        df_deal.drop("index",axis=1,inplace=True)


        confidence={}
        count={}

        for i in commonapriori:
            confidence[i]=0
            count[i]=0

        for i in df_deal.index:
            for j in df_deal.iloc[i].consequents:
                if j in commonapriori:
                    confidence[j]=confidence[j]+df_deal.iloc[i].confidence
                    count[j]=count[j]+1


        commonapriori_conf=[]

        for i in commonapriori:
            commonapriori_conf.append(confidence[i]/count[i])



        conf_to_cat={}
        i=0
        while i<len(commonapriori):
            conf_to_cat[commonapriori_conf[i]]=commonapriori[i]
            i=i+1

        commonapriori_conf.sort(reverse=True)

        cat_final=[]
        j=0
        while j<5:
            cat_final.append(conf_to_cat[commonapriori_conf[j]])
            j=j+1


        return cat_final









    allcorrectindex=True       
    categorylist=takeinput()  
    catlistemergency=categorylist.copy()
    commonapriori=[]

    if(len(categorylist)==0):
        st.write("One of the ProductID you entered doesn't exist")


    else:   
        for i in categorylist:
            if df_z[df_z["categoryid"]==i].empty:
                st.write("Some of the categoryid's that you entered don't exist ")
                allcorrectindex=False

        if allcorrectindex==True:
            while len(commonapriori)==0:
                print(categorylist)
                commonapriori=commoncategory(categorylist)
                if -1 in commonapriori:
                  commonapriori.remove(-1)
                if len(commonapriori)==0:
                  import random
                  categorylist.remove(random.choice(categorylist))
                if len(categorylist)==0:
                  break


            if len(commonapriori)>5:
              check=st.checkbox("Do you want to see All the Recommendations instead of Top 5",0)
              if check:
                commonapriori_top=commonapriori
              else: 
                  commonapriori_top=top5(commonapriori,categorylist)
            else:
              commonapriori_top=commonapriori    




            if len(categorylist)!=0:  
                st.write("The Category Recommendations are: ")
                a1=pd.Series(commonapriori_top)
                a1.index+=1
                a1=a1.rename("Category ID")
                st.write(a1)
                st.write("Do You Want a list of items in it (Y/N)")
                option = st.selectbox("Select",('No', 'Yes'))

                if option=="Yes":

                  final_item_list=finalprocessing(pd.Series(commonapriori_top))   
                  a2= pd.Series(list(set(final_item_list)))
                  a2.index+=1
                  a2=a2.rename("Product ID")
                  st.write(a2)

            else:
                st.write("Sorry, We Could not Find any category recommendation")
                st.write("Do you want to see the items in the same category as the items you have purchased ?")
                option = st.selectbox("Select",('No', 'Yes'))
                if(option=='Yes'):
                  final_item_list=all_in_each_cat(catlistemergency)
                  a2= pd.Series(list(set(final_item_list)))
                  a2.index+=1
                  a2=a2.rename("Product ID")
                  st.write(a2)






# In[ ]:


def cust_seg():
    import streamlit as st
    import numpy as np
    import pandas as pd

    @st.cache
    def load_data():
        df = pd.read_csv('transactions.csv')
        df = df.iloc[: , 1:]
        return df


        
    df = load_data()
    st.write("## Original Dataframe")

    if st.checkbox('Show dataframe'):
        st.dataframe(df)

    st.write("## Segment Based Visitor Information")
    option = st.selectbox("Select a Segment",('Low-Value', 'Mid-Value','High-Value'))

    st.write('### Here is the visitor information for ',option,'Customers')
    if st.checkbox('Show Information'):
        df_x =df.drop(['RecencyCluster','FrequencyCluster'],axis=1)
        st.write(df_x[df_x['Segment']==option])

    st.button("Re-run")


# In[3]:


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



# In[ ]:









def dynamic_pricing(prodid):
 
  def product_demand_forecast(product_id):
    import datetime
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.seasonal import seasonal_decompose
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import streamlit as st

    prod_group=pd.read_hdf("prod_group.hf5",key="prod_group")

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



    df_t = ts_set(prod_group,'wt_act',product_id)
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
    df_t[eval].plot(legend=True,figsize=(10,6))
    prediction.plot(legend=True,figsize=(10,6),label='pred')
    plt.legend()
    return prediction

  

  import pandas as pd  
  prod_group=pd.read_hdf("prod_group.hf5",key="prod_group") 

  if prod_group[prod_group.itemid==prodid].empty:
    st.write("Dynamic Pricing is not availible for this Product ID ")

  else:

    pred=product_demand_forecast(prodid)

    import pandas as pd
    import numpy as np
    df2=pd.read_hdf("df2.hf5")
    dynamic = df2[df2.itemid==prodid]
    dynamic.drop(columns=['timestamp','transactionid'],inplace=True)
    dummies = pd.get_dummies(dynamic.event)
    dynamic = pd.concat([dynamic,dummies],axis='columns')
    if 'transaction' not in dynamic.columns:
      dynamic['transaction'] = 0
    if 'addtocart' not in dynamic.columns:
      dynamic['addtocart'] = 0
    if 'view' not in dynamic.columns:
      dynamic['view'] = 0
    # if dynamic.event
    dynamic.drop(columns=['event'],inplace=True)

    dynamic['wt_act'] = 0.3*dynamic['addtocart'] + 0.5*dynamic['transaction'] + 0.2*dynamic['view']
    dynamic.drop(columns=['addtocart','view','transaction'],inplace=True)
    dynamic = dynamic.groupby('visitorid')['wt_act'].sum().reset_index()
    dynamic['prod_id'] = prodid
    dynamic['demand'] = np.exp(pred).max()
    dynamic['threshold'] = 1.8648

    dynamic['price'] = 'zero'
    for i in (dynamic.index):
      wt_act = dynamic.wt_act.iloc[i]
      dd = dynamic.demand.iloc[i]
      if wt_act == 0.2 or wt_act == 0.3:
        dynamic['price'].iloc[i] = 'No Change'
        continue
      if dd>1.864:
        dynamic['price'].iloc[i] = 'Decrease' 
      else:
        dynamic['price'].iloc[i] = 'Increase' 

    dynamic.index+=1
    st.write(dynamic)



# In[1]:


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



# In[ ]:





# In[ ]:




