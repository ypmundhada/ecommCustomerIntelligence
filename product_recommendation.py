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