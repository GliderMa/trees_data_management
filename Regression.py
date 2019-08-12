import ExcelDataManupilate.ColumnClean as CC
import statsmodels.api as sm
from scipy.stats.mstats import zscore
import sklearn.model_selection as sms
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn_features.transformers import DataFrameSelector
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy import sparse
from ExcelDataManupilate.ColumnClean import ScaleOperation as sop
'''

DataFrameSelector used in pipeline progress
'''


from sklearn.base import TransformerMixin #gives fit_transform method for free
class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)





class RegreesionModel(CC.ScaleOperation):
    def __init__(self,path):
        self.path=path
    def testmodel(self):
        df =self.get_working_df(self.path)
        df=df[['HealthIndex', 'AgeIndex', 'Intercept', 'Betweenness']].dropna()
        df.hist(bins=25)
        plt.show()
class MachineLearningProgress_create_test_set(CC.ScaleOperation):
    def __init__(self,path):
        self.path=path
    def get_possible_attributes(self):                #only leave useful informations
        df=self.get_working_df(self.path)
        attributes=['GEOID','HealthIndex', 'AgeIndex', 'Intercept', 'Betweenness','Orientation','ADJ_LAND','POSITION_I','PLANTER_C1']
        df_useful=df[attributes].dropna()
        return df_useful
    def create_test_set(self,ratio):            # actually divide the data into two parts, test group and train group
        df=self.get_possible_attributes()
        shuffled_df=df.sample(frac=1).reset_index(drop=True)           # index should also be reset to decide the loc to be obtained
        test_set_size=int(len(df)*ratio)

        test_set=shuffled_df.loc[:test_set_size]
        train_set=shuffled_df.loc[test_set_size:]
        return test_set,train_set
    def save_sets(self):
        test_set,train_set=self.create_test_set(0.2)
        test_set.to_csv('test_set.csv',index=False)
        train_set.to_csv('train_set.csv',index=False)


class MachineLearningProgress_data_cleaning(CC.ScaleOperation):
    def __init__(self,testset,trainset):
        self.testset=testset
        self.trainset=trainset
    def cross_validation_scores(self,score):
        print('Scores ',score)
        print('Mean ',score.mean())
        print('Standard deviation ',score.std())

    def predictor_processing(self):               # a process try to seperate different part
        tree_all=pd.read_csv(self.trainset)
        tree=tree_all.drop(['HealthIndex','GEOID'],axis=1)           #reamins predictors

        tree_num=tree.drop(['Orientation','ADJ_LAND','POSITION_I','PLANTER_C1'],axis=1)      #num variable
        tree_cata=tree.drop(['AgeIndex', 'Intercept', 'Betweenness',],axis=1)   #catagorical variable
        return tree,tree_num,tree_cata
    def outcome_processing(self):               # to get the outcome part
        tree_all = pd.read_csv(self.trainset)
        tree_health = tree_all['HealthIndex'].copy()  # outcome
        return tree_health
    '''
    Use pipeline processing to do data cleaning, for num data and catagory data seperately
    PROBLEM: category data can only be processed ONE BY ONE(MIGHT FIXED LATER)
    '''

    def pipeline_processing(self):
        tree,tree_num,tree_cata=self.predictor_processing()
        num_attributes=['AgeIndex', 'Intercept', ]
        cata_attributes=['Orientation','ADJ_LAND','POSITION_I','PLANTER_C1']
        num_pipeline=Pipeline([
            ('selector',DataFrameSelector(num_attributes)),
            ('imputer',Imputer(strategy='median')),
        ])

        cata_pipeline0=Pipeline([
            ('selector',DataFrameSelector(cata_attributes[0])),
            ('label_binarizer',MyLabelBinarizer())
        ])
        cata_pipeline1=Pipeline([
            ('selector',DataFrameSelector(cata_attributes[1])),
            ('label_binarizer',MyLabelBinarizer())
        ])
        cata_pipeline2=Pipeline([
            ('selector',DataFrameSelector(cata_attributes[2])),
            ('label_binarizer',MyLabelBinarizer())
        ])
        cata_pipeline3=Pipeline([
            ('selector',DataFrameSelector(cata_attributes[3])),
            ('label_binarizer',MyLabelBinarizer())
        ])
        # then do feature union
        full_pipeline=FeatureUnion(transformer_list=[
            ('num_pipeline',num_pipeline),
            ('cata_pipeline0', cata_pipeline0),
            ('cata_pipeline1', cata_pipeline1),
            ('cata_pipeline2', cata_pipeline2),
            ('cata_pipeline3', cata_pipeline3)


        ])
        tree_prepared=full_pipeline.fit_transform(tree)
        shape=tree_prepared.shape
        #print(tree_prepared,type(tree_prepared),shape)
        return tree_prepared
    def modeltraining(self):
        tree_prepared=self.pipeline_processing()
        tree_labeled=self.outcome_processing()

        '''
        Here is linear regression model
        '''
        print('liner regression processing...')
        lin_reg=LinearRegression()
        lin_reg.fit(tree_prepared,tree_labeled)

        lin_mse=mean_squared_error(tree_labeled,lin_predict)
        print(np.sqrt(lin_mse))
        scores_lin=cross_val_score(lin_reg,tree_prepared,tree_labeled,scoring='neg_mean_squared_error',cv=10)
        lin_reg_rmse_scores=np.sqrt(-scores_lin)
        self.cross_validation_scores(lin_reg_rmse_scores)
        #print(lin_reg.intercept_,lin_reg.coef_)

        print('decision tree precessing...')
        tree_reg=DecisionTreeRegressor()
        tree_reg.fit(tree_prepared,tree_labeled)
        tree_predict=tree_reg.predict(tree_prepared)
        tree_mes=mean_squared_error(tree_labeled,tree_predict)
        print(np.sqrt(tree_mes))
        scores_tree = cross_val_score(tree_reg, tree_prepared, tree_labeled, scoring='neg_mean_squared_error', cv=10)
        tree_reg_rmse_scores = np.sqrt(-scores_tree)
        self.cross_validation_scores(tree_reg_rmse_scores)

        print('randomforest processing...')

        ran_reg=RandomForestRegressor()
        ran_reg.fit(tree_prepared,tree_labeled)
        ran_predict=ran_reg.predict(tree_prepared)
        ran_mes=mean_squared_error(tree_labeled,ran_predict)
        print(np.sqrt(ran_mes))
        scores_ran = cross_val_score(ran_reg, tree_prepared, tree_labeled, scoring='neg_mean_squared_error', cv=10)
        ran_reg_rmse_scores = np.sqrt(-scores_ran)
        self.cross_validation_scores(ran_reg_rmse_scores)


        '''
        Save preferred model with all parameters
        '''
        joblib.dump(lin_reg,'linear_model_test.sav')
    def tra_linear_regression(self):
        tree_prepared=self.pipeline_processing()
        tree_labeled=self.outcome_processing()
        print(type(tree_prepared))
        model=sm.OLS(tree_labeled,tree_prepared).fit()
        z_model=sm.OLS(zscore(tree_labeled), zscore(tree_prepared)).fit()
        print(model.summary())
        print(z_model.summary())
'''
Self-Defined data pre-processing 
'''

'''
dataframe should have specie name first for assigning values, i.e derived from original species.xlxs, and contains 
specie name, num variables and cate variables (not the dummy version),after add dummys, then drop cate variables
'''
class DummyVariable:
    def __init__(self,dataframe,reference_sheets):
        self.df=dataframe
        self.excelfile=reference_sheets
    def get_specific_sheet(self,cate_name):
        file = pd.ExcelFile(self.excelfile)
        dict_df = file.parse(cate_name)
        return dict_df

    '''
    1.knowing which specie
    2. assign column
    3. according to column, fill the cell
    '''
    def transform_category(self,working_df,cate_name):
        specie_name=working_df.at[1,'SPECIES_FU']     #to get the name of specie

        landuse_table=self.get_specific_sheet(cate_name)

        category=[]
        for row in range(0,len(landuse_table),1):
            tag=landuse_table.at[row,specie_name]
            if (tag==1):
                category.append(landuse_table.at[row,'Row Labels'])
        for item in category:
            working_df[item]=None
            for i in range(0,len(working_df),1):
                if (working_df.at[i,cate_name]==item):
                    working_df.at[i,item]=1
                else:
                    working_df.at[i, item] = 0
        update_working_df=working_df.drop([cate_name],axis=1)
        return update_working_df
    def transform_category_forall(self,working_df,cate_name):
            #to get the name of specie

        landuse_table=self.get_specific_sheet(cate_name)

        category=[]
        for row in range(0,len(landuse_table),1):
            tag=landuse_table.at[row,'TAG']
            if (tag==1):
                category.append(landuse_table.at[row,'Row Labels'])
        for item in category:
            working_df[item]=None
            for i in range(0,len(working_df),1):
                if (working_df.at[i,cate_name]==item):
                    working_df.at[i,item]=1
                else:
                    working_df.at[i, item] = 0
        update_working_df=working_df.drop([cate_name],axis=1)
        return update_working_df
    def code_category(self,working_df,cate_name):
        tempset=sop.getScaleItem(a,working_df,cate_name)
        sort_set=[]
        for item in tempset:
            sort_set.append(item)
        sort_set.sort()
        dict_cate={}
        for i in range(0,len(sort_set),1):
            dict_cate[sort_set[i]]=i
        print(dict_cate)
        working_df[cate_name+'_code']=None
        for i in range(0, len(working_df), 1):
            item=working_df.at[i,cate_name]
            code=dict_cate[item]
            working_df.at[i,cate_name+'_code']=code
        return working_df




    def add_constant(self,working_df):
        working_df['const']=None
        for i in range(0,len(working_df),1):
            working_df.at[i,'const']=1
        return working_df
    '''
    actually preprocessing both num variable and category variable
    '''
    def data_preprocessing(self):
        attribute=['SPECIES_FU','HealthClassModifier', 'AgeIndex', 'Intercept', 'Betweenness','Orientation','ADJ_LAND','POSITION_I','PLANTER_C1']
        working_df=self.df[attribute].dropna()

        working_df=self.transform_category(working_df,'ADJ_LAND')
        working_df = self.transform_category(working_df, 'POSITION_I')
        working_df = self.transform_category(working_df, 'PLANTER_C1')

        working_df = working_df.drop(['SPECIES_FU'], axis=1)


        return working_df
    def data_preprocessing_all(self):
        attribute = ['HealthClassModifier', 'AgeClass', 'Intercept', 'Betweenness', 'Orientation',
                     'ADJ_LAND', 'POSITION_I', 'PLANTER_C2','SPECIES_FU','GEOID','AGE_CLASS']


        working_df = self.df[attribute].dropna()
        working_df=self.code_category(working_df,'ADJ_LAND')
        working_df = self.code_category(working_df, 'POSITION_I')
        working_df = self.code_category(working_df, 'PLANTER_C2')
        working_df = self.code_category(working_df, 'SPECIES_FU')

        #working_df = self.transform_category_forall(working_df, 'ADJ_LAND')
        #working_df = self.transform_category_forall(working_df, 'POSITION_I')
        #working_df = self.transform_category_forall(working_df, 'PLANTER_C2')
        #working_df = self.transform_category_forall(working_df, 'SPECIES_FU')

        return working_df

class OLSRegression:
    def __init__(self,working_df):
        self.working_df=working_df
    def outcome(self):
        tree_health = self.working_df['HealthClassModifier'].copy()  # outcome
        return tree_health
    def convert_dataframe_to_ndarray(self):
        df=self.working_df.drop(['HealthClassModifier'],axis=1)
        num_attributes=list(df)
        print(num_attributes)
        num_pipeline=Pipeline([
            ('selector',DataFrameSelector(num_attributes)),
            ('imputer',Imputer(strategy='median')),
        ])
        ndarray=num_pipeline.fit_transform(df)

        return ndarray
    def tra_linear_regression(self):
        tree_prepared=self.convert_dataframe_to_ndarray()
        tree_labeled=self.outcome()
        #tree_prepared=sm.add_constant(tree_prepared)
        model=sm.OLS(tree_labeled,tree_prepared).fit()
        z_model=sm.OLS(zscore(tree_labeled), zscore(tree_prepared)).fit()

        print(model.summary())

        with open('testt_PL.txt','wt') as f:
            print(z_model.summary(),file=f)
        tree_predict=model.predict(tree_prepared)
        z_tree_predict=z_model.predict(zscore(tree_prepared))
        res=tree_labeled-tree_predict
        #return zscore(tree_labeled),z_tree_predict
        return tree_labeled,tree_predict,res
    def vis_linear_result(self):
        labeled,predict,res=self.tra_linear_regression()
        plt.figure(figsize=(20, 10))
        plt.plot(range(len(labeled)),labeled,'b',label='real')
        plt.plot(range(len(predict)),predict, 'r', label='predict')
        plt.legend(loc='upper right')
        plt.xlabel('Number of trees')
        plt.ylabel('Health Index')
        plt.title('Mangifera indica')
        plt.show()
    def vis_res_histogram(self):
        labeled, predict, res = self.tra_linear_regression()
        plt.figure(figsize=(8, 8))
        plt.hist(res,bins=50)
        plt.xlabel('Residual')
        plt.ylabel('Count')
        plt.title('Residual Distribution for Mangifera indica')
        plt.show()




if __name__=='__main__':
    pathA='SpeciesaddPlanter_classify.xlsx'
    a=MachineLearningProgress_create_test_set(path=pathA)
    b=MachineLearningProgress_data_cleaning('',trainset='train_set.csv')
    #b.tra_linear_regression()
    '''
    file=pd.read_csv('PL1501new.csv')
    c=DummyVariable(file,'category_update.xlsx')
    df=c.data_preprocessing()
    writer=pd.ExcelWriter('PL_Ready.xlsx')
    df.to_excel(writer,sheet_name='1',index=False)
    writer.save()
    hh=OLSRegression(df)
    hh.tra_linear_regression()
    #hh.vis_linear_result()
    #hh.vis_res_histogram()
    '''
    file=pd.read_csv('AllSpecies.csv')
    d=DummyVariable(file,'categoryforALL.xlsx')


    df = d.data_preprocessing_all()
    writer=pd.ExcelWriter('PreparedData_ver7.xlsx')
    df.to_excel(writer,sheet_name='sheet1',index=False)
    writer.save()
