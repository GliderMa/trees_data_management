import pandas as pd
import scipy.stats as stats
import math
from matplotlib import pyplot as plt
import numpy as np

#import Diversity as dis
# to make the column record in order
class CleanField:
    def __init__(self,field,path):
        self.field=field
        self.path=path

    def get_field_record(self):                      # further operation place
        if (self.field=='CH'):
            field_list=['CROWN_HEAD','CROWN_HE_1','CROWN_HE_2']
            field_content=['Utility line','Awning/ Structure']
        elif (self.field=='CLR'):
            field_list=['CROWN_LATE','CROWN_LA_1','CROWN_LA_2','CROWN_LA_3']
            field_content=['Adjacent building / structure','Utility /Traffic poles','Adjacent tree']
        elif (self.field == 'TRG'):
            field_list =['TRUNK_ROOT','TRUNK_RO_1','TRUNK_RO_2','TRUNK_RO_3','TRUNK_RO_4']
            field_content =['Hard Paving to Trunk/Root-ball','Drainage Structure','Other Tree in Planter','Embedded Debris in Planter']
        elif(self.field == 'DIS'):
            field_list =['DISTURBANC','DISTURBA_1','DISTURBA_2','DISTURBA_3','DISTURBA_4','DISTURBA_5']
            field_content=['Road Widening','Drainage Works','Sidewalk Works','Building Works','Utility Works']
        elif(self.field == 'BC'):
            field_list=['BRANCH_CON','BRANCH_C_1','BRANCH_C_2','BRANCH_C_3','BRANCH_C_4']
            field_content=['Broken Limbs','Dead Limbs','Parasitic Plant','Asymmetric Growth']
        elif(self.field =='TC'):
            field_list = ['TRUNK_COND','TRUNK_COND_1','TRUNK_CO_1','TRUNK_CO_2','TRUNK_CO_3','TRUNK_CO_4']
            field_content=['Co-dominant Trunks','Trunk Cavity','Signs of Rot/Decay','Cracks /Splits','Mechanical Injury']
        elif(self.field =='RC'):
            field_list =['ROOT_CONDI','ROOT_CON_1','ROOT_CON_2','ROOT_CON_3']
            field_content=['Visible Roots','Cut Roots','Girdling']
        else:
            field_list=[]
            field_content=[]
        return [field_list,field_content]
    def get_working_df(self,path):
        file=pd.ExcelFile(path)
        sheetname=file.sheet_names[0]                 # default regard the first sheet as working sheet
        df=file.parse(sheetname)
        return df

    def get_field_clean(self):
        df=self.get_working_df(self.path)
        field_record=self.get_field_record()
        field_list=field_record[0]
        field_content=field_record[1]

        for item in field_list[:-1]:
            df[item+'_new']=None
            keyword=field_content[field_list.index(item)]
            #print(keyword)
            for i in range(0, len(df), 1):
                indicator=0
                for column_name in field_list[:-1]:
                    if (keyword==df.at[i,column_name]):
                        indicator=indicator+1
                if (indicator==0):
                    df.at[i,item+'_new']='(Empty)'
                else:
                    df.at[i, item + '_new'] = keyword

        df[field_list[-1]+'_new']=df[field_list[-1]]

        return df
    def update_data(self):
        data=self.get_field_clean()
        data.to_excel(self.path)
#to change one catagery into dummy variable columns
class ExtendField(CleanField):
    def __init__(self,column_name,path,outputpath):
        self.column_name=column_name
        self.path=path
        self.outputpath=outputpath
    def get_record_list(self,df,column_name):
        record=set(df[column_name])
        record_list=[]
        for item in record:
            record_list.append(item)
        record_list.sort()
        return record_list
    def extend_field(self,df,column_name,record_list):

        for item in record_list[:-1]:
            df[item]=None
            for i in range(0, len(df), 1):
                if (item==df.at[i,column_name]):
                    df.at[i,item]=1
                else:
                    df.at[i, item] = 0
        return df
    def update_extended_data(self):
        df=self.get_working_df(self.path)
        recordlist=self.get_record_list(df,self.column_name)
        data=self.extend_field(df,self.column_name,recordlist)
        data.to_excel(self.outputpath)

# merge two landuse column into one
class LanduseMarge(CleanField):
    def __init__(self,path,dictpath,outputpath):
        self.path=path
        self.dictpath=dictpath
        self.outputpath = outputpath

    def get_dict(self,dataframe):          #   Past_Landuse   & New_Landuse
        dict1={}
        for i in range(0,len(dataframe),1):
            dict1[dataframe.at[i,'Past_Landuse']]=dataframe.at[i,'New_Landuse']
        return dict1
    def landuse_merge(self,dataframe,dictionary):
        dataframe['ADJ_LAND']=None
        for i in range(0, len(dataframe), 1):
            try:
                dataframe.at[i,'ADJ_LAND']=dictionary[dataframe.at[i,'ADJACENT_L']]
            except:
                try:
                    dataframe.at[i, 'ADJ_LAND']=dictionary[dataframe.at[i,'ADJACENT_1']]
                except:KeyError
        return dataframe
    def updata_landuse_data(self):
        dictionary=self.get_dict(self.get_working_df(self.dictpath))
        data=self.landuse_merge(self.get_working_df(self.path),dictionary)
        data.to_excel(self.outputpath)


class ScaleOperation:
    def __init__(self,path,outputpath,scale_name,add_field):
        self.path=path
        self.output=outputpath
        self.scale_name=scale_name
        self.add_field=add_field

    def get_working_df(self,path):
        file=pd.ExcelFile(path)
        sheetname=file.sheet_names[0]                 # default regard the first sheet as working sheet
        df=file.parse(sheetname)
        return df
    def get_working_df_by_sheetindex(self,path,SheetIndex):
        file = pd.ExcelFile(path)
        sheetname = file.sheet_names[SheetIndex]
        df=file.parse(sheetname)
        return df
    def getScaleItem(self,dataframe, ScaleField):

        tempset = set()
        a = dataframe[ScaleField]
        for item in a:
            tempset.add(item)

        return tempset
    def sort_scaleItem(self,tempset):
        scaleitem=[]
        for word in tempset:
            try:
                scaleitem.append(int(word))
            except: ValueError
        scaleitem.sort()
        return scaleitem


    def get_avg(self,data):  # need to deal with blank value, data is a list of values
        sum = 0
        count = 0
        for item in data:
            try:
                itemvalue = float(item)
                sum = sum + itemvalue
                count = count + 1
            except:ValueError
        if (count == 0):
            avg_value = 0
        else:
            avg_value = sum / count
        return avg_value

    def getnumberinScale(self,dataframe, ScaleField, ScaleID):
        a = dataframe.loc[dataframe[ScaleField] == ScaleID]
        Total_num = len(a)
        return Total_num
    def getmean(self,dataframe,ScaleField,ScaleID,attribute):

        a=dataframe.loc[dataframe[ScaleField]==ScaleID]
        mean=self.get_avg(a[attribute])
        return mean
    def get_datalist(self,dataframe,ScaleField,ScaleID,attribute):
        a = dataframe.loc[dataframe[ScaleField] == ScaleID]
        b=[]
        try:
            for item in a[attribute]:
                value=float(item)
                b.append(value)
        except:ValueError
        return b
    def get_datalist_exclude(self,dataframe,ScaleField,ScaleID,attribute):
        a = dataframe.loc[dataframe[ScaleField] != ScaleID]
        b=[]
        try:
            for item in a[attribute]:
                value=float(item)
                b.append(value)
        except:ValueError
        return b
    def get_datalist_string(self,dataframe,ScaleField,ScaleID,attribute):
        a = dataframe.loc[dataframe[ScaleField] == ScaleID]
        b=[]

        for item in a[attribute]:

            b.append(item)

        return b
    def get_set_num(self,datalist):
        tempset = set()
        templist=[]
        a = datalist

        for item in a:
            tempset.add(item)
        for item in tempset:
            templist.append(item)
        return len(templist)




    def get_min(self,datalist):
        datalist.sort()
        return datalist[0]
    def get_max(self,datalist):
        datalist.sort()
        return datalist[-1]
    def get_mediannum(self,datalist):
        total_num=len(datalist)
        datalist.sort()
        if (total_num%2==1):
            i=int((total_num+1)/2)-1
            return datalist[i]
        else:
            i=int(total_num/2)-1
            return (datalist[i]+datalist[i+1])/2


    def get_sum(self,datalist):
        total=0
        for i in datalist:
            total=i+total
        return total

    def get_std(self, datalist):
        mean = self.get_avg(datalist)
        var_sum = 0
        valid_count = 0
        for item in datalist:
            try:
                item = float(item)
                var = (item - mean) * (item - mean)
                var_sum = var_sum + var
                valid_count = valid_count + 1
            except:
                ValueError
        if (valid_count == 0):
            std = 0
        else:
            variation = var_sum / valid_count
            std = math.sqrt(variation)
        return std
    def get_line(self):
        X1=197127.9766
        Y1=1857178.25
        X2=197087.7188
        Y2=1856404.5313
        a=(Y2-Y1)/(X2-X1)
        b=Y1-a*X1
        return [a,b]
    def get_intercept(self,ab_value,X_value,Y_value):
        a=ab_value[0]
        b=ab_value[1]
        a1=-1/a
        b1=Y_value-a1*X_value
        X0=(b1-b)/(a-a1)
        Y0=a1*X0+b1
        distance=math.sqrt((X0-X_value)*(X0-X_value)+ (Y0-Y_value)*(Y0-Y_value))
        return distance
    def calculate_intercept(self,df):
        df['Intercept']=None
        for i in range(0,len(df),1):
            Xvalue=df.at[i,'X']
            Yvalue=df.at[i,'Y']
            intercept=self.get_intercept(self.get_line(),Xvalue,Yvalue)
            if intercept>7000:
                intercept='INVALID'
            df.at[i,'Intercept']=intercept
        return df
    def get_SHDI(self,dataframe,ScaleField,ScaleID,attribute):
        datalist = []
        tempset = set()
        a = dataframe.loc[dataframe[ScaleField] == ScaleID]
        # print(type(a),len(a))
        Total_num = len(a)
        for item in a[attribute]:
            datalist.append(item)
            tempset.add(item)
        ShannonIndex = 0
        for species in tempset:
            num = datalist.count(species)
            ShannonIndexsub = (num / Total_num) * math.log((num / Total_num))
            ShannonIndex = ShannonIndex - ShannonIndexsub
        return ShannonIndex

    def Compute_SimpsonIndex(self,dataframe, ScaleField, ScaleID, attribute):
        datalist = []
        tempset = set()
        a = dataframe.loc[dataframe[ScaleField] == ScaleID]
        N = len(a)
        if (N == 1) or (N == 0):
            SimpsonIndexvalue = 0
        else:
            for item in a[attribute]:
                datalist.append(item)
                tempset.add(item)
            SimpsonReciprocalIndex = 0
            for species in tempset:
                n = datalist.count(species)
                subindex = (n * (n - 1)) / (N * (N - 1))
                SimpsonReciprocalIndex = SimpsonReciprocalIndex + subindex
            SimpsonIndexvalue = 1 - SimpsonReciprocalIndex

        return SimpsonIndexvalue
    def series_to_list(self,dataseries):
        list=[]
        for item in dataseries:
            list.append(item)
        return list
    def array_preprocessing(self,list_of_data):    # a list contain several list of numbers need in statistical analysis
        #Problem, if element=None, the nthe function does not work
        mark=set()
        list_operation=[]      #finally working on list operation
        for array in list_of_data:

            for i in range(0,len(array)):

                try:
                    float(array[i])
                except:
                    mark.add(i)
        #print(list(mark))

        # to decide which pairof data should be removed
        array_combo=[]

        #print(mark)
        for array in list_of_data:
            emptylist = []
            for i in range(0, len(array)):
                if (i in mark)==False:
                    emptylist.append(float(array[i]))
            array_reviewed = np.asarray(emptylist)
            array_combo.append(array_reviewed)


        return array_combo




    def operationfield_list(self):
        a=['Utility line','Awning/ Structure',
           'Adjacent building / structure','Utility /Traffic poles','Adjacent tree',
           'Hard Paving to Trunk/Root-ball','Drainage Structure','Other Tree in Planter','Embedded Debris in Planter',
           'Road Widening','Drainage Works','Sidewalk Works','Building Works','Utility Works',
           'Broken Limbs','Dead Limbs','Parasitic Plant','Asymmetric Growth',
           'Co-dominant Trunks','Trunk Cavity','Signs of Rot/Decay','Cracks /Splits','Mechanical Injury',
           'Visible Roots','Cut Roots','Girdling']
        return a

    def create_new_df(self,field_list,ScaleID):     #create a new df, and create new titles and labels
        columnlist=[]
        columnlist.append(ScaleID)
        columnlist.append('NUM')
        for item in field_list:
            columnlist.append(item)
        df_empty=pd.DataFrame(columns=columnlist)
        return df_empty



    def operation(self):               # fill the new table
        Scale=self.scale_name

        df=self.get_working_df(self.path)
        operationfield_list=self.operationfield_list()

        new_df=self.create_new_df(operationfield_list,Scale)          # new df for output

        ScaleID_list=self.sort_scaleItem(self.getScaleItem(df,Scale))
        # for every blockID, get it data from data source, and then record in new df
        #HOW TO ADD A NEW ROW IN A NEW DF???
        for ID in ScaleID_list:
            number=self.getnumberinScale(df,Scale,ID)   # HERE ADD-FIELD IS NUMBER OF TREE IN A SCALE UNIT
            new_df = new_df.append(pd.DataFrame({Scale: [ID],'NUM':number}), ignore_index=True, sort=False)

        for i in range(0, len(new_df), 1):
            scale_ID=new_df.at[i,Scale]
            for field in operationfield_list:
                meanvalue = self.getmean(df, Scale, scale_ID, field)
                new_df.at[i,field] = meanvalue
        new_df.to_csv(self.output,index=False)
        print(self.scale_name+' work done')

class SpeciesDescribe(ScaleOperation):
    def __init__(self, path, outputpath,operationfield):
        self.path = path
        self.output = outputpath
        self.oper_field=operationfield
    def species_operationlist(self):
        a=['Min','Max','Median','Mean','Std']
        return a
    def species_operation(self):
        df = self.get_working_df(self.path)
        new_df=self.create_new_df(self.species_operationlist(),'SPECIES_FU')
        species_list = self.getScaleItem(df, 'SPECIES_FU')
        operationlist=self.species_operationlist()
        for species in species_list:
            number = self.getnumberinScale(df, 'SPECIES_FU', species)
            new_df = new_df.append(pd.DataFrame({'SPECIES_FU': [species],'NUM':number}), ignore_index=True, sort=False)
        for i in range(0, len(new_df), 1):
            specie_name = new_df.at[i, 'SPECIES_FU']
            datalist=self.get_datalist(df,'SPECIES_FU',specie_name,self.oper_field)
            new_df.at[i,operationlist[0]]=self.get_min(datalist)
            new_df.at[i, operationlist[1]]=self.get_max(datalist)
            new_df.at[i, operationlist[2]]=self.get_mediannum(datalist)
            new_df.at[i, operationlist[3]]=self.get_avg(datalist)
            new_df.at[i, operationlist[4]]=self.get_std(datalist)

        new_df.to_csv(self.output, index=False)
        print('done')

    def visualization(self):
        df = self.get_working_df(self.path)
        species_list = self.getScaleItem(df, 'SPECIES_FU')
        column_list=[]
        label_list=[]
        for species in species_list:
            number = self.getnumberinScale(df, 'SPECIES_FU', species)
            if (number>50):
                a=str(number)
                label=species+'\n'+a.center(len(species))
                #print(label)
                label_list.append(label)                    #label for figure display
                column_list.append(species)
        vis_df = pd.DataFrame(columns=label_list)

        for item in column_list:
            datalist = self.get_datalist(df, 'SPECIES_FU', item, self.oper_field)
            series=pd.Series(np.array(datalist))
            label_in_df=label_list[column_list.index(item)]
            vis_df=vis_df.append(pd.DataFrame({label_in_df:series}),sort=True)
        #vis_df for boxplot display
        plt.figure(figsize=(20, 6))
        fig=vis_df.boxplot(sym='x',notch=False,fontsize=6,showmeans=False,patch_artist=True,return_type='dict',grid=False)
        for box in fig['boxes']:
            box.set(facecolor='w')
        for median in fig['medians']:
            median.set(color='r')

        plt.ylabel('DBH Range')
        plt.title('DBH Class By Species',fontsize=16)
        plt.show()
    def sortSecond(self,val):
        return val[1]
    def visualization_sorted(self):
        df = self.get_working_df(self.path)
        species_list = self.getScaleItem(df, 'SPECIES_FU')
        column_list=[]
        label_list=[]
        label_number=[]
        for species in species_list:
            number = self.getnumberinScale(df, 'SPECIES_FU', species)
            if (number>50):
                a=str(number)
                label=species+'\n'+a.center(len(species))
                #print(label)

                label_number.append([species,number,label])

        # to sort the label order
        label_number.sort(key=self.sortSecond,reverse=True)
        print(label_number)
        for item in label_number:
            label=item[2]
            species=item[0]
            label_list.append(label)  # label for figure display
            column_list.append(species)

        vis_df = pd.DataFrame(columns=label_list)

        for item in column_list:
            datalist = self.get_datalist(df, 'SPECIES_FU', item, self.oper_field)
            series=pd.Series(np.array(datalist))
            label_in_df=label_list[column_list.index(item)]
            vis_df=vis_df.append(pd.DataFrame({label_in_df:series}),sort=False)


        #vis_df for boxplot display
        plt.figure(figsize=(21, 6))
        fig=vis_df.boxplot(sym='x',notch=False,fontsize=5.8,showmeans=True,patch_artist=True,return_type='dict',grid=False)
        for box in fig['boxes']:
            box.set(facecolor='w')
        for median in fig['medians']:
            median.set(color='black')


        plt.ylabel('DBH Range')
        plt.title('DBH Class By Species',fontsize=16)
        plt.show()


class CalculateBlockLanduseEntropy(ScaleOperation):
    def __init__(self,path,output,weight):
        self.path=path
        self.output=output
        self.weight=weight


    #function for calculate entropy in each block
    def calculate_entropy(self,dataframe,blockID):
        all_weight_list=self.get_datalist(dataframe,'BlockID',blockID,self.weight)
        totalweight_landuse=self.get_sum(all_weight_list)        #get total weight
        landuselist=self.getScaleItem(dataframe,'LAND_ADJ')
        totalnum_landuse=len(landuselist)
        entropy=0
        if totalnum_landuse==1:
            return entropy
        else:
            for items in landuselist:
                sub_weightlist=self.get_datalist(dataframe,'LAND_ADJ',items,self.weight)
                subweight_landuse=self.get_sum(sub_weightlist)
                portion=subweight_landuse/totalweight_landuse
                entropy=entropy+portion*((math.log(portion))/(math.log(totalnum_landuse)))
            return entropy
    def entropy_operation_list(self):
        a=['Entropy']
        return a
    def run_entropy(self):
        df = self.get_working_df(self.path)
        new_df = self.create_new_df(self.entropy_operation_list(), 'BlockID')
        Blocklist=self.getScaleItem(df,'BlockID')
        for ID in Blocklist:
            print(ID)
            try:
                float(ID)


                number = self.getnumberinScale(df, 'BlockID', ID)
                sub_df = df.loc[df['BlockID'] == ID]
                entropy=self.calculate_entropy(sub_df,ID)
                new_df = new_df.append(pd.DataFrame({'BlockID': [ID], 'NUM': number,'Entropy':entropy}), ignore_index=True,
                                   sort=False)
            except:ValueError
        new_df.to_csv(self.output, index=False)


class OverallOperation(ScaleOperation):
    def __init__(self,path,output):
        self.path=path
        self.output=output
    def run_operations(self):
        df=self.get_working_df(self.path)
        DBH_list=self.get_datalist(df,'ALLTAG',1,'D_B_H_PROC')
        minvalue=self.get_min(DBH_list)
        maxvalue=self.get_max(DBH_list)
        meanvalue=self.get_avg(DBH_list)
        stdvalue=self.get_std(DBH_list)
        print('DBH ',minvalue,maxvalue,meanvalue,stdvalue)
        Height_list=self.get_datalist(df,'ALLTAG',1,'TREE_HEIGH')
        minvalue=self.get_min(Height_list)
        maxvalue=self.get_max(Height_list)
        meanvalue=self.get_avg(Height_list)
        stdvalue=self.get_std(Height_list)
        print('TreeHeight ',minvalue,maxvalue,meanvalue,stdvalue)
        SHDIvalue=self.get_SHDI(df,'ALLTAG',1,'SPECIES_FU')
        Simpsonvalue=self.Compute_SimpsonIndex(df,'ALLTAG',1,'SPECIES_FU')
        print('Simpson ',Simpsonvalue)
        print('SHDI ',SHDIvalue)


    def calculate_DBH_for_one(self,dataframe,ScaleField,ScaleItem,attribute):
        single_DBH_list=self.get_datalist(dataframe,ScaleField,ScaleItem,attribute)
        Total_DBH_for_a_specie=self.get_sum(single_DBH_list)
        return Total_DBH_for_a_specie

    def calculate_DBH_for_all(self,dataframe,attribute):
        DBH_list=self.get_datalist(dataframe,'ALLTAG',1,attribute)
        Total_DBH_for_ALL=self.get_sum(DBH_list)
        return Total_DBH_for_ALL

    def relative_dominance(self):
        df = self.get_working_df(self.path)
        specie_set=self.getScaleItem(df,'SPECIES_FU')
        specie_with_num=[]
        for species in specie_set:
            number = self.getnumberinScale(df, 'SPECIES_FU', species)
            combo=[species,number]
            specie_with_num.append(combo)
        specie_with_num.sort(key=lambda x: x[1],reverse=True)

        print(specie_with_num[:14])
        DBH_for_All=self.calculate_DBH_for_all(df,'D_B_H_PROC')
        tree_num=self.getnumberinScale(df,'ALLTAG',1)
        new_df = self.create_new_df(['Abundance Rank','Relative Abundance','Relative_Dominance'], 'SPECIES_FU')
        for item in specie_with_num[:15]:
            specie_name=item[0]
            number=item[1]
            Relative_Abundance=number/tree_num
            Relative_Dominance=self.calculate_DBH_for_one(df,'SPECIES_FU',specie_name,'D_B_H_PROC')/DBH_for_All
            try:
                new_df = new_df.append(pd.DataFrame({'SPECIES_FU': [specie_name], 'NUM': number, 'Abundance Rank':specie_with_num.index(item)+1,'Relative Abundance':Relative_Abundance,'Relative_Dominance':Relative_Dominance}),
                                   ignore_index=False,
                                   sort=False)
            except:ValueError
        new_df.to_csv(self.output, index=False)
    def DBH_range_operation(self):
        df = self.get_working_df(self.path)
        DBH_range_list=list(self.getScaleItem(df,'DBH_CLASS_'))
        DBH_range_list.sort()
        print(DBH_range_list)
        for group in DBH_range_list:
            group_num=self.getnumberinScale(df,'DBH_CLASS_',group)
            group_SHDI=self.get_SHDI(df,'DBH_CLASS_',group,'SPECIES_FU')
            print(group,group_num,group_SHDI)
    def seperate_data_by_species(self):
        df = self.get_working_df(self.path)
        specie_set=self.getScaleItem(df,'SPECIES_FU')
        specie_with_num=[]
        for species in specie_set:
            number = self.getnumberinScale(df, 'SPECIES_FU', species)
            combo=[species,number]
            specie_with_num.append(combo)
        specie_with_num.sort(key=lambda x: x[1],reverse=True)
        writer=pd.ExcelWriter('Species.xlsx')
        for item in specie_with_num[:15]:
            specie_name = item[0]
            new_df=df.loc[df['SPECIES_FU']==specie_name]
            try:
                new_df.to_excel(writer,specie_name,index=False)
            except:

                new_df.to_excel(writer, 'Peltophorum inerme(pterocarpum)',index=False)
            writer.save()
    def pearson_correlation_for_tree_age(self):
        columnlist=['species','NUM',
                    'DBH_HE_cor','DBH_HE_p','DBH_HE_mark',
                    'DBH_CR_cor', 'DBH_CR_p', 'DBH_CR_mark',
                    'HEI_CR_cor', 'HEI_CR_p', 'HEI_CR_mark']
        new_df=pd.DataFrame(columns=columnlist)
        for i in range(15):
            file=pd.ExcelFile(self.path)
            sheetname=file.sheet_names[i]
            df = file.parse(sheetname)
            num=len(df.index)
            DBH=self.series_to_list(df['D_B_H_PROC'])
            HEIGHT=self.series_to_list(df['TREE_HEIGH'])
            CROWN=self.series_to_list(df['CROWN_DIAM'])

            DBH_HE=self.array_preprocessing([DBH,HEIGHT])
            DBH_CR=self.array_preprocessing([DBH,CROWN])
            HEI_CR=self.array_preprocessing([HEIGHT,CROWN])
            # I need to figure out how many are not the nums, and give a mark, then delect in both array

            #cor_1, pval_1 = stats.pearsonr(DBH, Tree_Height)
            cor_0, pval_0 = stats.pearsonr(DBH_HE[0], DBH_HE[1])
            if (pval_0<0.05):
                mark0=1
            else:
                mark0=0
            cor_1, pval_1 = stats.pearsonr(DBH_CR[0], DBH_CR[1])
            if (pval_1<0.05):
                mark1=1
            else:
                mark1=0
            cor_2, pval_2 = stats.pearsonr(HEI_CR[0], HEI_CR[1])
            if (pval_2<0.05):
                mark2=1
            else:
                mark2=0
            new_df = new_df.append(pd.DataFrame({'species': [sheetname], 'NUM': num,
                                                 'DBH_HE_cor':cor_0,'DBH_HE_p':pval_0,'DBH_HE_mark':mark0,
                                                 'DBH_CR_cor':cor_1, 'DBH_CR_p':pval_1, 'DBH_CR_mark':mark1,
                                                 'HEI_CR_cor':cor_2, 'HEI_CR_p':pval_2, 'HEI_CR_mark':mark2
                                                 }), ignore_index=True, sort=False)
        new_df.to_csv(self.output,index=False)
    def pearson_correlation_with_health(self,factor):
        new_df = pd.DataFrame()
        for i in range(15):
            file=pd.ExcelFile(self.path)
            sheetname=file.sheet_names[i]
            df = file.parse(sheetname)

            groupA=self.series_to_list(df['HealthClass'])
            groupB=self.series_to_list(df[factor])
            pre_processed_group=self.array_preprocessing([groupA,groupB])

            try:
                num=len(pre_processed_group[0])
                cor, pval = stats.pearsonr(pre_processed_group[0], pre_processed_group[1])

                if (pval<0.05)&(num>=5):
                    mark=1
                else:
                    mark=0
            except:
                num=0
                cor=None
                pval=None

                mark=0
            new_df = new_df.append(pd.DataFrame(
                {'Species': [sheetname], 'NUM': num,  'Correlation': cor, 'PVALUE': pval, 'MARK': mark}),
                                   ignore_index=True,
                                   sort=False)
            new_df.to_csv(factor+'_Health.csv',index=False)
    def ageindex_calculation(self):
        writer=pd.ExcelWriter('Species_addAGE.xlsx')
        for i in range(15):
            file=pd.ExcelFile(self.path)
            sheetname=file.sheet_names[i]
            df = file.parse(sheetname)
            df['AgeIndex']=None
            df['AgeGroup']=None
            df['HealthIndex']=None
            DBH_datalist=self.get_datalist(df,'ALLTAG',1,'D_B_H_PROC')
            DBH_range=self.get_max(DBH_datalist)-self.get_min(DBH_datalist)
            for row in range(0,len(df),1):
                DBH_age=df.at[row, 'D_B_H_PROC']-self.get_min(DBH_datalist)
                DBH_age_index=DBH_age/DBH_range
                if (DBH_age_index<=0.4):
                    DBH_age_Group='YOUNG'
                elif (DBH_age_index>0.7):
                    DBH_age_Group = 'OLD'
                else:
                    DBH_age_Group = 'MATURE'
                df.at[row,'AgeIndex']=DBH_age_index
                df.at[row,'AgeGroup']=DBH_age_Group
                df.at[row,'HealthIndex']=df.at[row,'BRANCH_DMG']+df.at[row,'TRUNK_DMG']+df.at[row,'Root_DMG']
                #function for health indicator
            df.to_excel(writer, sheetname, index=False)
            writer.save()

    def getdict(self,dataframe, key_location, content_location):
        dict1 = {}
        for i in range(0, len(dataframe), 1):
            dict1[dataframe.at[i, key_location]] = dataframe.at[i, content_location]
        return dict1

    def planter_classification(self):
        dict_df=pd.read_csv('Planter_classify.csv')
        writer = pd.ExcelWriter('SpeciesaddPlanter_classify.xlsx')

        dictC1=self.getdict(dict_df,'PlanterType','Classify_Type1')
        dictC2=self.getdict(dict_df,'PlanterType','Classify_Type2')
        for i in range(15):
            file = pd.ExcelFile(self.path)
            sheetname = file.sheet_names[i]
            df = file.parse(sheetname)
            df['PLANTER_C1']=None
            df['PLANTER_C2']=None
            for row in range(0,len(df),1):
                Planter_type=df.at[row, 'PLANTER_TY']
                df.at[row,'PLANTER_C1']=dictC1[Planter_type]
                df.at[row,'PLANTER_C2']=dictC2[Planter_type]

            df.to_excel(writer, sheetname, index=False)
            writer.save()

    def ttest_for_factor_Health(self,factor):
        new_df=pd.DataFrame()
        for i in range(15):
            file=pd.ExcelFile(self.path)
            sheetname=file.sheet_names[i]
            df = file.parse(sheetname)
            num = len(df.index)

            factorlist=list(self.getScaleItem(df,factor))
            for item in factorlist:
                groupA=self.get_datalist(df,factor,item,'HealthIndex')
                groupB=self.get_datalist_exclude(df,factor,item,'HealthIndex')
                if ((len(groupA)<=5)or (len(groupB)<=5)):
                    mark=0
                    t_stat=None
                    p_val=None
                else:
                    t_stat,p_val=stats.ttest_ind(groupA,groupB,equal_var=False)
                    if p_val<0.05:
                        mark=1
                    else:
                        mark=0
                new_df = new_df.append(pd.DataFrame({'Species': [sheetname], 'NUM':num,factor:item,'T_TEST':t_stat,'PVALUE':p_val,'MARK':mark}), ignore_index=True,
                                       sort=False)
        outputname=factor+'_ttest_HC.csv'
        new_df.to_csv(outputname,index=False)
    def remove_non_digi(self,lista):
        emptylist = []
        for i in range(0,len(lista)):
            try:
                a=float(lista[i])
                emptylist.append(a)
            except:ValueError
        return emptylist

    '''
    to do the t-test for a binary categorical and an interval factor 
    '''
    def ttest_for_binaryfactor_continiousfactor(self,factor1,factor2):
        new_df=pd.DataFrame()
        item=factor1+'&'+factor2
        for i in range(15):
            file=pd.ExcelFile(self.path)
            sheetname=file.sheet_names[i]
            df = file.parse(sheetname)

            groupA=self.remove_non_digi(self.get_datalist(df,factor1,1,factor2))
            groupB=self.remove_non_digi(self.get_datalist(df,factor1,0,factor2))
            num=len(groupA)+len(groupB)
            if ((len(groupA)<=5)or (len(groupB)<=5)):
                mark=0
                t_stat=None
                p_val=None
            else:
                t_stat,p_val=stats.ttest_ind(groupA,groupB,equal_var=False)
                if p_val<0.05:
                    mark=1
                else:
                    mark=0
            new_df = new_df.append(pd.DataFrame({'Species': [sheetname], 'NUM':num,factor1+'&'+factor2:item,'T_TEST':t_stat,'PVALUE':p_val,'MARK':mark}), ignore_index=True,
                                       sort=False)
        outputname=factor1+factor2+'_ttest.csv'
        new_df.to_csv(outputname,index=False)
    def average_betweenness(self):
        df = self.get_working_df(self.path)
        new_df=pd.DataFrame()
        MergedStreetID_list = list(self.getScaleItem(df, 'MergedStreetID'))
        MergedStreetID_list.sort()
        for ID in MergedStreetID_list:
            betweennesslist=self.get_datalist(df,'MergedStreetID',ID,'betweenness')
            betweennessvalue=self.get_avg(betweennesslist)
            new_df=new_df.append(pd.DataFrame({'MergedStreetID':[ID],'Betweenness':betweennessvalue}),ignore_index=True,
                                       sort=False)
        new_df.to_csv(self.output,index=False)
    def add_information(self):                                   # could be change to add other information
        writer=pd.ExcelWriter('Species_addBetweenness.xlsx')
        ref_df = self.get_working_df('street_betweenness_upgrade_results.xlsx')           # the reference excel file
        for i in range(15):
            file=pd.ExcelFile(self.path)
            sheetname=file.sheet_names[i]
            df = file.parse(sheetname)
            df['Betweenness']=None
            for row in range(0,len(df),1):
                MergedStreetID=df.at[row,'StreetMergeID']
                Betweennesslist=self.get_datalist(ref_df,'MergedStreetID',MergedStreetID,'Betweenness')
                                   # ref_record contains many information that can be derived

                try:
                    df.at[row,'Betweenness']=Betweennesslist[0]
                except:
                    df.at[row,'Betweenness']='(Empty)'
            df.to_excel(writer,sheetname,index=False)
            writer.save()
    def add_distance(self):
        writer = pd.ExcelWriter('Species_addDistance.xlsx')
        for i in range(15):
            file=pd.ExcelFile(self.path)
            sheetname=file.sheet_names[i]
            df = file.parse(sheetname)
            new_df=self.calculate_intercept(df)
            new_df.to_excel(writer,sheetname,index=False)
            writer.save()


    def add_column(self,pathname,attri_column_name,attri_name,outputname):
        writer = pd.ExcelWriter(outputname)
        for i in range(15):
            file=pd.ExcelFile(pathname)
            sheetname=file.sheet_names[i]
            df = file.parse(sheetname)
            df[attri_name]=None
            for row in range(0,len(df),1):
                value=df.at[row,attri_column_name]
                if (value==attri_name):
                    tag=1
                else:
                    tag=0
                df.at[row,attri_name]=tag
            df.to_excel(writer,sheetname,index=False)
            writer.save()
    def add_column_canopy(self,pathname,outputname):
        writer = pd.ExcelWriter(outputname)
        for i in range(15):
            file=pd.ExcelFile(pathname)
            sheetname=file.sheet_names[i]
            df = file.parse(sheetname)
            df['DISCO&WILTED']=None
            for row in range(0,len(df),1):
                value=df.at[row,'CANOPY_APP']
                if (value=='Discolored'):
                    tag=1
                elif(value=='Wilted'):

                    tag=1
                else:
                    tag=0
                df.at[row,'DISCO&WILTED']=tag
            df.to_excel(writer,sheetname,index=False)
            writer.save()
    def add_merge_column(self,pathname,column1,column2,column_merged,outputname):
        writer = pd.ExcelWriter(outputname)
        for i in range(15):
            file=pd.ExcelFile(pathname)
            sheetname=file.sheet_names[i]
            df = file.parse(sheetname)
            df[column_merged]=None
            for row in range(0, len(df), 1):
                indicator1=df.at[row,column1]
                indicator2=df.at[row,column2]
                if (indicator1+indicator2>=1):
                    tag=1
                else:
                    tag=0
                df.at[row, column_merged] = tag
            df.to_excel(writer,sheetname,index=False)
            writer.save()
    def healthclass_modify(self,pathname,outputname):
        writer = pd.ExcelWriter(outputname)
        for i in range(15):
            file = pd.ExcelFile(pathname)
            sheetname = file.sheet_names[i]
            df = file.parse(sheetname)
            df['HealthClassModifier']=None
            for row in range(0, len(df), 1):
                healthclass=df.at[row,'HealthClass']
                if ((healthclass==0) or (healthclass==1) or (healthclass==2)):
                    healthModifier=1
                elif ((healthclass == 3) or (healthclass == 4)):
                    healthModifier=2
                else:
                    healthModifier=3
                df.at[row,'HealthClassModifier']=healthModifier
            df.to_excel(writer, sheetname, index=False)
            writer.save()
    def ageindex_modify(self,pathname,outputname):
        writer = pd.ExcelWriter(outputname)
        for i in range(15):
            file = pd.ExcelFile(pathname)
            sheetname = file.sheet_names[i]
            df = file.parse(sheetname)
            df['AgeClass']=None
            for row in range(0, len(df), 1):
                DBHIndex=df.at[row,'DBH_CLASS_']
                if (DBHIndex=='0-15'):
                    AgeClass=0
                elif (DBHIndex == '15-30'):
                    AgeClass = 1
                elif (DBHIndex == '30-60'):
                    AgeClass = 2
                elif (DBHIndex == '61-90'):
                    AgeClass = 3
                elif (DBHIndex == '90-150'):
                    AgeClass = 4
                else:
                    AgeClass = 5

                df.at[row,'AgeClass']=AgeClass
            df.to_excel(writer, sheetname, index=False)
            writer.save()
    def add_health_class(self,pathname,outputname):
        writer = pd.ExcelWriter(outputname)
        for i in range(15):
            file = pd.ExcelFile(pathname)
            sheetname = file.sheet_names[i]
            df = file.parse(sheetname)
            df['HealthClass']=None
            for row in range(0, len(df), 1):

                TrunkCavity=df.at[row,'Trunk Cavity']
                TrunkDecay=df.at[row,'Signs of Rot/Decay']
                CorG=df.at[row,'Cut or Gridling']
                Split=df.at[row,'Cracks /Splits']
                Severe=TrunkCavity+TrunkDecay+CorG+Split

                BorD=df.at[row,'BorD Limbs']
                Mechanical=df.at[row,'Mechanical Injury']
                CER=df.at[row,'CER']
                Lean=df.at[row,'Severe Lean']
                Disturbance=df.at[row,'Disturbance']
                Lean_Disturb=Lean*Disturbance
                Moderate=BorD+Mechanical+CER+Lean_Disturb

                Sparse=df.at[row,'Sparse']
                Disco_Wilted=df.at[row,'DISCO&WILTED']
                Parastic=df.at[row,'Parasitic Plant']
                Visible=df.at[row,'Visible Roots']
                Co_dominant=df.at[row,'Co-dominant Trunks']
                Minor=Sparse+Disco_Wilted+Parastic+Visible+Co_dominant

                Health_class=self.health_classifier(Severe,Moderate,Minor)
                df.at[row,'HealthClass']=Health_class

            df.to_excel(writer, sheetname, index=False)
            writer.save()
    def health_classifier(self,a,b,c):

        if a>1:
            return 6
        elif a==1:
            return 5
        else:
            if b>1:
                return 4
            elif b==1:
                return 3
            else:
                if c>1:
                    return 2
                elif c==1:
                    return 1
                else:
                    return 0
    def vis_healthclass(self):
        for i in range(5):
            file=pd.ExcelFile(self.path)
            sheetname=file.sheet_names[i]
            df = file.parse(sheetname)
            HealthIndex=df['HealthIndex'].copy()
            HealthClass=df['HealthClass'].copy()
            scale=HealthClass/HealthIndex
            #plt.plot(range(len(HealthIndex)), HealthIndex, 'b', label='HealthIndex')
            #plt.plot(range(len(HealthClass)), HealthClass, 'r', label='HealthClass')
            plt.hist(scale, bins=50)
            plt.show()
    def sheetmerge(self,pathname,outputname):
        writer = pd.ExcelWriter(outputname)
        new_df=pd.DataFrame()
        for i in range(15):
            file = pd.ExcelFile(pathname)
            sheetname = file.sheet_names[i]
            df = file.parse(sheetname)
            new_df=new_df.append(df)
        new_df.to_excel(writer,'sheet1',index=False)
        writer.save()
    def new_DBH_class(self,pathname,outputname):
        df=pd.read_csv(pathname)
        df['DBH_CLASS_NEW']=None
        for i in range(0,len(df),1):
            DBH=float(df.at[i,'D_B_H_PROC'])
            if DBH<15:
                DBH_CLASS='0-15'
            elif (DBH>=15)&(DBH<30):
                DBH_CLASS='15-30'
            elif (DBH < 45) & (DBH >= 30):
                DBH_CLASS = '30-45'
            elif (DBH < 60) & (DBH >= 45):
                DBH_CLASS = '45-60'
            elif (DBH < 75) & (DBH >= 60):
                DBH_CLASS = '60-75'
            elif (DBH < 90) & (DBH >= 75):
                DBH_CLASS = '75-90'
            else:
                DBH_CLASS ='90-310'
            df.at[i,'DBH_CLASS_NEW']=DBH_CLASS
        df.to_csv(outputname,index=False)

    def DBH_range_new_operation(self):
        df = self.get_working_df(self.path)
        DBH_range_list = list(self.getScaleItem(df, 'DBH_CLASS_NEW'))
        DBH_range_list.sort()
        print(DBH_range_list)
        for group in DBH_range_list:
            group_num = self.getnumberinScale(df, 'DBH_CLASS_NEW', group)
            group_SHDI = self.get_SHDI(df, 'DBH_CLASS_NEW', group, 'SPECIES_FU')
            print(group, group_num, group_SHDI)
    def dict_for_block_distance(self,dictsource):
        df=pd.read_csv(dictsource)
        dict1={}
        for i in range(0,len(df),1):
            block=df.at[i,'BlockID']
            distance=df.at[i,'Intercept']
            dict1[block]=distance

        return dict1
    def add_distance(self):
        df = self.get_working_df(self.path)
        dict1=self.dict_for_block_distance('block_intercept_done.csv')
        df['Block Distance']=None
        for i in range (0,len(df),1):
            block=df.at[i,'BlockID']
            try:
                distance=dict1[block]
            except:KeyError
            df.at[i,'Block Distance']=distance
        df.to_csv('new_DBHClass_add_dis.csv')
    def block_info_table(self):
        df=self.get_working_df(self.path)
        new_df=pd.DataFrame()
        distance_dict=self.dict_for_block_distance('block_intercept_done.csv')
        BlockID_list=list(self.getScaleItem(df,'BlockID'))

        for ID in BlockID_list:


            try:
                distance=distance_dict[ID]
            except:
                distance='NA'
            num = self.getnumberinScale(df,'BlockID',ID)
            SHDI = self.get_SHDI(df, 'BlockID', ID, 'SPECIES_FU')
            DBHclass_list=self.get_datalist_string(df,'BlockID',ID,'DBH_CLASS_NEW')

            c0015=DBHclass_list.count('0-15')
            c1530 = DBHclass_list.count('15-30')
            c3045 = DBHclass_list.count('30-45')
            c4560 = DBHclass_list.count('45-60')
            c6075 = DBHclass_list.count('60-75')
            c7590 = DBHclass_list.count('75-90')
            c90310 = DBHclass_list.count('90-310')

            new_df = new_df.append(pd.DataFrame({'BlockID': [ID], 'NUM':num,'Distance':distance,'SHDI':SHDI,'DBH:0-15':c0015,'DBH:15-30':c1530,'DBH:3045':c3045,
                                             'DBH:45-60':c4560,'DBH:60-75':c6075,'DBH:75-90':c7590,'DBH:90-310':c90310,}), ignore_index=True,
                                   sort=False)


        new_df=new_df.dropna(axis=0)
        #new_df=new_df.sort_values('Distance', ascending=True,axis=1)
        new_df.to_csv('Block_info.csv',index=False)


    def landuse_info_table(self):
        df=self.get_working_df(self.path)
        new_df=pd.DataFrame()

        Landuse_list=list(self.getScaleItem(df,'ADJ_LAND'))
        Landuse_list.sort()

        for landuse in Landuse_list:

            num = self.getnumberinScale(df,'ADJ_LAND',landuse)
            SHDI = self.get_SHDI(df, 'ADJ_LAND', landuse, 'SPECIES_FU')
            DBHclass_list=self.get_datalist_string(df,'ADJ_LAND',landuse,'DBH_CLASS_NEW')

            c0015=DBHclass_list.count('0-15')
            c1530 = DBHclass_list.count('15-30')
            c3045 = DBHclass_list.count('30-45')
            c4560 = DBHclass_list.count('45-60')
            c6075 = DBHclass_list.count('60-75')
            c7590 = DBHclass_list.count('75-90')
            c90310 = DBHclass_list.count('90-310')

            new_df = new_df.append(pd.DataFrame({'Landuse': [landuse], 'NUM':num,'SHDI':SHDI,'DBH:0-15':c0015,'DBH:15-30':c1530,'DBH:3045':c3045,
                                             'DBH:45-60':c4560,'DBH:60-75':c6075,'DBH:75-90':c7590,'DBH:90-310':c90310,}), ignore_index=True,
                                   sort=False)


        new_df.to_csv('Landuse_info.csv',index=False)
    def count_old_tree_on_street(self):
        df = self.get_working_df(self.path)
        new_df = pd.DataFrame()

        df['c90310']=None
        for i in range(0,len(df),1):
            DBH=float(df.at[i,'D_B_H_PROC'])
            if (DBH>=90):
                df.at[i,'c90310']=1
            else:
                df.at[i,'c90310']=0
        for number in range(0,312):
            count=0
            for record in range(0,len(df),1):
                try:
                    streetID=int(df.at[record,'StreetID_1'])
                except: ValueError

                if (streetID==number):
                    count=count+df.at[record,'c90310']
                    total_tree=df.at[record,'TotalTree']
            try:
                oldratio=count/total_tree
            except: ValueError
            new_df=new_df.append(pd.DataFrame({'StreetSGID':[number],'Old_count':count,'TotalTree':total_tree,'old_ratio':oldratio}),ignore_index=True,
                                   sort=False)
        new_df.to_csv('oldratio_info.csv', index=False)
    def count_richness(self):
        df=pd.read_csv(self.path)
        new_df = pd.DataFrame()

        streetMerge_list = list(self.getScaleItem(df, 'StreetMergeID'))
        streetMerge_list.sort()

        for item in streetMerge_list:
            specie_list=self.get_datalist_string(df,'StreetMergeID',item,'SPECIES_FU')
            print(len(specie_list))
            richness=self.get_set_num(specie_list)
            new_df = new_df.append(pd.DataFrame(
                {'StreetBlockID': [item], 'Richness': richness, }),
                                   ignore_index=True,
                                   sort=False)
        new_df.to_csv('richness_info.csv', index=False)







if __name__=='__main__':
    field=['CH','CLR','TRG','DIS','BC','TC','RC']
    filepath='Tree_Data_Street_review_1205.xls'
    new_path='DMG_Added.xls'
    #for item in field:
    #insta=CleanField(field[6],filepath)
    #insta.update_data()
    #columnname='ADJACENT_L'
    #a=CleanField(field,filepath)
    #a.update_data()
    #b=ExtendField(columnname,filepath,'landuse_ext.xls')
    #b.update_extended_data()
    #c=LanduseMarge(new_path,'landuse_dict.xlsx','landuse_updated.xls')
    #c.updata_landuse_data()
    columnname='ADJ_LAND'
    #d=ExtendField(columnname,'landuse_updated.xls','landuse_updated.xls')
    #d.update_extended_data()
    #streetmerge=ScaleOperation('visualization.xls','streetmerge_vis.csv','StreetMergeID','NUM')
    #streetmerge.operation()
    #street=ScaleOperation('visualization.xls','street_vis.csv','StreetID','NUM')
    #street.operation()
    #block=ScaleOperation('visualization.xls','block_vis.csv','BlockID','NUM')
    #block.operation()
    #test=SpeciesDescribe('visualization_heritage_out.xlsx','species.csv','Root_DMG')
    #test.visualization()
    #entropy=CalculateBlockLanduseEntropy('Tree_Data_Street_review_1205.xls','entropy.csv','Landuse_Weight')
    #entropy.run_entropy()
    overall=OverallOperation('Overall_Area_Trees.xlsx','Relative Dominanace.csv')

    #NS=OverallOperation('NS_Orientation_Trees.xlsx','')
    #EW=OverallOperation('EW_Orientation_Trees.xlsx','')
    Heritage=OverallOperation('Heritage_Area_Trees.xlsx','')
    #Heritage.DBH_range_operation()
    #overall.relative_dominance()
    #overall.DBH_range_operation()
    species=OverallOperation('Species.xlsx','pearson_stats.csv')
    #species.ageindex_calculation()
    #species.pearson_correlation_for_tree_age()

    landuse=OverallOperation('Species_addAGE.xlsx','')
    #landuse.ttest_for_factor_Health('ADJ_LAND')
    #landuse.ttest_for_factor_Health('PLANTER_TY')
    #landuse.ttest_for_factor_Health('Orientation')

    #landuse.planter_classification()
    plantertype=OverallOperation('Planter_classify.xlsx','')
    #plantertype.ttest_for_factor_Health('PLANTER_C1')
    #plantertype.ttest_for_factor_Health('PLANTER_C2')
    betweenness=OverallOperation('street_with_betweenness_upgrade.xls','street_with_betweenness_upgrade.csv')
    analysisbetweenness=OverallOperation('Species_addBetweenness.xlsx','')
    #analysisbetweenness.pearson_correlation_with_health('Betweenness')
    #analysisbetweenness.ttest_for_binaryfactor_continiousfactor('Orientation','Betweenness')
    analysisdistance=OverallOperation('Species_addDistance.xlsx','')

    #analysisdistance.add_column('SpeciesaddPlanter_classify.xlsx','LEAN','Severe Lean','Lean.xlsx')
    #analysisdistance.add_column('BorD.xlsx','CROWN_DENS','Sparse','Sparse.xlsx')
    #analysisdistance.add_merge_column('Cut&Gridled.xlsx','Broken Limbs','Dead Limbs','BorD Limbs','BorD.xlsx')
    #analysisdistance.add_health_class('Sparse.xlsx','SpeciesAddHealthClass.xlsx')
    Newhealth=OverallOperation('SpeciesAddHealthClass.xlsx','')
    #Newhealth.vis_healthclass()
    #Newhealth.healthclass_modify('SpeciesrenewAGE.xlsx','Species_new.xlsx')
    #Newhealth.ageindex_modify('Speciesrenewhealth.xlsx','SpeciesrenewAGE.xlsx')
    NewAgeClass=OverallOperation('Species_new.xlsx','')
    #NewAgeClass.sheetmerge('Species_new.xlsx','AllSpecies_new.xlsx')
    #NewAgeClass.new_DBH_class('block_landuse0221.csv','new_DBHClass.csv')
    NEWDBHCLASS=OverallOperation('new_DBHClass_add_dis.xlsx','')
    #NEWDBHCLASS.DBH_range_new_operation()
    #NEWDBHCLASS.add_distance()
    #NEWDBHCLASS.landuse_info_table()
    #a_20190402=OverallOperation('Overall_Area_Trees.xlsx','')
    #a_20190402.count_old_tree_on_street()


    test=SpeciesDescribe('Tree_Data_Street_review_0410.xls','species_0410.csv','D_B_H_PROC')
    #test.visualization_sorted()

    richness=OverallOperation('visualization.csv','')
    richness.count_richness()