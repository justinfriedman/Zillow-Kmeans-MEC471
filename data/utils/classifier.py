import pandas
import matplotlib.pyplot as plt
import numpy as np
import sys
import time


#read in hardcoded dta dataframe to pandas dataframe
dfOrig = pandas.read_stata('../ZillowData_realPrice-train.dta')
dfOrigCols = dfOrig.columns
df = dfOrig
#create new features for the average of every feature in each cluster
count = 0
numCols = len(dfOrig.index)
for feature in dfOrigCols:
    name = feature
    df[name +'_AVE_C1'] = pandas.Series(1200, index=df.index)

    df[name +'_AVE_C1'] = df[name][df['_clus_k5'] == 1].mean()
    count = count + 1
    sys.stdout.write('📟 on feature ' + str(count) + ' of ' + str(numCols) + '\r')
    sys.stdout.flush()

print("✅ cluster 1 done, starting cluster 2")
count = 0
for feature in dfOrigCols:
    name = feature
    df[name +'_AVE_C2'] = pandas.Series(1200, index=df.index)
    df[name +'_AVE_C2']= df[name][df['_clus_k5'] == 2].mean()
    count = count + 1
    sys.stdout.write('📟 on feature ' + str(count) + ' of ' + str(numCols) + '\r')
    sys.stdout.flush()
print("✅ cluster 2 done, starting cluster 3")
count = 0
for feature in dfOrigCols:
    name = feature
    df[name +'_AVE_C3'] = pandas.Series(1200, index=df.index)
    df[name +'_AVE_C3']= df[name][df['_clus_k5'] == 3].mean()
    count = count + 1
    sys.stdout.write('📟 on feature ' + str(count) + ' of ' + str(numCols) + '\r')
    sys.stdout.flush()
print("✅ cluster 3 done, starting cluster 4")
count = 0
for feature in dfOrigCols:
    name = feature
    df[name +'_AVE_C4'] = pandas.Series(1200, index=df.index)
    df[name +'_AVE_C4']= df[name][df['_clus_k5'] == 4].mean()
    count = count + 1
    sys.stdout.write('📟 on feature ' + str(count) + ' of ' + str(numCols) + '\r')
    sys.stdout.flush()
print("✅ cluster 4 done, starting cluster 5")
count = 0
for feature in dfOrigCols:
    name = feature
    df[name +'_AVE_C5'] = pandas.Series(1200, index=df.index)
    df[name +'_AVE_C5']= df[name][df['_clus_k5'] == 5].mean()
    count = count + 1
    sys.stdout.write('📟 on feature ' + str(count) + ' of ' + str(numCols) + '\r')
    sys.stdout.flush()




#
# for each in df.columns:
#     print(each)




print("starting classification")
#read in hardcoded dta dataframe to pandas dataframe
#test data
# dfTestOrig = pandas.read_stata('../ames_test.dta')
#train data
dfTestOrig = pandas.read_stata('../test_dataFILTERED_FINAL_11AM.dta')
dfTestOrigCols = dfTestOrig.columns
dfTest = dfTestOrig

dfTest['cluster_classification'] = pandas.Series(len(dfTest.index), index=dfTest.index)
dfTest['cluster_1_score'] = pandas.Series(len(dfTest.index), index=dfTest.index)
dfTest['cluster_2_score'] = pandas.Series(len(dfTest.index), index=dfTest.index)
dfTest['cluster_3_score'] = pandas.Series(len(dfTest.index), index=dfTest.index)
dfTest['cluster_4_score'] = pandas.Series(len(dfTest.index), index=dfTest.index)
dfTest['cluster_5_score'] = pandas.Series(len(dfTest.index), index=dfTest.index)


for row in dfTest.index:
    clusterTally = [0,0,0,0,0]
    colName = ''
    for feature in dfTestOrigCols:
        sys.stdout.write('📟 on row ' + str(row) + ' and col ' + str(feature) + '\r')
        sys.stdout.flush()
        colName = feature
        val = dfTest.get_value(row, feature, takeable=False)
        #generate observation residuals when compared to feature average for each cluster, 0.00001 added to stop div by 0 error
        # Mean Bruteforce
        # val_residual_cluster1Percent = (df.at[row, feature+'_AVE_C1'] - val)/(df.get_value(row, feature+'_AVE_C1', takeable=False) + 0.0000000001)
        # val_residual_cluster2Percent = (df.at[row, feature+'_AVE_C2'] - val)/(df.get_value(row, feature+'_AVE_C2', takeable=False)+ 0.0000000001)
        # val_residual_cluster3Percent = (df.at[row, feature+'_AVE_C3'] - val)/(df.get_value(row, feature+'_AVE_C3', takeable=False)+ 0.0000000001)
        # val_residual_cluster4Percent = (df.at[row, feature+'_AVE_C4'] - val)/(df.get_value(row, feature+'_AVE_C4', takeable=False)+ 0.0000000001)
        # val_residual_cluster5Percent = (df.at[row, feature+'_AVE_C5'] - val)/(df.get_value(row, feature+'_AVE_C5', takeable=False)+ 0.0000000001)

        #ln Bruteforce
        if abs(df.at[3, feature+'_AVE_C1'] - val) != 0:
            val_residual_cluster1Percent = np.log(abs(df.at[3, feature+'_AVE_C1'] - val))
        if (df.at[3, feature+'_AVE_C1'] - val) == 0:
            val_residual_cluster1Percent = 0

        if abs(df.at[3, feature+'_AVE_C2'] - val) != 0:
            val_residual_cluster2Percent = np.log(abs(df.at[3, feature+'_AVE_C2'] - val))
        if (df.at[3, feature+'_AVE_C2'] - val) == 0:
            val_residual_cluster2Percent = 0


        if abs(df.at[3, feature+'_AVE_C3'] - val) != 0:
            val_residual_cluster3Percent = np.log(abs(df.at[3, feature+'_AVE_C3'] - val))
        if (df.at[3, feature+'_AVE_C3'] - val) == 0:
            val_residual_cluster3Percent = 0


        if abs(df.at[3, feature+'_AVE_C4'] - val) != 0:
            val_residual_cluster4Percent = np.log(abs(df.at[3, feature+'_AVE_C4'] - val))
        if (df.at[3, feature+'_AVE_C4'] - val) == 0:
            val_residual_cluster4Percent = 0


        if abs(df.at[3, feature+'_AVE_C5'] - val) != 0:
            val_residual_cluster5Percent = np.log(abs(df.at[3, feature+'_AVE_C5'] - val))
        if (df.at[3, feature+'_AVE_C5'] - val) == 0:
            val_residual_cluster5Percent = 0


        residuals = [val_residual_cluster1Percent,val_residual_cluster2Percent,val_residual_cluster3Percent,val_residual_cluster4Percent,val_residual_cluster5Percent]










        #find the smallest residual % cluster
        smallest = residuals.index(min(residuals))


        #assign cluster
        if smallest == 0:
            clusterTally[0] = clusterTally[0] + 1
        if smallest == 1:
            clusterTally[1] = clusterTally[1] + 1
        if smallest == 2:
            clusterTally[2] = clusterTally[2] + 1
        if smallest == 3:
            clusterTally[3] = clusterTally[3] + 1
        if smallest == 4:
            clusterTally[4] = clusterTally[4] + 1





    dfTest.at[row,'cluster_1_score'] = clusterTally[0]
    dfTest.at[row,'cluster_2_score'] = clusterTally[1]
    dfTest.at[row,'cluster_3_score'] = clusterTally[2]
    dfTest.at[row,'cluster_4_score'] = clusterTally[3]
    dfTest.at[row,'cluster_5_score'] = clusterTally[4]

    #for entire observation, find which cluster is most descriptive
    winner = clusterTally.index(max(clusterTally)) + 1
    dfTest.at[row, 'cluster_classification'] = winner



    #export dta and excel because we couldnt export dta because sometimes python is a globalist cuck like jared kushner
print("✅ exporting train_output_with_feature_means.dta ⏬")
df.to_stata("output/train_output_with_feature_means.dta", convert_dates=None, write_index=True, encoding='latin-1', byteorder=None, time_stamp=None, data_label=None, variable_labels=None)

# dfTest.to_stata("output/classified_test_output.dta", convert_dates=None, write_index=True, encoding='latin-1', byteorder=None, time_stamp=None, data_label=None, variable_labels=dfTest.columns)
print("✅ exporting test_output_classified.xlsx ⏬")
dfTest.to_excel("output/test_output_classified.xlsx", sheet_name='Sheet1', na_rep='', float_format=None, columns=None, header=True, index=True, index_label=None, startrow=0, startcol=0, engine=None, merge_cells=True, encoding=None, inf_rep='inf', verbose=True, freeze_panes=None)
