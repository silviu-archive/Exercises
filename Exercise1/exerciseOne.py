import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats


def main():

    # Read in the dataset as a dataframe
    df = pd.read_csv('dataset1.csv')

    #Observe column names that have whitespaces in them - remove it
    df.rename(columns = lambda x: x.strip(), inplace=True)

    #Check the datatypes in the dataframe
    datatypes = df.dtypes

    #Check the number of unique countries in the promotion
    uniqueCountriesCountTuple = np.unique(df['Country'], return_counts=True)
    for country, count in zip(uniqueCountriesCountTuple[0], uniqueCountriesCountTuple[1]):
        print('Country: %s, Number of occurences: %s' % (country, count))

    #Check the number of unique groups in the promotion
    uniqueGroupCountTuple = np.unique(df['Group'], return_counts=True)
    for group, count in zip(uniqueGroupCountTuple[0], uniqueGroupCountTuple[1]):
        print('Group: %s, Number of occurences: %s' % (group, count))

    #Calculate T-test values for activity based on group membership
    group1Activity = df.loc[df['Group'] == 1]['Activity']
    group2Activity = df.loc[df['Group'] == 2]['Activity']
    group3Activity = df.loc[df['Group'] == 3]['Activity']
    group4Activity = df.loc[df['Group'] == 4]['Activity']
    groupControlActivity = df.loc[df['Group'] == 5]['Activity']

    group1Test = scipy.stats.ttest_ind(group1Activity, groupControlActivity)
    group2Test = scipy.stats.ttest_ind(group2Activity, groupControlActivity)
    group3Test = scipy.stats.ttest_ind(group3Activity, groupControlActivity)
    group4Test = scipy.stats.ttest_ind(group4Activity, groupControlActivity)

    #From the T-test results, we can observe that group 1 and group 3 have p-values <0.05
    #In this case, we reject the null hypothesis that the group membership does not have any impact on activity
    #These results show that groups 1&3, with bonuses of 20$ and 25$ respectively, had significant impact on
    #player activity compared to groups 2&3, with bonuses of 15$ and 5% respectively.

    group1vs3 = scipy.stats.ttest_ind(group1Activity, group3Activity)
    #By comparing group 1 and 3 against each other, we can conclude that there is no statistical significance
    #between the activity of the two, and thus making group 1 the better alternative, as it provides a significant
    #increase in player activity at a lower cost

    #Now that we have concluded that 2 campaigns were significant in modifying activity,
    #we need to understand what was the value of these campaigns irrespective of country influence
    #In order to do so, but outside the scope of the current task, we could create a linear model to see the influence
    #of promotions on activity, and then we could add features related to countries to see how the coefficients
    #for promotion change
    #However, for now, as we have concluded that groups 1 and 3 offer significant improvements, we shall describe
    #these improvements across countries

    #Remove groups 2 and 4 from the dataset and calculate Gross Revenue
    df = df.loc[(df['Group'] != 2) & (df['Group'] != 4)]
    df['GrossRev'] = df['Activity'] * 0.0213

    #Create basic statistics about the grouped dataframe by country
    dfDescribe = df.groupby(['Country', 'Group']).describe()
    #print(dfDescribe)

    #Plot activity and revenue grouped by 'Group' for each country
    sns.boxplot(x='Country', y='Activity', hue='Group', data=df, showfliers=False)
    plt.figure()
    sns.boxplot(x='Country', y='GrossRev', hue='Group', data=df, showfliers=False)

    #From these plots we can observe that Group 3 actually has a better influence on activity and, in turn, revenue.
    #Group 1 outperforms Group 3 only in Swains Island, but Group 3 increases revenue in all the other 5 locations.
    #This includes an increase in median revenue, as well as 75th and 100th percentile

    controlAvgRev = df.loc[df['Group'] == 5]['GrossRev'].mean()
    group3AvgRev = df.loc[df['Group'] == 3]['GrossRev'].mean()
    group1AvgRev = df.loc[df['Group'] == 1]['GrossRev'].mean()
    diff3to5 = group3AvgRev - controlAvgRev
    diff1to5 = group1AvgRev - controlAvgRev
    diff3to1 = group3AvgRev - group1AvgRev
    print('Average revenue difference between Group 3 and Control Group is %s' % diff3to5)
    print('Average revenue difference between Group 1 and Control Group is %s' % diff1to5)
    print('Average revenue difference between Group 3 and Group 1 is %s' % diff3to1)

    #If the bonus is awarded per player, but on average the increase in revenue is less than what has been
    #spent on the promotion, then in most cases the promotions are not sustainable
    #i.e. Group 3 increases average revenue per player by 24.74 but 25 is spent on the campaign

    #To conclude, we can state the following:
    #The marketing campaign provided statistical significance only for groups 1 and 3
    #Further research shows that the increase in average revenue from group 1 and 3's campaign are less than
    #the amount spent on these campaigns, and thus should not be pursued
    #Further research is required to test the relationship between countries and their impact on promotions

    plt.show()

if __name__ == '__main__':
    main()

