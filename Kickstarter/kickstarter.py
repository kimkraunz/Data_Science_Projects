import sys
reload(sys)
sys.setdefaultencoding('UTF8')

from __future__ import division

import pandas
from pandas import DatetimeIndex
import numpy as np
from pandas.tseries.tools import to_datetime
import numpy
import matplotlib.pylab as plt
from textblob import TextBlob
from nltk.corpus import stopwords

df = pandas.read_csv('/Users/frontlines/Documents/kickstarter projects.csv')

pandas.set_option('display.max_columns', None)
df.head()
df.drop('currency_symbol', axis = 1, inplace = True)

# Functions
def cleanup_data(df, cutoffPercent = .01):
   for col in df:
       sizes = df[col].value_counts(normalize = True)
       # get the names of the levels that make up less than 1% of the dataset
       values_to_delete = sizes[sizes<cutoffPercent].index
       df[col].ix[df[col].isin(values_to_delete)] = "Other"
   return df

def get_binary_values(data_frame):
   """encodes cateogrical features in Pandas.
   """
   all_columns = pandas.DataFrame( index = data_frame.index)
   for col in data_frame.columns:
       data = pandas.get_dummies(data_frame[col], prefix=col.encode('ascii', 'replace'))
       all_columns = pandas.concat([all_columns, data], axis=1)
   return all_columns


def find_zero_var(df):
   """finds columns in the dataframe with zero variance -- ie those
       with the same value in every observation.
   """   
   toKeep = []
   toDelete = []
   for col in df:
       if len(df[col].value_counts()) > 1:
           toKeep.append(col)
       else:
           toDelete.append(col)
       ##
   return {'toKeep':toKeep, 'toDelete':toDelete} 

   
def find_perfect_corr(df):
   """finds columns that are eother positively or negatively perfectly correlated (with correlations of +1 or -1), and creates a dict 
       that includes which columns to drop so that each remaining column
       is independent
   """  
   corrMatrix = df.corr()
   corrMatrix.loc[:,:] =  numpy.tril(corrMatrix.values, k = -1)
   already_in = set()
   result = []
   for col in corrMatrix:
       perfect_corr = corrMatrix[col][abs(numpy.round(corrMatrix[col],10)) == 1.00].index.tolist()
       if perfect_corr and col not in already_in:
           already_in.update(set(perfect_corr))
           perfect_corr.append(col)
           result.append(perfect_corr)
   toRemove = []
   for item in result:
       toRemove.append(item[1:(len(item)+1)])
   toRemove = sum(toRemove, [])
   return {'corrGroupings':result, 'toRemove':toRemove}

# Data manipulation

# turn upper cases into lower cases in category and sub_category
df.main_category = df.main_category.str.lower()
df.sub_category = df.sub_category.str.lower()


art = ('conceptual_art', 'digital_art', 'illustration', 'installations', 'mixed_media', 'painting', 'performance_art', 'public_art', 'sculpture', 'textiles', 'video_art', 'ceramics')

comics = ('anthologies', 'comic_books', 'events', 'graphic_novels', 'webcomics')

crafts = ('candles', 'crochet', 'diy', 'embroidery', 'glass', 'knitting', 'letterpress', 'pottery', 'printing', 'quilts', 'stationery', 'taxidermy', 'weaving', 'woodworking')

dance = ('performances', 'residencies', 'spaces', 'workshops')

design = ('architecture', 'civic_design', 'graphic_design', 'interactive_design', 'product_design', 'typography')

fashion = ('accessories', 'apparel', 'childrenswear', 'couture', 'footwear', 'jewelry', 'pet_fashion', 'ready-to-wear')

film_and_video = ('action', 'animation', 'comedy', 'documentary', 'drama', 'experimental', 'family', 'fantasy', 'festivals', 'horror', 'movie_theaters', 'music_videos', 'narrative_film', 'romance', 'science_fiction', 'shorts', 'television', 'thrillers', 'webseries')

food = ('bacon', 'community_gardens', 'cookbooks', 'drinks', 'events', 'farmers_markets', 'farms', 'food_trucks', 'restaurants', 'small_batch', 'spaces', 'vegan')

games = ('gaming_hardware', 'live_games', 'mobile_games', 'playing_cards', 'puzzles', 'tabletop_games', 'video_games')

journalism = ('audio', 'photo', 'prints', 'video', 'web')

music = ('blues', 'chiptune', 'classical_music', 'country_&_folk', 'electronic_music', 'faith', 'hip-hop', 'indie_rock', 'jazz', 'kids', 'latin', 'metal', 'pop', 'punk', 'r&b', 'rock', 'world_music')

animals = ('fine_art', 'nature', 'people', 'photobooks', 'places')

technology = ('3d_printing', 'apps','camera_equipment', 'diy_electronics', 'fabrication_tools', 'flight', 'gadgets', 'hardware', 'makerspaces', 'robots','software','sound', 'space_exploration', 'wearables', 'web', 'open_software')

theater = ('experimental', 'festivals', 'immersive', 'musical', 'plays', 'spaces')

publishing = ('academic', 'anthologies', 'art_books','calendars','childrens_books','fiction', 'literary_journals', 'nonfiction', 'periodicals', 'poetry', 'radio_and_podcasts', 'translations', 'young_adult', 'zines')

df.main_category[df.main_category == "children's_book"] = 'childrens_books'
df.main_category[df.main_category == "children's_books"] = 'childrens_books'
df.main_category[df.main_category == 'short_film'] = 'shorts'
df.main_category[df.main_category == 'art_book'] = 'art_books'
df.main_category[df.main_category == 'periodical'] = 'periodicals'
df.main_category[df.main_category == 'radio_&_podcast'] = 'radio_and_podcasts'
df.main_category[df.main_category == 'radio_&_podcasts'] = 'radio_and_podcasts'
df.main_category[df.main_category == "farmer's_markets"] = 'farmers_markets'
df.main_category[df.main_category == 'print'] = 'prints'
df.main_category[df.main_category == 'film_&_video'] = 'film_and_video'


for name in technology:
    df.sub_category[df.main_category == name] = name
    df.main_category[df.sub_category == name] = 'technology'

for name in theater:
    df.sub_category[df.main_category == name] = name
    df.main_category[df.sub_category == name] = 'theater'

for name in animals:
    df.sub_category[df.main_category == name] = name
    df.main_category[df.sub_category == name] = 'animals'

for name in music:
    df.sub_category[df.main_category == name] = name
    df.main_category[df.sub_category == name] = 'music'
    
for name in journalism:
    df.sub_category[df.main_category == name] = name
    df.main_category[df.sub_category == name] = 'journalism'
 
for name in games:
    df.sub_category[df.main_category == name] = name
    df.main_category[df.sub_category == name] = 'games' 
    
for name in food:
    df.sub_category[df.main_category == name] = name
    df.main_category[df.sub_category == name] = 'food' 

for name in film_and_video:
    df.sub_category[df.main_category == name] = name
    df.main_category[df.sub_category == name] = 'film_and_video'
    
for name in fashion:
    df.sub_category[df.main_category == name] = name
    df.main_category[df.sub_category == name] = 'fashion' 
    
for name in design:
    df.sub_category[df.main_category == name] = name
    df.main_category[df.sub_category == name] = 'design'
    
for name in dance:
    df.sub_category[df.main_category == name] = name
    df.main_category[df.sub_category == name] = 'dance' 
    
for name in crafts:
    df.sub_category[df.main_category == name] = name
    df.main_category[df.sub_category == name] = 'crafts' 
    
for name in comics:
    df.sub_category[df.main_category == name] = name
    df.main_category[df.sub_category == name] = 'comics' 
    
for name in art:
    df.sub_category[df.main_category == name] = name
    df.main_category[df.sub_category == name] = 'art' 
    
for name in publishing:
    df.sub_category[df.main_category == name] = name
    df.main_category[df.sub_category == name] = 'publishing' 
     
df.main_category.value_counts(normalize = True, dropna = False )
#film_and_video    0.176407
#music             0.140513
#publishing        0.105653
#games             0.090599
#design            0.079068
#art               0.072961
#food              0.070152
#technology        0.069614
#fashion           0.055507
#comics            0.029130
#theater           0.026819
#crafts            0.021554
#journalism        0.018874
#photography       0.017622
#animals           0.012848
#dance             0.012647
#unknown           0.000032
#NaN               0.000000

main_cats = ('film_and_video', 'music','publishing', 'games', 'design', 'art','food','technology','fashion','comics', 'theater', 'crafts', 'journalism','photography','animals','dance')

df.sub_category[df.main_category == 'unknown'] = 'unknown'
df.sub_category[df.sub_category == 'film_&_video'] = 'film_and_video'

for name in main_cats:
    df.sub_category[df.sub_category == name] = 'unknown'
    
df.sub_category.value_counts(normalize = True, dropna = False)
#unknown            0.302564
#product_design     0.058027
#documentary        0.045966
#tabletop_games     0.036938
#shorts             0.033792
#video_games        0.033559
#fiction            0.026161
#nonfiction         0.023561
#rock               0.017542
#childrens_books    0.017181
#webseries          0.015985
#narrative_film     0.015432
#indie_rock         0.015070
#hardware           0.013698
#apparel            0.013345
#...
#pet_fashion        0.000281
#art_book           0.000281
#embroidery         0.000233
#quilts             0.000225
#periodical         0.000225
#latin              0.000209
#weaving            0.000193
#residencies        0.000169
#pottery            0.000152
#radio_&_podcast    0.000128
#typography         0.000104
#chiptune           0.000088
#letterpress        0.000088
#taxidermy          0.000016
#NaN                0.000000

df['funded']= 2
df.funded[df.state == 'successful'] = 1
df.funded[df.state == 'failed'] = 0
df = df[df.funded != 2]


df.deadline = to_datetime(df.deadline)


df.describe()
df.head()


# Convert currency to USD
df.currency.value_counts(normalize = True)
#USD    0.716564
#GBP    0.088293
#CAD    0.039900
#AUD    0.020653
#EUR    0.007431
#NZD    0.004088
#SEK    0.001760
#DKK    0.001173
#NOK    0.000754

# Convert goals and pledged to USD
df['goal_USD'] = df.goal

df.goal_USD = df.goal_USD[df.currency == "GBP"] = df.goal * 1.48
df.goal_USD = df.goal_USD[df.currency == "CAD"] = df.goal * .79
df.goal_USD = df.goal_USD[df.currency == "AUD"] = df.goal * .76
df.goal_USD = df.goal_USD[df.currency == "EUR"] = df.goal * 1.07
df.goal_USD = df.goal_USD[df.currency == "NZD"] = df.goal * .75
df.goal_USD = df.goal_USD[df.currency == "DKK"] = df.goal * .14
df.goal_USD = df.goal_USD[df.currency == "NOK"] = df.goal * .12

df['pledged_USD'] = df.pledged

df.pledged_USD = df.pledged_USD[df.currency == "GBP"] = df.pledged * 1.48
df.pledged_USD = df.pledged_USD[df.currency == "CAD"] = df.pledged * .79
df.pledged_USD = df.pledged_USD[df.currency == "AUD"] = df.pledged * .76
df.pledged_USD = df.pledged_USD[df.currency == "EUR"] = df.pledged * 1.07
df.pledged_USD = df.pledged_USD[df.currency == "NZD"] = df.pledged * .75
df.pledged_USD = df.pledged_USD[df.currency == "DKK"] = df.pledged * .14
df.pledged_USD = df.pledged_USD[df.currency == "NOK"] = df.pledged * .12

df.pledged_USD = df.pledged_USD.astype('int')
df.goal_USD = df.goal_USD.astype('int')

# Create quarters

frame_sum = df.set_index(df.deadline)
frame_sum = frame_sum[: '12/31/2014']
frame_sum = frame_sum.resample('Q', how = 'sum')
frame_sum = frame_sum[['goal_USD', 'pledged_USD']]
frame_sum['prop_funded_q'] = frame_sum.pledged_USD / frame_sum.goal_USD
frame_sum.reset_index(inplace = True)
df['prop_funded_prev_q'] = 0


df.prop_funded_prev_q[(df.deadline >= '2009-7-1') & (df.deadline < '2009-10-1')] = 1.340893
df.prop_funded_prev_q[(df.deadline >= '2009-10-1') & (df.deadline < '2010-1-1')] = 1.378548

df.prop_funded_prev_q[(df.deadline >= '2010-1-1') & (df.deadline < '2010-4-1')] = 1.204869
df.prop_funded_prev_q[(df.deadline >= '2010-4-1') & (df.deadline < '2010-7-1')] = 1.239185
df.prop_funded_prev_q[(df.deadline >= '2010-7-1') & (df.deadline < '2010-10-1')] = 1.291623
df.prop_funded_prev_q[(df.deadline >= '2010-10-1') & (df.deadline < '2011-1-1')] = 1.267159

df.prop_funded_prev_q[(df.deadline >= '2011-1-1') & (df.deadline < '2011-4-1')] = 1.567291
df.prop_funded_prev_q[(df.deadline >= '2011-4-1') & (df.deadline < '2011-7-1')] = 1.405329
df.prop_funded_prev_q[(df.deadline >= '2011-7-1') & (df.deadline < '2011-10-1')] = 1.556862
df.prop_funded_prev_q[(df.deadline >= '2011-10-1') & (df.deadline < '2012-1-1')] = 1.570845

df.prop_funded_prev_q[(df.deadline >= '2012-1-1') & (df.deadline < '2012-4-1')] = 1.691398
df.prop_funded_prev_q[(df.deadline >= '2012-4-1') & (df.deadline < '2012-7-1')] = 2.251373
df.prop_funded_prev_q[(df.deadline >= '2012-7-1') & (df.deadline < '2012-10-1')] = 2.635929
df.prop_funded_prev_q[(df.deadline >= '2012-10-1') & (df.deadline < '2013-1-1')] = 2.354662

df.prop_funded_prev_q[(df.deadline >= '2013-1-1') & (df.deadline < '2013-4-1')] = 2.112577
df.prop_funded_prev_q[(df.deadline >= '2013-4-1') & (df.deadline < '2013-7-1')] = 2.150800
df.prop_funded_prev_q[(df.deadline >= '2013-7-1') & (df.deadline < '2013-10-1')] = 1.365337
df.prop_funded_prev_q[(df.deadline >= '2013-10-1') & (df.deadline < '2014-1-1')] = 0.411087

df.prop_funded_prev_q[(df.deadline >= '2014-1-1') & (df.deadline < '2014-4-1')] = 0.451956
df.prop_funded_prev_q[(df.deadline >= '2014-4-1') & (df.deadline < '2014-7-1')] = 0.438982
df.prop_funded_prev_q[(df.deadline >= '2014-7-1') & (df.deadline < '2014-10-1')] = 0.444080
df.prop_funded_prev_q[(df.deadline >= '2014-10-1') & (df.deadline < '2015-1-1')] = 0.147932

frame_sum = df.set_index(df.deadline)
frame_sum = frame_sum[: '12/31/2014']
frame_sum = frame_sum.resample('Q', how = 'count')
frame_sum = frame_sum[['goal_USD', 'pledged_USD']]



frame_sum_funded = df[df.funded == 1]
frame_sum_funded = frame_sum_funded.set_index(frame_sum_funded.deadline)
frame_sum = frame_sum[: '12/31/2014']
frame_sum_funded = frame_sum_funded.resample('Q', how = 'count')
frame_sum_funded = frame_sum_funded[['goal_USD', 'pledged_USD']]


# Text analysis

df.blurb = df.blurb.astype('str')


def polarity(text):
    return TextBlob(unicode(text, errors='ignore')).sentiment.polarity

def subjectivity(text):
    return TextBlob(unicode(text, errors='ignore')).sentiment.subjectivity


df['polarity'] = map(polarity, df.blurb)
df['subjectivity'] = map(subjectivity, df.blurb) 


#####################  Exploratory graphs  ###############################

# Polarity and subjectivity graph


plt.figure(figsize=(9, 12)) 
plt.subplots_adjust(hspace=.4)
plt.subplot(211)
plt.title('Polarity', fontsize = 15)
plt.hist(df.polarity, bins=30, histtype='step', color='b', label='Polarity')
plt.text(-.97, -13000, 'Negative', rotation = 'vertical', verticalalignment='bottom', horizontalalignment='right', fontsize=12)
plt.text(1.03, -12000, 'Positive', rotation = 'vertical', verticalalignment='bottom', horizontalalignment='right', fontsize=12)
plt.ylabel("Frequency", fontsize = 12)
plt.xlabel("Scale", fontsize = 12)

plt.figure(figsize=(9, 12)) 
plt.subplot(211)
plt.title('Subjectivity', fontsize = 15)
plt.hist(df.subjectivity, bins=30, histtype='step', color='r', label='Subjectivity')
plt.text(.03, -11000, 'Objective', rotation = 'vertical', verticalalignment='bottom', horizontalalignment='right', fontsize=12)
plt.text(1.02, -11500, 'Subjective', rotation = 'vertical', verticalalignment='bottom', horizontalalignment='right', fontsize=12)

plt.xlabel("Scale", fontsize = 12)
plt.ylabel("Frequency", fontsize = 12)
plt.show()

# Backer_counts

plt.figure(figsize=(6, 4)) 

plt.title('Project backers', fontsize = 15)
df[df.funded ==1].backers_count.hist(bins=100, range = (0, 200), label = 'funded')
df[df.funded ==0].backers_count.hist(bins=100, range = (0, 200), label = 'not funded')
plt.ylim(0,2000)
plt.xlabel("Number of project backers (limited to <200)", fontsize = 12)
plt.ylabel("Frequency", fontsize = 12)
plt.legend()

# Funded and asked line graph over time
frame = df.set_index(df.deadline)
frame = frame[: '12/31/2014']
frame = frame.resample('Q', how = 'mean')
frame.describe()
frame['ave_prop_funded'] = 100 * frame.pledged_USD / frame.goal_USD
frame_sum['perc_funded_q'] = 100 * frame_sum_funded.pledged_USD / frame_sum.pledged_USD

fig, ax1 = plt.subplots()
frame.goal_USD.plot(ax=ax1, linewidth = 3, label = 'goal')
frame.pledged_USD.plot(ax=ax1, linewidth = 3, label = 'pledged')
ax1.set_xlabel('Year (by quarter)', fontsize = 12)
ax1.set_ylabel('Average project in USD', fontsize = 12)

plt.title('Average goal and pledges per Kickstarter project', fontsize=16)
ax1.title.set_position((.5,1.08))


ax2 = ax1.twinx()
frame.ave_prop_funded.plot(ax=ax2, color = 'g', linestyle = '--', label = '% proportion pledged', fontsize = 12)
frame_sum.perc_funded_q.plot(ax=ax2, color = 'black', linestyle = '--', label = '% of projects funded', fontsize = 12)
ax2.set_ylabel('%')
ax1.legend(bbox_to_anchor=(1.12, .95), loc=2, borderaxespad=0.)
ax2.legend(bbox_to_anchor=(1.12, .6), loc=3, borderaxespad=0.)

plt.show()

# Sum of projects by quarter
frame_sum = df.set_index(df.deadline)
frame_sum = frame_sum[: '12/31/2014']
frame_sum = frame_sum.resample('Q', how = 'sum')

frame_count = df.set_index(df.deadline)
frame_count = frame_count[: '12/31/2014']
frame_count = frame_count.resample('Q', how = 'count')

frame_sum['log_pledged'] = np.log10(frame_sum.pledged_USD)
frame_sum['log_goal'] = np.log10(frame_sum.goal_USD)
frame_sum['ave_sum_pct_funded'] = 100 * frame.pledged_USD / frame.goal_USD

fig, ax1 = plt.subplots()

frame_sum.log_goal.plot(ax=ax1, linewidth = 3, label = 'goal')
frame_sum.log_pledged.plot(ax=ax1, linewidth = 3, label = 'pledged')
ax1.set_xlabel('Year (by quarter)', fontsize = 12)
ax1.set_ylabel('Log (Total sum of projects in USD)', fontsize = 12)
plt.title('Sum of Kickstarter goals and pledges in USD', fontsize=16)
ax1.title.set_position((.5,1.08))


ax2 = ax1.twinx()
frame_count.main_category.plot(ax=ax2, linewidth = 1, label = '# of projects')
ax2.set_ylabel('Number of projects', fontsize = 12)
ax1.legend(bbox_to_anchor=(1.2, .95), loc=2, borderaxespad=0.)
ax2.legend(bbox_to_anchor=(1.2, .6), loc=3, borderaxespad=0.)
plt.show()

# Sum of total 

fig, ax1 = plt.subplots()

frame_sum.goal_USD.plot(ax=ax1, color = 'b', linewidth = 3, label = 'goal')
frame_sum.pledged_USD.plot(ax=ax1, color = 'r', linewidth = 3, label = ' funded')
ax1.set_xlabel('Year (by quarter)', fontsize = 12)
ax1.set_ylabel('Total sum of projects in USD x 10^8', fontsize = 12)
plt.title('Sum of Kickstarter goals and funding in USD', fontsize=16)
ax1.title.set_position((.5,1.08))


ax2 = ax1.twinx()
frame_count.main_category.plot(ax=ax2, color = 'g', linewidth = 1, label = '# of projects')
ax2.set_ylabel('Number of projects', fontsize = 12)
ax1.legend(bbox_to_anchor=(1.2, .95), loc=2, borderaxespad=0.)
ax2.legend(bbox_to_anchor=(1.2, .6), loc=3, borderaxespad=0.)
plt.show()

##  Group by category
pandas.set_option('display.max_rows', 350)
cat_group = df[['deadline', 'main_category', 'goal_USD', 'pledged_USD']].set_index('deadline').groupby('main_category').resample('Q', how='sum')

cat_group.reset_index(inplace=True)
cat_group= cat_group[(cat_group.main_category != 'unknown') & (cat_group.deadline <'1/1/2015')]


groups = cat_group.groupby('main_category')

# Plot

plt.figure(figsize=(16, 12)) 

plt.rcParams.update(pandas.tools.plotting.mpl_stylesheet)
pandas.options.display.mpl_style = False
colors = pandas.tools.plotting._get_standard_colors(len(groups), color_type='random')

fig, ax1 = plt.subplots()
for name, group in groups:
    ax1.plot(group.deadline, group.pledged_USD, linestyle='-', linewidth = 2, label=name)
ax1.legend(bbox_to_anchor=(1.12, 1), loc=2, borderaxespad=0.)
plt.title('Sum of Kickstarter pledges in USD by main category', fontsize=16, y = 1.08)
ax1.set_ylabel('USD', fontsize = 12)
plt.show()


# Funded by currency

currency_group = df[['deadline', 'currency', 'goal_USD', 'pledged_USD']].set_index('deadline').groupby('currency').resample('Q', how='sum')

currency_group.reset_index(inplace=True) 
currency_group= currency_group[(currency_group.currency != 'unknown') & (currency_group.deadline <'1/1/2015')]

df.currency.value_counts(normalize = False)

by_currency = df.groupby(df.currency).sum()
N=9
ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
pledged = ax.bar(ind, by_currency.pledged_USD, width, color='r')
ask = ax.bar(ind+width, by_currency.goal_USD, width, color='b')

ax.set_ylabel('US Dollars in billions')
ax.set_title('Currency')
ax.set_xticks(ind+width)
ax.set_xticklabels(("Australian $", "Canadian $", "Danish Krone", "Euro", "British Pound", "Norwegian Krone", "New Zealand $", "Swedish Krona", "US $"), rotation = 'vertical')
ax.legend( (pledged[0], ask[0]), ('Funded', 'Asked') )



# Set up explanatory and response features
df = df[(df.deadline > '2009-6-30') & (df.deadline < '2015-01-01')]
df = df.drop(df.index[[0,10]])

explanatory_features = [col for col in df.columns if col in ['main_category', 'sub_category', 'backers_count', 'currency', 'goal_USD', 'pct_funded_prev_q', 'polarity', 'subjectivity']]

explanatory_df = df[explanatory_features]


explanatory_df.dropna(how = 'all', inplace = True)

explanatory_col_names = explanatory_df.columns

response_series = df.funded

response_series.dropna(how = 'all', inplace = True)

response_series.index[~response_series.index.isin(explanatory_df.index)]


# 1. Split data into categorical and numeric data

string_features = explanatory_df.ix[: , explanatory_df.dtypes == 'object']
numeric_features = explanatory_df.ix[: , explanatory_df.dtypes != 'object']

string_features.head()
numeric_features.head()

# 2. Fill numeric NaNs through imputation

#from sklearn.preprocessing import Imputer
#
#imputer_object = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#imputer_object.fit(numeric_features)
#numeric_features = pandas.DataFrame(imputer_object.transform(numeric_features), columns = numeric_features.columns)

# 3. Fill categorical NaNs with ‘Nothing’

string_features = string_features.fillna('unknown')

# 4. Detect low-frequency levels in categorical features and bin them under ‘other’

cleanup_data(string_features)


# Create list of column names for when used on testing data
string_features_cat = 	{}
for col in string_features.columns:
	string_features_cat[col] = string_features[col].unique()

# 5. Encode each categorical variable into a sequence of binary variables.

string_features = get_binary_values(string_features)


# 6. Merge your encoded categorical data with your numeric data

explanatory_df = pandas.concat([numeric_features, string_features], axis = 1)
explanatory_df.head()

explanatory_df.describe()


# 7. Remove features with no variation
   
find_zero_var(explanatory_df)

# No features had zero variance

# 8. Remove perfectly correlated features
   
find_perfect_corr(explanatory_df)

# No features had perfect correlation

# 9. Scale your data with zero mean and unit variance

from sklearn import preprocessing

scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pandas.DataFrame(scaler.transform(explanatory_df), columns = explanatory_df.columns)
from sklearn import ensemble
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.grid_search import GridSearchCV


rf = ensemble.RandomForestClassifier(n_estimators = 500)
et = ensemble.ExtraTreesClassifier(n_estimators = 500)


roc_scores_rf = cross_val_score(rf, explanatory_df, response_series, cv=10, scoring = 'roc_auc', n_jobs = -1)

roc_scores_tree = cross_val_score(tree.DecisionTreeClassifier(), explanatory_df, response_series, cv=10, scoring = 'roc_auc', n_jobs = -1)

roc_scores_et = cross_val_score(et, explanatory_df, response_series, cv=10, scoring = 'roc_auc', n_jobs = -1)

print roc_scores_rf.mean()
print roc_scores_tree.mean()
print roc_scores_et.mean()

xTrain, xTest, yTrain, yTest = train_test_split(explanatory_df, response_series, test_size = 0.3)

# Create pandas dataframe
tree_probabilities = pandas.DataFrame(tree.DecisionTreeClassifier().fit(xTrain, yTrain).predict_proba(xTest))
rf_probabilities = pandas.DataFrame(rf.fit(xTrain, yTrain).predict_proba(xTest))
et_probabilities = pandas.DataFrame(et.fit(xTrain, yTrain).predict_proba(xTest))

# plot with Y and 2nd column of pandas dataframe
tree_fpr, tree_tpr, thresholds = metrics.roc_curve(yTest, tree_probabilities[1])
rf_fpr, rf_tpr, thesholds = metrics.roc_curve(yTest, rf_probabilities[1])
et_fpr, et_tpr, thesholds = metrics.roc_curve(yTest, et_probabilities[1])

plt.figure()
plt.plot(tree_fpr, tree_tpr, color = 'g')
plt.plot(rf_fpr, rf_tpr, color = 'b')
plt.plot(et_fpr, et_tpr, color = 'r')
plt.xlabel('False Positive Rate(1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')


''' RFE and Grid Search once I find the best model'''

''' fix param grid for both params'''


#depth_range = range(3, 6)
#trees_range = range(10, 550, 10)
#param_grid = dict(estimator__max_depth=depth_range)
#
#class TreeClassifierWithCoef(tree.DecisionTreeClassifier):
#   def fit(self, *args, **kwargs):
#       super(tree.DecisionTreeClassifier, self).fit(*args, **kwargs)
#       self.coef_ = self.feature_importances_
#
## these are the default settings for the tree based classifier
#decision_tree = TreeClassifierWithCoef(criterion = 'gini', splitter = 'best', max_features = None, max_depth = None, min_samples_split = 2, min_samples_leaf = 2, max_leaf_nodes = None, random_state = 1)
#
#rfe_cv = RFECV(estimator=decision_tree, step=1, cv=10,
#              scoring='roc_auc', verbose = 1)
#
#rfe_grid_search = GridSearchCV(rfe_cv, param_grid, cv = 10, scoring = 'roc_auc')
#rfe_grid_search.fit(explanatory_df, response_series)
#
#print rfe_grid_search.grid_scores_
#
#rfe_grid_search.best_params_
#
#
#grid_mean_scores = [score[1] for score in rfe_grid_search.grid_scores_]
#
## Plot max_depth vs. ROC score
#plt.figure()
#plt.plot(depth_range, grid_mean_scores)
#plt.hold(True)
#plt.plot(rfe_grid_search.best_params_['estimator__max_depth'], rfe_grid_search.best_score_, 'ro', markersize=12, markeredgewidth=1.5,
#         markerfacecolor='None', markeredgecolor='r')
#plt.grid(True)
#
## pull out the winning estimator.
#best_decision_tree_rfe_grid = rfe_grid_search.best_estimator_
#
#features_used_rfecv_grid = explanatory_df.columns[best_decision_tree_rfe_grid.get_support()]
#
#print features_used_rfecv_grid
#
#best_features = explanatory_df[features_used_rfecv_grid]

'''importances for best model'''

#importances = pandas.DataFrame(gbm_grid.best_estimator_.feature_importances_, index = explanatory_df.columns, columns =['importance'])
#
#importances.sort(columns = ['importance'], ascending = False, inplace = True)
#print importances

explanatory_df_no_backs = explanatory_df.drop('backers_count', 1)

accuracy_rf_nb = cross_val_score(rf, explanatory_df_no_backs, response_series, cv=10, scoring = 'accuracy', n_jobs = -1)

accuracy_tree_nb = cross_val_score(tree.DecisionTreeClassifier(), explanatory_df_no_backs, response_series, cv=10, scoring = 'accuracy', n_jobs = -1)

accuracy_et_nb = cross_val_score(et, explanatory_df_no_backs, response_series, cv=10, scoring = 'accuracy', n_jobs = -1)

accuracy_rf_nb.mean()
accuracy_tree_nb.mean()
accuracy_et_nb.mean()

#w/o sentiment

#0.642830242708
#0.57791814837
#0.602440778334

# with sentiment
#0.60020032163560011
#0.57065206149835612
#0.59243482561913152

xTrain, xTest, yTrain, yTest = train_test_split(explanatory_df_no_backs, response_series, test_size = 0.3)

# Create pandas dataframe
tree_probabilities_nb = pandas.DataFrame(tree.DecisionTreeClassifier().fit(xTrain, yTrain).predict_proba(xTest))
rf_probabilities_nb = pandas.DataFrame(rf.fit(xTrain, yTrain).predict_proba(xTest))
et_probabilities_nb = pandas.DataFrame(et.fit(xTrain, yTrain).predict_proba(xTest))

# plot with Y and 2nd column of pandas dataframe
tree_fpr_nb, tree_tpr_nb, thresholds = metrics.roc_curve(yTest, tree_probabilities_nb[1])
rf_fpr_nb, rf_tpr_nb, thesholds = metrics.roc_curve(yTest, rf_probabilities_nb[1])
et_fpr_nb, et_tpr_nb, thesholds = metrics.roc_curve(yTest, et_probabilities_nb[1])

plt.figure()
plt.plot(tree_fpr_nb, tree_tpr_nb, color = 'g', label = 'decision tree')
plt.plot(rf_fpr_nb, rf_tpr_nb, color = 'b', label = 'random forest')
plt.plot(et_fpr_nb, et_tpr_nb, color = 'r', label = 'extra trees')
plt.xlabel('False Positive Rate(1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC scores (without backer counts)')
plt.legend(loc = 0)


from sklearn.feature_selection import RFECV
from sklearn import tree

# create new class with a .coef_ attribute.
class TreeClassifierWithCoef(tree.DecisionTreeClassifier):
    def fit(self, *args, **kwargs):
        super(tree.DecisionTreeClassifier, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_

# create your tree based estimator
decision_tree = TreeClassifierWithCoef(criterion = 'gini', splitter = 'best', max_features = None, max_depth = None, min_samples_split = 2, min_samples_leaf = 2, max_leaf_nodes = None, random_state = 1)

## set up the estimator. Score by AUC
rfe_cv = RFECV(estimator=decision_tree, step=1, cv=10,
              scoring='roc_auc', verbose = 1)
rfe_cv.fit(explanatory_df_no_backs, response_series)

print "Optimal number of features :{0} of {1} considered".format(rfe_cv.n_features_,len(explanatory_df_no_backs.columns))

# printing out scores as we increase the number of features -- the farter
# down the list, the higher the number of features considered.
print rfe_cv.grid_scores_

## let's plot out the results
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (ROC_AUC)")
plt.plot(range(1, len(rfe_cv.grid_scores_) + 1), rfe_cv.grid_scores_)
plt.show()
# notice you could have just as well have included the 10 most important 
# features and received similar accuracy.

# you can pull out the features used this way:
features_used = explanatory_df_no_backs.columns[rfe_cv.get_support()]
print features_used
# Index([u'goal_USD', u'polarity', u'subjectivity', u'main_category_animals', u'main_category_art', u'main_category_comics', u'main_category_crafts', u'main_category_dance', u'main_category_design', u'main_category_fashion', u'main_category_film_and_video', u'main_category_food', u'main_category_games', u'main_category_journalism', u'main_category_music', u'main_category_photography', u'main_category_publishing', u'main_category_technology', u'main_category_theater', u'sub_category_Other', u'sub_category_apparel', u'sub_category_childrens_books', u'sub_category_documentary', u'sub_category_fiction', u'sub_category_hardware', u'sub_category_indie_rock', u'sub_category_narrative_film', u'sub_category_nonfiction', u'sub_category_pop', u'sub_category_product_design', u'sub_category_rock', u'sub_category_shorts', u'sub_category_tabletop_games', u'sub_category_unknown', u'sub_category_video_games', u'sub_category_webseries', u'currency_AUD', u'currency_CAD', u'currency_GBP', u'currency_Other', u'currency_USD', u'currency_unknown'], dtype='object')
#you can extract the final selected model object this way:
best_model = explanatory_df_no_backs[features_used]
importances = pandas.DataFrame(rf.feature_importances_, index = best_model.columns, columns =['importance'])

importances.sort(columns = ['importance'], ascending = False, inplace = True)
print importances



#from sklearn.feature_selection import RFECV
#
#class ForestClassifierWithCoef(RandomForestClassifier):
#   def fit(self, *args, **kwargs):
#       super(RandomForestClassifier, self).fit(*args, **kwargs)
#       self.coef_ = self.feature_importances_
#
## these are the default settings for the tree based classifier
#random_forest = ForestClassifierWithCoef(n_estimators = 100)
#
#rfe_cv = RFECV(estimator=DecisionTree, step=1, cv=10,
#              scoring='accuracy', verbose = 1)
#rfe_cv.fit(explanatory_df_no_backs, response_series)
#
#"Optimal number of features :{0} of {1} considered".format(rfe_cv.n_features_, len(explanatory_df_no_backs.columns))
#
#
#rfe_cv.n_features_
#
#rfe_cv.grid_scores_
#
#plt.figure()
#plt.xlabel("Number of features selected")
#plt.ylabel("Cross validation score (ROC_AUC)")
#plt.plot(range(1, len(rfe_cv.grid_scores_) + 1), rfe_cv.grid_scores_)
#plt.show()
#
#features_used = explanatory_df_no_backs.columns[rfe_cv.get_support()]
#features_used

features_used = ['goal_USD', 'polarity', 'subjectivity', 'main_category_animals', 'main_category_art', 'main_category_comics', 'main_category_crafts', 'main_category_dance', 'main_category_design', 'main_category_fashion', 'main_category_film_and_video', 'main_category_food', 'main_category_games', 'main_category_journalism', 'main_category_music', 'main_category_photography', 'main_category_publishing', 'main_category_technology', 'main_category_theater', 'sub_category_Other', 'sub_category_apparel', 'sub_category_childrens_books', 'sub_category_documentary', 'sub_category_fiction', 'sub_category_hardware', 'sub_category_indie_rock', 'sub_category_narrative_film', 'sub_category_nonfiction', 'sub_category_pop', 'sub_category_product_design', 'sub_category_rock', 'sub_category_shorts', 'sub_category_tabletop_games', 'sub_category_unknown', 'sub_category_video_games', 'sub_category_webseries', 'currency_AUD', 'currency_CAD', 'currency_GBP', 'currency_Other', 'currency_USD', 'currency_unknown']

best_model = explanatory_df_no_backs[features_used]

rf = RandomForestClassifier(n_estimators = 100)

rf.fit(best_model, response_series)
rf.score(best_model, response_series)
# 0.93025459339210681
################################################################################
#######################  Import test data  #####################################
################################################################################

df_2015 = pandas.read_csv('/Users/frontlines/Documents/kickstarter_test.csv')

pandas.set_option('display.max_columns', None)
df_2015.head()
df_2015.drop('currency_symbol', axis = 1, inplace = True)

# turn upper cases into lower cases in category and sub_category
df_2015.main_category = df_2015.main_category.str.lower()
df_2015.sub_category = df_2015.sub_category.str.lower()



art = ('conceptual_art', 'digital_art', 'illustration', 'installations', 'mixed_media', 'painting', 'performance_art', 'public_art', 'sculpture', 'textiles', 'video_art', 'ceramics')

comics = ('anthologies', 'comic_books', 'events', 'graphic_novels', 'webcomics')

crafts = ('candles', 'crochet', 'diy', 'embroidery', 'glass', 'knitting', 'letterpress', 'pottery', 'printing', 'quilts', 'stationery', 'taxidermy', 'weaving', 'woodworking')

dance = ('performances', 'residencies', 'spaces', 'workshops')

design = ('architecture', 'civic_design', 'graphic_design', 'interactive_design', 'product_design', 'typography')

fashion = ('accessories', 'apparel', 'childrenswear', 'couture', 'footwear', 'jewelry', 'pet_fashion', 'ready-to-wear')

film_and_video = ('action', 'animation', 'comedy', 'documentary', 'drama', 'experimental', 'family', 'fantasy', 'festivals', 'horror', 'movie_theaters', 'music_videos', 'narrative_film', 'romance', 'science_fiction', 'shorts', 'television', 'thrillers', 'webseries')

food = ('bacon', 'community_gardens', 'cookbooks', 'drinks', 'events', 'farmers_markets', 'farms', 'food_trucks', 'restaurants', 'small_batch', 'spaces', 'vegan')

games = ('gaming_hardware', 'live_games', 'mobile_games', 'playing_cards', 'puzzles', 'tabletop_games', 'video_games')

journalism = ('audio', 'photo', 'prints', 'video', 'web')

music = ('blues', 'chiptune', 'classical_music', 'country_&_folk', 'electronic_music', 'faith', 'hip-hop', 'indie_rock', 'jazz', 'kids', 'latin', 'metal', 'pop', 'punk', 'r&b', 'rock', 'world_music')

animals = ('fine_art', 'nature', 'people', 'photobooks', 'places')

technology = ('3d_printing', 'apps','camera_equipment', 'diy_electronics', 'fabrication_tools', 'flight', 'gadgets', 'hardware', 'makerspaces', 'robots','software','sound', 'space_exploration', 'wearables', 'web', 'open_software')

theater = ('experimental', 'festivals', 'immersive', 'musical', 'plays', 'spaces')

publishing = ('academic', 'anthologies', 'art_books','calendars','childrens_books','fiction', 'literary_journals', 'nonfiction', 'periodicals', 'poetry', 'radio_and_podcasts', 'translations', 'young_adult', 'zines')

df_2015.sub_category = df_2015.sub_category.astype('str')
df_2015.main_category = df_2015.main_category.astype('str')
df_2015.main_category = df_2015.main_category.replace(' ', '_')
  
     

df_2015.main_category[df_2015.main_category == "children's_book"] = 'childrens_books'
df_2015.main_category[df_2015.main_category == "children's_books"] = 'childrens_books'
df_2015.main_category[df_2015.main_category == 'short_film'] = 'shorts'
df_2015.main_category[df_2015.main_category == 'art_book'] = 'art_books'
df_2015.main_category[df_2015.main_category == 'periodical'] = 'periodicals'
df_2015.main_category[df_2015.main_category == 'radio_&_podcast'] = 'radio_and_podcasts'
df_2015.main_category[df_2015.main_category == 'radio_&_podcasts'] = 'radio_and_podcasts'
df_2015.main_category[df_2015.main_category == "farmer's_markets"] = 'farmers_markets'
df_2015.main_category[df_2015.main_category == 'print'] = 'prints'
df_2015.main_category[df_2015.main_category == 'film_&_video'] = 'film_and_video'


df_2015.main_category[df_2015.main_category == 'product design'] = 'product_design'
df_2015.main_category[df_2015.main_category == 'tabletop games'] = 'tabletop_games'
df_2015.main_category[df_2015.main_category == 'video games'] = 'video_games'
df_2015.main_category[df_2015.main_category == 'film & video'] = 'film_and_video'
df_2015.main_category[df_2015.main_category == 'children\'s books'] = 'childrens_books'
df_2015.main_category[df_2015.main_category == 'playing cards'] = 'playing_cards'
df_2015.main_category[df_2015.main_category == 'country & folk'] = 'country_&_folk'
df_2015.main_category[df_2015.main_category == 'comic books'] = 'comic_books'
df_2015.main_category[df_2015.main_category == 'food trucks'] = 'food_trucks'
df_2015.main_category[df_2015.main_category == 'small batch'] = 'small_batch'
df_2015.main_category[df_2015.main_category == 'indie rock'] = 'indie_rock'
df_2015.main_category[df_2015.main_category == 'public art'] = 'public_art'
df_2015.main_category[df_2015.main_category == 'mobile games'] = 'mobile_games'
df_2015.main_category[df_2015.main_category == 'mixed media'] = 'mixed_media'
df_2015.main_category[df_2015.main_category == 'classical music'] = 'classical_music'
df_2015.main_category[df_2015.main_category == 'graphic novels']= 'graphic_novels'
df_2015.main_category[df_2015.main_category == 'narrative film'] = 'narrative_film'
df_2015.main_category[df_2015.main_category == 'electronic music'] = 'electronic_music'
df_2015.main_category[df_2015.main_category == 'art books'] = 'art_books'
df_2015.main_category[df_2015.main_category == 'world music'] = 'world_music'
df_2015.main_category[df_2015.main_category == 'graphic design'] = 'graphic_design'
df_2015.main_category[df_2015.main_category == 'diy electronics'] = 'diy_electronics'
df_2015.main_category[df_2015.main_category == 'live games'] = 'live_games'
df_2015.main_category[df_2015.main_category == 'performance art'] = 'performance_art'
df_2015.main_category[df_2015.main_category == 'science fiction'] = 'science_fiction'
df_2015.main_category[df_2015.main_category == '3d printing'] = '3d_printing'
df_2015.main_category[df_2015.main_category == 'fine art'] = 'fine_art'
df_2015.main_category[df_2015.main_category == 'young adult'] = 'young_adult'
df_2015.main_category[df_2015.main_category == 'digital art'] = 'digital_art'
df_2015.main_category[df_2015.main_category == 'music videos'] = 'music_videos'  
df_2015.main_category[df_2015.main_category == 'radio & podcasts'] = 'radio_and_podcasts'
df_2015.main_category[df_2015.main_category == 'community gardens'] = 'community_gardens'
df_2015.main_category[df_2015.main_category == 'farmer\'s markets'] = 'farmers_markets'
df_2015.main_category[df_2015.main_category == 'conceptual art'] = 'conceptual_art' 
df_2015.main_category[df_2015.main_category == 'interactive design'] = 'interactive_design'
df_2015.main_category[df_2015.main_category == 'space exploration'] = 'space_exploration'
df_2015.main_category[df_2015.main_category == 'children\'s book'] = 'childrens_books'
df_2015.main_category[df_2015.main_category == 'camera equipment'] = 'camera_equipment'  
df_2015.main_category[df_2015.main_category == 'civic design'] = 'civic_design'
df_2015.main_category[df_2015.main_category == 'gaming hardware'] = 'gaming_hardware'    
df_2015.main_category[df_2015.main_category == 'literary journals'] = 'literary_journals'    
df_2015.main_category[df_2015.main_category == 'fabrication tools'] = 'fabrication_tools'
df_2015.main_category[df_2015.main_category == 'movie theaters'] = 'movie_theaters'
df_2015.main_category[df_2015.main_category == 'pet fashion'] = 'pet_fashion'  
df_2015.main_category[df_2015.main_category == 'video art'] = 'video_art'        
df_2015.main_category[df_2015.main_category == 'art book'] = 'art_books'         
df_2015.main_category[df_2015.main_category == 'radio & podcast'] = 'radio_and_podcasts'
                 



for name in technology:
    df_2015.sub_category[df_2015.main_category == name] = name
    df_2015.main_category[df_2015.sub_category == name] = 'technology'

for name in theater:
    df_2015.sub_category[df_2015.main_category == name] = name
    df_2015.main_category[df_2015.sub_category == name] = 'theater'

for name in animals:
    df_2015.sub_category[df_2015.main_category == name] = name
    df_2015.main_category[df_2015.sub_category == name] = 'animals'

for name in music:
    df_2015.sub_category[df_2015.main_category == name] = name
    df_2015.main_category[df_2015.sub_category == name] = 'music'
    
for name in journalism:
    df_2015.sub_category[df_2015.main_category == name] = name
    df_2015.main_category[df_2015.sub_category == name] = 'journalism'
 
for name in games:
    df_2015.sub_category[df_2015.main_category == name] = name
    df_2015.main_category[df_2015.sub_category == name] = 'games' 
    
for name in food:
    df_2015.sub_category[df_2015.main_category == name] = name
    df_2015.main_category[df_2015.sub_category == name] = 'food' 

for name in film_and_video:
    df_2015.sub_category[df_2015.main_category == name] = name
    df_2015.main_category[df_2015.sub_category == name] = 'film_and_video'
    
for name in fashion:
    df_2015.sub_category[df_2015.main_category == name] = name
    df_2015.main_category[df_2015.sub_category == name] = 'fashion' 
    
for name in design:
    df_2015.sub_category[df_2015.main_category == name] = name
    df_2015.main_category[df_2015.sub_category == name] = 'design'
    
for name in dance:
    df_2015.sub_category[df_2015.main_category == name] = name
    df_2015.main_category[df_2015.sub_category == name] = 'dance' 
    
for name in crafts:
    df_2015.sub_category[df_2015.main_category == name] = name
    df_2015.main_category[df_2015.sub_category == name] = 'crafts' 
    
for name in comics:
    df_2015.sub_category[df_2015.main_category == name] = name
    df_2015.main_category[df_2015.sub_category == name] = 'comics' 
    
for name in art:
    df_2015.sub_category[df_2015.main_category == name] = name
    df_2015.main_category[df_2015.sub_category == name] = 'art' 
    
for name in publishing:
    df_2015.sub_category[df_2015.main_category == name] = name
    df_2015.main_category[df_2015.sub_category == name] = 'publishing' 
     
df_2015.main_category.value_counts(normalize = True, dropna = False )


main_cats = ('film_and_video', 'music','publishing', 'games', 'design', 'art','food','technology','fashion','comics', 'theater', 'crafts', 'journalism','photography','animals','dance')

df_2015.sub_category[df_2015.main_category == 'unknown'] = 'unknown'
df_2015.sub_category[df_2015.sub_category == 'film_&_video'] = 'film_and_video'

for name in main_cats:
    df_2015.sub_category[df_2015.sub_category == name] = 'unknown'
    
df_2015.sub_category[df_2015.sub_category == 'nan'] = 'unknown'
    
df_2015.sub_category.value_counts(normalize = True, dropna = False)

df_2015['funded']= 2
df_2015.funded[df_2015.state == 'successful'] = 1
df_2015.funded[df_2015.state == 'failed'] = 0
df_2015 = df_2015[df_2015.funded != 2]

df_2015.deadline = to_datetime(df_2015.deadline)
df_2015['year'] = DatetimeIndex(df_2015['deadline']).year
df_2015['month'] = DatetimeIndex(df_2015['deadline']).month

df_2015.describe()
df_2015.head()

# Convert currency to USD

df_2015.currency.value_counts(normalize = True, dropna = False)


# Convert goals and pledged to USD
df_2015['goal_USD'] = df_2015.goal

df_2015.goal_USD = df_2015.goal_USD[df_2015.currency == "GBP"] = df_2015.goal * 1.48
df_2015.goal_USD = df_2015.goal_USD[df_2015.currency == "CAD"] = df_2015.goal * .79
df_2015.goal_USD = df_2015.goal_USD[df_2015.currency == "AUD"] = df_2015.goal * .76
df_2015.goal_USD = df_2015.goal_USD[df_2015.currency == "EUR"] = df_2015.goal * 1.07
df_2015.goal_USD = df_2015.goal_USD[df_2015.currency == "NZD"] = df_2015.goal * .75
df_2015.goal_USD = df_2015.goal_USD[df_2015.currency == "DKK"] = df_2015.goal * .14
df_2015.goal_USD = df_2015.goal_USD[df_2015.currency == "NOK"] = df_2015.goal * .12

df_2015['pledged_USD'] = df_2015.pledged

df_2015.pledged_USD = df_2015.pledged_USD[df_2015.currency == "GBP"] = df_2015.pledged * 1.48
df_2015.pledged_USD = df_2015.pledged_USD[df_2015.currency == "CAD"] = df_2015.pledged * .79
df_2015.pledged_USD = df_2015.pledged_USD[df_2015.currency == "AUD"] = df_2015.pledged * .76
df_2015.pledged_USD = df_2015.pledged_USD[df_2015.currency == "EUR"] = df_2015.pledged * 1.07
df_2015.pledged_USD = df_2015.pledged_USD[df_2015.currency == "NZD"] = df_2015.pledged * .75
df_2015.pledged_USD = df_2015.pledged_USD[df_2015.currency == "DKK"] = df_2015.pledged * .14
df_2015.pledged_USD = df_2015.pledged_USD[df_2015.currency == "NOK"] = df_2015.pledged * .12

df_2015.pledged_USD = df_2015.pledged_USD.astype('int')
df_2015.goal_USD = df_2015.goal_USD.astype('int')

# Create quarters

frame_sum = df_2015.set_index(df_2015.deadline)
frame_sum = frame_sum['12/31/2014':]
frame_sum = frame_sum.resample('Q', how = 'sum')
frame_sum = frame_sum[['goal_USD', 'pledged_USD']]
frame_sum['perc_funded_q'] = frame_sum.pledged_USD / frame_sum.goal_USD
frame_sum.reset_index(inplace = True)
df_2015['pct_funded_prev_q'] = 0


df_2015.pct_funded_prev_q[(df_2015.deadline >= '2015-1-1') & (df_2015.deadline < '2015-4-1')] = 0.157881

df_2015.pct_funded_prev_q[(df_2015.deadline >= '2015-4-1') & (df_2015.deadline < '2015-7-1')] = 0.225512

# Text analysis

df_2015.blurb = df_2015.blurb.astype('str')


def polarity(text):
    return TextBlob(unicode(text, errors='ignore')).sentiment.polarity

def subjectivity(text):
    return TextBlob(unicode(text, errors='ignore')).sentiment.subjectivity


df_2015['polarity'] = map(polarity, df_2015.blurb)
df_2015['subjectivity'] = map(subjectivity, df_2015.blurb) 


# Set up explanatory and response features

expl_df_2015_no_backs = df_2015

explanatory_features_no_backs = [col for col in expl_df_2015_no_backs.columns if col in ['main_category', 'sub_category', 'currency', 'goal_USD', 'pct_funded_prev_q', 'subjectivity', 'polarity', 'backers_count']]

explanatory_df_2015_no_backs = expl_df_2015_no_backs[explanatory_features_no_backs]

explanatory_df_2015_no_backs.dropna(how = 'all', inplace = True)

explanatory_col_names_no_backs = explanatory_df_2015_no_backs.columns

response_series_2015_no_backs = expl_df_2015_no_backs.funded

response_series_2015_no_backs.dropna(how = 'all', inplace = True)

response_series_2015_no_backs.index[~response_series_2015_no_backs.index.isin(explanatory_df_2015_no_backs.index)]

~explanatory_df_no_backs.columns.isin(explanatory_df_2015_no_backs.columns)
# 1. Split data into categorical and numeric data

string_features_2015_no_backs = explanatory_df_2015_no_backs.ix[: , explanatory_df_2015_no_backs.dtypes == 'object']
numeric_features_2015_no_backs = explanatory_df_2015_no_backs.ix[: , explanatory_df_2015_no_backs.dtypes != 'object']

# 3. Fill categorical NaNs with ‘unknown’

string_features_2015_no_backs = string_features_2015_no_backs.fillna('unknown')

# 4. Matches categorical data to pre-2015 data

# If there is a value that is not in the training data set, replaces value with "Other"

for col in string_features_2015_no_backs:
    string_features_2015_no_backs[col].ix[~string_features_2015_no_backs[col].isin(string_features_cat[col])] = "Other"


# 5. Encode each categorical variable into a sequence of binary variables.

string_features_2015_no_backs = get_binary_values(string_features_2015_no_backs)

for col in string_features_2015_no_backs:
	if col not in string_features_2015_no_backs:
		string_features_2015_no_backs[col] = 0
 

# 6. Merge your encoded categorical data with your numeric data

explanatory_df_2015_no_backs = pandas.concat([numeric_features_2015_no_backs, string_features_2015_no_backs], axis = 1)
explanatory_df_2015_no_backs.head()



# 7. Remove features with no variation
   
# No features had zero variance

# 8. Remove perfectly correlated features
   
# No features had perfect correlation


# 9. Scale your data with zero mean and unit variance

explanatory_df_2015_no_backs = pandas.DataFrame(scaler.transform(explanatory_df_2015_no_backs), columns = explanatory_df_2015_no_backs.columns, index = explanatory_df_2015_no_backs.index)

##########################  Predicting on new data ##############################


best_model_2015 = explanatory_df_2015_no_backs[features_used]

kickstarter_pred_2015_no_backs = rf.predict(best_model_2015)

from __future__ import division

number_correct = len(response_series_2015_no_backs[response_series_2015_no_backs == kickstarter_pred_2015_no_backs])
total = len(response_series_2015_no_backs)
accuracy = number_correct / total

accuracy
#0.63422230874051

cm = pandas.crosstab(response_series_2015_no_backs, kickstarter_pred_2015_no_backs, rownames=['True Label'], colnames=['Predicted Label'], margins=True)

print cm
'''
Predicted Label     0     1    All
True Label                        
0                6981  3258  10239
1                2379  2793   5172
All              9360  6051  15411
'''


