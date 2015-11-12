# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 18:05:43 2015

@author: frontlines
"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


import pandas
from pandas import DatetimeIndex
import numpy as np
from pandas.tseries.tools import to_datetime
import numpy
import matplotlib.pylab as plt
from __future__ import division
from sklearn.cross_validation import cross_val_score
import statsmodels.formula.api as smf

df = pandas.read_csv('/NO NAME/kim/kickstarter projects.csv')

pandas.set_option('display.max_columns', None)
df.head()
df.drop('currency_symbol', axis = 1, inplace = True)

# Data manipulation

# turn upper cases into lower cases in category and sub_category
df.main_category = df.main_category.str.lower()
df.sub_category = df.sub_category.str.lower()

art = ('conceptual_art', 'digital_art', 'illustration', 'installations', 'mixed_media', 'painting', 'performance_art', 'public_art', 'sculpture', 'textiles', 'video_art', 'ceramics')

comics = ('anthologies', 'comic_books', 'events', 'graphic_novels', 'webcomics')

crafts = ('candles', 'crochet', 'diy', 'embroidery', 'glass', 'knitting', 'letterpress', 'pottery', 'printing', 'quilts', 'stationery', 'taxidermy', 'weaving', 'woodworking')

dance = ('performances', 'residencies', 'spaces', 'workshops')

design = ('architecture', 'civic_design', 'graphic_design', 'interactive_design', 'product_design', 'typography')

food = ('accessories', 'apparel', 'childrenswear', 'couture', 'footwear', 'jewelry', 'pet_food', 'ready-to-wear')

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

df['funded']= 2
df.funded[df.state == 'successful'] = 1
df.funded[df.state == 'failed'] = 0
df = df[df.funded != 2]

df.deadline = to_datetime(df.deadline)

# Convert pledged to USD

df['pledged_USD'] = df.pledged

df.pledged_USD = df.pledged_USD[df.currency == "GBP"] = df.pledged * 1.48
df.pledged_USD = df.pledged_USD[df.currency == "CAD"] = df.pledged * .79
df.pledged_USD = df.pledged_USD[df.currency == "AUD"] = df.pledged * .76
df.pledged_USD = df.pledged_USD[df.currency == "EUR"] = df.pledged * 1.07
df.pledged_USD = df.pledged_USD[df.currency == "NZD"] = df.pledged * .75
df.pledged_USD = df.pledged_USD[df.currency == "DKK"] = df.pledged * .14
df.pledged_USD = df.pledged_USD[df.currency == "NOK"] = df.pledged * .12

df.pledged_USD = df.pledged_USD.astype('int')


df = df.set_index(df.deadline)
#########################   All categories  #########################################


df_total_Qsum = df[['deadline', 'pledged_USD']].sort('deadline').resample('Q', how = 'sum')

y_total = pandas.DataFrame(columns=('x1', 'x2', 'x3', 'x4', 'x5', 'y'))

for i in range(0,(len(df_total_Qsum) - 6)):
    y_total.loc[i] = [df_total_Qsum.pledged_USD.ix[i], df_total_Qsum.pledged_USD.ix[i+1], df_total_Qsum.pledged_USD.ix[i+2], df_total_Qsum.pledged_USD.ix[i+3], df_total_Qsum.pledged_USD.ix[i+4], df_total_Qsum.pledged_USD.ix[i+5]]   
    
# Run linear regression with cross validation
    
X = y_total[['x1', 'x2', 'x3', 'x4', 'x5']]
y = y_total['y']
  
est = smf.ols(formula='y ~ x1 + x2 + x3 + x4 + x5', data=y_total).fit()
# now, let's print out the results.
print est.summary()

yp = est.predict(X)
plt.plot(yp, y, 'o')
plt.plot(y, y, 'r-')
plt.xlabel("Predicted")
plt.ylabel("Observed")

# create y_total_pred dataframe with predicted y_est
y_total_pred = pandas.DataFrame(columns=y_total.columns)
k = len(y_total)
for i in range(0, k):
   totalList = y_total.ix[i,0:5].tolist()
   y_est = est.predict(y_total.ix[i,0:5])
   totalList.append(y_est[0])
   totalSeries = pandas.Series(totalList, index = y_total.columns)
   y_total_pred = y_total_pred.append(totalSeries, ignore_index = True)

j = len(y_total) - 1
for i in range(j, j+13):
   totalList = y_total_pred.ix[i,1:6].tolist()
   y_est = est.predict(y_total_pred.ix[i,0:5])
   totalList.append(y_est[0])
   totalSeries = pandas.Series(totalList, index = y_total.columns)
   y_total_pred = y_total_pred.append(totalSeries, ignore_index = True)
        
y_total_pred
import matplotlib.pyplot as plt
    
fig, ax1 = plt.subplots()
y_total.y.plot(ax=ax1, color = 'b', linewidth = 3, label = 'sum of pledges')
y_total_pred.y.plot(ax=ax1, color = 'b', linewidth = 3, linestyle = '--', label = 'predicted sum of pledges')
ax1.set_xlabel('Year (by quarter)', fontsize = 12)
ax1.set_ylabel('USD x 10^7', fontsize = 12)

a=ax1.get_xticks().tolist()
a=('Q4-2010','Q1-2012', 'Q2-2013', 'Q3-2014', 'Q4-2015', 'Q1-2017', 'Q2-2018')
ax1.set_xticklabels(a, rotation = 'vertical')
ax1.text(20, 3000000, 'R squared =', fontsize = 12)
ax1.text(20, 1000000, round(est.rsquared, 3), fontsize = 12)

ax1.legend(bbox_to_anchor=(.5, .5), loc = 2)

plt.title('Prediction of Kickstarter pledges', fontsize=16)
ax1.title.set_position((.5,1.08))

#########################   Art  #########################################

df_art_Qsum = df[df.main_category == 'art']
df_art_Qsum = df_art_Qsum[['deadline', 'pledged_USD']].sort('deadline').resample('Q', how = 'sum')

y_art = pandas.DataFrame(columns=('x1', 'x2', 'x3', 'x4', 'x5', 'y'))

for i in range(0,(len(df_art_Qsum) - 6)):
    y_art.loc[i] = [df_art_Qsum.pledged_USD.ix[i], df_art_Qsum.pledged_USD.ix[i+1], df_art_Qsum.pledged_USD.ix[i+2], df_art_Qsum.pledged_USD.ix[i+3], df_art_Qsum.pledged_USD.ix[i+4], df_art_Qsum.pledged_USD.ix[i+5]]   
    
# Run linear regression with cross validation
    
X = y_art[['x1', 'x2', 'x3', 'x4', 'x5']]
y = y_art['y']
  
est = smf.ols(formula='y ~ x1 + x2 + x3 + x4 + x5', data=y_art).fit()
# now, let's print out the results.
print est.summary()

yp = est.predict(X)
plt.plot(yp, y, 'o')
plt.plot(y, y, 'r-')
plt.xlabel("Predicted")
plt.ylabel("Observed")

# create y_art_pred dataframe with predicted y_est
y_art_pred = pandas.DataFrame(columns=y_art.columns)
k = len(y_art)
for i in range(0, k):
   artList = y_art.ix[i,0:5].tolist()
   y_est = est.predict(y_art.ix[i,0:5])
   artList.append(y_est[0])
   artSeries = pandas.Series(artList, index = y_art.columns)
   y_art_pred = y_art_pred.append(artSeries, ignore_index = True)

j = len(y_art) - 1
for i in range(j, j+13):
   artList = y_art_pred.ix[i,1:6].tolist()
   y_est = est.predict(y_art_pred.ix[i,0:5])
   artList.append(y_est[0])
   artSeries = pandas.Series(artList, index = y_art.columns)
   y_art_pred = y_art_pred.append(artSeries, ignore_index = True)
          
fig, ax1 = plt.subplots()
y_art.y.plot(ax=ax1, color = 'g', linewidth = 3, label = 'sum of pledges')
y_art_pred.y.plot(ax=ax1, color = 'g', linewidth = 3, linestyle = '--', label = 'predicted sum of pledges')
ax1.set_xlabel('Year (by quarter)', fontsize = 12)
ax1.set_ylabel('USD', fontsize = 12)

a=ax1.get_xticks().tolist()
a=('Q4-2010','Q1-2012', 'Q2-2013', 'Q3-2014', 'Q4-2015', 'Q1-2017', 'Q2-2018')
ax1.set_xticklabels(a, rotation = 'vertical')
ax1.text(20, 150000, 'R squared =', fontsize = 12)
ax1.text(20, 100000, round(est.rsquared, 3), fontsize = 12)

ax1.legend(bbox_to_anchor=(.45, .55), loc = 2)

plt.title('Prediction of Kickstarter pledges for Art', fontsize=16)
ax1.title.set_position((.5,1.08))

#############################  Technology   ########################################


df_tech_Qsum = df[df.main_category == 'technology']
df_tech_Qsum = df_tech_Qsum[['deadline', 'pledged_USD']].sort('deadline').resample('Q', how = 'sum')

y_tech = pandas.DataFrame(columns=('x1', 'x2', 'x3', 'x4', 'y'))

for i in range(0,(len(df_tech_Qsum) - 5)):
    y_tech.loc[i] = [df_tech_Qsum.pledged_USD.ix[i], df_tech_Qsum.pledged_USD.ix[i+1], df_tech_Qsum.pledged_USD.ix[i+2], df_tech_Qsum.pledged_USD.ix[i+3], df_tech_Qsum.pledged_USD.ix[i+4]]   
    
# Run linear regression with cross validation
    
X = y_tech[['x1', 'x2', 'x3', 'x4']]
y = y_tech['y']
  
est = smf.ols(formula='y ~ x1 + x2 + x3 + x4', data=y_tech).fit()
# now, let's print out the results.
print est.summary()

yp = est.predict(X)
plt.plot(yp, y, 'o')
plt.plot(y, y, 'r-')
plt.xlabel("Predicted")
plt.ylabel("Observed")

# create y_tech_pred dataframe with predicted y_est
y_tech_pred = pandas.DataFrame(columns=y_tech.columns)
k = len(y_tech)
for i in range(0, k):
   techList = y_tech.ix[i,0:4].tolist()
   y_est = est.predict(y_tech.ix[i,0:4])
   techList.append(y_est[0])
   techSeries = pandas.Series(techList, index = y_tech.columns)
   y_tech_pred = y_tech_pred.append(techSeries, ignore_index = True)

j = len(y_tech) - 1
for i in range(j, j+13):
   techList = y_tech_pred.ix[i,1:5].tolist()
   y_est = est.predict(y_tech_pred.ix[i,0:4])
   techList.append(y_est[0])
   techSeries = pandas.Series(techList, index = y_tech.columns)
   y_tech_pred = y_tech_pred.append(techSeries, ignore_index = True)
          
fig, ax1 = plt.subplots()
y_tech.y.plot(ax=ax1, color = 'b', linewidth = 3, label = 'sum of pledges')
y_tech_pred.y.plot(ax=ax1, color = 'b', linewidth = 3, linestyle = '--', label = 'predicted sum of pledges')
ax1.set_xlabel('Year (by quarter)', fontsize = 12)
ax1.set_ylabel('USD x 10^7', fontsize = 12)

a=ax1.get_xticks().tolist()
a=('Q4-2010','Q1-2012', 'Q2-2013', 'Q3-2014', 'Q4-2015', 'Q1-2017', 'Q2-2018')
ax1.set_xticklabels(a, rotation = 'vertical')
ax1.text(20, 3000000, 'R squared =', fontsize = 12)
ax1.text(20, 1000000, round(est.rsquared, 3), fontsize = 12)

ax1.legend( loc = 'best')

plt.title('Prediction of Kickstarter pledges for Technology', fontsize=16)
ax1.title.set_position((.5,1.08))


#############################  Comics  ######################################
# Nan for Q3 2009- need to replace with 0
'''
df_comics_Qsum = df[df.main_category == 'comics']
df_comics_Qsum = df_comics_Qsum[['deadline', 'pledged_USD']].sort('deadline').resample('Q', how = 'sum')

y_comics = pandas.DataFrame(columns=('x1', 'x2', 'x3', 'x4', 'x5', 'y'))

for i in range(0,(len(df_comics_Qsum) - 6)):
    y_comics.loc[i] = [df_comics_Qsum.pledged_USD.ix[i], df_comics_Qsum.pledged_USD.ix[i+1], df_comics_Qsum.pledged_USD.ix[i+2], df_comics_Qsum.pledged_USD.ix[i+3], df_comics_Qsum.pledged_USD.ix[i+4], df_comics_Qsum.pledged_USD.ix[i+5]]   
   
# Run linear regression with cross validation
    
X = y_comics[['x1', 'x2', 'x3', 'x4', 'x5']]
y = y_comics['y']
  
est = smf.ols(formula='y ~ x1 + x2 + x3 + x4 + x5', data=y_comics).fit()
# now, let's print out the results.
print est.summary()

yp = est.predict(X)
plt.plot(yp, y, 'o')
plt.plot(y, y, 'r-')
plt.xlabel("Predicted")
plt.ylabel("Observed")

# create y_comics_pred dataframe with predicted y_est
y_comics_pred = pandas.DataFrame(columns=y_comics.columns)
k = len(y_comics)
for i in range(0, k):
   comicsList = y_comics.ix[i,0:5].tolist()
   y_est = est.predict(y_comics.ix[i,0:5])
   comicsList.append(y_est[0])
   comicsSeries = pandas.Series(comicsList, index = y_comics.columns)
   y_comics_pred = y_comics_pred.append(comicsSeries, ignore_index = True)

j = len(y_comics) - 1
for i in range(j, j+13):
   comicsList = y_comics_pred.ix[i,1:6].tolist()
   y_est = est.predict(y_comics_pred.ix[i,0:5])
   comicsList.append(y_est[0])
   comicsSeries = pandas.Series(comicsList, index = y_comics.columns)
   y_comics_pred = y_comics_pred.append(comicsSeries, ignore_index = True)
          
fig, ax1 = plt.subplots()
y_comics.y.plot(ax=ax1, color = 'g', linewidth = 3, label = 'sum of pledges')
y_comics_pred.y.plot(ax=ax1, color = 'g', linewidth = 3, linestyle = '--', label = 'predicted sum of pledges')
ax1.set_xlabel('Year (by quarter)', fontsize = 12)
ax1.set_ylabel('USD x 10^7', fontsize = 12)

a=ax1.get_xticks().tolist()
a=('Q4-2010','Q1-2012', 'Q2-2013', 'Q3-2014', 'Q4-2015', 'Q1-2017', 'Q2-2018')
ax1.set_xticklabels(a, rotation = 'vertical')
ax1.text(20, 150000, 'R squared =', fontsize = 12)
ax1.text(20, 100000, round(est.rsquared, 3), fontsize = 12)

ax1.legend(bbox_to_anchor=(.45, .55), loc = 2)

plt.title('Prediction of Kickstarter pledges for Comics', fontsize=16)
ax1.title.set_position((.5,1.08))
'''

############################  Crafts  ###############################################

df_crafts_Qsum = df[df.main_category == 'crafts']
df_crafts_Qsum = df_crafts_Qsum[['deadline', 'pledged_USD']].sort('deadline').resample('Q', how = 'sum')

y_crafts = pandas.DataFrame(columns=('x1', 'x2', 'x3', 'x4', 'x5', 'y'))

for i in range(0,(len(df_crafts_Qsum) - 6)):
    y_crafts.loc[i] = [df_crafts_Qsum.pledged_USD.ix[i], df_crafts_Qsum.pledged_USD.ix[i+1], df_crafts_Qsum.pledged_USD.ix[i+2], df_crafts_Qsum.pledged_USD.ix[i+3], df_crafts_Qsum.pledged_USD.ix[i+4], df_crafts_Qsum.pledged_USD.ix[i+5]]   
    
# Run linear regression with cross validation
    
X = y_crafts[['x1', 'x2', 'x3', 'x4', 'x5']]
y = y_crafts['y']
  
est = smf.ols(formula='y ~ x1 + x2 + x3 + x4 + x5', data=y_crafts).fit()
# now, let's print out the results.
print est.summary()

yp = est.predict(X)
plt.plot(yp, y, 'o')
plt.plot(y, y, 'r-')
plt.xlabel("Predicted")
plt.ylabel("Observed")

# create y_crafts_pred dataframe with predicted y_est
y_crafts_pred = pandas.DataFrame(columns=y_crafts.columns)
k = len(y_crafts)
for i in range(0, k):
   craftsList = y_crafts.ix[i,0:5].tolist()
   y_est = est.predict(y_crafts.ix[i,0:5])
   craftsList.append(y_est[0])
   craftsSeries = pandas.Series(craftsList, index = y_crafts.columns)
   y_crafts_pred = y_crafts_pred.append(craftsSeries, ignore_index = True)

j = len(y_crafts) - 1
for i in range(j, j+13):
   craftsList = y_crafts_pred.ix[i,1:6].tolist()
   y_est = est.predict(y_crafts_pred.ix[i,0:5])
   craftsList.append(y_est[0])
   craftsSeries = pandas.Series(craftsList, index = y_crafts.columns)
   y_crafts_pred = y_crafts_pred.append(craftsSeries, ignore_index = True)
          
fig, ax1 = plt.subplots()
y_crafts.y.plot(ax=ax1, color = 'g', linewidth = 3, label = 'sum of pledges')
y_crafts_pred.y.plot(ax=ax1, color = 'g', linewidth = 3, linestyle = '--', label = 'predicted sum of pledges')
ax1.set_xlabel('Year (by quarter)', fontsize = 12)
ax1.set_ylabel('USD', fontsize = 12)

a=ax1.get_xticks().tolist()
a=('Q4-2010','Q1-2012', 'Q2-2013', 'Q3-2014', 'Q4-2015', 'Q1-2017', 'Q2-2018')
ax1.set_xticklabels(a, rotation = 'vertical')
ax1.text(3, 300000, 'R squared =', fontsize = 12)
ax1.text(3, 220000, round(est.rsquared, 3), fontsize = 12)

ax1.legend(bbox_to_anchor=(.1, .55), loc = 2)

plt.title('Prediction of Kickstarter pledges for Crafts', fontsize=16)
ax1.title.set_position((.5,1.08))

############################  Dance  ###############################################

df_dance_Qsum = df[df.main_category == 'dance']
df_dance_Qsum = df_dance_Qsum[['deadline', 'pledged_USD']].sort('deadline').resample('Q', how = 'sum')

y_dance = pandas.DataFrame(columns=('x1', 'x2', 'x3', 'x4', 'x5', 'y'))

for i in range(0,(len(df_dance_Qsum) - 6)):
    y_dance.loc[i] = [df_dance_Qsum.pledged_USD.ix[i], df_dance_Qsum.pledged_USD.ix[i+1], df_dance_Qsum.pledged_USD.ix[i+2], df_dance_Qsum.pledged_USD.ix[i+3], df_dance_Qsum.pledged_USD.ix[i+4], df_dance_Qsum.pledged_USD.ix[i+5]]   
    
# Run linear regression with cross validation
    
X = y_dance[['x1', 'x2', 'x3', 'x4', 'x5']]
y = y_dance['y']
  
est = smf.ols(formula='y ~ x1 + x2 + x3 + x4 + x5', data=y_dance).fit()
# now, let's print out the results.
print est.summary()

yp = est.predict(X)
plt.plot(yp, y, 'o')
plt.plot(y, y, 'r-')
plt.xlabel("Predicted")
plt.ylabel("Observed")

# create y_dance_pred dataframe with predicted y_est
y_dance_pred = pandas.DataFrame(columns=y_dance.columns)
k = len(y_dance)
for i in range(0, k):
   danceList = y_dance.ix[i,0:5].tolist()
   y_est = est.predict(y_dance.ix[i,0:5])
   danceList.append(y_est[0])
   danceSeries = pandas.Series(danceList, index = y_dance.columns)
   y_dance_pred = y_dance_pred.append(danceSeries, ignore_index = True)

j = len(y_dance) - 1
for i in range(j, j+13):
   danceList = y_dance_pred.ix[i,1:6].tolist()
   y_est = est.predict(y_dance_pred.ix[i,0:5])
   danceList.append(y_est[0])
   danceSeries = pandas.Series(danceList, index = y_dance.columns)
   y_dance_pred = y_dance_pred.append(danceSeries, ignore_index = True)
          
fig, ax1 = plt.subplots()
y_dance.y.plot(ax=ax1, color = 'g', linewidth = 3, label = 'sum of pledges')
y_dance_pred.y.plot(ax=ax1, color = 'g', linewidth = 3, linestyle = '--', label = 'predicted sum of pledges')
ax1.set_xlabel('Year (by quarter)', fontsize = 12)
ax1.set_ylabel('USD', fontsize = 12)

a=ax1.get_xticks().tolist()
a=('Q4-2010','Q1-2012', 'Q2-2013', 'Q3-2014', 'Q4-2015', 'Q1-2017', 'Q2-2018')
ax1.set_xticklabels(a, rotation = 'vertical')
ax1.text(3, 300000, 'R squared =', fontsize = 12)
ax1.text(3, 220000, round(est.rsquared, 3), fontsize = 12)

ax1.legend(bbox_to_anchor=(.1, .55), loc = 2)

plt.title('Prediction of Kickstarter pledges for Dance', fontsize=16)
ax1.title.set_position((.5,1.08))

############################  Design  ###############################################

df_design_Qsum = df[df.main_category == 'design']
df_design_Qsum = df_design_Qsum[['deadline', 'pledged_USD']].sort('deadline').resample('Q', how = 'sum')

y_design = pandas.DataFrame(columns=('x1', 'x2', 'x3', 'x4', 'x5', 'y'))

for i in range(0,(len(df_design_Qsum) - 6)):
    y_design.loc[i] = [df_design_Qsum.pledged_USD.ix[i], df_design_Qsum.pledged_USD.ix[i+1], df_design_Qsum.pledged_USD.ix[i+2], df_design_Qsum.pledged_USD.ix[i+3], df_design_Qsum.pledged_USD.ix[i+4], df_design_Qsum.pledged_USD.ix[i+5]]   
    
# Run linear regression with cross validation
    
X = y_design[['x1', 'x2', 'x3', 'x4', 'x5']]
y = y_design['y']
  
est = smf.ols(formula='y ~ x1 + x2 + x3 + x4 + x5', data=y_design).fit()
# now, let's print out the results.
print est.summary()

yp = est.predict(X)
plt.plot(yp, y, 'o')
plt.plot(y, y, 'r-')
plt.xlabel("Predicted")
plt.ylabel("Observed")

# create y_design_pred dataframe with predicted y_est
y_design_pred = pandas.DataFrame(columns=y_design.columns)
k = len(y_design)
for i in range(0, k):
   designList = y_design.ix[i,0:5].tolist()
   y_est = est.predict(y_design.ix[i,0:5])
   designList.append(y_est[0])
   designSeries = pandas.Series(designList, index = y_design.columns)
   y_design_pred = y_design_pred.append(designSeries, ignore_index = True)

j = len(y_design) - 1
for i in range(j, j+13):
   designList = y_design_pred.ix[i,1:6].tolist()
   y_est = est.predict(y_design_pred.ix[i,0:5])
   designList.append(y_est[0])
   designSeries = pandas.Series(designList, index = y_design.columns)
   y_design_pred = y_design_pred.append(designSeries, ignore_index = True)
          
fig, ax1 = plt.subplots()
y_design.y.plot(ax=ax1, color = 'g', linewidth = 3, label = 'sum of pledges')
y_design_pred.y.plot(ax=ax1, color = 'g', linewidth = 3, linestyle = '--', label = 'predicted sum of pledges')
ax1.set_xlabel('Year (by quarter)', fontsize = 12)
ax1.set_ylabel('USD', fontsize = 12)

a=ax1.get_xticks().tolist()
a=('Q4-2010','Q1-2012', 'Q2-2013', 'Q3-2014', 'Q4-2015', 'Q1-2017', 'Q2-2018')
ax1.set_xticklabels(a, rotation = 'vertical')
ax1.text(15, 1500000, 'R squared =', fontsize = 12)
ax1.text(15, 1000000, round(est.rsquared, 3), fontsize = 12)

ax1.legend(bbox_to_anchor=(.1, .9), loc = 2)

plt.title('Prediction of Kickstarter pledges for design', fontsize=16)
ax1.title.set_position((.5,1.08))

############################  Fashion  ###############################################

df_fashion_Qsum = df[df.main_category == 'fashion']
df_fashion_Qsum = df_fashion_Qsum[['deadline', 'pledged_USD']].sort('deadline').resample('Q', how = 'sum')

y_fashion = pandas.DataFrame(columns=('x1', 'x2', 'x3', 'x4', 'x5', 'y'))

for i in range(0,(len(df_fashion_Qsum) - 6)):
    y_fashion.loc[i] = [df_fashion_Qsum.pledged_USD.ix[i], df_fashion_Qsum.pledged_USD.ix[i+1], df_fashion_Qsum.pledged_USD.ix[i+2], df_fashion_Qsum.pledged_USD.ix[i+3], df_fashion_Qsum.pledged_USD.ix[i+4], df_fashion_Qsum.pledged_USD.ix[i+5]]   
    
# Run linear regression with cross validation
    
X = y_fashion[['x1', 'x2', 'x3', 'x4', 'x5']]
y = y_fashion['y']
  
est = smf.ols(formula='y ~ x1 + x2 + x3 + x4 + x5', data=y_fashion).fit()
# now, let's print out the results.
print est.summary()

yp = est.predict(X)
plt.plot(yp, y, 'o')
plt.plot(y, y, 'r-')
plt.xlabel("Predicted")
plt.ylabel("Observed")

# create y_fashion_pred dataframe with predicted y_est
y_fashion_pred = pandas.DataFrame(columns=y_fashion.columns)
k = len(y_fashion)
for i in range(0, k):
   fashionList = y_fashion.ix[i,0:5].tolist()
   y_est = est.predict(y_fashion.ix[i,0:5])
   fashionList.append(y_est[0])
   fashionSeries = pandas.Series(fashionList, index = y_fashion.columns)
   y_fashion_pred = y_fashion_pred.append(fashionSeries, ignore_index = True)

j = len(y_fashion) - 1
for i in range(j, j+13):
   fashionList = y_fashion_pred.ix[i,1:6].tolist()
   y_est = est.predict(y_fashion_pred.ix[i,0:5])
   fashionList.append(y_est[0])
   fashionSeries = pandas.Series(fashionList, index = y_fashion.columns)
   y_fashion_pred = y_fashion_pred.append(fashionSeries, ignore_index = True)
          
fig, ax1 = plt.subplots()
y_fashion.y.plot(ax=ax1, color = 'g', linewidth = 3, label = 'sum of pledges')
y_fashion_pred.y.plot(ax=ax1, color = 'g', linewidth = 3, linestyle = '--', label = 'predicted sum of pledges')
ax1.set_xlabel('Year (by quarter)', fontsize = 12)
ax1.set_ylabel('USD', fontsize = 12)

a=ax1.get_xticks().tolist()
a=('Q4-2010','Q1-2012', 'Q2-2013', 'Q3-2014', 'Q4-2015', 'Q1-2017', 'Q2-2018')
ax1.set_xticklabels(a, rotation = 'vertical')
ax1.text(3, 800000, 'R squared =', fontsize = 12)
ax1.text(3, 650000, round(est.rsquared, 3), fontsize = 12)

ax1.legend(bbox_to_anchor=(.1, .9), loc = 2)

plt.title('Prediction of Kickstarter pledges for fashion', fontsize=16)
ax1.title.set_position((.5,1.08))

############################  Film and Video ###############################################


df_film_and_video_Qsum = df[df.main_category == 'film_and_video']
df_film_and_video_Qsum = df_film_and_video_Qsum[['deadline', 'pledged_USD']].sort('deadline').resample('Q', how = 'sum')

y_film_and_video = pandas.DataFrame(columns=('x1', 'x2', 'x3', 'x4', 'y'))

for i in range(0,(len(df_film_and_video_Qsum) - 5)):
    y_film_and_video.loc[i] = [df_film_and_video_Qsum.pledged_USD.ix[i], df_film_and_video_Qsum.pledged_USD.ix[i+1], df_film_and_video_Qsum.pledged_USD.ix[i+2], df_film_and_video_Qsum.pledged_USD.ix[i+3], df_film_and_video_Qsum.pledged_USD.ix[i+4]]   
    
# Run linear regression with cross validation
    
X = y_film_and_video[['x1', 'x2', 'x3', 'x4']]
y = y_film_and_video['y']
  
est = smf.ols(formula='y ~ x1 + x2 + x3 + x4', data=y_film_and_video).fit()
# now, let's print out the results.
print est.summary()

yp = est.predict(X)
plt.plot(yp, y, 'o')
plt.plot(y, y, 'r-')
plt.xlabel("Predicted")
plt.ylabel("Observed")

# create y_film_and_video_pred dataframe with predicted y_est
y_film_and_video_pred = pandas.DataFrame(columns=y_film_and_video.columns)
k = len(y_film_and_video)
for i in range(0, k):
   film_and_videoList = y_film_and_video.ix[i,0:4].tolist()
   y_est = est.predict(y_film_and_video.ix[i,0:4])
   film_and_videoList.append(y_est[0])
   film_and_videoSeries = pandas.Series(film_and_videoList, index = y_film_and_video.columns)
   y_film_and_video_pred = y_film_and_video_pred.append(film_and_videoSeries, ignore_index = True)

j = len(y_film_and_video) - 1
for i in range(j, j+13):
   film_and_videoList = y_film_and_video_pred.ix[i,1:5].tolist()
   y_est = est.predict(y_film_and_video_pred.ix[i,0:4])
   film_and_videoList.append(y_est[0])
   film_and_videoSeries = pandas.Series(film_and_videoList, index = y_film_and_video.columns)
   y_film_and_video_pred = y_film_and_video_pred.append(film_and_videoSeries, ignore_index = True)
          
fig, ax1 = plt.subplots()
y_film_and_video.y.plot(ax=ax1, color = 'r', linewidth = 3, label = 'sum of pledges')
y_film_and_video_pred.y.plot(ax=ax1, color = 'r', linewidth = 3, linestyle = '--', label = 'predicted sum of pledges')
ax1.set_xlabel('Year (by quarter)', fontsize = 12)
ax1.set_ylabel('USD', fontsize = 12)

a=ax1.get_xticks().tolist()
a=('Q4-2010','Q1-2012', 'Q2-2013', 'Q3-2014', 'Q4-2015', 'Q1-2017', 'Q2-2018')
ax1.set_xticklabels(a, rotation = 'vertical')
ax1.text(1, 2500000, 'R squared =', fontsize = 12)
ax1.text(1, 2300000, round(est.rsquared, 3), fontsize = 12)

ax1.legend(loc = 'best')

plt.title('Prediction of Kickstarter pledges for film and video', fontsize=16)
ax1.title.set_position((.5,1.08))


############################  Food  ###############################################

df_food_Qsum = df[df.main_category == 'food']
df_food_Qsum = df_food_Qsum[['deadline', 'pledged_USD']].sort('deadline').resample('Q', how = 'sum')

y_food = pandas.DataFrame(columns=('x1', 'x2', 'x3', 'x4', 'x5', 'y'))

for i in range(0,(len(df_food_Qsum) - 6)):
    y_food.loc[i] = [df_food_Qsum.pledged_USD.ix[i], df_food_Qsum.pledged_USD.ix[i+1], df_food_Qsum.pledged_USD.ix[i+2], df_food_Qsum.pledged_USD.ix[i+3], df_food_Qsum.pledged_USD.ix[i+4], df_food_Qsum.pledged_USD.ix[i+5]]   
    
# Run linear regression with cross validation
    
X = y_food[['x1', 'x2', 'x3', 'x4', 'x5']]
y = y_food['y']
  
est = smf.ols(formula='y ~ x1 + x2 + x3 + x4 + x5', data=y_food).fit()
# now, let's print out the results.
print est.summary()

yp = est.predict(X)
plt.plot(yp, y, 'o')
plt.plot(y, y, 'r-')
plt.xlabel("Predicted")
plt.ylabel("Observed")

# create y_food_pred dataframe with predicted y_est
y_food_pred = pandas.DataFrame(columns=y_food.columns)
k = len(y_food)
for i in range(0, k):
   foodList = y_food.ix[i,0:5].tolist()
   y_est = est.predict(y_food.ix[i,0:5])
   foodList.append(y_est[0])
   foodSeries = pandas.Series(foodList, index = y_food.columns)
   y_food_pred = y_food_pred.append(foodSeries, ignore_index = True)

j = len(y_food) - 1
for i in range(j, j+13):
   foodList = y_food_pred.ix[i,1:6].tolist()
   y_est = est.predict(y_food_pred.ix[i,0:5])
   foodList.append(y_est[0])
   foodSeries = pandas.Series(foodList, index = y_food.columns)
   y_food_pred = y_food_pred.append(foodSeries, ignore_index = True)
          
fig, ax1 = plt.subplots()
y_food.y.plot(ax=ax1, color = 'g', linewidth = 3, label = 'sum of pledges')
y_food_pred.y.plot(ax=ax1, color = 'g', linewidth = 3, linestyle = '--', label = 'predicted sum of pledges')
ax1.set_xlabel('Year (by quarter)', fontsize = 12)
ax1.set_ylabel('USD', fontsize = 12)

a=ax1.get_xticks().tolist()
a=('Q4-2010','Q1-2012', 'Q2-2013', 'Q3-2014', 'Q4-2015', 'Q1-2017', 'Q2-2018')
ax1.set_xticklabels(a, rotation = 'vertical')
ax1.text(3, 1200000, 'R squared =', fontsize = 12)
ax1.text(3, 650000, round(est.rsquared, 3), fontsize = 12)

ax1.legend(bbox_to_anchor=(.1, .9), loc = 2)

plt.title('Prediction of Kickstarter pledges for food', fontsize=16)
ax1.title.set_position((.5,1.08))

############################  Games  ###############################################


df_games_Qsum = df[df.main_category == 'games']
df_games_Qsum = df_games_Qsum[['deadline', 'pledged_USD']].sort('deadline').resample('Q', how = 'sum')

y_games = pandas.DataFrame(columns=('x1', 'x2', 'x3', 'x4', 'y'))

for i in range(0,(len(df_games_Qsum) - 6)):
    y_games.loc[i] = [df_games_Qsum.pledged_USD.ix[i], df_games_Qsum.pledged_USD.ix[i+1], df_games_Qsum.pledged_USD.ix[i+2], df_games_Qsum.pledged_USD.ix[i+3], df_games_Qsum.pledged_USD.ix[i+4]]   
    
# Run linear regression with cross validation
    
X = y_games[['x1', 'x2', 'x3', 'x4']]
y = y_games['y']
  
est = smf.ols(formula='y ~ x1 + x2 + x3 + x4', data=y_games).fit()
# now, let's print out the results.
print est.summary()

yp = est.predict(X)
plt.plot(yp, y, 'o')
plt.plot(y, y, 'r-')
plt.xlabel("Predicted")
plt.ylabel("Observed")

# create y_games_pred dataframe with predicted y_est
y_games_pred = pandas.DataFrame(columns=y_games.columns)
k = len(y_games)
for i in range(0, k):
   gamesList = y_games.ix[i,0:4].tolist()
   y_est = est.predict(y_games.ix[i,0:4])
   gamesList.append(y_est[0])
   gamesSeries = pandas.Series(gamesList, index = y_games.columns)
   y_games_pred = y_games_pred.append(gamesSeries, ignore_index = True)

j = len(y_games) - 1
for i in range(j, j+13):
   gamesList = y_games_pred.ix[i,1:5].tolist()
   y_est = est.predict(y_games_pred.ix[i,0:4])
   gamesList.append(y_est[0])
   gamesSeries = pandas.Series(gamesList, index = y_games.columns)
   y_games_pred = y_games_pred.append(gamesSeries, ignore_index = True)
          
fig, ax1 = plt.subplots()
y_games.y.plot(ax=ax1, color = 'b', linewidth = 3, label = 'sum of pledges')
y_games_pred.y.plot(ax=ax1, color = 'b', linewidth = 3, linestyle = '--', label = 'predicted sum of pledges')
ax1.set_xlabel('Year (by quarter)', fontsize = 12)
ax1.set_ylabel('USD', fontsize = 12)

a=ax1.get_xticks().tolist()
a=('Q4-2010','Q1-2012', 'Q2-2013', 'Q3-2014', 'Q4-2015', 'Q1-2017', 'Q2-2018')
ax1.set_xticklabels(a, rotation = 'vertical')
ax1.text(1, 1200000, 'R squared =', fontsize = 12)
ax1.text(1, 900000, round(est.rsquared, 3), fontsize = 12)

ax1.legend(loc = 'best')

plt.title('Prediction of Kickstarter pledges for games', fontsize=16)
ax1.title.set_position((.5,1.08))


############################  Journalism  ###############################################
# unstable
'''
df_journalism_Qsum = df[df.main_category == 'journalism']
df_journalism_Qsum = df_journalism_Qsum[['deadline', 'pledged_USD']].sort('deadline').resample('Q', how = 'sum')

y_journalism = pandas.DataFrame(columns=('x1', 'x2', 'x3', 'x4', 'x5', 'y'))

for i in range(0,(len(df_journalism_Qsum) - 6)):
    y_journalism.loc[i] = [df_journalism_Qsum.pledged_USD.ix[i], df_journalism_Qsum.pledged_USD.ix[i+1], df_journalism_Qsum.pledged_USD.ix[i+2], df_journalism_Qsum.pledged_USD.ix[i+3], df_journalism_Qsum.pledged_USD.ix[i+4], df_journalism_Qsum.pledged_USD.ix[i+5]]   
    
# Run linear regression with cross validation
    
X = y_journalism[['x1', 'x2', 'x3', 'x4', 'x5']]
y = y_journalism['y']
  
est = smf.ols(formula='y ~ x1 + x2 + x3 + x4 + x5', data=y_journalism).fit()
# now, let's print out the results.
print est.summary()

yp = est.predict(X)
plt.plot(yp, y, 'o')
plt.plot(y, y, 'r-')
plt.xlabel("Predicted")
plt.ylabel("Observed")

# create y_journalism_pred dataframe with predicted y_est
y_journalism_pred = pandas.DataFrame(columns=y_journalism.columns)
k = len(y_journalism)
for i in range(0, k):
   journalismList = y_journalism.ix[i,0:5].tolist()
   y_est = est.predict(y_journalism.ix[i,0:5])
   journalismList.append(y_est[0])
   journalismSeries = pandas.Series(journalismList, index = y_journalism.columns)
   y_journalism_pred = y_journalism_pred.append(journalismSeries, ignore_index = True)

j = len(y_journalism) - 1
for i in range(j, j+13):
   journalismList = y_journalism_pred.ix[i,1:6].tolist()
   y_est = est.predict(y_journalism_pred.ix[i,0:5])
   journalismList.append(y_est[0])
   journalismSeries = pandas.Series(journalismList, index = y_journalism.columns)
   y_journalism_pred = y_journalism_pred.append(journalismSeries, ignore_index = True)
          
fig, ax1 = plt.subplots()
y_journalism.y.plot(ax=ax1, color = 'g', linewidth = 3, label = 'sum of pledges')
y_journalism_pred.y.plot(ax=ax1, color = 'g', linewidth = 3, linestyle = '--', label = 'predicted sum of pledges')
ax1.set_xlabel('Year (by quarter)', fontsize = 12)
ax1.set_ylabel('USD x 10^7', fontsize = 12)

a=ax1.get_xticks().tolist()
a=('Q4-2010','Q1-2012', 'Q2-2013', 'Q3-2014', 'Q4-2015', 'Q1-2017', 'Q2-2018')
ax1.set_xticklabels(a, rotation = 'vertical')
ax1.text(3, 1200000, 'R squared =', fontsize = 12)
ax1.text(3, 650000, round(est.rsquared, 3), fontsize = 12)

ax1.legend(bbox_to_anchor=(.1, .9), loc = 2)

plt.title('Prediction of Kickstarter pledges for journalism', fontsize=16)
ax1.title.set_position((.5,1.08))
'''

############################  Music  ###############################################
# unstable

'''
df_music_Qsum = df[df.main_category == 'music']
df_music_Qsum = df_music_Qsum[['deadline', 'pledged_USD']].sort('deadline').resample('Q', how = 'sum')

y_music = pandas.DataFrame(columns=('x1', 'x2', 'x3', 'x4', 'x5', 'y'))

for i in range(0,(len(df_music_Qsum) - 6)):
    y_music.loc[i] = [df_music_Qsum.pledged_USD.ix[i], df_music_Qsum.pledged_USD.ix[i+1], df_music_Qsum.pledged_USD.ix[i+2], df_music_Qsum.pledged_USD.ix[i+3], df_music_Qsum.pledged_USD.ix[i+4], df_music_Qsum.pledged_USD.ix[i+5]]   
    
# Run linear regression with cross validation
    
X = y_music[['x1', 'x2', 'x3', 'x4', 'x5']]
y = y_music['y']
  
est = smf.ols(formula='y ~ x1 + x2 + x3 + x4 + x5', data=y_music).fit()
# now, let's print out the results.
print est.summary()

yp = est.predict(X)
plt.plot(yp, y, 'o')
plt.plot(y, y, 'r-')
plt.xlabel("Predicted")
plt.ylabel("Observed")

# create y_music_pred dataframe with predicted y_est
y_music_pred = pandas.DataFrame(columns=y_music.columns)
k = len(y_music)
for i in range(0, k):
   musicList = y_music.ix[i,0:5].tolist()
   y_est = est.predict(y_music.ix[i,0:5])
   musicList.append(y_est[0])
   musicSeries = pandas.Series(musicList, index = y_music.columns)
   y_music_pred = y_music_pred.append(musicSeries, ignore_index = True)

j = len(y_music) - 1
for i in range(j, j+13):
   musicList = y_music_pred.ix[i,1:6].tolist()
   y_est = est.predict(y_music_pred.ix[i,0:5])
   musicList.append(y_est[0])
   musicSeries = pandas.Series(musicList, index = y_music.columns)
   y_music_pred = y_music_pred.append(musicSeries, ignore_index = True)
          
fig, ax1 = plt.subplots()
y_music.y.plot(ax=ax1, color = 'g', linewidth = 3, label = 'sum of pledges')
y_music_pred.y.plot(ax=ax1, color = 'g', linewidth = 3, linestyle = '--', label = 'predicted sum of pledges')
ax1.set_xlabel('Year (by quarter)', fontsize = 12)
ax1.set_ylabel('USD x 10^7', fontsize = 12)

a=ax1.get_xticks().tolist()
a=('Q4-2010','Q1-2012', 'Q2-2013', 'Q3-2014', 'Q4-2015', 'Q1-2017', 'Q2-2018')
ax1.set_xticklabels(a, rotation = 'vertical')
ax1.text(3, 1200000, 'R squared =', fontsize = 12)
ax1.text(3, 650000, round(est.rsquared, 3), fontsize = 12)

ax1.legend(bbox_to_anchor=(.1, .9), loc = 2)

plt.title('Prediction of Kickstarter pledges for music', fontsize=16)
ax1.title.set_position((.5,1.08))
'''
# publishing, theater
