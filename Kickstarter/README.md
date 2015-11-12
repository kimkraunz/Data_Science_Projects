#Kickstarter


I created this Capstone project for the General Assembly course that I took in the spring of 2015.  I used python (scikit learn, pandas, numpy) to predict whether or not a Kickstarter project would be funded as well as the amount of funding per Kickstarter category over time. 

The presentation of the results is [here](Kickstarter%20presentation.pdf).

##Exploring the data
I first looked at how project funding has changed over time.  I saw that there was a dramatic increase in funding around 2013.
![Average project by year](ave_project_year_graph.png)

![Log sum of projects](Log sum of projects.png)

![Backers](backers.png)
![Backers funding](backers_funding.png)
![Categories](Categories.png)
![Polarity and subjectivity](polarity and subjectivity.png)

##Predicting whether or not a Kickstarter project would be funded

![RFE](RFE.png)
![ROC score with sentiment](ROC score with sentiment and back.png)
![ROC score without the number of backers](ROC sent no backs .png)

## Predicting the amount of funding by Kickstarter category
I then used a time series linear regression predictive model for each category to predict how funding would change over time.  The r-squared (on each figure) indicates the goodness of fit of each model.  The models that had more consistent time series data had more consistent predictions.

###Technology
![Technology](technology_prediction.png)
###Gaming
![Gaming](games2_prediction.png)
###Film and Videos
![Film and Videos](film_and_videos_prediction.png)
###Design
![Design](design_prediction.png)
###Art
![Art](Art_prediction.png)
###Fashion
![Fashion](fashion_prediction.png)
###Crafts
![Crafts](Crafts_prediction.png)
###Dance
![Dance](Dance_prediction.png)
###Food
![Food](Food_prediction.png)

