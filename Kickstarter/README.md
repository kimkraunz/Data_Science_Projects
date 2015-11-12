#Kickstarter


I created this Capstone project for the General Assembly course that I took in the spring of 2015.  I used python (scikit learn, pandas, numpy) to predict whether or not a Kickstarter project would be funded as well as the amount of funding per Kickstarter category over time. 

The presentation of the results is [here](Kickstarter%20presentation.pdf).
The python code used for the analysis is [here](kickstarter.py) and [here](need.py)

##Exploring the data
I first looked at how project funding has changed over time.  I saw that there was a dramatic increase in funding around 2012.  I looked at both the proportion of projects funded and proportion of pledged dollars as well.  We see that starting in 2012 that that proportion of projects funded peaks and that the proportion of pledged dollars is over 100%.  In early 2013, the amount of projects increases while we see a decrease in the amount of funding in both the proportion of projects funded and proportion of dollars pledged.  Meanwhile, there is a dramatic increase in the amount of the proposed projects goals (in dollars).

![Average project by year](ave_project_year_graph.png)

I looked at the relationship between the log sum of the amounts in both the goals and pledges over time in comparison to the number of projects.  This again highlights the change in both proposed projects and funding in early 2013. 

![Log sum of projects](Log sum of projects.png)

I examined the frequency of the number of backers for each project.  As expected, most projects have a few number of backers while there are few projects that have a high number of backers.

![Backers](backers.png)

I also compared the number of backers for projects that were funded versus those that were not funded.  Not suprisingly, the projects that were funded had more backers.

![Backers funding](backers_funding.png)

Each project has a category.  I looked at the distribution of categories over time (by month).  The figure (messy!) shows that the categories with the highest pledged dollars are technology, gaming, design, and film and videos.

![Categories](Categories.png)

Each project also has a description to try to convice backers to support the project.  I used TextBlob, a text processing library in python to look at the distribution of polarity and subjectivity of the project description.

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

