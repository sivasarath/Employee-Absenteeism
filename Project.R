rm(list=ls(all=T))
setwd("C:\Users\sarath chandra\Desktop\data science\project 2")

#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')

#install.packages(x)
lapply(x, require, character.only = TRUE)
rm(x)

## Read the data
marketing_train = read.csv("project.csv", header = T, na.strings = c(" ", "", "NA"))
missing_val = data.frame(apply(marketing_train,2,function(x){sum(is.na(x))}))
marketing_train = knnImputation(marketing_train, k = 3)

factor_data=('Reason for absence','Day of the week','Seasons','Month of absence',
      'Disciplinary failure','Education','Son','Social drinker','Social smoker','Absenteeism time in hours')
cnames=('ID','Transportation expense','Distance from Residence to Work','Service time','Age',
      'Work load Average','Hit target','Pet','Weight','Height','Body mass index')
for (i in 1:10)
{
  print(names(factor_data)[i])
  print(chisq.test(table(factor_data$responded,factor_data[,i])))
  
}

marketing_train = subset(marketing_train,select = -c(education,socila drinker))

for(i in cnames){
  print(i)
  marketing_train[,i] = (marketing_train[,i] - min(marketing_train[,i]))/
    (max(marketing_train[,i] - min(marketing_train[,i])))
}

train.index = createDataPartition(marketing_train$responded, p = .80, list = FALSE)
train = marketing_train[ train.index,]
test  = marketing_train[-train.index,]

##Decision tree for classification
#Develop Model on training data
C50_model = C5.0(responded ~., train, trials = 100, rules = TRUE)

C50_Predictions = predict(C50_model, test[,-17], type = "class")

ConfMatrix_C50 = table(test$absenteeisminhours, C50_Predictions)
cm=confusionMatrix(ConfMatrix_C50)

tp=0
tot=0
for i in cm.columns
{
  for j in cm.columns{
  tot=tot+cm[i][j]
if(i==j)
{
  tp=tp+cm[i][j]
}
print(tp/tot)
  }
}
RF_model = randomForest(responded ~ ., train, importance = TRUE, ntree = 500)

RF_Predictions = predict(RF_model, test[,-18])

##Evaluate the performance of classification model
ConfMatrix_RF = table(test$absenteeisminhours, RF_Predictions)
cm=confusionMatrix(ConfMatrix_RF)

for i in cm.columns
{
  for j in cm.columns{
    tot=tot+cm[i][j]
    if(i==j)
    {
      tp=tp+cm[i][j]
    }
    print(tp/tot)
  }
}

NB_model = naiveBayes(responded ~ ., data = train)

#predict on test cases #raw
NB_Predictions = predict(NB_model, test[,1:17], type = 'class')

Conf_matrix = table(observed = test[,18], predicted = NB_Predictions)
confusionMatrix(Conf_matrix)
