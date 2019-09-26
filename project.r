#Installing and importing the necessary libraries.
#install.packages("e1071")
library(e1071)
library(rpart)
#install.packages('randomForest')
library(randomForest)
#install.packages('pROC')
library(pROC)
#nstall.packages('caret')
library(caret)
#Clearing all the unnecessary variables from the environment.
rm(list=ls(all=T))
setwd("/Users/anchitchaturvedi/Desktop/Study/Edwisor/project 3")
getwd()
df = read.csv('train.csv')
head(df)
#Dropping the ID_code column from the dataframe, as it is not required for the analysis.
df = subset(df, select = -c(ID_code) )
#Converting the target column to categorical type.
df[,1] = sapply(df[,1], as.factor)
head(df)
#Checking null values in the data.
sapply(df, function(x) sum(is.na (x)))
set.seed(100)
#Performing train-test split.
train_index = sample(1:nrow(df), 0.75*nrow(df))
train = df[train_index,]
test = df[-train_index,]
#Implementing the logistic regression model.
logit_model = glm(target ~ ., data = train, family = binomial)
#Getting the probability scores from the model.
lr_prob = predict(logit_model, test[,-1], type = 'response')
lr_prob
#Getting the class labels from the model.
pred_lr = ifelse(lr_prob > 0.5, 1, 0)
#Calculating AUC score for the model.
auc_lr = auc(test[,1], lr_prob)
auc_lr
cf_lr = as.matrix(table(test[,1], pred_lr))
#Confusion Matrix for the Logistic regression model.
confusionMatrix(cf_lr)
#Precision score of the model
precision_lr = cf_lr['1','1'] / (cf_lr['1','1'] + cf_lr['1','0'])
precision_lr
#Recall of the model
recall_lr = cf_lr['1','1'] / (cf_lr['1','1'] + cf_lr['0','1'])
recall_lr
#Training the decision tree model
dt = rpart(target ~., train)
#Getting class labels using the model
pred_dt = predict(dt, test[,-1], type = 'class')
pred_dt
#Getting the probability scores using the model.
dt_prob = predict(dt, test[,-1], type = 'prob')
#AUC score of the model
auc_dt = auc(test[,1], dt_prob[,2])
auc_dt
#Confusion Matrix of the model
cf_dt = as.matrix(table(test[,1], pred_dt))
confusionMatrix(cf_rf)
#Precision score for the model
precision_dt = cf_dt['1','1'] / (cf_dt['1','1'] + cf_dt['1','0'])
precision_dt
#Recall of the decision tree model.
recall_dt = cf_dt['1','1'] / (cf_dt['1','1'] + cf_dt['0','1'])
recall_dt
#Implementing the Naive Bayes algorithm
nb = naiveBayes(target ~., train)
#Predicting class labels using the trained model
pred_nb = predict(nb, test[,-1], type = 'class')
pred_nb
#Predicting probability scores using the model.
nb_prob = predict(nb, test[,-1], type = 'raw')
nb_prob
#AUC score for the model
auc_nb = auc(test[,1], nb_prob[,2], levels = c(0, 1), direction = "<")
auc_nb
#Confusion matrix of the Naive Bayes model
cf_nb = as.matrix(table(test[,1], pred_nb))
confusionMatrix(cf_nb)
#Precision score for the model
precision_nb = cf_nb['1','1'] / (cf_nb['1','1'] + cf_nb['1','0'])
precision_nb
#Recall for the model
recall_nb = cf_nb['1','1'] / (cf_nb['1','1'] + cf_nb['0','1'])
recall_nb
#Training the Random Forest model
set.seed(100)
rf = randomForest(target ~., train, ntree=100, importance= TRUE)
#Predicting class labels using the trained model.
pred_rf = predict(rf, test[,-1], type = 'class')
rf
#Getting probability scores using the trained model
rf_prob = predict(rf, test[,-1], type = 'prob')
rf_prob
#AUC for the model
auc_rf = auc(test[,1], rf_prob[,2])
auc_rf
#Confusion matrix for the Random forest model.
cf_rf = as.matrix(table(test[,1], pred_rf))
confusionMatrix(cf_rf)
#Precision score of the model
precision_rf = cf_rf['1','1'] / (cf_rf['1','1'] + cf_rf['1','0'])
precision_rf
#Recall score of the model
recall_rf = cf_rf['1','1'] / (cf_rf['1','1'] + cf_rf['0','1'])
recall_rf
#Variable Importance using the Random Forest model
varImpPlot(rf, type = '2')
test_set = read.csv('test.csv')
#Training the Naive Bayes model on the whole training dataset
nbfinal = naiveBayes(target ~., df)
#Performing final predictions on the test dataset using the model
final_pred = predict(nbfinal, test_set[,-1], type = 'class')
