rm(list = ls())
######Prepare ------
library("rpart")
library(tree)
setwd("D://Study//Other_MS//GeorgeTechOnlineCSMaster//OMSCS//CS7641_ML//Assignments//AdultData")
data <- read.csv("adult_data.csv")

str(TrainData)
summary(TrainData)



############ 1) Learning Curve -------
data$WORKCLASS_n <- as.numeric(data$WORKCLASS)
data$EDUCATION_n <- as.numeric(data$EDUCATION)
data$MARITAL_STATUS_n <- as.numeric(data$MARITAL_STATUS)
data$OCCUPATION_n <- as.numeric(data$OCCUPATION)
data$RELATIONSHIP_n <- as.numeric(data$RELATIONSHIP)
data$RACE_n <- as.numeric(data$RACE)
data$SEX_n <- as.numeric(data$SEX)
data$NATIVE_COUNTRY_n <- as.numeric(data$NATIVE_COUNTRY)
data$SALARY_CLASS_n <- as.numeric(data$SALARY_CLASS)
data_filted <- subset(data,select=c("AGE","WORKCLASS_n","FNLWGT","EDUCATION_n","EDUCATION_NUM","MARITAL_STATUS_n","OCCUPATION_n",
                                    "RELATIONSHIP_n","RACE_n","SEX_n","CAPITAL_GAIN","CAPITAL_LOSS","HOURS_PER_WEEK",# "NATIVE_COUNTRY_n",
                                    "SALARY_CLASS_n"))
data_filted$SALARY_CLASS_n[data_filted$SALARY_CLASS_n == 1] <- "Less50k"
data_filted$SALARY_CLASS_n[data_filted$SALARY_CLASS_n == 2] <- "Larger50k"
train <- sample (1:nrow(data_filted), .8*nrow(data_filted))
TrainData = data_filted[train,]
TestData = data_filted[-train,]

LearnCurve_svm <- learing_curve_dat(dat = TrainData,
                                    outcome = "SALARY_CLASS_n",
                                    test_prop = 1/4, 
                                    ## `train` arguments:
                                    method = "svmLinear2", 
                                    metric = "ROC",
                                    trControl = trainControl(classProbs = TRUE,summaryFunction = twoClassSummary))
ggplot(LearnCurve_svm, aes(x = Training_Size, y = ROC, color = Data)) + ggtitle("Learning Curve for svm") +
  geom_smooth(method = loess, span = .8) + theme_bw()

############ 2) Build Decision Tree Model with cross validation#################
library(tree)
data$SALARY_CLASS_f <- as.factor(data$SALARY_CLASS)
data_filted <- subset(data,select=c("AGE","WORKCLASS","FNLWGT","EDUCATION","EDUCATION_NUM","MARITAL_STATUS","OCCUPATION",
                                    "RELATIONSHIP","RACE","SEX","CAPITAL_GAIN","CAPITAL_LOSS","HOURS_PER_WEEK",# "NATIVE_COUNTRY_n",
                                    "SALARY_CLASS_f"))
train <- sample (1:nrow(data_filted), .8*nrow(data_filted))
TrainData = data_filted[train,]
TestData = data_filted[-train,]

hist(TrainData$AGE,freq=FALSE,col='blue',xlim=c(1,100))
hist(TestData$AGE,freq=FALSE,col='red',xlim=c(1,100))

treeMod <- tree(TrainData$SALARY_CLASS_f~., data=TrainData)

plot(treeMod)
text(treeMod,pretty=0)
out_Train <- predict(treeMod) 
actual_Train <- as.character(TrainData$SALARY_CLASS_f) # actuals
out_Train.response <- colnames(out_Train)[max.col(out_Train, ties.method = c("first"))] # predicted
mean (actual_Train != out_Train.response) # misclassification %
##--------Prune the tree
cvTree <- cv.tree(treeMod, FUN = prune.misclass) # run the cross validation
plot(cvTree)
treePrunedMod <- prune.misclass(treeMod, best = 5) # set size corresponding to lowest value in below plot. try 4 or 16.
plot(treePrunedMod)
text(treePrunedMod, pretty = 0)
#--------Check Pruned results in TrainData
out_Train_Prune <- predict(treePrunedMod) # fit the pruned tree

out_Train.response.prune <- colnames(out_Train_Prune)[max.col(out_Train_Prune, ties.method = c("random"))] # predicted

mean(actual_Train != out_Train.response.prune) # Calculate Mis-classification error.
#--------Compare Pruned and not Pruned results in TestData
out_Test <- predict(treeMod, TestData)  # Predict testData with Pruned tree
out_Test.response <- colnames(out_Test)[max.col(out_Test, ties.method = c("first"))] # predicted
actual_Test <- as.character(TestData$SALARY_CLASS_f) # actuals
mean (actual_Test != out_Test.response) # misclassification %

out_Test <- predict(treePrunedMod, TestData)  # Predict testData with Pruned tree
out_Test.response <- colnames(out_Test)[max.col(out_Test, ties.method = c("first"))] # predicted
actual_Test <- as.character(TestData$SALARY_CLASS_f) # actuals
mean (actual_Test != out_Test.response) # misclassification %

############ 3) Build Neural Network Model###########
data$WORKCLASS_n <- as.numeric(data$WORKCLASS)
data$EDUCATION_n <- as.numeric(data$EDUCATION)
data$MARITAL_STATUS_n <- as.numeric(data$MARITAL_STATUS)
data$OCCUPATION_n <- as.numeric(data$OCCUPATION)
data$RELATIONSHIP_n <- as.numeric(data$RELATIONSHIP)
data$RACE_n <- as.numeric(data$RACE)
data$SEX_n <- as.numeric(data$SEX)
data$NATIVE_COUNTRY_n <- as.numeric(data$NATIVE_COUNTRY)
data$SALARY_CLASS_n <- as.numeric(data$SALARY_CLASS)
data_filted <- subset(data,select=c("AGE","WORKCLASS_n","FNLWGT","EDUCATION_n","EDUCATION_NUM","MARITAL_STATUS_n","OCCUPATION_n",
                                    "RELATIONSHIP_n","RACE_n","SEX_n","CAPITAL_GAIN","CAPITAL_LOSS","HOURS_PER_WEEK","NATIVE_COUNTRY_n",
                                    "SALARY_CLASS_n"))

maxs <- apply(data_filted, 2, max) 
mins <- apply(data_filted, 2, min)
scaled <- as.data.frame(scale(data_filted, center = mins, scale = maxs - mins))


train <- sample (1:nrow(scaled), .8*nrow(scaled))
TrainData = scaled[train,]
TestData = scaled[-train,]

library(neuralnet)
nn <- neuralnet(SALARY_CLASS_n ~ AGE + WORKCLASS_n + FNLWGT + EDUCATION_n + EDUCATION_NUM + MARITAL_STATUS_n + 
                  OCCUPATION_n + RELATIONSHIP_n + RACE_n + SEX_n + 
                  CAPITAL_GAIN + CAPITAL_LOSS + HOURS_PER_WEEK + NATIVE_COUNTRY_n, data=TrainData,  act.fct = "tanh", hidden = 4, linear.output=T) 

plot(nn)
pr_train.nn <- compute(nn,TrainData[,1:14])
pr_train.nn_ <- pr_train.nn$net.result*(max(data_filted$SALARY_CLASS_n)-min(data_filted$SALARY_CLASS_n))+min(data_filted$SALARY_CLASS_n)
train.r <- (TrainData$SALARY_CLASS_n)*(max(data_filted$SALARY_CLASS_n)-min(data_filted$SALARY_CLASS_n))+min(data_filted$SALARY_CLASS_n)
for (i in 1:length(pr_train.nn_)) {if (pr_train.nn_[i] >= 0) {pr_train.nn__[i] = 1} else {pr_train.nn__[i] = -1}}
mean(pr_train.nn__ != train.r) # misclassification % in train dataset


pr.nn <- compute(nn,TestData[,1:14])
pr.nn_ <- pr.nn$net.result*(max(data_filted$SALARY_CLASS_n)-min(data_filted$SALARY_CLASS_n))+min(data_filted$SALARY_CLASS_n)
test.r <- (TestData$SALARY_CLASS_n)*(max(data_filted$SALARY_CLASS_n)-min(data_filted$SALARY_CLASS_n))+min(data_filted$SALARY_CLASS_n)
for (i in 1:length(pr.nn_)) {if (pr.nn_[i] >= 0) {pr.nn__[i] = 1} else {pr.nn__[i] = -1}}
mean(pr.nn__ != test.r) # misclassification % in test dataset

# print(nn)
# gwplot(nn,selected.covariate = "goout")


# ----------------Cross Validation by running 10 times neural networks
cv_train.error = NULL
cv_test.error = NULL
for(ii in 1:10) {
  train <- sample (1:nrow(scaled), .8*nrow(scaled))
  TrainData = scaled[train,]
  TestData = scaled[-train,]
  
  library(neuralnet)
  nn <- neuralnet(SALARY_CLASS_n ~ AGE + WORKCLASS_n + FNLWGT + EDUCATION_n + EDUCATION_NUM + MARITAL_STATUS_n + 
                    OCCUPATION_n + RELATIONSHIP_n + RACE_n + SEX_n + 
                    CAPITAL_GAIN + CAPITAL_LOSS + HOURS_PER_WEEK + NATIVE_COUNTRY_n, data=TrainData,  act.fct = "tanh", hidden = 4, linear.output=T) 
  
  # plot(nn)
  pr_train.nn <- compute(nn,TrainData[,1:14])
  pr_train.nn_ <- pr_train.nn$net.result*(max(data_filted$SALARY_CLASS_n)-min(data_filted$SALARY_CLASS_n))+min(data_filted$SALARY_CLASS_n)
  train.r <- (TrainData$SALARY_CLASS_n)*(max(data_filted$SALARY_CLASS_n)-min(data_filted$SALARY_CLASS_n))+min(data_filted$SALARY_CLASS_n)
  MSE_train.nn <- sum((train.r - pr_train.nn_)^2)/nrow(TrainData)
  MSE_train.nn
  
  for (i in 1:length(pr_train.nn_)) {if (pr_train.nn_[i] >= 0) {pr_train.nn__[i] = 1} else {pr_train.nn__[i] = -1}}
  cv_train.error[ii] = mean(pr_train.nn__ != train.r) # misclassification % in train dataset
  
  
  pr.nn <- compute(nn,TestData[,1:22])
  pr.nn_ <- pr.nn$net.result*(max(data_filted$SALARY_CLASS_n)-min(data_filted$SALARY_CLASS_n))+min(data_filted$SALARY_CLASS_n)
  test.r <- (TestData$SALARY_CLASS_n)*(max(data_filted$SALARY_CLASS_n)-min(data_filted$SALARY_CLASS_n))+min(data_filted$SALARY_CLASS_n)
  MSE.nn <- sum((test.r - pr.nn_)^2)/nrow(TestData)
  MSE.nn
  
  for (i in 1:length(pr.nn_)) {if (pr.nn_[i] >= 0) {pr.nn__[i] = 1} else {pr.nn__[i] = -1}}
  cv_test.error[ii] = mean(pr.nn__ != test.r) # misclassification % in test dataset
  
}

boxplot(cv_train.error,xlab='Misclassification% CV',col='cyan',
        border='blue',names='CV error (Misclassification%)',
        main='CV error (Misclassification%) for NN in TrainData',horizontal=TRUE)

boxplot(cv_test.error,xlab='Misclassification% CV',col='cyan',
        border='blue',names='CV error (Misclassification%)',
        main='CV error (Misclassification%) for NN in TestData',horizontal=TRUE)

############ 4) Build Boosting using on the decision tree -------
data$SALARY_CLASS_f <- as.factor(data$SALARY_CLASS)
data_filted <- subset(data,select=c("AGE","WORKCLASS","FNLWGT","EDUCATION","EDUCATION_NUM","MARITAL_STATUS","OCCUPATION",
                                    "RELATIONSHIP","RACE","SEX","CAPITAL_GAIN","CAPITAL_LOSS","HOURS_PER_WEEK","NATIVE_COUNTRY_n",
                                    "SALARY_CLASS_f"))
train <- sample (1:nrow(data_filted), .8*nrow(data_filted))
TrainData = data_filted[train,]
TestData = data_filted[-train,]

install.packages("C50")
library(C50)
library(gmodels)

# --build first decision tree using C50
treeMod <- C5.0(TrainData[-15], TrainData$SALARY_CLASS_f,control = 
                  C5.0Control(noGlobalPruning = TRUE,earlyStopping = FALSE,CF=0.25,winnow=FALSE))
# treeMod
# plot(treeMod,trial=0, subtree = NULL)
# summary(treeMod)
test_pred <- predict(treeMod, TestData)
CrossTable(TestData$SALARY_CLASS_f, test_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default')) # check the cross validation by running test dataset
mean (TestData$SALARY_CLASS_f != test_pred) # misclassification %

# --Prune the decision tree
treeMod_Pruning <- C5.0(TrainData[-15], TrainData$SALARY_CLASS_f,control = 
                          C5.0Control(noGlobalPruning = FALSE,earlyStopping = FALSE,CF=0.25,winnow=FALSE))
# summary(treeMod_Pruning)
test_pred_Pruning <- predict(treeMod_Pruning, TestData)
CrossTable(TestData$SALARY_CLASS_f, test_pred_Pruning,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default')) # check the cross validation by running test dataset
# plot(treeMod_Pruning,trial=0, subtree = NULL)
mean (TestData$SALARY_CLASS_f != test_pred_Pruning) # misclassification %
# --Boosting
treeMod_boosting10 <- C5.0(TrainData[-15], TrainData$SALARY_CLASS_f,trials = 10,control = 
                             C5.0Control(noGlobalPruning = FALSE,earlyStopping = TRUE,CF=0.25,winnow=FALSE))
# summary(treeMod_boosting10)
test_pred_boosting10 <- predict(treeMod_boosting10, TestData)
CrossTable(TestData$SALARY_CLASS_f, test_pred_boosting10,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))
mean (TestData$SALARY_CLASS_f != test_pred_boosting10) #
# plot(treeMod_boosting10,trial=9, subtree = NULL)
#-----Boosting with more aggressive
treeMod_boosting10_winnow <- C5.0(TrainData[-15], TrainData$SALARY_CLASS_f,trials = 10,control = 
                                    C5.0Control(noGlobalPruning = FALSE,earlyStopping = TRUE,CF=0.25,winnow=TRUE))
# summary(treeMod_boosting10_winnow)
test_pred_boosting10_winnow <- predict(treeMod_boosting10_winnow, TestData)
CrossTable(TestData$SALARY_CLASS_f, test_pred_boosting10_winnow,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))
mean (TestData$SALARY_CLASS_f != test_pred_boosting10_winnow) #
# plot(treeMod_boosting10_winnow,trial=9, subtree = NULL)

############ 5) SVM using ksvm Use this bs with cross validation! ------
data$WORKCLASS_n <- as.numeric(data$WORKCLASS)
data$EDUCATION_n <- as.numeric(data$EDUCATION)
data$MARITAL_STATUS_n <- as.numeric(data$MARITAL_STATUS)
data$OCCUPATION_n <- as.numeric(data$OCCUPATION)
data$RELATIONSHIP_n <- as.numeric(data$RELATIONSHIP)
data$RACE_n <- as.numeric(data$RACE)
data$SEX_n <- as.numeric(data$SEX)
data$NATIVE_COUNTRY_n <- as.numeric(data$NATIVE_COUNTRY)
data$SALARY_CLASS_n <- as.numeric(data$SALARY_CLASS)
data_filted <- subset(data,select=c("AGE","WORKCLASS_n","FNLWGT","EDUCATION_n","EDUCATION_NUM","MARITAL_STATUS_n","OCCUPATION_n",
                                    "RELATIONSHIP_n","RACE_n","SEX_n","CAPITAL_GAIN","CAPITAL_LOSS","HOURS_PER_WEEK","NATIVE_COUNTRY_n",
                                    "SALARY_CLASS_n"))

data_filted$SALARY_CLASS_n[data_filted$SALARY_CLASS_n == 1] <- "Less50k"
data_filted$SALARY_CLASS_n[data_filted$SALARY_CLASS_n == 2] <- "Larger50k"

train <- sample (1:nrow(data_filted), .8*nrow(data_filted))
TrainData = data_filted[train,]
TestData = data_filted[-train,]

# Training SVM Models
library(caret)
library(dplyr)         # Used by caret
library(kernlab)       # support vector machine 
library(pROC)	       # plot the ROC curves


for( C1 in c(0.25,0.5,1,2,4,8,16,32,64)) {
svm_mdl_gaussian <- ksvm(SALARY_CLASS_n~.,data = TrainData,scaled = TRUE,kernel="rbfdot",kpar=list(sigma=0.05),C=C1,cross=3)
#svm_mdl
pred_svm_train <- predict(svm_mdl_gaussian,TrainData[,-15])
mean(pred_svm_train!=TrainData[,15])
## predict mail type on the test set
pred_svm <- predict(svm_mdl_gaussian,TestData[,-15])
## Check results
#table(pred_svm,TestData[,15])
mean(pred_svm!=TestData[,15])
message("When C = ", C1,": TrainDataErr = ",mean(pred_svm_train!=TrainData[,15]),"; TestDataErr = ", mean(pred_svm!=TestData[,15]))
}

for( C1 in c(0.25,0.5,1,2,4,16,32,64)) {
  svm_mdl <- ksvm(SALARY_CLASS_n~.,data = TrainData,scaled = TRUE,kernel="vanilladot",C=C1,cross=3)
  #svm_mdl
  pred_svm_train <- predict(svm_mdl,TrainData[,-15])
  mean(pred_svm_train!=TrainData[,15])
  ## predict mail type on the test set
  pred_svm <- predict(svm_mdl,TestData[,-15])
  ## Check results
  #table(pred_svm,TestData[,15])
  mean(pred_svm!=TestData[,15])
  message("When C = ", C1,": TrainDataErr = ",mean(pred_svm_train!=TrainData[,15]),"; TestDataErr = ", mean(pred_svm!=TestData[,15]))
}

############ 6) KNN for HW ------
library(class)
data$WORKCLASS_n <- as.numeric(data$WORKCLASS)
data$EDUCATION_n <- as.numeric(data$EDUCATION)
data$MARITAL_STATUS_n <- as.numeric(data$MARITAL_STATUS)
data$OCCUPATION_n <- as.numeric(data$OCCUPATION)
data$RELATIONSHIP_n <- as.numeric(data$RELATIONSHIP)
data$RACE_n <- as.numeric(data$RACE)
data$SEX_n <- as.numeric(data$SEX)
data$NATIVE_COUNTRY_n <- as.numeric(data$NATIVE_COUNTRY)
data_filted <- subset(data,select=c("AGE","WORKCLASS_n","FNLWGT","EDUCATION_n","EDUCATION_NUM","MARITAL_STATUS_n","OCCUPATION_n",
                                    "RELATIONSHIP_n","RACE_n","SEX_n","CAPITAL_GAIN","CAPITAL_LOSS","HOURS_PER_WEEK","NATIVE_COUNTRY_n",
                                    "SALARY_CLASS"))
train <- sample (1:nrow(data_filted), .8*nrow(data_filted))
TrainData = data_filted[train,]
TestData = data_filted[-train,]

for( k in c(20,25,30,35,40,45,50)) {
  pred_test_knn = knn(TrainData[,-15], TestData[,-15], cl=TrainData[,15], k, l = 0, prob = FALSE, use.all = TRUE)
  pred_train_knn = knn.cv(TrainData[,-15], cl=TrainData[,15], k, l = 0, prob = FALSE, use.all = TRUE)
  message("When k = ", k,": TrainDataErr = ",mean(TrainData[,15] != pred_train_knn),"; TestDataErr = ", mean(TestData[,15] != pred_test_knn))
  mean(TrainData[,15] != pred_train_knn)
  mean(TestData[,15] != pred_test_knn)
}
