Practical Machine Learning Project
================
Kate Xia
January 23, 2019

Introduction
============

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

In this project, we will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did the exercise.

Data Processing
===============

``` r
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(corrplot)
library(rattle)
```

Download and read the data
--------------------------

``` r
trainUrl <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
training <- read.csv(url(trainUrl))
test <- read.csv(url(testUrl))
dim(training)
```

    ## [1] 19622   160

``` r
dim(test)
```

    ## [1]  20 160

The training data set contains 19622 observations and 160 variables, while the test data set contains 20 observations and 160 variables. The "classe" variable in the training set is the outcome to predict.

Clean the data
--------------

``` r
sum(complete.cases(training))
```

    ## [1] 406

First, we remove columns that contain NA missing values.

``` r
training <- training[, colSums(is.na(training)) == 0]
test <- test[, colSums(is.na(test)) == 0]
```

Next, we get rid of some columns that do not contribute much to the accelerometer measurements.

``` r
classe <- training$classe
trainRemove <- grepl("^X|timestamp|window", names(training))
training <- training[, !trainRemove]
trainCleaned <- training[, sapply(training, is.numeric)]
trainCleaned$classe <- classe
testRemove <- grepl("^X|timestamp|window", names(test))
test <- test[, !testRemove]
testCleaned <- test[, sapply(test, is.numeric)]
dim(trainCleaned)
```

    ## [1] 19622    53

``` r
dim(testCleaned)
```

    ## [1] 20 53

After cleaning, the new training and test data set has only 53 columns.

Slice the data
--------------

Then, we can split the cleaned training set into a pure training data set (75%) and a validation data set (25%). We will use the validation data set to conduct cross validation in future steps.

``` r
set.seed(18974) # For reproducibile purpose
inTrain <- createDataPartition(trainCleaned$classe, p=0.75, list=F)
trainData <- trainCleaned[inTrain, ]
testData <- trainCleaned[-inTrain, ]
```

Data Modeling
=============

In the following sections, we will test 3 different models : classification tree, random forest, and gradient boosting method.

In order to limit the effects of overfitting, and improve the efficicency of the models, we will use the cross-validation technique. We will use 5 folds (usually, 5 or 10 can be used, but 10 folds gives higher run times with no significant increase of the accuracy).

Prediction with classification tree
-----------------------------------

``` r
trControl <- trainControl(method="cv", number=5)
model_CT <- train(classe~., data=trainData, method="rpart", trControl=trControl)
fancyRpartPlot(model_CT$finalModel)
```

![](PML_files/figure-markdown_github/unnamed-chunk-7-1.png)

``` r
trainpred <- predict(model_CT,newdata=testData)
confMatCT <- confusionMatrix(testData$classe,trainpred)
confMatCT$table
```

    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1258   24  108    0    5
    ##          B  416  259  274    0    0
    ##          C  412   12  431    0    0
    ##          D  329  142  333    0    0
    ##          E  136   57  301    0  407

``` r
confMatCT$overall[1]
```

    ##  Accuracy 
    ## 0.4802202

We can notice that the accuracy of this first model is very low (about 48%). This means that the outcome class will not be predicted very well by the other predictors.

Prediction with random forests
------------------------------

``` r
model_RF <- train(classe~., data=trainData, method="rf", trControl=trControl, verbose=FALSE)
print(model_RF)
```

    ## Random Forest 
    ## 
    ## 14718 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 11775, 11775, 11773, 11775, 11774 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    2    0.9908960  0.9884836
    ##   27    0.9906920  0.9882256
    ##   52    0.9851887  0.9812609
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was mtry = 2.

``` r
plot(model_RF,main="Accuracy of Random forest model by number of predictors")
```

![](PML_files/figure-markdown_github/unnamed-chunk-8-1.png)

``` r
trainpred <- predict(model_RF,newdata=testData)
confMatRF <- confusionMatrix(testData$classe,trainpred)
confMatRF$table
```

    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1395    0    0    0    0
    ##          B    6  940    3    0    0
    ##          C    0    6  849    0    0
    ##          D    0    0   11  792    1
    ##          E    0    0    0    2  899

``` r
confMatRF$overall[1]
```

    ##  Accuracy 
    ## 0.9940865

``` r
names(model_RF$finalModel)
```

    ##  [1] "call"            "type"            "predicted"      
    ##  [4] "err.rate"        "confusion"       "votes"          
    ##  [7] "oob.times"       "classes"         "importance"     
    ## [10] "importanceSD"    "localImportance" "proximity"      
    ## [13] "ntree"           "mtry"            "forest"         
    ## [16] "y"               "test"            "inbag"          
    ## [19] "xNames"          "problemType"     "tuneValue"      
    ## [22] "obsLevels"       "param"

``` r
model_RF$finalModel$classes
```

    ## [1] "A" "B" "C" "D" "E"

``` r
plot(model_RF$finalModel,main="Model error of Random forest model by number of trees")
```

![](PML_files/figure-markdown_github/unnamed-chunk-8-2.png)

``` r
MostImpVars <- varImp(model_RF)
MostImpVars
```

    ## rf variable importance
    ## 
    ##   only 20 most important variables shown (out of 52)
    ## 
    ##                      Overall
    ## roll_belt             100.00
    ## yaw_belt               74.56
    ## magnet_dumbbell_z      67.36
    ## magnet_dumbbell_y      62.76
    ## pitch_forearm          58.53
    ## pitch_belt             57.82
    ## roll_forearm           48.40
    ## magnet_dumbbell_x      48.35
    ## magnet_belt_z          42.66
    ## accel_dumbbell_y       42.61
    ## roll_dumbbell          42.48
    ## accel_belt_z           41.74
    ## magnet_belt_y          40.27
    ## roll_arm               35.62
    ## accel_dumbbell_z       34.56
    ## accel_forearm_x        32.09
    ## accel_dumbbell_x       29.35
    ## yaw_dumbbell           29.24
    ## gyros_belt_z           28.55
    ## total_accel_dumbbell   27.71

With random forest, we reach an accuracy of 99.4% using cross-validation with 5 steps.

Prediction with gradient boosting method
----------------------------------------

``` r
model_GBM <- train(classe~., data=trainData, method="gbm", trControl=trControl, verbose=FALSE)
print(model_GBM)
```

    ## Stochastic Gradient Boosting 
    ## 
    ## 14718 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 11775, 11774, 11773, 11774, 11776 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  Accuracy   Kappa    
    ##   1                   50      0.7549268  0.6894056
    ##   1                  100      0.8221231  0.7748389
    ##   1                  150      0.8558917  0.8176259
    ##   2                   50      0.8538532  0.8147655
    ##   2                  100      0.9068509  0.8820984
    ##   2                  150      0.9304271  0.9119521
    ##   3                   50      0.8957075  0.8679283
    ##   3                  100      0.9396663  0.9236505
    ##   3                  150      0.9590979  0.9482448
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## 
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final values used for the model were n.trees = 150,
    ##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.

``` r
plot(model_GBM)
```

![](PML_files/figure-markdown_github/unnamed-chunk-9-1.png)

``` r
trainpred <- predict(model_GBM,newdata=testData)
confMatGBM <- confusionMatrix(testData$classe,trainpred)
confMatGBM$table
```

    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1380    3    8    4    0
    ##          B   25  898   21    1    4
    ##          C    0   22  818   13    2
    ##          D    1    2   24  769    8
    ##          E    0    6    7   16  872

``` r
confMatGBM$overall[1]
```

    ##  Accuracy 
    ## 0.9659462

Precision with 5 folds is 96.6%.

Conclusion
==========

This shows that the random forest model is the best one. We will then use it to predict the values of classe for the test data set.

``` r
FinalTestPred <- predict(model_RF,newdata=testCleaned)
FinalTestPred
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
