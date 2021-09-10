---
title: "Activity Recognition Using Predictive Analysis"
output:
  html_document:
    keep_md: yes
  pdf_document: default
author: "Created By : Rishab Raina"
---
## Introduction

Using devices such as Jawbone Up, Nike Fuel Band, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerators on the belt, forearm, arm, and dumbbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: [http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har] (see the section on the Weight Lifting Exercise Data set).

For the purpose of this project, the following steps would be followed:

1. Data Loading
2. Data Cleaning & Preprocessing
2. Exploratory Data Analysis
3. Prediction Model Selection
4. Predicting Test Set Output



## Loading the Dataset & Libraries

The Data set has been downloaded from the internet and has been loaded into two seperate dataframes, __“training and “testing”__. The __“training__ data set has 19622 number of records and the __“testing__ data set has 20 records. The number of variables is 160.


```r
# Package names
packages <- c("caret","corrplot","rpart","rpart.plot","RColorBrewer","RGtk2","rattle","randomForest","gbm","lubridate","dplyr")

# Install packages not yet installed
installed_packages <- packages %in% rownames(installed.packages())
if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages])
}

# Packages loading
invisible(lapply(packages, library, character.only = TRUE))

# Loading Dataset using URL
train_url <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test_url  <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"


init_training_data <- read.csv(url(train_url))
init_testing_data <- read.csv(url(test_url))
```
## Data Cleaning

First, we load the training and testing set from the online sources and then split the training set further into training and test sets. 

*Removing Variables which have nearly zero variance*

```r
non_zero_var <- nearZeroVar(init_training_data)

training_data <- init_training_data[,-non_zero_var]
testing_data <- init_testing_data[,-non_zero_var]
```
*Removing Variables which are having NA values. Our threshold is 95%.*

```r
na_val_col <- sapply(training_data, function(x) mean(is.na(x))) > 0.95

training_data <- training_data[,na_val_col == FALSE]
testing_data <- testing_data[,na_val_col == FALSE]
```
*Removing variables which will not to used for model building*

```r
training_data <- training_data[,-(1:5)]
testing_data <- testing_data[,-(1:5)]
```
Data Cleaning Complete. As a result of data processing and cleaning, we are able to reduce the variables from 160 to 54

## EDA
Now that we have filtered out variables we are going to use in the model, we shall look at the dependence of these variables on each other through a correlation plot.

```r
library(corrplot)
corrMat <- cor(training_data[,-54])
corrplot(corrMat, method = "color", type = "lower", tl.cex = 0.8, tl.col = rgb(0,0,0))
```

![](RMarkdown-testing_files/figure-html/CorrelationPlot-1.png)<!-- -->

### Data Partitioning
Using the training set, we will be further splitting it into 2 sets - one for training and the other test set for validation to choose the best model. That model would be chosen to run on the actual testing dataset which we left out

```r
inTrain <- createDataPartition(training_data$classe, p=0.6, list=FALSE)
training <- training_data[inTrain,]
testing <- training_data[-inTrain,]
```
## Prediction Model Selection

*We will use 3 methods to model the training set and thereby choose the one having the best accuracy to predict the outcome variable in the testing set.* 
*The methods are Decision Tree, Random Forest and Generalized Boosted Model.*

Note : A confusion matrix plotted at the end of each model will help visualize the analysis better.

### Decision Tree

```r
set.seed(7867)
modelDT <- rpart(classe ~ ., data = training, method = "class")
fancyRpartPlot(modelDT)
```

![](RMarkdown-testing_files/figure-html/DecisionTree-1.png)<!-- -->

```r
predictDT <- predict(modelDT, testing, type = "class")
confMatDT <- confusionMatrix(predictDT, as.factor(testing$classe))
confMatDT
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1979  287   57  105   72
##          B   71  873   43   38   32
##          C   30  116 1163  185  102
##          D  131  178   68  825  166
##          E   21   64   37  133 1070
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7533          
##                  95% CI : (0.7436, 0.7628)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6869          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8866   0.5751   0.8501   0.6415   0.7420
## Specificity            0.9072   0.9709   0.9332   0.9172   0.9602
## Pos Pred Value         0.7916   0.8259   0.7287   0.6031   0.8075
## Neg Pred Value         0.9527   0.9050   0.9672   0.9288   0.9430
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2522   0.1113   0.1482   0.1051   0.1364
## Detection Prevalence   0.3186   0.1347   0.2034   0.1744   0.1689
## Balanced Accuracy      0.8969   0.7730   0.8917   0.7794   0.8511
```


### Random Forest


```r
library(caret)
set.seed(13908)
control <- trainControl(method = "cv", number = 3, verboseIter=FALSE)
modelRF <- train(classe ~ ., data = training, method = "rf", trControl = control, ntree =100)
modelRF$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, ntree = 100, mtry = min(param$mtry,      ncol(x))) 
##                Type of random forest: classification
##                      Number of trees: 100
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.37%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3347    0    0    0    1 0.0002986858
## B    6 2265    6    2    0 0.0061430452
## C    0    8 2044    2    0 0.0048685492
## D    0    0    8 1921    1 0.0046632124
## E    0    2    0    8 2155 0.0046189376
```

```r
predictRF <- predict(modelRF, testing)
confMatRF <- confusionMatrix(predictRF, as.factor(testing$classe))
confMatRF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    5    0    0    0
##          B    0 1511    3    0    0
##          C    0    2 1365    8    0
##          D    0    0    0 1278    2
##          E    0    0    0    0 1440
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9975          
##                  95% CI : (0.9961, 0.9984)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9968          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9954   0.9978   0.9938   0.9986
## Specificity            0.9991   0.9995   0.9985   0.9997   1.0000
## Pos Pred Value         0.9978   0.9980   0.9927   0.9984   1.0000
## Neg Pred Value         1.0000   0.9989   0.9995   0.9988   0.9997
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1926   0.1740   0.1629   0.1835
## Detection Prevalence   0.2851   0.1930   0.1752   0.1631   0.1835
## Balanced Accuracy      0.9996   0.9975   0.9981   0.9967   0.9993
```

```r
plot(confMatRF$table, col = confMatRF$byClass, 
     main = paste("Random Forest - Accuracy Level =",
                  round(confMatRF$overall['Accuracy'], 4)))
```

![](RMarkdown-testing_files/figure-html/RandomForest-1.png)<!-- -->
From the Confusion Matrix, we can clearly see that the prediction accuracy of Random Forest model is 99% which is satisfactory.

### Generalized Boosted Model


```r
library(caret)
set.seed(13908)
control <- trainControl(method = "repeatedcv", number = 5, repeats = 1, verboseIter = FALSE)
modelGBM <- train(classe ~ ., data = training, trControl = control, method = "gbm", verbose = FALSE)
modelGBM$finalModel
```

```
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 53 predictors of which 52 had non-zero influence.
```

```r
predictGBM <- predict(modelGBM, testing)
confMatGBM <- confusionMatrix(predictGBM, as.factor(testing$classe))
confMatGBM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2223    8    0    1    0
##          B    8 1491   17    4    4
##          C    0   19 1350   14    5
##          D    1    0    0 1267   14
##          E    0    0    1    0 1419
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9878          
##                  95% CI : (0.9851, 0.9901)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9845          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9960   0.9822   0.9868   0.9852   0.9840
## Specificity            0.9984   0.9948   0.9941   0.9977   0.9998
## Pos Pred Value         0.9960   0.9783   0.9726   0.9883   0.9993
## Neg Pred Value         0.9984   0.9957   0.9972   0.9971   0.9964
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2833   0.1900   0.1721   0.1615   0.1809
## Detection Prevalence   0.2845   0.1942   0.1769   0.1634   0.1810
## Balanced Accuracy      0.9972   0.9885   0.9905   0.9915   0.9919
```

```r
plot(confMatGBM$table, col = confMatGBM$byClass, 
     main = paste("Gradient Boosting - Accuracy Level =",
                  round(confMatGBM$overall['Accuracy'], 4)))
```

![](RMarkdown-testing_files/figure-html/GBM-1.png)<!-- -->
From Gradient Boost Model, the prediction accuracy is 98.7% which is satisfactory.

**Now we need to see how each model has predicted the validation dataset across the classifications.** We are not considering Decision Tree model as it didn’t reach the satisfactory prediction accuracy level. So only Random Forest and Gradient Boosting methods are being compared.
As Random Forest offers the maximum accuracy of 99.6%, we will go with Random Forest Model to predict our test data class variable.

## Predicting Test Set Output


```r
predictRF <- predict(modelRF, testing_data)
```
### The final prediction are as follows:

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

