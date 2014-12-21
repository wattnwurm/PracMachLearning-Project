## Practical-Machine-Learning-Project

**Author: Nea, Date: 21. Dezember 2014**


###Synopsis
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity. In most kind of using this devices they are interested of the quantified self movement. The goal of this course project is to predict the quality of executing an activity.

In this project we will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. "Participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in ﬁve diﬀerent fashions: exactly according to the speciﬁcation (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E). Class A corresponds to the speciﬁed execution of the exercise, while the other 4 classes correspond to common mistakes."

More information is available from the website here: http://groupware.les.inf.puc-rio.br/har
(http://groupware.les.inf.puc-rio.br/har) (see the section 5: Detection of mistakes). 

###Preliminary and Libraries
These libraries will be needed for further use.

```r
# rm(list=ls()) # clear environment, delete all
setwd("C:/Users/Alfred/Documents/GitHub/PracMachLearning-Project")
suppressMessages(library(dplyr))
suppressMessages(library(caret))
```

###Data Processing
###1. Loading and preprocessing the data
The input data contains two csv-files, downloaded from the Coursera website. The data are already divided in the training and the testing file. 

```r
# load the data
data.train <- read.csv("pml-training.csv", stringsAsFactors=FALSE)
data.test <- read.csv("pml-testing.csv", stringsAsFactors=FALSE)

# convert to a local data frame for easier use
d.train <- tbl_df(data.train)
d.test <- tbl_df(data.test)
```



###2. Cleaning the data
The first look shows, that there are many data records, which are not complete. There are a lot of NA-Values. 



```r
## Preprocessing the data - original
dim(d.train)
```

```
## [1] 19622   160
```

```r
names(d.train)
```

```
##   [1] "X"                        "user_name"               
##   [3] "raw_timestamp_part_1"     "raw_timestamp_part_2"    
##   [5] "cvtd_timestamp"           "new_window"              
##   [7] "num_window"               "roll_belt"               
##   [9] "pitch_belt"               "yaw_belt"                
##  [11] "total_accel_belt"         "kurtosis_roll_belt"      
##  [13] "kurtosis_picth_belt"      "kurtosis_yaw_belt"       
##  [15] "skewness_roll_belt"       "skewness_roll_belt.1"    
##  [17] "skewness_yaw_belt"        "max_roll_belt"           
##  [19] "max_picth_belt"           "max_yaw_belt"            
##  [21] "min_roll_belt"            "min_pitch_belt"          
##  [23] "min_yaw_belt"             "amplitude_roll_belt"     
##  [25] "amplitude_pitch_belt"     "amplitude_yaw_belt"      
##  [27] "var_total_accel_belt"     "avg_roll_belt"           
##  [29] "stddev_roll_belt"         "var_roll_belt"           
##  [31] "avg_pitch_belt"           "stddev_pitch_belt"       
##  [33] "var_pitch_belt"           "avg_yaw_belt"            
##  [35] "stddev_yaw_belt"          "var_yaw_belt"            
##  [37] "gyros_belt_x"             "gyros_belt_y"            
##  [39] "gyros_belt_z"             "accel_belt_x"            
##  [41] "accel_belt_y"             "accel_belt_z"            
##  [43] "magnet_belt_x"            "magnet_belt_y"           
##  [45] "magnet_belt_z"            "roll_arm"                
##  [47] "pitch_arm"                "yaw_arm"                 
##  [49] "total_accel_arm"          "var_accel_arm"           
##  [51] "avg_roll_arm"             "stddev_roll_arm"         
##  [53] "var_roll_arm"             "avg_pitch_arm"           
##  [55] "stddev_pitch_arm"         "var_pitch_arm"           
##  [57] "avg_yaw_arm"              "stddev_yaw_arm"          
##  [59] "var_yaw_arm"              "gyros_arm_x"             
##  [61] "gyros_arm_y"              "gyros_arm_z"             
##  [63] "accel_arm_x"              "accel_arm_y"             
##  [65] "accel_arm_z"              "magnet_arm_x"            
##  [67] "magnet_arm_y"             "magnet_arm_z"            
##  [69] "kurtosis_roll_arm"        "kurtosis_picth_arm"      
##  [71] "kurtosis_yaw_arm"         "skewness_roll_arm"       
##  [73] "skewness_pitch_arm"       "skewness_yaw_arm"        
##  [75] "max_roll_arm"             "max_picth_arm"           
##  [77] "max_yaw_arm"              "min_roll_arm"            
##  [79] "min_pitch_arm"            "min_yaw_arm"             
##  [81] "amplitude_roll_arm"       "amplitude_pitch_arm"     
##  [83] "amplitude_yaw_arm"        "roll_dumbbell"           
##  [85] "pitch_dumbbell"           "yaw_dumbbell"            
##  [87] "kurtosis_roll_dumbbell"   "kurtosis_picth_dumbbell" 
##  [89] "kurtosis_yaw_dumbbell"    "skewness_roll_dumbbell"  
##  [91] "skewness_pitch_dumbbell"  "skewness_yaw_dumbbell"   
##  [93] "max_roll_dumbbell"        "max_picth_dumbbell"      
##  [95] "max_yaw_dumbbell"         "min_roll_dumbbell"       
##  [97] "min_pitch_dumbbell"       "min_yaw_dumbbell"        
##  [99] "amplitude_roll_dumbbell"  "amplitude_pitch_dumbbell"
## [101] "amplitude_yaw_dumbbell"   "total_accel_dumbbell"    
## [103] "var_accel_dumbbell"       "avg_roll_dumbbell"       
## [105] "stddev_roll_dumbbell"     "var_roll_dumbbell"       
## [107] "avg_pitch_dumbbell"       "stddev_pitch_dumbbell"   
## [109] "var_pitch_dumbbell"       "avg_yaw_dumbbell"        
## [111] "stddev_yaw_dumbbell"      "var_yaw_dumbbell"        
## [113] "gyros_dumbbell_x"         "gyros_dumbbell_y"        
## [115] "gyros_dumbbell_z"         "accel_dumbbell_x"        
## [117] "accel_dumbbell_y"         "accel_dumbbell_z"        
## [119] "magnet_dumbbell_x"        "magnet_dumbbell_y"       
## [121] "magnet_dumbbell_z"        "roll_forearm"            
## [123] "pitch_forearm"            "yaw_forearm"             
## [125] "kurtosis_roll_forearm"    "kurtosis_picth_forearm"  
## [127] "kurtosis_yaw_forearm"     "skewness_roll_forearm"   
## [129] "skewness_pitch_forearm"   "skewness_yaw_forearm"    
## [131] "max_roll_forearm"         "max_picth_forearm"       
## [133] "max_yaw_forearm"          "min_roll_forearm"        
## [135] "min_pitch_forearm"        "min_yaw_forearm"         
## [137] "amplitude_roll_forearm"   "amplitude_pitch_forearm" 
## [139] "amplitude_yaw_forearm"    "total_accel_forearm"     
## [141] "var_accel_forearm"        "avg_roll_forearm"        
## [143] "stddev_roll_forearm"      "var_roll_forearm"        
## [145] "avg_pitch_forearm"        "stddev_pitch_forearm"    
## [147] "var_pitch_forearm"        "avg_yaw_forearm"         
## [149] "stddev_yaw_forearm"       "var_yaw_forearm"         
## [151] "gyros_forearm_x"          "gyros_forearm_y"         
## [153] "gyros_forearm_z"          "accel_forearm_x"         
## [155] "accel_forearm_y"          "accel_forearm_z"         
## [157] "magnet_forearm_x"         "magnet_forearm_y"        
## [159] "magnet_forearm_z"         "classe"
```

```r
head(d.train)
```

```
## Source: local data frame [6 x 160]
## 
##   X user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp
## 1 1  carlitos           1323084231               788290 05/12/2011 11:23
## 2 2  carlitos           1323084231               808298 05/12/2011 11:23
## 3 3  carlitos           1323084231               820366 05/12/2011 11:23
## 4 4  carlitos           1323084232               120339 05/12/2011 11:23
## 5 5  carlitos           1323084232               196328 05/12/2011 11:23
## 6 6  carlitos           1323084232               304277 05/12/2011 11:23
## Variables not shown: new_window (chr), num_window (int), roll_belt (dbl),
##   pitch_belt (dbl), yaw_belt (dbl), total_accel_belt (int),
##   kurtosis_roll_belt (chr), kurtosis_picth_belt (chr), kurtosis_yaw_belt
##   (chr), skewness_roll_belt (chr), skewness_roll_belt.1 (chr),
##   skewness_yaw_belt (chr), max_roll_belt (dbl), max_picth_belt (int),
##   max_yaw_belt (chr), min_roll_belt (dbl), min_pitch_belt (int),
##   min_yaw_belt (chr), amplitude_roll_belt (dbl), amplitude_pitch_belt
##   (int), amplitude_yaw_belt (chr), var_total_accel_belt (dbl),
##   avg_roll_belt (dbl), stddev_roll_belt (dbl), var_roll_belt (dbl),
##   avg_pitch_belt (dbl), stddev_pitch_belt (dbl), var_pitch_belt (dbl),
##   avg_yaw_belt (dbl), stddev_yaw_belt (dbl), var_yaw_belt (dbl),
##   gyros_belt_x (dbl), gyros_belt_y (dbl), gyros_belt_z (dbl), accel_belt_x
##   (int), accel_belt_y (int), accel_belt_z (int), magnet_belt_x (int),
##   magnet_belt_y (int), magnet_belt_z (int), roll_arm (dbl), pitch_arm
##   (dbl), yaw_arm (dbl), total_accel_arm (int), var_accel_arm (dbl),
##   avg_roll_arm (dbl), stddev_roll_arm (dbl), var_roll_arm (dbl),
##   avg_pitch_arm (dbl), stddev_pitch_arm (dbl), var_pitch_arm (dbl),
##   avg_yaw_arm (dbl), stddev_yaw_arm (dbl), var_yaw_arm (dbl), gyros_arm_x
##   (dbl), gyros_arm_y (dbl), gyros_arm_z (dbl), accel_arm_x (int),
##   accel_arm_y (int), accel_arm_z (int), magnet_arm_x (int), magnet_arm_y
##   (int), magnet_arm_z (int), kurtosis_roll_arm (chr), kurtosis_picth_arm
##   (chr), kurtosis_yaw_arm (chr), skewness_roll_arm (chr),
##   skewness_pitch_arm (chr), skewness_yaw_arm (chr), max_roll_arm (dbl),
##   max_picth_arm (dbl), max_yaw_arm (int), min_roll_arm (dbl),
##   min_pitch_arm (dbl), min_yaw_arm (int), amplitude_roll_arm (dbl),
##   amplitude_pitch_arm (dbl), amplitude_yaw_arm (int), roll_dumbbell (dbl),
##   pitch_dumbbell (dbl), yaw_dumbbell (dbl), kurtosis_roll_dumbbell (chr),
##   kurtosis_picth_dumbbell (chr), kurtosis_yaw_dumbbell (chr),
##   skewness_roll_dumbbell (chr), skewness_pitch_dumbbell (chr),
##   skewness_yaw_dumbbell (chr), max_roll_dumbbell (dbl), max_picth_dumbbell
##   (dbl), max_yaw_dumbbell (chr), min_roll_dumbbell (dbl),
##   min_pitch_dumbbell (dbl), min_yaw_dumbbell (chr),
##   amplitude_roll_dumbbell (dbl), amplitude_pitch_dumbbell (dbl),
##   amplitude_yaw_dumbbell (chr), total_accel_dumbbell (int),
##   var_accel_dumbbell (dbl), avg_roll_dumbbell (dbl), stddev_roll_dumbbell
##   (dbl), var_roll_dumbbell (dbl), avg_pitch_dumbbell (dbl),
##   stddev_pitch_dumbbell (dbl), var_pitch_dumbbell (dbl), avg_yaw_dumbbell
##   (dbl), stddev_yaw_dumbbell (dbl), var_yaw_dumbbell (dbl),
##   gyros_dumbbell_x (dbl), gyros_dumbbell_y (dbl), gyros_dumbbell_z (dbl),
##   accel_dumbbell_x (int), accel_dumbbell_y (int), accel_dumbbell_z (int),
##   magnet_dumbbell_x (int), magnet_dumbbell_y (int), magnet_dumbbell_z
##   (dbl), roll_forearm (dbl), pitch_forearm (dbl), yaw_forearm (dbl),
##   kurtosis_roll_forearm (chr), kurtosis_picth_forearm (chr),
##   kurtosis_yaw_forearm (chr), skewness_roll_forearm (chr),
##   skewness_pitch_forearm (chr), skewness_yaw_forearm (chr),
##   max_roll_forearm (dbl), max_picth_forearm (dbl), max_yaw_forearm (chr),
##   min_roll_forearm (dbl), min_pitch_forearm (dbl), min_yaw_forearm (chr),
##   amplitude_roll_forearm (dbl), amplitude_pitch_forearm (dbl),
##   amplitude_yaw_forearm (chr), total_accel_forearm (int),
##   var_accel_forearm (dbl), avg_roll_forearm (dbl), stddev_roll_forearm
##   (dbl), var_roll_forearm (dbl), avg_pitch_forearm (dbl),
##   stddev_pitch_forearm (dbl), var_pitch_forearm (dbl), avg_yaw_forearm
##   (dbl), stddev_yaw_forearm (dbl), var_yaw_forearm (dbl), gyros_forearm_x
##   (dbl), gyros_forearm_y (dbl), gyros_forearm_z (dbl), accel_forearm_x
##   (int), accel_forearm_y (int), accel_forearm_z (int), magnet_forearm_x
##   (int), magnet_forearm_y (dbl), magnet_forearm_z (dbl), classe (chr)
```

The first eight colums are used as identifier for the implementation and realization of the study. They would be removed. 
The filtering process must be done for both - the training and the test set. This will be done in a function.


```r
#--------------------------------------------
# function for cleaning of train and test data in identical manner
clearData <- function(df){
      # delete unnecessary columns
      df <- select(df, -(X : num_window)) 
      # delete colums with NA      
      no.na <- !sapply(df, function(x) any(is.na(x)))
      df <- df[, no.na]
      # delete columns with ""-value
      no.zero <- !sapply(df, function(x) any(x==""))
      df <- df[, no.zero]
}
#-------------------------------------------

# tidy up data - containing NA, 0 and Whitespace
d.train <- clearData(d.train)
head(d.train)
```

```
## Source: local data frame [6 x 53]
## 
##   roll_belt pitch_belt yaw_belt total_accel_belt gyros_belt_x gyros_belt_y
## 1      1.41       8.07    -94.4                3         0.00         0.00
## 2      1.41       8.07    -94.4                3         0.02         0.00
## 3      1.42       8.07    -94.4                3         0.00         0.00
## 4      1.48       8.05    -94.4                3         0.02         0.00
## 5      1.48       8.07    -94.4                3         0.02         0.02
## 6      1.45       8.06    -94.4                3         0.02         0.00
## Variables not shown: gyros_belt_z (dbl), accel_belt_x (int), accel_belt_y
##   (int), accel_belt_z (int), magnet_belt_x (int), magnet_belt_y (int),
##   magnet_belt_z (int), roll_arm (dbl), pitch_arm (dbl), yaw_arm (dbl),
##   total_accel_arm (int), gyros_arm_x (dbl), gyros_arm_y (dbl), gyros_arm_z
##   (dbl), accel_arm_x (int), accel_arm_y (int), accel_arm_z (int),
##   magnet_arm_x (int), magnet_arm_y (int), magnet_arm_z (int),
##   roll_dumbbell (dbl), pitch_dumbbell (dbl), yaw_dumbbell (dbl),
##   total_accel_dumbbell (int), gyros_dumbbell_x (dbl), gyros_dumbbell_y
##   (dbl), gyros_dumbbell_z (dbl), accel_dumbbell_x (int), accel_dumbbell_y
##   (int), accel_dumbbell_z (int), magnet_dumbbell_x (int),
##   magnet_dumbbell_y (int), magnet_dumbbell_z (dbl), roll_forearm (dbl),
##   pitch_forearm (dbl), yaw_forearm (dbl), total_accel_forearm (int),
##   gyros_forearm_x (dbl), gyros_forearm_y (dbl), gyros_forearm_z (dbl),
##   accel_forearm_x (int), accel_forearm_y (int), accel_forearm_z (int),
##   magnet_forearm_x (int), magnet_forearm_y (dbl), magnet_forearm_z (dbl),
##   classe (chr)
```

```r
dim(d.train)
```

```
## [1] 19622    53
```

```r
# the same for the test data
d.test <- clearData(d.test)
head(d.test)
```

```
## Source: local data frame [6 x 53]
## 
##   roll_belt pitch_belt yaw_belt total_accel_belt gyros_belt_x gyros_belt_y
## 1    123.00      27.00    -4.75               20        -0.50        -0.02
## 2      1.02       4.87   -88.90                4        -0.06        -0.02
## 3      0.87       1.82   -88.50                5         0.05         0.02
## 4    125.00     -41.60   162.00               17         0.11         0.11
## 5      1.35       3.33   -88.60                3         0.03         0.02
## 6     -5.92       1.59   -87.70                4         0.10         0.05
## Variables not shown: gyros_belt_z (dbl), accel_belt_x (int), accel_belt_y
##   (int), accel_belt_z (int), magnet_belt_x (int), magnet_belt_y (int),
##   magnet_belt_z (int), roll_arm (dbl), pitch_arm (dbl), yaw_arm (dbl),
##   total_accel_arm (int), gyros_arm_x (dbl), gyros_arm_y (dbl), gyros_arm_z
##   (dbl), accel_arm_x (int), accel_arm_y (int), accel_arm_z (int),
##   magnet_arm_x (int), magnet_arm_y (int), magnet_arm_z (int),
##   roll_dumbbell (dbl), pitch_dumbbell (dbl), yaw_dumbbell (dbl),
##   total_accel_dumbbell (int), gyros_dumbbell_x (dbl), gyros_dumbbell_y
##   (dbl), gyros_dumbbell_z (dbl), accel_dumbbell_x (int), accel_dumbbell_y
##   (int), accel_dumbbell_z (int), magnet_dumbbell_x (int),
##   magnet_dumbbell_y (int), magnet_dumbbell_z (int), roll_forearm (dbl),
##   pitch_forearm (dbl), yaw_forearm (dbl), total_accel_forearm (int),
##   gyros_forearm_x (dbl), gyros_forearm_y (dbl), gyros_forearm_z (dbl),
##   accel_forearm_x (int), accel_forearm_y (int), accel_forearm_z (int),
##   magnet_forearm_x (int), magnet_forearm_y (int), magnet_forearm_z (int),
##   problem_id (int)
```

```r
dim(d.test)
```

```
## [1] 20 53
```

All of the data records have complete data and are prepared for use.

###3. Modelling

The next step is to consider a predicting model. 


```r
# looking for zero covariates, only numeric variables can be evaluated in this way.
nsv <- nearZeroVar(d.train, saveMetrics=TRUE)
head(nsv)
```

```
##                  freqRatio percentUnique zeroVar   nzv
## roll_belt         1.101904     6.7781062   FALSE FALSE
## pitch_belt        1.036082     9.3772296   FALSE FALSE
## yaw_belt          1.058480     9.9734991   FALSE FALSE
## total_accel_belt  1.063160     0.1477933   FALSE FALSE
## gyros_belt_x      1.058651     0.7134849   FALSE FALSE
## gyros_belt_y      1.144000     0.3516461   FALSE FALSE
```

```r
table(d.train$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

All variables would be used for modeling. Two models will be sampled -  KNN and Random Forest. The calculation of RF takes a lot time. Maybe the use of the doMC-library for multiple cores will speed-up the calculations. The models are fitted with the training set.


```r
# making outcome as factor
d.train$classe <- factor(d.train$classe)
# from: user_caret_2up.pdf, page 34
cvCtrl <- trainControl(method = "cv", repeats = 3)
##--------KNN-----------
mod.KNN  <- train(classe ~ ., data = d.train, method="knn", trControl = cvCtrl)
#------Random Forests ------------------
mod.RF  <- train(classe ~ ., data = d.train, method="rf", trControl = cvCtrl)
# how are the results
mod.KNN
```

```
## k-Nearest Neighbors 
## 
## 19622 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 17659, 17658, 17660, 17659, 17660, 17661, ... 
## 
## Resampling results across tuning parameters:
## 
##   k  Accuracy   Kappa      Accuracy SD  Kappa SD   
##   5  0.9310983  0.9128285  0.006092019  0.007714683
##   7  0.9115806  0.8881057  0.009391071  0.011889320
##   9  0.8960873  0.8684892  0.009137506  0.011600449
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was k = 5.
```

```r
mod.RF
```

```
## Random Forest 
## 
## 19622 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 17661, 17661, 17661, 17661, 17659, 17660, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9956175  0.9944564  0.001204355  0.001523429
##   27    0.9950567  0.9937469  0.001226224  0.001551188
##   52    0.9905714  0.9880726  0.002154418  0.002726050
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

```r
# how accuracy is the model
max(head(mod.KNN$results)$Accuracy)
```

```
## [1] 0.9310983
```

```r
max(head(mod.RF$results)$Accuracy)
```

```
## [1] 0.9956175
```

The result of the RF-model appears to have the highest value of accuracy. This model will be used for prediction.

###4. Predictions

The cleaned test dataset in the separated loaded file will be used for prediction. 

```r
## prediction -------------------------
pred.RF  <- predict(mod.RF, newdata = d.test)
pred.RF
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

The classifications of the prediction for the 20 test cases should now be evaluated on thew Coursera website. For each test case we submit a text file with a single capital letter corresponding to the prediction for the corresponding problem in the test data set. 


```r
##------------------------------
answers <- pred.RF
# from submission website
pml_write_files = function(x){
      n = length(x)
      for(i in 1:n){
            filename = paste0("problem_id_",i,".txt")
            write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
      }
}

pml_write_files(answers)
##-----------------------------
```

All of the submitted predictions were found to be correct.

###5. Executive summary



