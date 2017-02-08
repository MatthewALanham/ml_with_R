################################################################################
# Machine Learning Prototyping with R
# Copyright (c)
# Authors: Matthew A. Lanham (lanhamm@purdue.edu)
# Key Takeaways: (1) Ease of modeling building using caret package
#                (2) Evaluation of binary class models using prob cal plots
###############################################################################
getMKLthreads()
#setMKLthreads(6)

################################################################################
# read data file from the web
myUrl <- "https://raw.githubusercontent.com/MatthewALanham/ml_with_R/master/data.csv"
records <- read.table(file=myUrl, header=T, sep=",", quote="")
rm(myUrl)

# review fields
str(records)

# make target variable a factor
records$Y <- as.factor(records$Y)
str(records)

###############################################################################
# Define what we are going to study
# methods to try
myMethods = c("rpart","lda","C5.0")
# resampling to try
balancingType = c("raw","up","down")

###############################################################################
# Create a dataframe to store performance measures for each set
n <- 1                     #number of datasets
m <- length(myMethods)     #number of methods
b <- length(balancingType) #number of ways to balance the training data
N <- n*m*b                 #number of iterations

perfMeasures <- data.frame(matrix(nrow = N, ncol=17))
names(perfMeasures)=c('Method','balancingType', 'Runtime','trCohensKappa',
                      'trMCC','trF1','trSe','trSp','trOA','trAUC',
                      'teCohensKappa','teMCC','teF1',
                      'teSe','teSp','teOA','teAUC')
head(perfMeasures)
###############################################################################
# Fill in the various parameter combinations in the perfMeasures table
# add data set information
names(perfMeasures)

# add methods
seq(from=1, to=N, by=N/m)
perfMeasures[1:3,"Method"] <- rep(myMethods[1],N/m)
perfMeasures[4:6,"Method"] <- rep(myMethods[2],N/m)
perfMeasures[7:9,"Method"] <- rep(myMethods[3],N/m)
# add balancing types
perfMeasures[,"balancingType"] <- rep(c(rep(balancingType[1],n)
                                        , rep(balancingType[2],n)
                                        , rep(balancingType[3],n)),b)

###############################################################################
# are there any missing values? if so, go ahead and remove them
sum(complete.cases(records))/nrow(records)
# [1] 0.99968
records <- na.omit(records)
###############################################################################
# clean up environoment
rm(b,m,n,N,balancingType,myMethods)
###############################################################################
# load libraries
library(ISLR)    #containts various data sets
library(lattice) #used for plotting
library(ggplot2) #used for plotting
library(caret)   #used for caret package functions
library(MASS)    #use for LDA/QDA
library(pROC)    #to generate ROC curves and capture AUC

###############################################################################
# Loop through each data set and build a model based of the type of balancing
setwd("C:\\Users\\Matthew A. Lanham\\Dropbox\\_Conferences\\_Talks this year\\2017 Genscape")
set.seed(123)   #set seed to replicate results
for (i in 1:9){

    # specify data set of interest
    df <- records
    
    #make names for targets if not already made 'X1' as 'positive' class
    levels(df$Y) <- make.names(levels(factor(df$Y)))
    df$Y <- relevel(df$Y,"X1")
    
    # identify records to be used for training model
    inTrain <- createDataPartition(y = df$Y,       # outcome variable
                                   p = .75,        # % of training data
                                   list = FALSE)
    # raw train
    raw <- df[inTrain,]
    
    # up-sampled training set
    upSampTrain <- upSample(x=raw
                            ,y=raw$Y
                            ,yname = 'Y')
    upSampTrain <- upSampTrain[,c(1:ncol(upSampTrain)-1)] # remove the added target
    
    # down-sampled training set
    downSampTrain <- downSample(x=raw
                                ,y=raw$Y
                                ,yname = 'Y')
    downSampTrain <- downSampTrain[,c(1:ncol(upSampTrain)-1)] # remove the added target
    
    # test data set
    test <- df[-inTrain,]
    ############################################################################
    n <- 100000
    ### Make sure if the number of records is greater than "n" sample it down more
    ### This should help with runtime
    if (nrow(raw) > n){
        inRaw <- createDataPartition(y = raw$Y, #outcome variable
                                     p = round(n/nrow(raw),2), #% of training data
                                     list = FALSE)
        # reduced raw train
        raw <- raw[inRaw,]
    }
    if (nrow(downSampTrain) > n){
        inDown <- createDataPartition(y = downSampTrain$Y, #outcome variable
                                      p = round(n/nrow(downSampTrain),2), #% of training data
                                      list = FALSE)
        # reduced down-sampled training set
        downSampTrain <- downSampTrain[inDown,]
    }
    if (nrow(upSampTrain) > n){
        inUp <- createDataPartition(y = upSampTrain$Y, #outcome variable
                                    p = round(n/nrow(upSampTrain),2), #% of training data
                                    list = FALSE)
        # reduced raw train
        upSampTrain <- upSampTrain[inUp,]
    }
    # remove these temporary variables
    suppressWarnings(rm(inTrain, n, inRaw, inDown, inUp))
    ############################################################################
    # remove outliers as they can lead to parameter stability issues
    #raw2 <- raw
    #for (k in 2:ncol(raw)) {
    #    raw2 <- raw[which(!raw2[,k] %in% boxplot.stats(raw2[,k])$out),]
    #}
    #raw <- raw2; rm(raw2)
    
    #upSampTrain2 <- upSampTrain
    #for (k in 2:ncol(upSampTrain2)) {
    #    upSampTrain2 <- raw[which(!upSampTrain2[,k] %in% boxplot.stats(upSampTrain2[,k])$out),]
    #}
    #upSampTrain <- upSampTrain2; rm(upSampTrain2)
    
    #downSampTrain2 <- downSampTrain
    #for (k in 2:ncol(downSampTrain2)) {
    #    downSampTrain2 <- raw[which(!downSampTrain2[,k] %in% boxplot.stats(downSampTrain2[,k])$out),]
    #}
    #downSampTrain <- downSampTrain2; rm(downSampTrain2)
    
    ############################################################################
    # remove features where the values they take on is limited
    # here we make sure to keep the target variable and only those input
    # features with enough variation
    nzv <- nearZeroVar(raw, saveMetrics = TRUE)
    raw <- raw[, c(TRUE,!nzv$zeroVar[2:ncol(raw)])]
    
    nzv <- nearZeroVar(upSampTrain, saveMetrics = TRUE)
    upSampTrain <- upSampTrain[, c(TRUE,!nzv$zeroVar[2:ncol(upSampTrain)])]
    
    nzv <- nearZeroVar(downSampTrain, saveMetrics = TRUE)
    downSampTrain <- downSampTrain[, c(TRUE,!nzv$zeroVar[2:ncol(downSampTrain)])]
    
    ############################################################################
    # remove highly correlated features. highly correlated features can lead to
    # collinearity and rank deficient issues which can have a drastic impact
    # depending on the methodology used
    descrCor <-  cor(raw[,2:ncol(raw)])
    highCorr <- sum(abs(descrCor[upper.tri(descrCor)]) > .999)
    highlyCorDescr <- findCorrelation(descrCor, cutoff = .85)
    highlyCorDescr <- highlyCorDescr+1
    if (length(highlyCorDescr) >= 1){
        raw <- raw[,-highlyCorDescr]
    }
    
    descrCor <-  cor(upSampTrain[,2:ncol(upSampTrain)])
    highCorr <- sum(abs(descrCor[upper.tri(descrCor)]) > .999)
    highlyCorDescr <- findCorrelation(descrCor, cutoff = .85)
    highlyCorDescr <- highlyCorDescr+1
    if (length(highlyCorDescr) >= 1){
        upSampTrain <- upSampTrain[,-highlyCorDescr]
    }
    
    descrCor <-  cor(downSampTrain[,2:ncol(downSampTrain)])
    highCorr <- sum(abs(descrCor[upper.tri(descrCor)]) > .999)
    highlyCorDescr <- findCorrelation(descrCor, cutoff = .85)
    highlyCorDescr <- highlyCorDescr+1
    if (length(highlyCorDescr) >= 1){
        downSampTrain <- downSampTrain[,-highlyCorDescr]
    }
    
    ############################################################################
    # define the model equation
    equ_v2=Y ~ .

    ############################################################################
    # specify the cross-validation approach to use
    ctrl <- trainControl(method = "cv", number=5   #5-fold CV
                         , classProbs = TRUE
                         , summaryFunction = twoClassSummary)
    
    # tryCatch ends at line 313
    tryCatch(
        # capture runtime while model learns
        runTime <- system.time(
            if (perfMeasures[i,"balancingType"] == "raw") {
                if (perfMeasures[i,"Method"] == "rpart") {
                    myModel <- suppressWarnings(
                                train(equ_v2              #model specification
                                     ,data = raw         #training set used
                                     ,method = perfMeasures[i,"Method"] 
                                     ,trControl = ctrl
                                     ,metric = "ROC")
                                )
                } else if (perfMeasures[i,"Method"] == "lda") {
                    myModel <- train(equ_v2              #model specification
                                     ,data = raw         #training set used
                                     ,method = perfMeasures[i,"Method"] 
                                     ,trControl = ctrl
                                     ,metric = "ROC")
                } else if (perfMeasures[i,"Method"] == "C5.0") {
                    myModel <- train(equ_v2              #model specification
                                     ,data = raw         #training set used
                                     ,method = perfMeasures[i,"Method"] 
                                     ,trControl = ctrl
                                     ,tuneLength=5
                                     ,metric = "ROC")
                } 
                # train model estimated probs and classes
                estTrainProbs <- predict(myModel, newdata = raw, type='prob')[,1]
                estTrainClasses <- predict(myModel, newdata = raw)
                # capture performance of the trained model
                cm <- confusionMatrix(data=estTrainClasses, raw$Y)
                # calculate ROC curve
                rocCurve <- roc(response = raw$Y
                                , predictor = estTrainProbs
                                # reverse the labels.
                                , levels = rev(levels(raw$Y)))
            } else if (perfMeasures[i,"balancingType"] == "down") { 
                if (perfMeasures[i,"Method"] == "rpart") {
                    myModel <- suppressWarnings(
                                train(equ_v2                  #model specification
                                     ,data = downSampTrain   #training set used
                                     ,method = perfMeasures[i,"Method"] 
                                     ,trControl = ctrl
                                     ,metric = "ROC")
                                )
                } else if (perfMeasures[i,"Method"] == "lda") {
                    myModel <- train(equ_v2                   #model specification
                                     ,data = downSampTrain    #training set used
                                     ,method = perfMeasures[i,"Method"] 
                                     ,trControl = ctrl
                                     ,metric = "ROC")
                } else if (perfMeasures[i,"Method"] == "C5.0") {
                    myModel <- train(equ_v2                   #model specification
                                     ,data = downSampTrain    #training set used
                                     ,method = perfMeasures[i,"Method"] 
                                     ,trControl = ctrl
                                     ,tuneLength=5
                                     ,metric = "ROC")
                } 
                # train model estimated probs and classes
                estTrainProbs <- predict(myModel, newdata = downSampTrain, type='prob')[,1]
                estTrainClasses <- predict(myModel, newdata = downSampTrain)
                # capture performance of the trained model
                cm <- confusionMatrix(data=estTrainClasses, downSampTrain$Y)
                # calculate ROC curve
                rocCurve <- roc(response = downSampTrain$Y
                                , predictor = estTrainProbs
                                # reverse the labels.
                                , levels = rev(levels(downSampTrain$Y)))
            } else if (perfMeasures[i,"balancingType"] == "up") { 
                if (perfMeasures[i,"Method"] == "rpart") {
                    myModel <- suppressWarnings(
                                train(equ_v2                #model specification
                                     ,data = upSampTrain   #training set used
                                     ,method = perfMeasures[i,"Method"] 
                                     ,trControl = ctrl
                                     ,metric = "ROC")
                                )
                } else if (perfMeasures[i,"Method"] == "lda") {
                    myModel <- train(equ_v2                 #model specification
                                     ,data = upSampTrain    #training set used
                                     ,method = perfMeasures[i,"Method"] 
                                     ,trControl = ctrl
                                     ,metric = "ROC")
                } else if (perfMeasures[i,"Method"] == "C5.0") {
                    myModel <- train(equ_v2                 #model specification
                                     ,data = upSampTrain    #training set used
                                     ,method = perfMeasures[i,"Method"] 
                                     ,trControl = ctrl
                                     ,tuneLength=5
                                     ,metric = "ROC")
                }
                # train model estimated probs and classes
                estTrainProbs <- predict(myModel, newdata = upSampTrain, type='prob')[,1]
                estTrainClasses <- predict(myModel, newdata = upSampTrain)
                # capture performance of the trained model
                cm <- confusionMatrix(data=estTrainClasses, upSampTrain$Y)
                # calculate ROC curve
                rocCurve <- roc(response = upSampTrain$Y
                                , predictor = estTrainProbs
                                # reverse the labels.
                                , levels = rev(levels(upSampTrain$Y)))
                
            }
        )[[1]] # end runTime
        ,
        error = function(f) {print("model can't be trained")}
    ) # end tryCatch
    #########################################################################
    # if the model could not be trained, skip trying to capture stats
    if (!exists("myModel")) {
        print("No model exists so no stats saved")
        # save run time (in minutes)
        perfMeasures[i,"Runtime"] <- NA
        
    } else if (exists("myModel")) {
        
        # save run time (in minutes)
        perfMeasures[i,"Runtime"] <- runTime/60
        
        # save training performance measures
        perfMeasures[i,"trCohensKappa"] <- cm$overall["Kappa"][[1]]
        perfMeasures[i,"trOA"] <- cm$overall["Accuracy"][[1]]
        perfMeasures[i,"trSe"] <- cm$byClass['Sensitivity'][[1]]
        perfMeasures[i,"trSp"] <- cm$byClass['Specificity'][[1]]
        perfMeasures[i,"trAUC"] <- auc(rocCurve)
        # https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
        TP <- as.numeric(cm$table[1,1])
        FP <- as.numeric(cm$table[1,2])
        FN <- as.numeric(cm$table[2,1])
        TN <- as.numeric(cm$table[2,2])
        perfMeasures[i,"trMCC"] <- ((TP*TN)-(FP*FN))/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        # https://en.wikipedia.org/wiki/F1_score
        perfMeasures[i,"trF1"] <- 2*(cm$byClass["Pos Pred Value"][[1]]*cm$byClass['Sensitivity'][[1]])/
            (cm$byClass["Pos Pred Value"][[1]]+cm$byClass['Sensitivity'][[1]])
        
        ########################################################################
        # calculate the testing predictions
        estTestProbs <- predict(myModel, newdata = test, type='prob')[,1]
        estTestClasses <- predict(myModel, newdata = test)
        ########################################################################
        # capture performance of the trained model
        testCM <- confusionMatrix(data=estTestClasses, test$Y)
        # calculate ROC curve
        testRocCurve <- roc(response = test$Y
                            , predictor = estTestProbs
                            # reverse the labels.
                            , levels = rev(levels(test$Y)))
        # save test performance measures
        perfMeasures[i,"teCohensKappa"] <- testCM$overall["Kappa"][[1]]
        perfMeasures[i,"teOA"] <- testCM$overall["Accuracy"][[1]]
        perfMeasures[i,"teSe"] <- testCM$byClass['Sensitivity'][[1]]
        perfMeasures[i,"teSp"] <- testCM$byClass['Specificity'][[1]]
        perfMeasures[i,"teAUC"] <- auc(testRocCurve)
        # https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
        TP <- as.numeric(testCM$table[1,1])
        FP <- as.numeric(testCM$table[1,2])
        FN <- as.numeric(testCM$table[2,1])
        TN <- as.numeric(testCM$table[2,2])
        perfMeasures[i,"teMCC"] <- ((TP*TN)-(FP*FN))/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        # https://en.wikipedia.org/wiki/F1_score
        perfMeasures[i,"teF1"] <- 2*(testCM$byClass["Pos Pred Value"][[1]]*testCM$byClass['Sensitivity'][[1]])/
            (testCM$byClass["Pos Pred Value"][[1]]+testCM$byClass['Sensitivity'][[1]])
        
        ############################################################################
        # save results for probability calibration plots
        forPlots <- data.frame(rep(perfMeasures[i,1],length(estTestProbs)),
                               rep(perfMeasures[i,2],length(estTestProbs)),
                               rep(perfMeasures[i,5],length(estTestProbs)),
                               rep(perfMeasures[i,6],length(estTestProbs)),
                               ifelse(test$Y=="X1",1,0),
                               estTestProbs
        )
        names(forPlots) <- c("Method","balancingType","Y","Yhat")
        
        ############################################################################
        # save each result to the database
        write.table(forPlots, file="test.csv" ,sep=",", quote=F, row.names=F
                    , append=T)
       
        ############################################################################
        # remove any iteration results to make sure they dont end up in another
        # data-model-balancetype combination
        rm(myModel,estTrainProbs,estTrainClasses,cm, rocCurve, estTestProbs, 
           estTestClasses, testCM, testRocCurve, df, TP,TN,FP,FN,forPlots)
        
    } # end if - for calculating model stats
}

# clean up environment
rm(descrCor, downSampTrain, nzv, raw, records, test, upSampTrain, ctrl,
   equ_v2, highCorr, highlyCorDescr, i, runTime)
################################################################################
# The following section of code creates probability calibration plots
################################################################################
getwd()
testPreds <- read.table(file="test.csv", header=T, sep=",", quote="")
testPreds <- testPreds[,c(1,2,5,6)]
names(testPreds) <- c("Method","balancingType","Y","Yhat")
testPreds$Y <- as.factor(testPreds$Y)
par(mfrow=c(1,1))
tmp <- testPreds

# fetch the dataset of interest
raw <- tmp[which(tmp$balancingType == "raw"),]
up <- tmp[which(tmp$balancingType == "up"),]
down <- tmp[which(tmp$balancingType == "down"),]

# generate statistics for calibration plot
curveRaw <- calibration(Y ~ Yhat, cuts=20, data=raw)$data
curveUp <- calibration(Y ~ Yhat, cuts=10, data=up)$data
curveDown <- calibration(Y ~ Yhat, cuts=20, data=down)$data

# remove bins where probabilities do not exist
curveRaw <- data.frame(curveRaw[curveRaw$Count>0,])
curveUp <- data.frame(curveUp[curveUp$Count>0,])
curveDown <- data.frame(curveDown[curveDown$Count>0,])

# fix percentages
curveRaw$Percent <- 100-curveRaw$Percent
curveUp$Percent <- 100-curveUp$Percent
curveDown$Percent <- 100-curveDown$Percent

# define parameters of calibration plots
numBins = 20
labelSize = 1

# generate custom calibration plot
#mypar = par(bg="white")
plot(y=curveRaw$Percent/100, x=curveRaw$midpoint/100
     , xaxt="n", yaxt="n", main="Probability calibration plot"
     , xlab="Predicted Probabilities (Bin Midpoint)", ylab="Observed Event %", xlim=c(0,1)
     , ylim=c(0,1), col="blue", bg="blue", pch=21, type="b", cex.lab=1.2)
axis(1, at=seq(from=0, to=1, by=0.05), labels=format(seq(from=0, to=1, by=0.05))
     , tick=T, lwd=0.5, cex.axis=labelSize)
axis(2, at=seq(from=0, to=1, by=0.05), labels=format(seq(from=0, to=1, by=0.05))
     , tick=T, lwd=0.5, cex.axis=labelSize)
# add 45-degree target line
abline(a=0, b=1, col="gray60", lty=2)
## Up points
points(y=curveUp$Percent/100, x=curveUp$midpoint/100, col="darkgreen"
       , bg="darkgreen", pch=22, type="b", cex.lab=1.2)
## Down points
points(y=curveDown$Percent/100, x=curveDown$midpoint/100, col="red"
       , bg="red", pch=24, type="b", cex.lab=1.2)
# add legend
legend("top", inset=0, #title="Balance Approach"
       , legend=c("raw","up","down"), fill=c("blue","darkgreen","red")
       , border="white", bty="n", cex=1.2, horiz=TRUE)

# clean up environment
rm(curveDown, curveRaw, curveUp, down, raw, testPreds, tmp, up, labelSize, numBins)