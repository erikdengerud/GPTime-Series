#This code can be used to reproduce the forecasts of the M4 Competition STATISTICAL Benchmarks and evaluate their accuracy

library(forecast) #Requires v8.2

#################################################################################
#In this example let us produce forecasts for 100 randomly generated timeseries
fh <- 6 #The forecasting horizon examined
frq <- 1 #The frequency of the data
data_train = data_test <- NULL #Train and test sample
for (i in 1:100){
  data_all <- 2+ 0.15*(1:20) + rnorm(20) 
  data_train[length(data_train)+1] <- list(ts(head(data_all,length(data_all)-fh),frequency = frq))
  data_test[length(data_test)+1] <- list(tail(data_all,fh))
}
#################################################################################
# Calculating benchmark scores using the M4 dataset
y_train = data.frame(read.csv("data/M4train/Yearly-train.csv"))
q_train = data.frame(read.csv("data/M4train/Quarterly-train.csv"))
m_train = data.frame(read.csv("data/M4train/Monthly-train.csv"))
w_train = data.frame(read.csv("data/M4train/Weekly-train.csv"))
d_train = data.frame(read.csv("data/M4train/Daily-train.csv"))
h_train = data.frame(read.csv("data/M4train/Hourly-train.csv"))

y_test = data.frame(read.csv("data/M4test/Yearly-test.csv"))
q_test = data.frame(read.csv("data/M4test/Quarterly-test.csv"))
m_test = data.frame(read.csv("data/M4test/Monthly-test.csv"))
w_test = data.frame(read.csv("data/M4test/Weekly-test.csv"))
d_test = data.frame(read.csv("data/M4test/Daily-test.csv"))
h_test = data.frame(read.csv("data/M4test/Hourly-test.csv"))

train_dfs = list(y_train, q_train, m_train, w_train, d_train, h_train)
test_dfs = list(y_test, q_test, m_test, w_test, d_test, h_test)
#################################################################################

smape_cal <- function(outsample, forecasts){
  #Used to estimate sMAPE
  outsample <- as.numeric(outsample) ; forecasts<-as.numeric(forecasts)
  smape <- (abs(outsample-forecasts)*200)/(abs(outsample)+abs(forecasts))
  return(smape)
}

mase_cal <- function(insample, outsample, forecasts){
  #Used to estimate MASE
  frq <- frequency(insample)
  forecastsNaiveSD <- rep(NA,frq)
  for (j in (frq+1):length(insample)){
    forecastsNaiveSD <- c(forecastsNaiveSD, insample[j-frq])
  }
  masep<-mean(abs(insample-forecastsNaiveSD),na.rm = TRUE)
  
  outsample <- as.numeric(outsample) ; forecasts <- as.numeric(forecasts)
  mase <- (abs(outsample-forecasts))/masep
  return(mase)
}

naive_seasonal <- function(input, fh){
  #Used to estimate Seasonal Naive
  frcy <- frequency(input)
  frcst <- naive(input, h=fh)$mean 
  if (frcy>1){ 
    frcst <- head(rep(as.numeric(tail(input,frcy)), fh), fh) + frcst - frcst
  }
  return(frcst)
}

Theta.classic <- function(input, fh){
  #Used to estimate Theta classic
  
  #Set parameters
  wses <- wlrl<-0.5 ; theta <- 2
  #Estimate theta line (0)
  observations <- length(input)
  xt <- c(1:observations)
  xf <- c((observations+1):(observations+fh))
  train <- data.frame(input=input, xt=xt)
  test <- data.frame(xt = xf)
  
  estimate <- lm(input ~ poly(xt, 1, raw=TRUE))
  thetaline0In <- as.numeric(predict(estimate))
  thetaline0Out <- as.numeric(predict(estimate,test))
  
  #Estimate theta line (2)
  thetalineT <- theta*input+(1-theta)*thetaline0In
  sesmodel <- ses(thetalineT, h=fh)
  thetaline2In <- sesmodel$fitted
  thetaline2Out <- sesmodel$mean
  
  #Theta forecasts
  forecastsIn <- (thetaline2In*wses)+(thetaline0In*wlrl)
  forecastsOut <- (thetaline2Out*wses)+(thetaline0Out*wlrl)
  
  #Zero forecasts become positive
  for (i in 1:length(forecastsOut)){
    if (forecastsOut[i]<0){ forecastsOut[i]<-0 }
  }
  
  output=list(fitted = forecastsIn, mean = forecastsOut,
              fitted0 = thetaline0In, mean0 = thetaline0Out,
              fitted2 = thetaline2In, mean2 = thetaline2Out)
  
  return(output)
}

SeasonalityTest <- function(input, ppy){
  #Used to determine whether a time series is seasonal
  tcrit <- 1.645
  if (length(input)<3*ppy){
    test_seasonal <- FALSE
  }else{
    xacf <- acf(input, plot = FALSE)$acf[-1, 1, 1]
    clim <- tcrit/sqrt(length(input)) * sqrt(cumsum(c(1, 2 * xacf^2)))
    test_seasonal <- ( abs(xacf[ppy]) > clim[ppy] )
    
    if (is.na(test_seasonal)==TRUE){ test_seasonal <- FALSE }
  }
  
  return(test_seasonal)
}


Benchmarks <- function(input, fh){
  print("Benchmarks")
  #Used to estimate the statistical benchmarks of the M4 competition
  
  #Estimate seasonaly adjusted time series
  ppy <- frequency(input) ; ST <- F
  if (ppy>1){ ST <- SeasonalityTest(input,ppy) }
  if (ST==T){
    Dec <- decompose(input,type="multiplicative")
    des_input <- input/Dec$seasonal
    SIout <- head(rep(Dec$seasonal[(length(Dec$seasonal)-ppy+1):length(Dec$seasonal)], fh), fh)
  }else{
    des_input <- input ; SIout <- rep(1, fh)
  }
  print("B1")
  f1 <- naive(input, h=fh)$mean #Naive
  print("B2")
  f2 <- naive_seasonal(input, fh=fh) #Seasonal Naive
  print("B3")
  f3 <- naive(des_input, h=fh)$mean*SIout #Naive2
  print("B4")
  f4 <- ses(des_input, h=fh)$mean*SIout #Ses
  print("B5")
  f5 <- holt(des_input, h=fh, damped=F)$mean*SIout #Holt
  print("B6")
  f6 <- holt(des_input, h=fh, damped=T)$mean*SIout #Damped
  print("B7")
  #f7 <- Theta.classic(input=des_input, fh=fh)$mean*SIout #Theta
  #print("B8")
  f8 <- (f4+f5+f6)/3 #Comb
  print("B9")
  
  return(list(f1,f2,f3,f4,f5,f6,f7,f8))
}

Names_benchmarks <- c("Naive", "sNaive", "Naive2", "SES", "Holt", "Damped", "Com")#"Theta"
fhs = c(6, 8, 18, 13, 14, 48)
mase_scores = array(NA, dim=c(length(Names_benchmarks), length(train_dfs)+1))
smape_scores = array(NA, dim=c(length(Names_benchmarks), length(train_dfs)+1))
owa_scores = array(NA, dim=c(length(Names_benchmarks), length(train_dfs)+1))

for (i in 1:length(train_dfs)){
  print(c("df #", 1))
  fh = fhs[i]
  data_train = train_dfs[[i]]
  data_test = test_dfs[[i]]
  freq_smape=freq_mase <- array(NA,dim = c(length(Names_benchmarks), fh, length(data_train)))
  for (j in 1:length(data_train)){
    insample <- data_train[[j]]
    outsample <- data_test[[j]]
    insample<-insample[!is.na(insample)]
    outsample<-outsample[!is.na(outsample)]
    forecasts <- Benchmarks(input=insample, fh=fh)
    
    #sMAPE
    for (k in 1:length(Names_benchmarks)){
      freq_smape[k,,j] <- smape_cal(outsample, forecasts[[k]]) #k the # of the benchmark
    }
    #MASE
    for (k in 1:length(Names_benchmarks)){
      freq_mase[k,,j] <- mase_cal(insample, outsample, forecasts[[k]]) #k the # of the benchmark
    }
  }
  ########### sMAPE ###############
  for (j in 1:length(Names_benchmarks)){
    mase_scores[i,j] = round(mean(freq_smape[j,,]), 3)
  }  
  ########### MASE ################
  for (j in 1:length(Names_benchmarks)){
    smape_scores[i,j] = round(mean(freq_mase[j,,]), 3)
  }
  ########### OWA ################
  for (j in 1:length(Names_benchmarks)){
    owa_scores[i,j] = round(((mean(freq_mase[j,,])/mean(freq_mase[3,,]))+(mean(freq_smape[j,,])/mean(freq_smape[3,,])))/2, 3)
  }
}

weights = c(nrow(train_dfs[[1]]), nrow(train_dfs[[2]]), nrow(train_dfs[[3]]), nrow(train_dfs[[4]]), nrow(train_dfs[[5]]), nrow(train_dfs[[6]]))
weights = weights/sum(weights)
### MASE ###
for (j in 1:length(Names_benchmarks)){
  mase_scores[length(train_dfs)+1, j] = mase_scores[,j] %*% weights
}

### SMAPE ###
for (j in 1:length(Names_benchmarks)){
  smape_scores[length(train_dfs)+1, j] = smape_scores[,j] %*% weights
}

### OWA ###
for (j in 1:length(Names_benchmarks)){
  owa_scores[length(train_dfs)+1, j] = (mase_scores[length(train_dfs)+1,j] / mase_scores[length(train_dfs)+1,3] + smape_scores[length(train_dfs)+1,j] / smape_scores[length(train_dfs)+1,3])/2
}




Names_benchmarks <- c("Naive", "sNaive", "Naive2", "SES", "Holt", "Damped", "Theta", "Com")
Total_smape=Total_mase <- array(NA,dim = c(length(Names_benchmarks), fh, length(data_train)))
#Methods, Horizon, time-series
for (i in 1:length(data_train)){
  
  insample <- data_train[[i]]
  outsample <- data_test[[i]]
  forecasts <- Benchmarks(input=insample, fh=fh)
  
  #sMAPE
  for (j in 1:length(Names_benchmarks)){
    Total_smape[j,,i] <- smape_cal(outsample, forecasts[[j]]) #j the # of the benchmark
  }
  #MASE
  for (j in 1:length(Names_benchmarks)){
    Total_mase[j,,i] <- mase_cal(insample, outsample, forecasts[[j]]) #j the # of the benchmark
  }
  
}

print("########### sMAPE ###############")
for (i in 1:length(Names_benchmarks)){
  print(paste(Names_benchmarks[i], round(mean(Total_smape[i,,]), 3)))
}
print("########### MASE ################")
for (i in 1:length(Names_benchmarks)){
  print(paste(Names_benchmarks[i], round(mean(Total_mase[i,,]), 3)))
}
print("########### OWA ################")
for (i in 1:length(Names_benchmarks)){
  print(paste(Names_benchmarks[i],
              round(((mean(Total_mase[i,,])/mean(Total_mase[3,,]))+(mean(Total_smape[i,,])/mean(Total_smape[3,,])))/2, 3)))
}

