rm(list=ls());

#load required library
library(glmnet);
library(randomForest);
library(deepTL);

set.seed(1000);
sample.size = 500;
trainsiz = round(sample.size*0.9,0);
p = 500;
connum = 20;
nuinum = p-connum;
n.sim = 1;
pvacut = 0.1;
#######################
# Scenario 1 setting  #
#######################
alpha_con = c(rnorm(connum/2)*2.5,rnorm(connum/2));
alpha_con[1] = alpha_con[1]*1.2;
alpha_con[3] = alpha_con[3]*12;
alpha_con[6] = alpha_con[6]*2;
alpha_con[7] = alpha_con[7]*1.2;
alpha_con[9] = alpha_con[9]*29;

#######################
# Scenario 2 setting  #
#######################
# alpha_con = (rnorm(connum)+20)/10;
# alpha_int = (rnorm(conint)-16)/10;
# alpha_qua = (rnorm(connum)-8)/10;

set.seed(1000*seeds);

resultfile = "~/HdFIT/result/sim1_seeds.csv";

#####################################################################
# This is Scenario 1 data generating function, i.e. all are linear  #
# summation of some of observed covariates (important variables).   #
#####################################################################
data_gen = function()  {  

	##################################
	# Generate important covariates  #
	##################################
	conmat = NULL;  
	for (i in 1:connum)  {
	  conxvar = rnorm(sample.size,mean=0,sd=1);
	  conmat = cbind(conmat,conxvar);
	}

	#######################
	# Generate y          #
	#######################
	y = 10 + conmat%*%alpha_con + rnorm(sample.size);

	#################################  
	# Generate nuisance covariates  #
	#################################
	nuimat = NULL;
	for (i in 1:nuinum)  {
	  nuixvar = rnorm(sample.size,mean=0,sd=1);	  
	  nuimat = cbind(nuimat,nuixvar);
	}
	zconmat = conmat[,1:round(connum/2,0)];
	xconmat = conmat[,(round(connum/2,0)+1):connum];
	datmat = data.frame(cbind(y,zconmat,nuimat,xconmat));
	for (i in 1:dim(datmat)[2])  {
	  if (i == 1)  colnames(datmat)[1] = "y";
	  if (i > 1)  colnames(datmat)[i] = paste0("V",i-1); 
	}
	return (datmat);
}

######################################################################
# This is Scenario 2 data generating function, i.e. some are linear  #
# and some are non-linear.                                           # 
######################################################################
# data_gen = function()  {  
#   
#   ##################################
#   # Generate important covariates  #
#   ##################################
#   conmat = quamat = NULL;  
#   for (i in 1:connum)  {
#     conxvar = rnorm(sample.size,mean=0,sd=1);
#     conmat = cbind(conmat,conxvar);
#     quamat = cbind(quamat,conxvar*conxvar);
#   }
#   
#   intmat = NULL;
#   for (i in 1:conint)  {
#     # temp = conmat[,i] + conmat[,i+conint];
#     temp = conmat[,i] * conmat[,i+conint];
#     intmat = cbind(intmat, temp);
#   } 
#   
#   #######################
#   # Generate y          #
#   #######################
#   y = 10 + conmat%*%alpha_con + intmat%*%alpha_int + quamat%*%alpha_qua + rnorm(sample.size);
#   
#   #################################  
#   # Generate nuisance covariates  #
#   #################################
#   nuimat = NULL;
#   for (i in 1:nuinum)  {
#     nuixvar = rnorm(sample.size,mean=0,sd=1);	  
#     nuimat = cbind(nuimat,nuixvar);
#   }
#   zconmat = conmat[,1:round(connum/2,0)];
#   xconmat = conmat[,(round(connum/2,0)+1):connum];
#   datmat = data.frame(cbind(y,zconmat,nuimat,xconmat));
#   for (i in 1:dim(datmat)[2])  {
#     if (i == 1)  colnames(datmat)[1] = "y";
#     if (i > 1)  colnames(datmat)[i] = paste0("V",i-1); 
#   }
#   return (datmat);
# }

truimp = 1:round(connum/2,0);
esCtrl = list(n.hidden=c(50,40,30,20), activate="relu", l1.reg=10**-4, early.stop.det=1000, n.batch=30, 
              n.epoch=1000, learning.rate.adaptive="adam", plot=FALSE);

for (i in 1:n.sim)  {
  dat = data_gen(); 
  datdim = dim(dat)[2];
  xvarmat = dat[,2:datdim];

  atrainx = xvarmat[1:trainsiz,];
  atestx = xvarmat[(trainsiz+1):sample.size,];
  ytrain = dat[1:trainsiz,1];
  ytest = dat[(trainsiz+1):sample.size,1];

  xvarcolm = NULL;
  xtrainmat = as.matrix(atrainx[,(datdim-connum):(datdim-1)]);
  #####################
  # Feature screening #
  #####################
  for (j in 1:(datdim-connum-1))  {
    lmfit = lm(ytrain~atrainx[,j]+xtrainmat);
    xvarcolm = c(xvarcolm,summary(lmfit)$coefficients[2,4]);
  }
  scrindx = order(xvarcolm,decreasing=F);
  selimpx = length(intersect(truimp,scrindx[1:80]));
  selxmat = cbind(xvarmat[,scrindx[1:80]],xvarmat[,(datdim-connum):(datdim-1)]);
  strainx = selxmat[1:trainsiz,];
  stestx = selxmat[(trainsiz+1):sample.size,];

  train_obj = importDnnet(x=as.matrix(atrainx), y=ytrain);
  strain_obj = importDnnet(x=as.matrix(strainx), y=ytrain);
  nshuffle = sample(trainsiz);
  
  ###############################################################  
  # LASSO fitting 
  ###############################################################
  cv.out = cv.glmnet(as.matrix(atrainx),ytrain,alpha=1,family='gaussian',nfolds=10,type.measure='mse');
  lyhattrain = predict(cv.out,newx=as.matrix(atrainx),s=cv.out$lambda.1se);
  lmsetrain = mean((ytrain-lyhattrain)**2);
  lpcctrain = cor(ytrain,lyhattrain);
  lyhattest = predict(cv.out,newx=as.matrix(atestx),s=cv.out$lambda.1se);
  lmsetest = mean((ytest-lyhattest)**2);
  lpcctest = cor(ytest,lyhattest);
  
  ###############################################################  
  # S-LASSO fitting 
  ###############################################################
  scv.out = cv.glmnet(as.matrix(strainx),ytrain,alpha=1,family='gaussian',nfolds=10,type.measure='mse');
  slyhattrain = predict(scv.out,newx=as.matrix(strainx),s=scv.out$lambda.1se);
  slmsetrain = mean((ytrain-slyhattrain)**2);
  slpcctrain = cor(ytrain,slyhattrain);
  slyhattest = predict(scv.out,newx=as.matrix(stestx),s=scv.out$lambda.1se);
  slmsetest = mean((ytest-slyhattest)**2);
  slpcctest = cor(ytest,slyhattest);
  
  ###############################################################  
  # RF
  ###############################################################
  rlmfit = randomForest(as.matrix(atrainx),ytrain,ntree=1000,nodesize=5);
  ryhattrain = predict(rlmfit,as.matrix(atrainx));
  rmsetrain = mean((ytrain-ryhattrain)**2);
  rpcctrain = cor(ytrain,ryhattrain);
  ryhattest = predict(rlmfit,as.matrix(atestx));
  rmsetest = mean((ytest-ryhattest)**2);
  rpcctest = cor(ytest,ryhattest);
  
  ###############################################################  
  # PermFIT-RF
  ###############################################################
  permfit_rf = permfit(train=train_obj, k_fold=10, n_perm=100, pathway_list=NULL, method="random_forest", shuffle=nshuffle, n.ensemble=100, ntree=1000, nodesize=5, verbose=0);
  rf_feature = which(permfit_rf@importance$importance_pval <= pvacut);
  prfimpx = length(intersect(truimp,rf_feature));
  prffit = randomForest(as.matrix(atrainx[,rf_feature]),ytrain,ntree=1000,nodesize=5);
  pryhattrain = predict(prffit,as.matrix(atrainx[,rf_feature]));
  prmsetrain = mean((ytrain-pryhattrain)**2);
  prpcctrain = cor(ytrain,pryhattrain);
  pryhattest = predict(prffit,as.matrix(atestx[,rf_feature]));
  prmsetest = mean((ytest-pryhattest)**2);
  prpcctest = cor(ytest,pryhattest);
  
  ###############################################################  
  # HdFIT-RF
  ###############################################################
  hdfit_rf = permfit(train=strain_obj, k_fold=10, n_perm=100, pathway_list=NULL, method="random_forest", shuffle=nshuffle, n.ensemble=100, ntree=1000, nodesize=5, verbose=0);
  srf_feature = which(hdfit_rf@importance$importance_pval <= pvacut);
  sprfimpx = length(intersect(truimp,scrindx[srf_feature]));
  sprffit = randomForest(as.matrix(strainx[,srf_feature]),ytrain,ntree=1000,nodesize=5);
  spryhattrain = predict(sprffit,as.matrix(strainx[,srf_feature]));
  sprmsetrain = mean((ytrain-spryhattrain)**2);
  sprpcctrain = cor(ytrain,spryhattrain);
  spryhattest = predict(sprffit,as.matrix(stestx[,srf_feature]));
  sprmsetest = mean((ytest-spryhattest)**2);
  sprpcctest = cor(ytest,spryhattest);  
  
  ###############################################################  
  # DNN
  ###############################################################  
  dlmfit = ensemble_dnnet(train_obj,100,esCtrl=esCtrl);  
  dyhattrain = predict(dlmfit,as.matrix(atrainx));
  dmsetrain = mean((ytrain-dyhattrain)**2);
  dpcctrain = cor(ytrain,dyhattrain);
  dyhattest = predict(dlmfit,as.matrix(atestx));
  dmsetest = mean((ytest-dyhattest)**2);
  dpcctest = cor(ytest,dyhattest);

  ###############################################################  
  # PermFIT-DNN
  ###############################################################
  permfit_dnn = permfit(train=train_obj, k_fold=10, n_perm=100, pathway_list=NULL, method="ensemble_dnnet", shuffle=nshuffle, n.ensemble=100, esCtrl=esCtrl, verbose=0);
  dnn_feature = which(permfit_dnn@importance$importance_pval <= pvacut);
  pdnnimpx = length(intersect(truimp,dnn_feature));
  pdnnfit = ensemble_dnnet(importDnnet(x=as.matrix(atrainx[,dnn_feature]),y=ytrain),100,esCtrl=esCtrl);
  pdyhattrain = predict(pdnnfit,as.matrix(atrainx[,dnn_feature]));
  pdmsetrain = mean((ytrain-pdyhattrain)**2);
  pdpcctrain = cor(ytrain,pdyhattrain);
  pdyhattest = predict(pdnnfit,as.matrix(atestx[,dnn_feature]));
  pdmsetest = mean((ytest-pdyhattest)**2);
  pdpcctest = cor(ytest,pdyhattest);

  ###############################################################  
  # HdFIT-DNN
  ###############################################################
  hdfit_dnn = permfit(train=strain_obj, k_fold=10, n_perm=100, pathway_list=NULL, method="ensemble_dnnet", shuffle=nshuffle, n.ensemble=100, esCtrl=esCtrl, verbose=0);
  sdnn_feature = which(hdfit_dnn@importance$importance_pval <= pvacut);
  spdnnimpx = length(intersect(truimp,scrindx[sdnn_feature]));
  spdnnfit = ensemble_dnnet(importDnnet(x=as.matrix(strainx[,sdnn_feature]),y=ytrain),100,esCtrl=esCtrl);
  spdyhattrain = predict(spdnnfit,as.matrix(strainx[,sdnn_feature]));
  spdmsetrain = mean((ytrain-spdyhattrain)**2);
  spdpcctrain = cor(ytrain,spdyhattrain);
  spdyhattest = predict(spdnnfit,as.matrix(stestx[,sdnn_feature]));
  spdmsetest = mean((ytest-spdyhattest)**2);
  spdpcctest = cor(ytest,spdyhattest);
  
  predictions = c(selimpx, prfimpx, sprfimpx, pdnnimpx, spdnnimpx, lmsetrain, lpcctrain, slmsetrain, slpcctrain, rmsetrain, rpcctrain, prmsetrain, prpcctrain, sprmsetrain, sprpcctrain, dmsetrain, dpcctrain, pdmsetrain, pdpcctrain, spdmsetrain, spdpcctrain);
  predictions = c(predictions, lmsetest, lpcctest, slmsetest, slpcctest, rmsetest, rpcctest, prmsetest, prpcctest, sprmsetest, sprpcctest, dmsetest, dpcctest, pdmsetest, pdpcctest, spdmsetest, spdpcctest);
  predictions = rbind(predictions);
  
  if (i < 2)  write.table(predictions,file=resultfile,append=F,row.names=F,col.names=F,sep=",");  
  if (i > 1)  write.table(predictions,file=resultfile,append=T,row.names=F,col.names=F,sep=",");
}
