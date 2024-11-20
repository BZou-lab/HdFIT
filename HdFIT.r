rm(list=ls());

#load required library
library(glmnet);
library(randomForest);
library(deepTL);

#################################################################################
# Random seed 1000 is used to generate the parameters for data generating model #
# in each round of Monte Carlo simulation to exactly duplicate reported results #
#################################################################################
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

########################################################################
# Random seed used for generating synthetic data in each of 1000 round #
# Monte Carlo simulation is 1000*1, 1000*2, ..., 1000*1000             #
########################################################################
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
  
  #################################
  # Informative Feature screening #
  #################################
  for (j in 1:(datdim-connum-1))  {
    lmfit = lm(ytrain~atrainx[,j]+xtrainmat);
    xvarcolm = c(xvarcolm,summary(lmfit)$coefficients[2,4]);
  }
  scrindx = order(xvarcolm,decreasing=F);
  selimpx = length(intersect(truimp,scrindx[1:80]));
  selxmat = cbind(xvarmat[,scrindx[1:80]],xvarmat[,(datdim-connum):(datdim-1)]);
  strainx = selxmat[1:trainsiz,];
  stestx = selxmat[(trainsiz+1):sample.size,];

  ############################
  # Random Feature Screening #
  ############################
  ramindx = sample(1:(datdim-connum-1),80);
  relxmat = cbind(xvarmat[,ramindx],xvarmat[,(datdim-connum):(datdim-1)]);
  rtrainx = relxmat[1:trainsiz,];
  rtestx = relxmat[(trainsiz+1):sample.size,];  
  
  ########################################
  # training data including all features #
  ########################################
  train_obj = importDnnet(x=as.matrix(atrainx), y=ytrain);
  #########################################################
  # training data including screened informative features #
  #########################################################  
  strain_obj = importDnnet(x=as.matrix(strainx), y=ytrain);
  ######################################################
  # training data including randomly screened features #
  ######################################################    
  rtrain_obj = importDnnet(x=as.matrix(rtrainx), y=ytrain);    

  nshuffle = sample(trainsiz);
  
  ###############################################################  
  # LASSO fitting based on all features (LASSO)
  ###############################################################
  cv.out = cv.glmnet(as.matrix(atrainx),ytrain,alpha=1,family='gaussian',nfolds=10,type.measure='mse');
  lyhattrain = predict(cv.out,newx=as.matrix(atrainx),s=cv.out$lambda.1se);
  lmsetrain = mean((ytrain-lyhattrain)**2);
  lpcctrain = cor(ytrain,lyhattrain);
  lyhattest = predict(cv.out,newx=as.matrix(atestx),s=cv.out$lambda.1se);
  lmsetest = mean((ytest-lyhattest)**2);
  lpcctest = cor(ytest,lyhattest);
  
  ###############################################################  
  # LASSO fitting based on screened features (SLASSO)
  ###############################################################
  scv.out = cv.glmnet(as.matrix(strainx),ytrain,alpha=1,family='gaussian',nfolds=10,type.measure='mse');
  slyhattrain = predict(scv.out,newx=as.matrix(strainx),s=scv.out$lambda.1se);
  slmsetrain = mean((ytrain-slyhattrain)**2);
  slpcctrain = cor(ytrain,slyhattrain);
  slyhattest = predict(scv.out,newx=as.matrix(stestx),s=scv.out$lambda.1se);
  slmsetest = mean((ytest-slyhattest)**2);
  slpcctest = cor(ytest,slyhattest);
  
  ###############################################################  
  # Random Forest based on all features (RF)
  ###############################################################
  rlmfit = randomForest(as.matrix(atrainx),ytrain,ntree=1000,nodesize=5);
  ryhattrain = predict(rlmfit,as.matrix(atrainx));
  rmsetrain = mean((ytrain-ryhattrain)**2);
  rpcctrain = cor(ytrain,ryhattrain);
  ryhattest = predict(rlmfit,as.matrix(atestx));
  rmsetest = mean((ytest-ryhattest)**2);
  rpcctest = cor(ytest,ryhattest);
  
  ###############################################################  
  # PermFIT for RF based on all features (PermFIT-RF)
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
  
  ##################################################################  
  # HdFIT for RF based on screened informative features (HdFIT-RF) #
  ##################################################################
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
  # DNN based on all features (DNN)
  ###############################################################  
  dlmfit = ensemble_dnnet(train_obj,100,esCtrl=esCtrl);  
  dyhattrain = predict(dlmfit,as.matrix(atrainx));
  dmsetrain = mean((ytrain-dyhattrain)**2);
  dpcctrain = cor(ytrain,dyhattrain);
  dyhattest = predict(dlmfit,as.matrix(atestx));
  dmsetest = mean((ytest-dyhattest)**2);
  dpcctest = cor(ytest,dyhattest);

  ###############################################################  
  # PermFIT for DNN based on all features (PermFIT-DNN)
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

  ####################################################################  
  # HdFIT for DNN based on screened informative features (HdFIT-DNN) #
  ####################################################################
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
  
  #################################################################  
  # HdFIT for RF based on randomly screened features (Control-RF) #
  #################################################################
  rpermfit_rf = permfit(train=rtrain_obj, k_fold=10, n_perm=100, pathway_list=NULL, method="random_forest", shuffle=nshuffle, n.ensemble=100, ntree=1000, nodesize=5, verbose=0);
  rrf_feature = which(rpermfit_rf@importance$importance_pval <= pvacut);
  rprfimpx = length(intersect(truimp,ramindx[rrf_feature]));
  rprffit = randomForest(as.matrix(rtrainx[,rrf_feature]),ytrain,ntree=1000,nodesize=5);
  rpryhattrain = predict(rprffit,as.matrix(rtrainx[,rrf_feature]));
  rprmsetrain = mean((ytrain-rpryhattrain)**2);
  rprpcctrain = cor(ytrain,rpryhattrain);
  rpryhattest = predict(rprffit,as.matrix(rtestx[,rrf_feature]));
  rprmsetest = mean((ytest-rpryhattest)**2);
  rprpcctest = cor(ytest,rpryhattest);    
  
  ###################################################################  
  # HdFIT for DNN based on randomly screened features (Control-DNN) #
  ###################################################################
  rpermfit_dnn = permfit(train=rtrain_obj, k_fold=10, n_perm=100, pathway_list=NULL, method="ensemble_dnnet", shuffle=nshuffle, n.ensemble=100, esCtrl=esCtrl, verbose=0);
  rdnn_feature = which(rpermfit_dnn@importance$importance_pval <= pvacut);
  rpdnnimpx = length(intersect(truimp,ramindx[rdnn_feature]));
  rpdnnfit = ensemble_dnnet(importDnnet(x=as.matrix(rtrainx[,rdnn_feature]),y=ytrain),100,esCtrl=esCtrl);
  rpdyhattrain = predict(rpdnnfit,as.matrix(rtrainx[,rdnn_feature]));
  rpdmsetrain = mean((ytrain-rpdyhattrain)**2);
  rpdpcctrain = cor(ytrain,rpdyhattrain);
  rpdyhattest = predict(rpdnnfit,as.matrix(rtestx[,rdnn_feature]));
  rpdmsetest = mean((ytest-rpdyhattest)**2);
  rpdpcctest = cor(ytest,rpdyhattest);
  
  ###############################################################  
  # Traditional DNN (tDNN)
  ###############################################################
  tdat_spl = splitDnnet(train_obj, 0.8);
  tdnnfit = do.call("dnnet",c(list("train"=tdat_spl$train,"validate"=tdat_spl$valid),esCtrl));  
  tdyhattrain = predict(tdnnfit,as.matrix(atrainx));
  tdmsetrain = mean((ytrain-tdyhattrain)**2);
  tdpcctrain = cor(ytrain,tdyhattrain);
  tdyhattest = predict(tdnnfit,as.matrix(atestx));
  tdmsetest = mean((ytest-tdyhattest)**2);
  tdpcctest = cor(ytest,tdyhattest);  
  
  predictions = c(selimpx, prfimpx, sprfimpx, pdnnimpx, spdnnimpx, lmsetrain, lpcctrain, slmsetrain, slpcctrain, rmsetrain, rpcctrain, prmsetrain, prpcctrain, sprmsetrain, sprpcctrain, dmsetrain, dpcctrain, pdmsetrain, pdpcctrain, spdmsetrain, spdpcctrain, rprmsetrain, rprpcctrain, rpdmsetrain, rpdpcctrain, tdmsetrain, tdpcctrain);
  predictions = c(predictions, lmsetest, lpcctest, slmsetest, slpcctest, rmsetest, rpcctest, prmsetest, prpcctest, sprmsetest, sprpcctest, dmsetest, dpcctest, pdmsetest, pdpcctest, spdmsetest, spdpcctest, rprmsetest, rprpcctest, rpdmsetest, rpdpcctest, tdmsetest, tdpcctest);
  predictions = rbind(predictions);
  
  if (i < 2)  write.table(predictions,file=resultfile,append=F,row.names=F,col.names=F,sep=",");  
  if (i > 1)  write.table(predictions,file=resultfile,append=T,row.names=F,col.names=F,sep=",");
}

