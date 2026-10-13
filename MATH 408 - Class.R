
#install.packages("torch")
#install.packages("luz")
#install.packages("pROC") 
#install.packages("e1071")
#install.packages("tree")
#install.packages("ISLR2")

library(rpart)

library(tree)
library(e1071)
library(boot)
library(ISLR2) 
library(pROC)
library(data.table)

library(tidyverse)
library(torch)
library(glmnet)
library(caret)
library(luz)
 
torch_manual_seed(13)
# cleaning dataset
heart = read.csv("C:\\Users\\gagel\\OneDrive\\Desktop\\heart(1).csv")
heart2 = na.omit(heart);
str(heart2)
#rudimeent Log regression plot, but this provided three different plots which are very useful 
Logistic_regression <- glm(TenYearCHD ~ ., data = heart2, family = binomial)
plot(Logistic_regression)
table1 <- data.table(Logistic_regression)
print(table1)
# Important for p-values and other attributes of the predictor variables. 
sum <- summary(Logistic_regression)

    

Ytesting <- predict(Logistic_regression)
sum <- summary(Logistic_regression)
dplyr::glimpse(heart2)
hx <- heart2 |>
  model.matrix(TenYearCHD ~ . - 1, data = _) |> 
  scale() |> 
  torch_tensor(dtype = torch_float())

hy <- heart2 |> 
  dplyr::select(TenYearCHD) |> 
  as.matrix() |> 
  torch_tensor(dtype = torch_float())
heart2 <- heart2 |> drop_na()
training <- heart2 |> slice_sample(prop = 0.8)
pre <- heart2|> anti_join(training)
validate <- pre |> slice_sample(prop =  0.5)
testing <- pre |> anti_join(validate)
Xtraining <- training |> 
  model.matrix(TenYearCHD ~ . - 1, data = _) |> 
  scale() |> 
  torch_tensor(dtype = torch_float())

Ytraining <- training |> 
  dplyr::select(TenYearCHD) |> 
  as.matrix() |>
  as.factor() |> 
  torch_tensor()
Xvalidate <- validate |> 
  model.matrix(TenYearCHD ~ . - 1, data = _) |> 
  scale() |> 
  torch_tensor(dtype = torch_float())

Yvalidate <- validate |> 
  dplyr::select(TenYearCHD) |> 
  as.matrix() |>
  as.factor() |> 
  torch_tensor()
Xtesting <- testing |> 
  model.matrix(TenYearCHD ~ . - 1, data = _) |> 
  scale() |> 
  torch_tensor(dtype = torch_float())

Ytesting <- testing |> 
  dplyr::select(TenYearCHD) |> 
  as.matrix() |>
  as.factor() |> 
  torch_tensor(dtype = torch_float())
# model has 3 hidden layers with input 16 and output 2. hideen layers 1 and 2 have dropout to corect '
#overfitting.
modelnn <- nn_module(
  initialize = function(input_size) {
    self$hidden1 <- nn_linear(in_features = input_size, out_features = 16)                           
  
    self$hidden2 <- nn_linear(in_features = 16, 
                              out_features = 8)
    
    self$hidden3 <- nn_linear(in_features = 8,
                               out_features = 4)
    
    self$output <- nn_linear(in_features = 4, 
                             out_features = 2)
                        
    self$activation <- nn_relu()
    self$drop1 <- nn_dropout(0.4)
    self$drop2 <- nn_dropout(0.3)
  }
,
    forward = function(x) {
    x |>  
      self$hidden1() |>  
      self$activation() |>  
      self$drop1() |>
     
      
      self$hidden2() |>
      self$activation() |>
      self$drop2() |>

      
     
      
      self$hidden3() |>
      self$activation() |>
     
 
     
      
      self$output()
    }
)
modelnn <- modelnn |> 
  setup(
    loss = nn_cross_entropy_loss(), # Used for numerical counts
    optimizer = optim_rmsprop
  ) |>
  set_hparams(input_size = ncol(Xtraining)) |> 
  set_opt_hparams(weight_decay = 0.0001)
fitted <- modelnn |> 
  fit(
    data = list(Xtraining, Ytraining),
    #Tuning advises less iterations becasue too many iterations will result into overfitting 
    #maximum iterations attempted is 120 and it ruined the ROC curve and gave little precsiion
    #furthermore, this error is notable in the tree CV plot in the report - depending on the traiing and validation loss, 
    #loss, there will be more error per epoch and this is due to a weak model and can be improved 
    # Hence, the reason for early stopping 
    epochs = 40,
    valid_data = list(Xvalidate, Yvalidate))
    luz_callback_early_stopping(patience = 2, monitor="valid_loss")
      
  


res <- fitted |>  
  predict(Xtesting) |>  
  torch_argmax(dim = 2) |>   
  as.array() 
mean(as.numeric(Ytesting)==res)


pred_class <- as.factor(res)
plot(fitted)






#cross validation of MSE where n = 10, which is useful with high varaince 

cv10 <- cv.glm(heart2, Logistic_regression,K = 10)

cv10$delta[1]  

pred <- fitted |>
  predict(Xtesting) |>
  torch_argmax(dim = 2) |>
  as_array()

#Accuracy of the model, but just projecting the accuracy of determining CHD
#The accuracy is 0.80 to 0.85 which is subpar, and this can be fixed in extensive coding

acc <- mean(pred == as_array(Ytesting)) 
acc
prob <- fitted |>
  predict(Xtesting) |>
  nnf_softmax(dim = 2) |>
  as_array()

y_true <- as.numeric(as_array(Ytesting)) - 1 #balanced out iterations and indexing
# pnull is utilizi=es the mean of the testing data 
#loglik null takes into cosnideration the mean of the testing data 
# loglik_model is the summation of y_true, and this is code is derived 
#by the equation for Mcfadden Psuedo R-Squared which is typically used 
# for classification R-squared.
loglik_model <- sum(log(prob[cbind(1:length(y_true), y_true)]))
p_null <- mean(training$TenYearCHD)

loglik_null <- sum(y_true * log(p_null) +(1-y_true) * log(1-p_null)
)
#R-squared is around 0.89 to 0.93 and this very good, and this varies ebcasue of utilziing 
#testing data and this can be further analyzed 
R2 <- 1 - (loglik_model / loglik_null)
R2

roc_nn <- roc(response = y_true, predictor = prob[,2]) # outputs p class 1 and pclass 2

plot(
  roc_nn,
  col = "#1f77b4",
  lwd = 4,
  legacy.axes = TRUE,
  main = "Neural Network ROC Curve"
)

#smoothing of the ROC curv which wil obviously introduce lack of precision for AUC
# AUC of regular ROC curve without smoothing 
auc(roc_nn)
rocs <- smooth(roc_nn, method = "density") 
#plotting ROC smooth curve, #AUC highest was 0.74
plot(rocs,legacy.axes = TRUE)
auc(rocs)


testing$TenYearCHD <- as.factor(testing$TenYearCHD) #establishing testing of predictor varibale as a qualitative intital




testing <- testing |> 
  mutate(across(where(is.character), as.factor)) 
# changing all data types into caterogircal data types for the tree plot 
#want to implement boosting but chose to do an error graph based on the tree size 

tree.CHD <- tree(TenYearCHD ~., data = testing)
prune <- prune.misclass(tree.CHD, best = 20) #initailizing pruned tree, where this possibly more correction 
plot(prune) #plotting pruned tree 
text(prune)
summary(prune)
summary(tree.CHD)
plot(tree.CHD) #plotting unpruned tree
text(tree.CHD)
rgraph <- rpart(TenYearCHD ~ ., data = testing, method = "class")
rrgraph <- plotcp(rgraph) #plotting CV for every branch point. 


