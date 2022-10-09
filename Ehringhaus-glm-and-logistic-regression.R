# █▀█ █▀▀ █▀ █▀▀ ▀█▀ █▀
# █▀▄ ██▄ ▄█ ██▄ ░█░ ▄█
# clear console
cat("\014")
# clear global environment
rm(list = ls())
# clear plots
try(dev.off(dev.list()["RStudioGD"]), silent = TRUE)
# clear packages
try(p_unload(p_loaded(), character.only = TRUE), silent = TRUE)
# disables scientific notion for entire R session
options(scipen = 100)
# set seed for reproducibility
set.seed(44)

# █▀█ ▄▀█ █▀▀ █▄▀ ▄▀█ █▀▀ █▀▀ █▀
# █▀▀ █▀█ █▄▄ █░█ █▀█ █▄█ ██▄ ▄█
library(pacman)      # package manager
p_load(tidyverse)    # for ggplot and various other useful packages
p_load(ISLR)         # for the College dataset
p_load(caret)        # for making train/test partitions
p_load(skimr)        # for descriptive statistics, alternative to summary()
p_load(corrplot)     # for creating correlation matrix
p_load(glue)         # for pasting strings together with R code
p_load(janitor)      # for tabyl, alternative to table()
p_load(pscl)         # to calculate pseudo r-squared values
p_load(pROC)         # to visualize ROC curve
p_load(equatiomatic) # extract equation from model


# █░█ █▀▀ █░░ █▀█ █▀▀ █▀█ █▀
# █▀█ ██▄ █▄▄ █▀▀ ██▄ █▀▄ ▄█
# Outputs a string, prettified for printing to the console
pretty_glue <- function(string) {
  border <- strrep('*', 80)
  glue("{border}\n{toString(string)}\n{border}")
}

# Outputs a list containing the modified data and confusion matrix
glm.model.results <- function(data, dep.variable, model, cutoff) {
  data$probs <- predict(model, newdata = data, type = 'response')
  data$preds <- as.factor(ifelse(data$probs >= cutoff, 'Yes', 'No'))
  confusion.matrix <- confusionMatrix(data = data$preds, 
                                      reference = dep.variable, 
                                      positive = 'Yes')
  return(list('confusion.matrix' = confusion.matrix, 'data' = data))
}

# █ █▄░█ ▀█▀ █▀█ █▀█ █▀▄ █░█ █▀▀ ▀█▀ █ █▀█ █▄░█
# █ █░▀█ ░█░ █▀▄ █▄█ █▄▀ █▄█ █▄▄ ░█░ █ █▄█ █░▀█
pretty_glue("Using the College dataset from the ISLR library to build a logistic 
regression model to predict whether a university is private or public.")

# █░░ █▀█ ▄▀█ █▀▄   █▀▄ ▄▀█ ▀█▀ ▄▀█
# █▄▄ █▄█ █▀█ █▄▀   █▄▀ █▀█ ░█░ █▀█
college.data <- as_tibble(ISLR::College)
glimpse(college.data)

# ▀█▀ █▀█ ▄▀█ █ █▄░█   ░░▄▀   ▀█▀ █▀▀ █▀ ▀█▀
# ░█░ █▀▄ █▀█ █ █░▀█   ▄▀░░   ░█░ ██▄ ▄█ ░█░
pretty_glue("Creating a 70% / 30% split")

train.index <- 
  createDataPartition(y = college.data$Private, p = 0.70, list = FALSE)
college.train <- college.data[train.index, ]
college.test <- college.data[-train.index, ]
rm(train.index) # no longer needed
tabyl(college.train, Private)
tabyl(college.test, Private)

pretty_glue("The relative frequency distributions of the train and test set are 
almost equal. Also, note the class imbalance (72.66% Private and 27.34% Public).
The Accuracy Paradox may be something to look out for, which can occur when the 
model finds it expeditious to produce more accurate results simply by predicting
the larger class. Due to the class imbalance in this dataset, precision and
recall will serve as better metrics.")

# █▀▀ ▀▄▀ █▀█ █░░ █▀█ █▀█ ▄▀█ ▀█▀ █▀█ █▀█ █▄█
# ██▄ █░█ █▀▀ █▄▄ █▄█ █▀▄ █▀█ ░█░ █▄█ █▀▄ ░█░
myskim <- skim_with(numeric = sfl(min, max))
myskim(college.train) %>% 
  select(-n_missing, -complete_rate, -starts_with('numeric.p'))

pretty_glue("One-hot encoding converts the dependent variable `Private` into
numeric variables (`PrivateNo` and `PrivateYes`), where 0 and 1 represent
whether or not the school is private. A correlation plot can be used to assess 
which predictors are the most predictive, as variables with highest correlations 
to PrivateNo or PrivateYes will generally serve as good predictors.")

cor(model.matrix(~0+., data = college.train)) %>% 
      corrplot(title = "Correlation Plot\nDependent Variable One-Hot Encoded",
               type = c("lower"),
               mar= c(0, 0, 2, 0),
               method = "circle", 
               insig = "blank", 
               diag = FALSE)

pretty_glue("The two variables with the highest correlations to private and
public schools appear to be `F.Undergrad` (number of fulltime undergraduates) 
and `Outstate` (out-of-state tuition).\n
`F.Undergrad` is positively correlated with public schools. This means public
schools generally have higher numbers of full time undergraduate students. One
possible explanation is that public schools are generally larger, and thus they
would have more full time undergraduate students than private schools.\n
`Outstate` is negatively correlated with public schools. This means public 
schools generally have lower out-of-state tuitions. One possible explanation
is that public schools are generally cheaper, and thus the out-of-state tuition,
too, is cheaper than those of private schools.")

college.train %>% 
  ggplot + 
  aes(x = Outstate, y = F.Undergrad, color = Private, shape = Private) +
  geom_point() +
  labs(title = "Plot #1: Predictive Predictors",
       subtitle = "Note the Separation in the Dependent Variable",
       x = "Out-of-state tuition (US dollars)",
       y = "Full-time undergraduate students (number)")

pretty_glue("The above graph exemplifies the good separation between the 
predictors. Likely, the model will have a harder time at classifying private 
schools with a low out-of-state tuition, as there is overlap with public schools
in this lower-left area of the graph. This could lead to an increase in
false-negatives and/or false-positives when private is misclassified as public
or vice-versa.")

# █▀▄▀█ █▀█ █▀▄ █▀▀ █░░
# █░▀░█ █▄█ █▄▀ ██▄ █▄▄
model.glm <- 
  glm(Private ~ F.Undergrad + Outstate, data = college.train, 
      family = binomial(link = 'logit'))
summary(model.glm)
coef(model.glm)
extract_eq(model.glm, wrap = TRUE, use_coefs = TRUE)

pretty_glue("There is a small, significant p-value for both predictors.\n
The negative coefficient for `F.Undergrad` suggests that the more full-time 
undergrad students a school has, the less likely it is to be private.\n
The positive coefficient for `Outstate` suggests that the higher the out-of-state 
tuition is, the more likely it is to be private.\n
This concurs with the findings from the correlation plot.")

pR2(model.glm)

pretty_glue("McFadden's pseudo r-squared value results in a value between 0 and 1
and approximates the role of r-squared in a linear regression model to assess the
model's goodness of fit. The high value of 0.68 indicates the model fits the data 
well and has good predictive powers.")

# █▀▄▀█ ▄▀█ ▀█▀ █▀█ █ ▀▄▀   █▀▄▀█ █▀▀ ▀█▀ █▀█ █ █▀▀ █▀
# █░▀░█ █▀█ ░█░ █▀▄ █ █░█   █░▀░█ ██▄ ░█░ █▀▄ █ █▄▄ ▄█
results <- glm.model.results(data = college.train, 
                             dep.variable = college.train$Private, 
                             model = model.glm, 
                             cutoff = 0.50)

pretty_glue("The following can be extracted from the confusion matrix:\n
TN: 132  FP: 17 
FN: 16   TP: 380  
Accuracy -> 93.94%
Recall (sensitivity) -> 95.96%
Precision (pos pred value) -> 95.72%
Specificity -> 88.59%\n
Due to the class imbalance, the accuracy of 93.94% can be ignored for the time
being.\n
Positive observations represent private schools and the recall is 95.96%, which
means 95.96% of private schools were detected by the model and 4.04% were not.\n
The precision is 95.72%, which means 95.75% of predicted private schools are
indeed private and 4.25% of predicted private schools are actually public.\n
The specificity is 88.59%, which means 88.59% of predicted public schools are 
indeed public and 11.41% of predicted public schools are actually private.\n
For classifying whether a school is private or public where positive observations
represent private schools, it is more damaging to have false-positives In this
case, a false-positives represents a prediction that a school is private when it 
is, in fact, public. If the goal is to isolate private schools for further 
analysis, then false-positives would skew the later analyses by introducing 
data from public schools. False-negatives, on the other hand, would simply take
away some data points on private schools, but the analysis would be more accurate
without the influence of public school data.")

# ▀█▀ █▀▀ █▀ ▀█▀   █▀ █▀▀ ▀█▀
# ░█░ ██▄ ▄█ ░█░   ▄█ ██▄ ░█░
results <- glm.model.results(data = college.test, 
                             dep.variable = college.test$Private, 
                             model = model.glm, 
                             cutoff = 0.50)
results['confusion.matrix']

pretty_glue("The following can be extracted from the confusion matrix:\n
TN: 55  FP: 8 
FN: 10  TP: 159  
Accuracy -> 92.24% (compared to train set: 93.94%)
Recall (sensitivity) -> 94.08% (compared to train set: 95.96%)
Precision (pos pred value) -> 95.21% (compared to train set: 95.72%)
Specificity -> 87.30% (compared to train set: 88.59%)")

# █▀█ █▀█ █▀▀   █▀▀ █░█ █▀█ █░█ █▀▀
# █▀▄ █▄█ █▄▄   █▄▄ █▄█ █▀▄ ▀▄▀ ██▄
test.roc <- 
  roc(response = results['data'][[1]]$Private, 
      predictor = results['data'][[1]]$probs, 
      plot = TRUE, print.auc = TRUE)
ideal.cutoff <- as.numeric(test.roc$auc)

pretty_glue("An ROC curve tests all the various cutoffs and plot sensitivity and
specificity. The goal is to limit the number of false positives (i.e., increase
specificity and decrease sensitivity) as it is desirable to limit the number of 
public schools that are predicted to be private, which would skew future analyses.
The AUC is 0.970, which will be used as the cutoff below.")

results <- glm.model.results(data = college.test, 
                             dep.variable = college.test$Private, 
                             model = model.glm, 
                             cutoff = ideal.cutoff)
results['confusion.matrix']

pretty_glue("The following can be extracted from the confusion matrix:\n
TN: 62  FP: 1 
FN: 52  TP: 117  
Accuracy -> 77.16%
Recall (sensitivity) -> 69.23%
Precision (pos pred value) -> 99.15%
Specificity -> 98.41%")

pretty_glue("Indeed, the high cutoff of 0.970 decreases the overall accuracy of
the model, but there is only one false-positive that got through. For purposes
of creating a robust model that predicts private schools that are private and
does not let public schools slip through as a false-positive, this cutoff may be
the most desirable. However, on the downside, 52 potential private schools that 
could be included in further analyses have been disregarded as public, even when
they are private. If enough private schools are ultimately included in the dataset,
then this may not be an issue.")

# █▄░█ █▀▀ ▀▄▀ ▀█▀   █▀ ▀█▀ █▀▀ █▀█ █▀
# █░▀█ ██▄ █░█ ░█░   ▄█ ░█░ ██▄ █▀▀ ▄█
pretty_glue("Applying a log-transformation to predictor variables results in 
a clearer separation in the dependent variable.")

college.train %>% 
  ggplot + 
  aes(x = log(Outstate), y = log(F.Undergrad), color = Private, shape = Private) +
  geom_point() +
  labs(title = "Plot #2: Log-Transformed Predictors",
       subtitle = "Even clearer separation",
       x = "log (Out-of-state tuition)",
       y = "log (Full-time undergraduate students)")

model.glm.predictors.ln <- 
  glm(Private ~ log(F.Undergrad) + log(Outstate), data = college.train, 
      family = binomial(link = 'logit'))
summary(model.glm.predictors.ln)
coef(model.glm.predictors.ln)
extract_eq(model.glm.predictors.ln, wrap = TRUE, use_coefs = TRUE)

pR2(model.glm.predictors.ln)

results <- glm.model.results(data = college.test, 
                             dep.variable = college.test$Private, 
                             model = model.glm.predictors.ln, 
                             cutoff = 0.50)
results['confusion.matrix']

roc <- 
  roc(response = results['data'][[1]]$Private, 
      predictor = results['data'][[1]]$probs, 
      plot = TRUE, print.auc = TRUE)
ideal.cutoff <- as.numeric(roc$auc)

results <- glm.model.results(data = college.test, 
                             dep.variable = college.test$Private, 
                             model = model.glm.predictors.ln, 
                             cutoff = ideal.cutoff)
results['confusion.matrix']