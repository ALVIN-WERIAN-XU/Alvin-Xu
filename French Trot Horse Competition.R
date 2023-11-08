library(arrow)
library(dplyr)
library(MASS)
library(glmnet)
horse_race <- read_parquet("D:/Edge Download/horserace.parquet")
horse_race

# Rearrange the dataset in the ascending order
df <- horse_race[order(horse_race$RaceStartTime), ]

# Create the training dataset 
training_data <- df[1:1172292,]
training_data

# Create the test dataset 
test_data <- df[1172293:1200412,]

# Set a random winprobability for each horse within each RaceID
set.seed(123)  # For reproducibility, you can change the seed
training_data <- training_data %>%
  group_by(RaceID) %>%
  mutate(winprobability = runif(n()),  # Generates random values between 0 and 1
         winprobability = winprobability / sum(winprobability)) %>%  # Normalize to sum to 1
  ungroup()

# Transforming categorical data columns into numeric values
training_data$Gender <- ifelse(training_data$Gender == "M", 0, 1)
training_data$HandicapType <- ifelse(training_data$Gender=="Hcp",1,0)

# Use Forward Selection to choose model by AIC criterion
fit0 <- lm(winprobability~1,data=training_data)

fit1 <- lm(winprobability~Gender+HandicapType+FrontShoes+WetnessScale+HindShoes
           +WeightCarried+HorseAge+RacePrizemoney,data=training_data)

fit.forward <- step(fit0,scope=list(lower=winprobability~1,upper=fit1),direction="forward")

summary(fit.forward)

# Use Backward Selection to choose model by AIC criterion
fit.backward <- step(fit1,scope=list(lower=winprobability~1,upper=fit1),direction="backward")
summary(fit.backward)

# Use Stepwise Regression to choose model by AIC criterion
fit.both <- step(fit0,scope=list(lower=winprobability~1,upper=fit1),direction="both")

# From three regressions above, we find appropriate variables for predicting the
# winprobability

fit.ridge <- lm.ridge(winprobability~HorseAge+WeightCarried+RacePrizemoney+HindShoes
                      +FrontShoes+WetnessScale,data=training_data,
                      lambda=seq(0,20,0.1))
plot(fit.ridge)

select(fit.ridge) 

fit.ridge$coef

plot(seq(0,20,0.1),fit.ridge$GCV,xlab= expression(lambda),ylab="GCV")

# Use the model to predict the winprobability in the test dataset
# output <- sum(c(1,a,b,c,d,e,f)*fit.both$coefficients)
# winprobability <- 1-(exp(output)/(1+exp(output)))


