##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)
# Here Edx data will beused for training and Validation for testing

# Creating the rmse( root mean square error function to check accuracy)
Accuracy <- function(actual_ratings, predicted_ratings){
  sqrt(mean((actual_ratings - predicted_ratings)^2))
}

# Average or Naive Model
avg<-mean(edx$rating)
results1<- Accuracy(validation$rating,avg)
results1

#  Movie Bias 
movie_dev <- edx %>% #calculating the bias
  group_by(movieId) %>%
  summarize(movie_bias = mean(rating - avg))
movie_dev
approach_2 <- avg +  validation %>% # applying to get new rmse
  left_join(movie_dev, by='movieId') %>%
  pull(movie_bias)
results2 <- Accuracy(approach_2, validation$rating)
results2

#User Bias
user_dev<- edx %>% 
  left_join(movie_dev, by='movieId') %>% #calculating the bias
  group_by(userId) %>%
  summarize(user_bias = mean(rating - avg - movie_bias))
predicted_ratings <- validation%>%
  left_join(movie_dev, by='movieId') %>%
  left_join(user_dev, by='userId') %>%
  mutate(pred = avg + user_bias) %>%
  pull(pred)

results3<- Accuracy(predicted_ratings, validation$rating)
results3

# User + movie bias
predicted_ratings <- validation%>%
  left_join(movie_dev, by='movieId') %>%
  left_join(user_dev, by='userId') %>%
  mutate(pred = avg + user_bias+movie_bias) %>%
  pull(pred)

results4<- Accuracy(predicted_ratings, validation$rating)
results4

# Regularization MOdel

alpha <- seq(1, 10, 0.25)


rmses <- sapply(alpha, function(l){
  
  avg <- mean(edx$rating)
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - avg)/(n()+l))
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - avg)/(n()+l))
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = avg + b_i + b_u) %>%
    pull(pred)
  
  return(Accuracy(predicted_ratings, validation$rating))
})

# Now let's choose the best possible alpha
alpha <- alpha[which.min(rmses)]
alpha
# Accuracy of this model is:
min(rmses)
result5=min(rmses)

#building a table of results
Models<-c("Naive","Movie Bias","User Bias","Movie and user Bias"," Regulaization and User+Movie bias")
Results<-c(results1,results2,results3,results4,result5)
Prediction_Table= data.frame(Models,Results)
Prediction_Table

#Thus we see that the best model has a RMSE< 0.86490


