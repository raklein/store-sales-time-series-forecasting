# Load required packages
library(dplyr)
library(tidyr)
library(lubridate)
library(ggplot2)
library(forecast)
library(caret)
library(prophet)
library(zoo)
library(readr)
library(future.apply)

# Set the path to the input files
path <- "G:/My Drive/GitHub/store-sales-time-series-forecasting"

# Load data
train <- read_csv(file.path(path, "train.csv"))
test <- read_csv(file.path(path, "test.csv"))
stores <- read_csv(file.path(path, "stores.csv"))
oil <- read_csv(file.path(path, "oil.csv"))
holidays_events <- read_csv(file.path(path, "holidays_events.csv"))

# Preprocessing
train$date <- as.Date(train$date)
test$date <- as.Date(test$date)
oil$date <- as.Date(oil$date)
holidays_events$date <- as.Date(holidays_events$date)

# interpolate missing oil values
oil$dcoilwtico <- na.approx(oil$dcoilwtico, na.rm = FALSE)
oil$dcoilwtico <- na.locf(oil$dcoilwtico, na.rm = FALSE, fromLast = TRUE)
oil$dcoilwtico <- na.locf(oil$dcoilwtico, na.rm = FALSE, fromLast = FALSE)

# Rename the date column in the oil data frame to "ds"
oil <- oil %>% rename(ds = date)

# Aggregate holidays_events data
holidays_events_agg <- holidays_events %>%
  mutate(transferred = as.logical(transferred)) %>%
  mutate(transferred = as.numeric(transferred)) %>%
  group_by(date) %>%
  summarize(
    num_holidays = n(),
    num_transferred = sum(transferred, na.rm = TRUE)
  )

# Update the join operations in the train and test preprocessing steps
train <- train %>% 
  left_join(stores, by = "store_nbr") %>%
  left_join(oil, by = c("date" = "ds")) %>%
  left_join(holidays_events_agg, by = "date") %>%
  mutate(
    year = lubridate::year(date),
    month = lubridate::month(date),
    day = lubridate::day(date),
    day_of_week = lubridate::wday(date),
    week_of_year = lubridate::isoweek(date)
  )

test <- test %>%
  left_join(stores, by = "store_nbr") %>%
  left_join(oil, by = c("date" = "ds")) %>%
  left_join(holidays_events_agg, by = "date") %>%
  mutate(
    year = lubridate::year(date),
    month = lubridate::month(date),
    day = lubridate::day(date),
    day_of_week = lubridate::wday(date),
    week_of_year = lubridate::isoweek(date)
  )

# Create a train-validation split
set.seed(123)
#train <- train %>% sample_frac(.01) # Reducing the training set for faster code experimentation, remove for full run
train_indices <- createDataPartition(train$sales, p = 0.8, list = FALSE)
train_data <- train[train_indices, ]
validation_data <- train[-train_indices, ]

### prophet ###

# add parallel processing
plan(multisession)

# Prepare the data for the Prophet model
train_prophet <- train_data %>%
  group_by(date, store_nbr, family) %>%
  summarize(sales = sum(sales)) %>%
  ungroup()

# Rename columns as required by the Prophet model
train_prophet <- train_prophet %>%
  rename(ds = date, y = sales)

unique_stores <- unique(train_data$store_nbr)
unique_families <- unique(train_data$family)
options(future.rng.onMisuse = "ignore")
future::plan(strategy = future::multisession, workers = availableCores, future.seed = TRUE)

# Fit prophet model
fit_predict_prophet <- function(data, future_dates, oil_data) {
  data <- left_join(data, oil_data, by = "ds")
  future_dates <- left_join(future_dates, oil_data, by = "ds")
  
  # Interpolate missing values in the 'dcoilwtico' column
  data$dcoilwtico <- na.approx(data$dcoilwtico, na.rm = FALSE)
  data$dcoilwtico <- na.locf(data$dcoilwtico, na.rm = FALSE, fromLast = TRUE)
  data$dcoilwtico <- na.locf(data$dcoilwtico, na.rm = FALSE, fromLast = FALSE)
  
  future_dates$dcoilwtico <- na.approx(future_dates$dcoilwtico, na.rm = FALSE)
  future_dates$dcoilwtico <- na.locf(future_dates$dcoilwtico, na.rm = FALSE, fromLast = TRUE)
  future_dates$dcoilwtico <- na.locf(future_dates$dcoilwtico, na.rm = FALSE, fromLast = FALSE)
  
  m <- prophet::prophet(daily.seasonality = TRUE) 
  m <- prophet::add_regressor(m, 'dcoilwtico')
  m <- prophet::fit.prophet(m, data) # fit the model after adding the regressor
  
  forecast <- prophet:::predict.prophet(m, future_dates)
  
  return(forecast$yhat)
}


results <- data.frame()
for (store in unique(validation_data$store_nbr)) {
  store_data <- validation_data[validation_data$store_nbr == store, ]
  
  store_results <- do.call(rbind, future_lapply(unique(train$family), function(family) {
    family_data <- store_data[store_data$family == family, ]
    family_data_prophet <- data.frame(ds = family_data$date, y = family_data$sales)
    
    future_dates <- data.frame(ds = seq(min(validation_data$date), max(validation_data$date), by = "day"))
    # Fit prophet model with oil data
    pred_sales <- fit_predict_prophet(family_data_prophet, future_dates, oil)
    
    data.frame(date = future_dates$ds, store_nbr = store, family = family, pred_sales = pred_sales)
  }))
  
  results <- rbind(results, store_results)
}

# Join the predictions with the validation dataset
validation_data <- left_join(validation_data, results, by = c("date", "store_nbr", "family"))

# Ensure non-negative predictions
validation_data$pred_sales <- pmax(validation_data$pred_sales, 0)

# Calculate RMSLE for the Prophet model
rmsle_prophet <- sqrt(mean((log1p(validation_data$pred_sales) - log1p(validation_data$sales))^2, na.rm = TRUE))
cat("RMSLE for Prophet model:", rmsle_prophet, "\n")

# RMSLE for Prophet model: 1.073685 

# predict on test data and prepare submission file
test_prophet <- test %>%
  group_by(date, store_nbr, family) %>%
  ungroup()

test_results <- data.frame()
for (store in unique(test$store_nbr)) {
  store_data <- test[test$store_nbr == store, ]
  
  store_results <- do.call(rbind, future_lapply(unique(train$family), function(family) {
    family_data <- store_data[store_data$family == family, ]
    
    # Get training data for this specific store and family
    family_data_train <- train_prophet[train_prophet$store_nbr == store & train_prophet$family == family, ]
    
    future_dates <- data.frame(ds = seq(min(test$date), max(test$date), by = "day"))
    # Predict on test data with oil data
    pred_sales <- fit_predict_prophet(family_data_train, future_dates, oil)
    
    data.frame(date = future_dates$ds, store_nbr = store, family = family, pred_sales = pred_sales)
  }))
  
  test_results <- rbind(test_results, store_results)
}

# Join the predictions with the test dataset
test <- left_join(test, test_results, by = c("date", "store_nbr", "family"))

# Ensure non-negative predictions
test$pred_sales <- pmax(test$pred_sales, 0)

# Create the submission file
submission <- data.frame(id = test$id, sales = test$pred_sales)

# This submission improved on the original score to 0.50559

# Write the submission file to a CSV
#write.csv(submission, "submission_prophet_oil.csv", row.names = FALSE)
