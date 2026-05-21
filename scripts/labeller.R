# Load libraries
source("scripts/library_loader.R")
source("scripts/utils.R")

# Load the stance detector model
nb_big_nohash <- readRDS("Data/correct_model_no_hashtags.rds")

# Read the data
submissions <- readSubmissions()
comments <- readComments()

# Sanitizing ...
submissions$Content <- str_replace_all(str_replace_all(submissions$Content, "&gt;.*\n", ""), "\n", "")
comments$Content <- str_replace_all(str_replace_all(comments$Content, "&gt;.*\n", ""), "\n", "")

# Get predictions for submissions
submissions_pred <- getPredictions(submissions, contentColumnNumber = which(names(submissions) == "Content"))
submissions$leave_probability <- submissions_pred$leave_prob
submissions$stance <- ifelse(submissions$leave_probability >= 0.75, "Pro-Brexit", 
                             ifelse(submissions$leave_probability >= 0.25, "Neutral", "Against-Brexit"))

# Get predictions for comments
comments_pred <- getPredictions(comments, contentColumnNumber = which(names(comments) == "Content"))
comments$leave_probability <- comments_pred$leave_prob
comments$stance <- ifelse(comments$leave_probability >= 0.75, "Pro-Brexit", 
                          ifelse(comments$leave_probability >= 0.25, "Neutral", "Against-Brexit"))

# Write to CSV
write.csv(submissions, "Data/diffusions_submissions_labeled.csv", row.names = FALSE) # nolint: line_length_linter.
write.csv(comments, "Data/diffusions_comments_labeled.csv", row.names = FALSE)
