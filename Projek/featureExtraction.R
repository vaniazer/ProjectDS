#Library
library(RTextTools)
library(tm)

features <- readRDS("fitur.rds")

convert_count <- function(x) {
  y <- ifelse(x > 0, 1, 0)
  y <- factor(y, levels = c(0,1), labels = c("No", "Yes"))
  y
} 
create_dtm <- function(data) {
  corpus <- Corpus(VectorSource(data))
  
  corpus_clean <- corpus %>%
    tm_map(content_transformer(tolower)) %>%
    tm_map(removePunctuation) %>%
    tm_map(removeNumbers) %>%
    tm_map(removeWords, stopwords(kind = "en")) %>%
    tm_map(stripWhitespace)
  
  create_dtm <- corpus_clean %>%
    DocumentTermMatrix(control = list(dictionary = features))
  
}

extract_feature <- function(data) {
  apply(create_dtm(data), 2, convert_count)
}

