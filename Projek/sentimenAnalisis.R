
# sentiment analisis
## Library
library(e1071)
library(caret)
library(syuzhet)
library(RTextTools)
library(tm)
library(tidytext)
library(tidyverse)
library(tidymodels)

datanya <- read.csv("data_clean.csv")
glimpse(datanya)
datanya <- datanya %>%
  select(comment, rating)

# menambahkan kelas baru dengan persyaratan rating lebih dari 3 maka bernilai 1 atau good
datanya <- datanya %>% mutate(kelas = ifelse(rating>=3, "1", "0"))

# mengubah rating dan kelas menjadi faktor
datanya$rating <- as.factor(datanya$rating)
datanya$kelas <- as.factor(datanya$kelas)

set.seed(18940)
datanya <- datanya[sample(nrow(datanya)),]
datanya <- datanya[sample(nrow(datanya)),]
glimpse(datanya)



corpus <- Corpus(VectorSource(datanya$comment))
corpus
inspect(corpus[1:10])
corpus_clean <- corpus %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind = "en")) %>%
  tm_map(stripWhitespace)

dtm <- DocumentTermMatrix(corpus_clean)
inspect(dtm[1:10, 1:10])



data_split <- initial_split(datanya)
data_split

data_train <- datanya[1:1868,]
data_test <- datanya[1869:2490,]

dtm_train <- dtm[1:1868,]
dtm_test <- dtm[1869:2490,]

# cc = corpus clean
cc_train <- corpus_clean[1:1868]
cc_test <- corpus_clean[1869:2490]



dim(dtm_train)

# menyeleksi feature sehingga yg diambil adalah kata yang muncul setidaknya 75 kali
rungpuluh_freq <- findFreqTerms(dtm_train,75)
length(rungpuluh_freq)
saveRDS(rungpuluh_freq, "fitur.rds")

# menyesuaikan fitur pada data train dan test dengan fitur yg sudah diseleksi
dtm_train_a <- cc_train %>%
  DocumentTermMatrix(control = list(dictionary = rungpuluh_freq))

dtm_test_a <- cc_test %>%
  DocumentTermMatrix(control = list(dictionary = rungpuluh_freq))

dim(dtm_train_a)
dim(dtm_test_a)


# fungsi untuk mengubah nilai 0 dan 1 menjadi no dan yes
ngubah <- function(x) {
  y <- ifelse(x>0, 1, 0)
  y <- factor(y, levels = c(0,1), labels = c("No","Yes"))
  y
}


train_b <- apply(dtm_train_a, 2, ngubah)
test_b <- apply(dtm_test_a, 2, ngubah)

# membuat model naive bayes untuk 5 rating
classifier <- naiveBayes(train_b, data_train$rating, laplace = 1)

# menyimpan model untuk aplikasi
save(classifier, file = "NaiveBayesClassifier.rda")

# test model naive bayes
prediksi <- predict(classifier, test_b)

# membuat tabel hasil prediksi
table("Prediksi" = prediksi, "Asli" = data_test$rating)

# mengecek akurasi
conf <- confusionMatrix(prediksi, data_test$rating)
conf$overall['Accuracy']




# membuat model naive bayes
classifier2 <- naiveBayes(train_b, data_train$kelas, laplace = 1)

# menyimpan model untuk aplikasi
save(classifier2, file = "NaiveBayesClassifier2.rda")

# test model naive bayes
prediksi2 <- predict(classifier2, test_b)

# membuat tabel hasil prediksi
table("Prediksi" = prediksi2, "Asli" = data_test$kelas)

# mengecek akurasi
conf2 <- confusionMatrix(prediksi2, data_test$kelas)
conf2$overall['Accuracy']





