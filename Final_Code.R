#set seed
set.seed(55411)
#set wd
setwd("C:/Physik/Data Science/Capstone Project")
#load review data
#review_data <- readRDS("review_flat.rds")
library(jsonlite)
file_review <- "yelp_academic_dataset_review.json"
document_review <- fromJSON(file_review)
review_data <- stream_in(file(document_review))
review_data <- flatten(review_data)
saveRDS(review_data, "review_flat.rds")
#lexicon approach
#load up word polarity list and format it
afinn_list <- read.delim(file='AFINN/AFINN-111.txt', header=FALSE, stringsAsFactors=FALSE)
names(afinn_list) <- c('word', 'score')
afinn_list$word <- tolower(afinn_list$word)

#categorize words as very negative to very positive and add some movie-specific words
vNegTerms <- afinn_list$word[afinn_list$score==-5 | afinn_list$score==-4]
negTerms <- c(afinn_list$word[afinn_list$score==-3 | afinn_list$score==-2 | afinn_list$score==-1], "second-rate", "moronic", "third-rate", "flawed", "juvenile", "boring", "distasteful", "ordinary", "disgusting", "senseless", "static", "brutal", "confused", "disappointing", "bloody", "silly", "tired", "predictable", "stupid", "uninteresting", "trite", "uneven", "outdated", "dreadful", "bland")
posTerms <- c(afinn_list$word[afinn_list$score==3 | afinn_list$score==2 | afinn_list$score==1], "first-rate", "insightful", "clever", "charming", "comical", "charismatic", "enjoyable", "absorbing", "sensitive", "intriguing", "powerful", "pleasant", "surprising", "thought-provoking", "imaginative", "unpretentious")
vPosTerms <- c(afinn_list$word[afinn_list$score==5 | afinn_list$score==4], "uproarious", "riveting", "fascinating", "dazzling", "legendary")

#function to calculate number of words in each category within a sentence
data <- data.frame(text=review_data[,5], stringsAsFactors=FALSE)
sentences <- as.list(data[,1])
library(plyr)
library(stringr)
sentimentScore <- function(sentences, vNegTerms, negTerms, posTerms, vPosTerms){
  final_scores <- matrix('', 0, 5)
  scores <- laply(sentences, function(sentence, vNegTerms, negTerms, posTerms, vPosTerms){
    initial_sentence <- sentence
    #remove unnecessary characters and split up by word 
    sentence <- gsub('[[:punct:]]', '', sentence)
    sentence <- gsub('[[:cntrl:]]', '', sentence)
    sentence <- gsub('\\d+', '', sentence)
    sentence <- tolower(sentence)
    wordList <- str_split(sentence, '\\s+')
    words <- unlist(wordList)
    #build vector with matches between sentence and each category
    vPosMatches <- match(words, vPosTerms)
    posMatches <- match(words, posTerms)
    vNegMatches <- match(words, vNegTerms)
    negMatches <- match(words, negTerms)
    #sum up number of words in each category
    vPosMatches <- sum(!is.na(vPosMatches))
    posMatches <- sum(!is.na(posMatches))
    vNegMatches <- sum(!is.na(vNegMatches))
    negMatches <- sum(!is.na(negMatches))
    score <- c(vNegMatches, negMatches, posMatches, vPosMatches)
    #add row to scores table
    newrow <- c(initial_sentence, score)
    final_scores <- rbind(final_scores, newrow)
    return(final_scores)
  }, vNegTerms, negTerms, posTerms, vPosTerms)
  return(scores)
}    

#counted the pos and neg words
results <- as.data.frame(sentimentScore(sentences, vNegTerms, negTerms, posTerms, vPosTerms))
results$stars <- review_data$stars
results <- within(results, rating <- "positive")
results[(results$stars<4), "rating"] <- "neutral"
results[(results$stars<2), "rating"] <- "negative"
results$rating <- as.factor(results$rating)
#create training and testing partition
library(caret)
inTrain <- createDataPartition(y=review_data$stars, p=0.5, list=FALSE)
training <- results[inTrain,]
testing <- results[-inTrain,]
#train the model
library(e1071)
classifier <- naiveBayes(training, training$rating)
#predict if reviews of testing are positive
pred <- predict(classifier, testing[, 2:5])
print(confusionMatrix(pred, testing[,7]), digits=4)
testing$prediction <- pred

#TestData (positive prediction and 1 star rating)
data_final <- subset(testing, stars < 2)
data_final <- subset(data_final, prediction == "positive")
#CompareData - sample (positive prediction and 5 star rating)
data_compare <- subset(testing, stars == 5) 
data_compare <- subset(data_compare, prediction == "positive")
data_compare <- data_compare[sample(nrow(data_compare), 282190/5), ]
#saveRDS(data_final, "data_final.rds")
#saveRDS(data_compare, "data_compare.rds")
#data_final <- readRDS("data_final.rds")
#data_compare <- readRDS("data_compare.rds")

#create the Term Matrix
#text Data.Frame
library(tm)
data <- data_final[,1]
data2 <- data_compare[,1]
data <- data.frame(text=data, stringsAsFactors=FALSE)
data2 <- data.frame(text=data2, stringsAsFactors=FALSE)
data_source <- DataframeSource(data)
data_source2 <- DataframeSource(data2)
docss <- Corpus(data_source)
docss2 <- Corpus(data_source2)
#preprocessing
docs <- tm_map(docss, removePunctuation)   # *Removing punctuation:*    
docs <- tm_map(docs, removeNumbers)      # *Removing numbers:*    
docs <- tm_map(docs, tolower)   # *Converting to lowercase:*    
docs <- tm_map(docs, removeWords, stopwords("english"))   # *Removing "stopwords" 
library(SnowballC)
docs <- tm_map(docs, stemDocument)   # *Removing common word endings* (e.g., "ing", "es")   
docs <- tm_map(docs, stripWhitespace)   # *Stripping whitespace   
docs <- tm_map(docs, PlainTextDocument) 

docs2 <- tm_map(docss2, removePunctuation)   # *Removing punctuation:*    
docs2 <- tm_map(docs2, removeNumbers)      # *Removing numbers:*    
docs2 <- tm_map(docs2, tolower)   # *Converting to lowercase:*    
docs2 <- tm_map(docs2, removeWords, stopwords("english"))   # *Removing "stopwords" 
docs2 <- tm_map(docs2, stemDocument)   # *Removing common word endings* (e.g., "ing", "es")   
docs2 <- tm_map(docs2, stripWhitespace)   # *Stripping whitespace   
docs2 <- tm_map(docs2, PlainTextDocument) 

#DocumentTermMatrix
dtm <- DocumentTermMatrix(docs)
dtms <- removeSparseTerms(dtm, 0.93)
dtm2 <- DocumentTermMatrix(docs2)
dtms2 <- removeSparseTerms(dtm2, 0.93)

#list of most frequent words
freqr <- colSums(as.matrix(dtms))
order <- sort(freqr, decreasing=TRUE)
order[1:50]

freqr2 <- colSums(as.matrix(dtms2))
order2 <- sort(freqr2, decreasing=TRUE)
order2[1:50]


#find associated words
mostfreq <- findFreqTerms(dtms, lowfreq=15000)
assocs <- findAssocs(dtms, mostfreq, 0.1)
mostfreq2 <- findFreqTerms(dtms2, lowfreq=15000)
assocs2 <- findAssocs(dtms2, mostfreq, 0.15)

#saveRDS(dtm, "dtm.rds")
#saveRDS(dtm2, "dtm2.rds")
#dtm <- readRDS("dtm.rds")
#dtm2 <- readRDS("dtm2.rds")

#cluster associated terms
f <- matrix (0, ncol=ncol(dtms), nrow=ncol(dtms))
colnames (f) <- colnames(dtms)
rownames (f) <- colnames(dtms)
for (i in colnames(dtms)) {
  ff <- findAssocs(dtms,i,0)
  for  (j in 1:length(ff[[1]])) {
    n <- names(ff[[1]])[j]
    f[n,i]= as.numeric(ff[[1]])[j]
  }
}
fd <- as.dist(f) # calc distance matrix
fd <- dist(scale(as.matrix(f)))
fit<-hclust(fd, method="ward.D") # plot dendrogram
plot(fit)
rect.hclust(fit, k=10)

f2 <- matrix (0, ncol=ncol(dtms2), nrow=ncol(dtms2))
colnames (f2) <- colnames(dtms2)
rownames (f2) <- colnames(dtms2)
for (i in colnames(dtms2)) {
  ff2 <- findAssocs(dtms2,i,0)
  for  (j in 1:length(ff2[[1]])) {
    n2 <- names(ff2[[1]])[j]
    f2[n2,i]= as.numeric(ff2[[1]])[j]
  }
}
fd2 <- as.dist(f2) # calc distance matrix
fd2 <- dist(scale(as.matrix(f2)))
fit2<-hclust(fd2, method="ward.D") # plot dendrogram
plot(fit2)
rect.hclust(fit2, k=10)

