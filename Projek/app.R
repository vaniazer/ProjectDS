# Load library
library(shiny)
library(shinycssloaders)
library(here)
library(vroom)
library(wordcloud2)
library(ggplot2)
library(shinydashboard)
library(dplyr)
library(tidytext)
library(DT)

nbClassifier <- load("NaiveBayesClassifier2.rda")
british <- vroom(here("Data_Review_British_Museum.csv"))
option_tanggal <- unique(british[["written"]])


source("featureExtraction.R")

ui <- dashboardPage(
  
  dashboardHeader(title = "British Museum Review"),
  
  dashboardSidebar(
    
    selectInput(
                  inputId = "selectDate",
                  label =  "PILIH TANGGAL",
                  choices = option_tanggal,
                  multiple = TRUE,
                  selected = NULL
    ),
    fluidPage(
      hr(),
      helpText(
        "Data review tempat wisata British Museum di London yang diambil dengan scrapping dari website ",
        a("Tripadvisor", href = "https://www.tripadvisor.com/Attraction_Review-g186338-d187555-Reviews-The_British_Museum-London_England.html#REVIEWS"),
        ". Mohon tunggu beberapa saat."
      ),
      hr(),
      helpText(
        "Review British Museum di London yang di-scrape akan di klasifikasikan dengan Naive Bayes"
      ),
      hr(),
      helpText(
        "Author : Rifka Canalisa Rahayu (123180062) dan Vania Zerlinda (123180086)" 
      ),
      hr(),
      helpText(
        "Peringatan: Mungkin terjadi lost connection saat scraping data. Refresh halaman jika terjadi error.", style = "color:#d9534f"
      )
    )
  ),
  
  dashboardBody(
    fluidRow(
      valueBoxOutput("total_review"),
      valueBoxOutput("positif_review"),
      valueBoxOutput("negatif_review")
    ),
    fluidRow(
      box(
        title = "British Review + Klasifikasi Sentiment",
        solidHeader = T,
        width = 12,
        collapsible = T,
        div(DT::dataTableOutput("table_review") %>% withSpinner(color="#1167b1"), style = "font-size: 70%;")
      ),
    ),
    fluidRow(
      box(title = "Wordcloud",
          solidHeader = T,
          width = 6,
          collapsible = T,
          wordcloud2Output("wordcloud") %>% withSpinner(color="#1167b1")
      ),
      box(title = "Word Count",
          solidHeader = T,
          width = 6,
          collapsible = T,
          plotOutput("word_count") %>% withSpinner(color="#1167b1")
      )
    ),
    fluidRow(
      box(title = "Sentimen Negatif / Positif yang Paling Umum",
          solidHeader = T,
          width = 12,
          collapsible = T,
          plotOutput("kontribusi_sentimen") %>% withSpinner(color="#1167b1")
      )
    )
  )
)

server <- function(input, output) {
  #Masukan Data Tanggal yang dipilih
  data <- reactive({
    british %>%
      filter(written %in% input$selectDate)

  })
  #Data disimpan ke dataNB
  dataNB <- reactive({
    reviews <- data()$comment
    withProgress({
      setProgress(message = "Ekstrak Fitur...")
      newData <- extract_feature(reviews)
    })
    withProgress({
      setProgress(message = "Klasifikasi Sentiment...")
      pred <- predict(get(nbClassifier), newData)
    })
    #data disimpan satu frame
    data.frame(Title = data()$title, Review = data()$comment, Trip = data()$trip, Writer = data()$writer, Written = data()$written, Rating = data()$rating,  Prediksi = as.factor(pred), stringsAsFactors = FALSE)
  })
  
  dataWord <- reactive({
    v <- sort(colSums(as.matrix(create_dtm(data()$comment))), decreasing = TRUE)
    data.frame(Kata=names(v), Jumlah=as.integer(v), row.names=NULL, stringsAsFactors = FALSE) %>%
      filter(Jumlah > 0)
  })
  #data ditampilkan dengan bentuk tabel
  output$table_review <- renderDataTable(datatable({
    dataNB()
  }))
  #total review
  output$total_review <- renderValueBox({
    valueBox(
      "Total", 
      paste0(nrow(dataNB()), " review"),
      icon = icon("pen"),
      color = "blue"
    )
  })
  
  output$positif_review <- renderValueBox({
    valueBox(
      "Positif", 
      paste0(nrow(dataNB() %>% filter(Prediksi == "1")), " wisatawan merasa senang"),
      icon = icon("smile"),
      color = "green")
  })
  
  output$negatif_review <- renderValueBox({
    valueBox(
      "Negatif",
      paste0(nrow(dataNB() %>% filter(Prediksi == "0")), " wisatawan merasa tidak senang"), 
      icon = icon("frown"),
      color = "red")
  })
  
  output$wordcloud <- renderWordcloud2({
    wordcloud2(top_n(dataWord(), 50, Jumlah))
  })
  
  output$word_count <- renderPlot({
    countedWord <- dataWord() %>%
      top_n(10, Jumlah) %>%
      mutate(Kata = reorder(Kata, Jumlah))
    
    ggplot(countedWord, aes(Kata, Jumlah, fill = -Jumlah)) +
      geom_col() +
      guides(fill = FALSE) +
      theme_minimal()+
      labs(x = NULL, y = "Word Count") +
      ggtitle("Most Frequent Words") +
      coord_flip()
  })
  
  output$kontribusi_sentimen <- renderPlot({
    sentiments <- dataWord() %>% 
      inner_join(get_sentiments("bing"), by = c("Kata" = "word")) 
    
    positive <- sentiments %>% filter(sentiment == "positive") %>% top_n(5) 
    negative <- sentiments %>% filter(sentiment == "negative") %>% top_n(5)
    sentiments <- rbind(positive, negative)
    
    sentiments <- sentiments %>%
      mutate(Jumlah=ifelse(sentiment =="negative", -Jumlah, Jumlah))%>%
      mutate(Kata = reorder(Kata, Jumlah))
    
    ggplot(sentiments, aes(Kata, Jumlah, fill=sentiment))+
      geom_bar(stat = "identity")+
      theme(axis.text.x = element_text(angle = 90, hjust = 1))+
      ylab("Kontibusi Sentimen")
  })
}

shinyApp(ui, server)