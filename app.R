library(tidyverse)
library(shiny)
library(shinydashboard)
library(DT)
library(factoextra)
library(caret)
library(rpart)
library(randomForest)
library(neuralnet)
library(matrixStats)
library(plotly)

# load data
data      <- read_csv("train.csv")
dataNames <- names(data)
dataMeans <- colMeans(data)
dataSds   <- colSds(data %>% as.matrix)
n         <- nrow(data)
p         <- ncol(data)

# variable names with response variable first
rNames    <- c(dataNames[p], dataNames[1:(p-1)])

# summarize data
sumData <- data %>% 
  gather(1:82, key="Variable", value="Value") %>%
  group_by(Variable) %>%
  summarize(Mean=mean(Value), 
            SD=sd(Value), 
            Var=var(Value),
            Median=median(Value),
            Min=min(Value),
            Max=max(Value)) %>%
  mutate_at(2:7, round, 2)

##### PCA #####
# do PCA of scaled data (minus response)
pcaData  <- data %>% 
  select(-critical_temp) %>%
  sapply(FUN=scale) %>% 
  as.matrix() %>%
  prcomp()

# calculate some PCA statistics
cutoff     <- 0.95
pv         <- pcaData$sdev^2
cv         <- cumsum(pv)/sum(pv)
critP      <- which.min(cv < cutoff)

##### modeling #####
# test, training data split
pData <- pcaData$x[,1:critP] %>%
  as.data.frame() %>%
  mutate(critical_temp = data$critical_temp)

# build pca data, training and test subsets
set.seed(123)
trainSplit <- 0.8
trainI     <- sample(1:n, size=n*trainSplit)
pData      <- pData
trainData  <- pData[trainI,]
testData   <- pData[-trainI,]

##specify modeling control
myControl <- trainControl(method="cv", number=10)

## Note: Commenting out default models to help app load efficiently!!

## do linear model
# lmFit <- train(critical_temp ~ .,
#                data=trainData,
#                trControl=myControl,
#                method="lmStepAIC",
#                trace=F)
# save(lmFit, file="lmFit.robject")
load("lmFit.robject")

## do regression tree model
# treeFit <- train(critical_temp ~ .,
#                  data=trainData,
#                  trControl=myControl,
#                  method="rpart",
#                  tuneGrid=expand.grid(
#                    cp=10^(seq(-10,-3,by=0.1))
#                  ))
# save(treeFit, file="treeFit.robject")
load("treeFit.robject") # loading default model for efficiency

## do random forest model
## mtry determined through tuneGrid search
# rfFit  <- randomForest(critical_temp ~ ., data=trainData, mtry=6)
# save(rfFit, file="rfFit.robject")
load("rfFit.robject") # loading default model for efficiency

## do neural net model
# DVs     <- trainData %>% select(-critical_temp) %>% names %>% paste(collapse=" + ")
# formula <- paste0("critical_temp ~ ", DVs)
# nnFit   <- neuralnet(formula, data=trainData)
# save(nnFit, file="nnFit.robject")
load("nnFit.robject")

# get function to calculate RMSE for model fits
getRMSE <- function(fit, testData, resp){
  respCol <- which(names(testData) == resp)
  pred    <- predict(fit, testData[,-respCol])
  actual  <- testData[,respCol] %>% unlist() # in case tibble
  sqrt(mean((pred-actual)^2))
}

# get RMSEs
lmRMSE   <- getRMSE(lmFit, testData, "critical_temp") 
treeRMSE <- getRMSE(treeFit, testData, "critical_temp")
rfRMSE   <- getRMSE(rfFit, testData, "critical_temp")
nnRMSE   <- getRMSE(nnFit, testData, "critical_temp")

##### other #####
## code for numerous numeric inputs
## Because hand-coding 81 numeric inputs for 
##   that one prediction option would otherwise be tedious
##   will just import the text dump as an HTML file in
##   the relevant section of the code 
##   (see Prediction subtab of the Modeling tab)
# htmlNumInputs <- as.character(p-1)
# for(i in 1:(p-1)){
#   htmlNumInputs[i] <- numericInput(paste0(dataNames[i],"_NI"),
#                                           label=dataNames[i],
#                                           value=dataMeans[i] %>% round()) %>% 
#                       as.character()
# }
# htmlNumInputs <- paste(htmlNumInputs, sep="", collapse="")
# write(htmlNumInputs, file="manyNumericInputs.html")

##### UI #####
ui <- dashboardPage(skin="green",
  dashboardHeader(title="ST558 Project 3"),
  
  # icon source: https://fontawesome.com/icons?d=gallery
  dashboardSidebar(sidebarMenu(
    menuItem("About", tabName="about", icon=icon("sticky-note")),
    menuItem("Data", tabName="data", icon=icon("table")),
    menuItem("Investigate", tabName="invest", icon=icon("binoculars")),
    menuItem("Modeling", tabName="model", icon=icon("desktop"))
  )),
  
  dashboardBody(
    tabItems(
      # ABOUT TAB
      tabItem(tabName="about",
        fluidPage(
          h1("Modeling Critical Temperatures in Superconductors"),
          "By:   Peter Scarbrough",
          br(),
          "Date: November 26, 2019",
          h3("Introduction"),
          "Superconductors are materials with special physical properties of great research interest. The key feature of all superconductors is that as they reach their 'critical temperature', their electrical resistance drops to zero. In addition to becoming an extremely efficient conductor of electricity, the material resists degradation from sustained electrical currents and requires no additional power input to sustain the electrical currents currently in motion. The transition of a material to a superconducting state requires a physical state transition that ejects the local magnetic field. The materials capable of doing this and an understanding of their relationship to their critical temperature is known to be a quantum mechanical effect; however, the details governing superconductor physics are otherwise poorly understood. Given the potential applications to engineering, superconductivity is an important area of current physics research.",
          br(""),
          "Large amounts of high-dimensional chemical data and complex quantum mechanical effects make empirical advances in superconductor research generally slow and incremental. However, machine learning has the potential to make insights through tools that leverage inherent data complexity to power sophisticated models for prediction. In this study, high dimensional chemical data (containing over 80 features) from over 20,000 superconductor chemicals is used to predict the critical temperature of superconductors. The goal is to identify model strategies that provide the best predictions for critical temperature based off the available chemical data.",
          br(""),
          "In general, superconductors are categorized as belonging to one of two types: with Type I having low critical temperatures and Type II having higher critical temperatures. These materials also tend to broadly differ in terms of their physical properties and specific manifestation of their superconductivity state. The data set included in this research includes superconductors of both types. Therefore one of the additional goals of this research will be to informally observe whether modeling is able to provide a consistent data-driven explanation of the superconductivity phenomenon between the two superconductor types.",
          h3("Data"),
          uiOutput("urlUCI"),
          h3("App Features"),
          "The purpose of this app is to allow the exploration and dynamic visualization of the data with the primary goal of being able to showcase, dynamically interact with, and evaluate different models of superconductor critical temperature. The app itself is broken into 4 tabs. The purpose of the current tab is to provide a context for this app and an overview of its features and findings.",
          br(""),
          "The 'Data' tab will allow one to explore and examine the entire data set or produce any of a selected numbef of summary statistics from the data. The 'Investigate' tab will allow one to examine histograms for each of the variables in the data set or inspect bivariate scatterplots from any of the selected variables. The charts in this section may be clicked on, scaled, and saved as desired. The 'Investigate' tab also contains the ability to investigate the structure of the data through Principal Component Analysis (PCA). Here, one can see how efficiently principal components are able to summarize the variance in the data set and visually inspect various PC diagnostics which include a screeplot and biplot. The 'Model' tab includes information and opportunities to dynamically interact with and compare various machine learning methods that were used to model superconductor critical temperature. Linear and regression tree model input may be modified by the user to compare their custom models to the final models. This section also includes the ability to produce predictions of critical temperature based on a select model and given an input data set. The input data set can include the full data set, a subset of the data, or the user may input specific data if they wish.",
          h3("Modeling"),
          "Four supervised learning methods were used in this study: stepwise AIC-selected linear regression, regression tree, random forest, and neural net modeling. Due to the number of available predictors in the study (81), dimensional reduction of the data was performed using PCA. A smaller number of predictors was selected to build the indicated models that summarized 95% of the variance in the total data set. This included 17 principal components. Models were compared and evaluated based on their root mean squared error (RMSE) of prediction versus actual values of the response variable (critical temperature). Lower RMSE was interpreted as indicating better model fit. Prior to modeling fitting data were randomly split 80/20 into training and test sets, respectively. The training data was used to perform all modeling selection steps. The final test (hold-out) set was used to calculate final model RMSE and evaluate model performance.",
          br(""),
          "In the modeling tab, one may build different linear and regression tree models by choosing to include different numbers of principal components than the default number chosen. To speed up performance of this app, the random forest and neural net models were calculated separately and then loaded into this app for evaluation. Linear modeling selection was performed using both forward- and backward-selection and using AIC as a selection criterion. Both linear modeling and regression tree modeling were done using the 'caret' package in R to train their respective models. For these models, 10-fold cross-validation was performed to help control against over-fitting of the data.",
          br(""),
          "For random forest modeling an interative search of the hyperparameter space was performed to find an optimal value for the tuning parameters, the number of random parameters to include in the regression tree model (mtry). Modeling was performed using the 'randomForest' function from the 'randomForest' package. For neural net modeling, the 'neuralnet' library was used and the model was built using the default arguments from the 'neuralnet' function in this package.",
          h3("Results"),
          "The random forest model of critical temperature performed the best, having the lowest RMSE of all the models tested. A plot of predicted versus actual values also demonstrated fairly consistent variation in random forest model performance across the entire range of critical temperature values, which is another desirable model feature and reason to prefer the random forest model over the results from other methods. In contrast, linear and neural net models appeared to otherwise struggle to provide a consistent data-driven framework to predict critical temperatures between the two types of superconductor materials: low versus high temperature types (see: Prediction subtab of the Modeling tab).",
          br(""),
          "Random forest model performance provides a subtle indication that a unified model to explain critical temperature variation between the two different types of superconductor material likely exists. However, the linear and neural net modeling provide an indication that attempting to understand this model in terms of direct action from the available chemical features is likely not-straight-forward. It seems that understanding the interactions between some subset of these chemical features would be required to more fully understand how superconductors take on specific values of critical temperatures."
        )
      ),
      # DATA TAB
      tabItem(tabName="data",
        tabsetPanel(
          tabPanel("Exploration",
            fluidPage(
              fluidRow(
                h1("Data Exploration"),
                "Explore the full data set.",
                br("")
              ),
              sidebarPanel(
                checkboxInput("dataDisplayDefaultVars",
                              label="Select Custom Variables to Display ",
                              value=0),
                actionButton("dataSaveDataButton",
                             label="Save Data"),
                br(""),
                span(textOutput("dataSavedMsg"), style="color:red"),
                conditionalPanel(condition = "input.dataDisplayDefaultVars == 1",
                  checkboxGroupInput("dataVarsToDisplay",
                                     label="Select Variables to Display",
                                     choices=c(dataNames[p], dataNames[1:(p-1)]),
                                     selected=dataNames)
                )
              ),
              mainPanel(
                dataTableOutput("dataFulldataDatatable"),
                style = "overflow-x: scroll;"
              )
            )
          ),
          tabPanel("Summary",
            fluidPage(
              fluidRow(
                h1("Data Summaries"),
                "Get and save statistical summaries of chemical data."
              ),
              br(),
              sidebarPanel(
                checkboxInput("dataCheckUseDefault",
                              label="Use Default Statistics",
                              value=T
                ),
                conditionalPanel(condition = "input.dataCheckUseDefault == 0",
                                 checkboxGroupInput("dataCheckStats",
                                                    label="Select Statistics: ",
                                                    choices=c("Mean",
                                                              "SD",
                                                              "Var",
                                                              "Median",
                                                              "Min",
                                                              "Max"
                                                    ),
                                                    selected=c("Mean",
                                                               "SD",
                                                               "Median",
                                                               "Min",
                                                               "Max")
                                 )
                ),
                actionButton("dataSaveSummary",
                             label="Save Summary"),
                br(""),
                span(textOutput("dataSavedStatsMsg"), style="color:red")
              ),
              mainPanel(
                dataTableOutput("dataDatatable"),
                style = "overflow-x: scroll;"
              )
            )
          )
        )
      ),
      # INVESTIGATE TAB
      tabItem(tabName="invest",
        tabsetPanel(
          tabPanel("Distributions",
            fluidPage(
              fluidRow(
                h1("Investigate Distributions"),
                "Use scatterplots and histograms to investigate data distributions.",
                br(""),
                tags$i("Note: Plots are made using the `plotly` library in R and may be slow to load. Feel free to manipulate plots and save your results by clicking the image.")
              ),
              br(),
              sidebarPanel(
                selectInput("investDistributionPlotType",
                  label="Select Plot Type:",
                  choices=c("Histogram", 
                            "Scatterplot"),
                  selected="Histogram"
                ),
                conditionalPanel(condition = "input.investDistributionPlotType == 'Scatterplot'",
                  selectInput("investDistributionScatterVar1",
                              label="Select First Variable:",
                              choices=rNames,
                              selected=rNames[1]
                  ),
                  selectInput("investDistributionScatterVar2",
                              label="Select Second Variable:",
                              choices=rNames,
                              selected=rNames[2]
                  )
                ),
                conditionalPanel(condition = "input.investDistributionPlotType == 'Histogram'",
                  selectInput("investDistributionHistogramVar",
                              label="Select Variable:",
                              choices=rNames,
                              selected=rNames[1]
                  )
                )
              ),
              mainPanel(
                plotlyOutput("investDistributionPlot")
              )
            )
          ),
          tabPanel("Structure",
            fluidPage(
              fluidRow(
                h1("Investigate Data Structure"),
                "Observe the results of Principal Component Analysis (PCA). Use biplots to investigate how pairwise principal component (PC) relationships discriminate between all the different dataset variables. Use the cumulative variance plot to discover how many principal components (PCs) are required to include the indicated proportion of data variance.",
                br(""),
                conditionalPanel(condition = "input.investStructurePlotType == 'Biplot'",
                  tags$i("Note: The biplot may take some time to load. It is made using the `plotly` library in R. Feel free to click, resize, crop, and save the image by clicking on it.")
                )  
              ),
              br(),
              sidebarPanel(
                selectInput("investStructurePlotType",
                            label="Select Plot to Show: ",
                            choices=c("Cumulative Variance",
                                      "Screeplot",
                                      "Biplot"),
                            selected="Cumulative Variance"
                ),
                conditionalPanel(
                  condition = "input.investStructurePlotType == 'Cumulative Variance'",
                  sliderInput("investStructurecVPC",
                              label="Select PC:",
                              min=1,
                              max=(p-1),
                              value=20,
                              step=1
                  )
                ),
                conditionalPanel(
                  condition = "input.investStructurePlotType == 'Biplot'",
                  checkboxInput("investBiplotCheckbox",
                                label="Hide Variable Names",
                                value=1),
                  sliderInput("investStructureBPPC1",
                              label="Select 1st PC for Biplot",
                              min=1,
                              max=(p-1),
                              value=1,
                              step=1
                  ),
                  sliderInput("investStructureBPPC2",
                              label="Select 2nd PC for Biplot",
                              min=1,
                              max=(p-1),
                              value=2,
                              step=1
                  )
                )
              ),
              mainPanel(
                conditionalPanel(
                  condition = "input.investStructurePlotType == 'Cumulative Variance'",
                  plotOutput("pcaCumvarplot")
                ),
                conditionalPanel(
                  condition = "input.investStructurePlotType == `Screeplot`",
                  plotOutput("pcaScreeplot")
                ),
                conditionalPanel(
                  condition = "input.investStructurePlotType == 'Biplot'",
                  plotlyOutput("pcaBiplot")
                )
              )
            ) 
          )
        )
      ),    
      # MODELING TAB
      tabItem(tabName="model",
        tabsetPanel(
          # LINEAER TAB
          tabPanel("Linear",
            fluidPage(
              fluidRow(
                h1("Linear Modeling"),
                "PCA identified an optimal number of principal components (17) to summarize at least 95% of the total variance of the data set. These were the predictors used for all subsequent modeling steps. The linear model considered all direct effects of the predictors but no interactions. The final terms to include in the model was determined using both forward- and backward-stepwise selection. Models were compared using the AIC fit criterion and 10-fold cross-validation was used to help limit the effects of selection bias in modeling fitting. The final linear model was compared to other machine learning methods by calculating and comparing RMSE values.",
                br(""),
                "Linear models attempted to fit the optimal linear combination of predictors (x) to minimize the sum of squared errors in the predicted versus actual response values (y). In other words, linear modeling was attempting to optimize the following equation for the ith observation across j predictors: ",
                withMathJax(
                  '$$y_i = \\beta_1x_{1i}+\\beta_2x_{2i}+...+\\beta_jx_{ji}$$'
                ),
                "For efficiency, the default model was calculated, saved, and loaded into this application (i.e. the model including 17 principal components). However, one may manipulate the custom model controls by adjusting the number of principal components to be included in the linear stepwise selection model process. The custom model can be compared to the default linear model by comparing the reported RMSE on the right hand side of this page.",
                br("")
              ),
              sidebarPanel(
                h2("Build a Custom Model"),
                br(),
                sliderInput("modelLinearSlider",
                            label="Select PCs to include in model: ",
                            min=1,
                            max=(p-1),
                            value=critP),
                tags$i("Caution: Large models may take long to run."),
                br(),
                tags$i("Selecting less than 20 PCs is recommended."),
                br(""),
                actionButton("modelLinearButton",
                             label="Run Custom Model")
              ),
              mainPanel(
                h2("Modeling Results"),
                textOutput("modelLinearDefaultRMSE"),
                textOutput("modelLinearCustomRMSE")
              )
            )
          ),
          # REGRESSION TREE TAB
          tabPanel("Regression Tree",
            fluidPage(
              fluidRow(
                h1("Regression Tree Modeling"),
                "The regression tree model was fit using the smallest number of principal components (17) to include 95% of the variance of all 81 predictors. The regression tree was pruned using cost-complexity pruning as implemented by the `rpart` package in R. The optimal tuning parameter was selected using 10-fold cross validation, as managed by the `caret` package in R. For efficiency, the optimal model was previously calculated, saved, and loaded into this application. Users may create their own custom models to compare to the final regression tree model by varying modeling parameters on the left-hand side of the page and compare to the default root mean square error (RMSE) on the right-hand side of the page.",
                br("")
              ),
              sidebarPanel(
                h2("Custom Modeling"),
                br(),
                sliderInput("modelTreeSlider",
                            label="Select PCs to include in model: ",
                            min=1,
                            max=(p-1),
                            value=critP),
                tags$i("Caution: Large models may take long to run."),
                br(),
                tags$i("Selecting less than 20 PCs is recommended."),
                br(""),
                sliderInput("modelTreeCPSlider",
                            label="Select Cost-Complexity Pruning Parameter: 10^(x) ",
                            min=-10,
                            max=-3,
                            step=0.1,
                            value=-7),
                br(""),
                actionButton("modelTreeButton",
                             label="Run Custom Model")                
              ),
              mainPanel(
                h2("Modeling Results"),
                textOutput("modelTreeDefaultRMSE"),
                textOutput("modelTreeCustomRMSE")
              )
            )
          ),
          # ADVANCED MODELING TAB
          tabPanel("Advanced",
            fluidPage(
              fluidRow(
                h1("Advanced Modeling"),
                "To model superconductor critical temperature, other machine learning methods were also employed: Random Forest and Neural Net. Due to the excessively long computation time it takes to run these models with a data set of this size, no custom model building options have been provided. Instead, these models were separately run, saved, and loaded into this application. All models are compared to each other by RMSE in the Summary tab. While there is no user control over the building of these models, one may use these models to make dynamic predictions based on these or custom data queries (see: Prediction tab).",
                h3("Random Forest"),
                "Random forest modeling was performed using the `randomForest` function from its namesake package in R. The model can be tuned by varying the number of random model parameters to include in regression tree generation (i.e. the 'mtry' parameter). A putative optimal value for this parameter was found by exploring the parameter space with the training data set. The final random forest model was fit with mtry = 6.",
                h3("Neural Net"),
                "Neural net modeling was performed using the 'neuralnet' function from its namesake package in R. In general, neural net modeling has a large number of different ways it may be constructed and tuned. It is generally computational extensive, especially for a dataset of this size. Therefore, for the purposes of this experiment, simply the default values from the 'neuralnet' function were used as a rough starting point in order to compare it to the other models, with acknowledgement that further optimization and tuning of the method could conceivably result in a better performing model."
              )
            )
          ),
          tabPanel("Summary",
            fluidPage(
              fluidRow(
                h1("Modeling Summary")
              ),
              mainPanel(
                fluidRow(
                  column(8, 
                    "The random forest model performed better than any of the other models tested. Analysis of the plots of observed versus predicted values (see: Prediction tab) also suggest it is as or more consistent between low and high values of critical temperature than other models tested.",
                    br(""),
                    "Random forest modeling has several advantages as both an ensemble and tree model. First, the use of tree modeling naturally provides fits for different regions of predictor space, naturally incorporating interactions between predictors, making the model very flexible. Usually this low bias would result in a model with high variance. However, use of ensemble modeling and the averaging of predictions across different models in bootstrap aggregation is usually able to reduce this variation. The trick of random forest to build each of its tree models off a randomly selected subset of predictors helps to make the models more independent from each other and thus is generally capable of reducing model variance even further.",
                    br(""),
                    "It is interesting to note that although the neural net model did not perform the best, it is already outperforming the linear model even with just a preliminary build using the default modeling hyperparameters. This suggests that neural net modeling has the potential for significant improvement through further tuning of model parameters and inclusion of more specialized nodes, node structure, and network depth."
                  ),
                  column(4,
                    tableOutput("modelSummaryTable")
                  )
                )
              )
            )
          ),
          tabPanel("Predictions",
            fluidPage(
              fluidRow(
                h1("Modeling Predictions"),
                "Select model and data; visualize and save predictions.",
                br(""),
                tags$i("Note: The red line is a zero-intercept line of slope=1, which would indicate a perfect model fit."),
                br("")
              ),
              sidebarPanel(
                selectInput("modelPredictionsModelInput",
                            label="Select Model for Predictions:",
                            choices=c("Linear",
                                      "Regression Tree",
                                      "Random Forest",
                                      "Neural Net"),
                            selected="Linear"),
                br(),
                selectInput("modelPredictionsDataInput",
                            label="Select Data for Predictions:",
                            choices=c("Full Dataset",
                                      "Subset of Data",
                                      "Manual Data Entry"),
                            selected="Full Dataset"),
                conditionalPanel(condition = "input.modelPredictionsDataInput == 'Subset of Data'",
                  br(""),
                  sliderInput("modelPredictionsMinData",
                              label="Select Start Data Index: ",
                              min=1,
                              max=n,
                              value=1),
                  sliderInput("modelPredictionsMaxData",
                              label="Select End Data Index: ",
                              min=1,
                              max=n,
                              value=n)
                ),
                br(""),
                actionButton("modelPredictionsSavePredictions",
                             label="Save Predictions"),
                span(textOutput("predictionSavedText"), style="color:red")
              ),
              mainPanel(
                conditionalPanel(condition = "input.modelPredictionsDataInput != 'Manual Data Entry'",
                  plotOutput("modelPredictionsPlot")
                ),
                conditionalPanel(condition = "input.modelPredictionsDataInput == 'Manual Data Entry'",
                  # programmatically generated HTML code earlier in this file (commented out) 
                  # to generate numeric inputs for each of the predictors variables
                  # that HTML code was saved to an HTML file (also commented out)
                  # now referencing in that HTML code directly so I don't have to hard code 80+ widgets...
                  includeHTML("manyNumericInputs.html")
                )
              )
            )
          )
        ) # end tab set panel for model
      )   # end modeling tab
    )     # end tabset items for dashboard body
  )       # end dashboard body
)         # end dashboard page

##### SERVER #####
server <- shinyServer(function(input, output, session){
  # INTRODUCTION TAB FUNCTIONS
  urlUCI      <- a("University of California Irvine (UCI) Machine Learning Repository website", 
                    href="http://archive.ics.uci.edu/ml/datasets/Superconductivty+Data")
  urlSUPERCON <- a("[Link]",
                   href="https://supercon.nims.go.jp/index_en.html")
  output$urlUCI <- renderUI({
    tagList("The data were obtained from the ",
          urlUCI,
          ". The data originated from the SUPERCON database of superconducting materials ",
          urlSUPERCON,
          ". The data contain information of 82 chemical features from over 21,263 unique chemicals. All features are numeric and interpreteed as continuous variables and there was no missing data. Data were available in a csv format and were read-in and analyzed using R Studio (Version 1.2.5018) with R (Version 3.6.1). Data were read using the 'read_csv' function from the 'readr' package.")
    })
  # DATA TAB FUNCTIONS
  # i) display whole data set (given user selected vars to display)
  dataFulldataDatatable <- reactive({
    data %>% 
      select(input$dataVarsToDisplay) %>%
      # reorder data set to have resp var first if present
      select(contains("critical_temp"), everything()) 
  })
  output$dataFulldataDatatable <- renderDataTable({
      dataFulldataDatatable()
    },
    escape=F
  )
  # ii) save data (or subset data) and display message
  observeEvent(input$dataSaveDataButton,{
    write_csv(dataFulldataDatatable(), "chemical_data_custom.csv")
  })
  # iii) display saved custom data message
  output$dataSavedMsg <- eventReactive(input$dataSaveDataButton, {
    paste0("Summary statistics saved to: ",
           getwd(),
           "/chemical_data_custom.csv")
  })
  # iv) get and display summary level stats
  dataDatatable <- reactive({
      sumData %>% 
        select(input$dataCheckStats) %>%
        mutate(Variable = sumData$Variable) %>%
        select(Variable, everything())
  })
  output$dataDatatable <- renderDataTable({
    dataDatatable()
    },
    escape=F
  )
  # v) save summary level data
  observeEvent(input$dataSaveSummary,{
     write_csv(dataDatatable(), "chemical_data_summary.csv")
  })
  # vi) display saved summary data mesasge
  output$dataSavedStatsMsg <- eventReactive(input$dataSaveSummary, {
    paste0("Summary statistics saved to: ",
           getwd(),
           "/chemical_data_summary.csv")
  })
  # INVESTIGATE TAB FUNCTIONS
  # A) DISTRIBUTION SECTION
  output$investDistributionPlot <- renderPlotly({
    if(input$investDistributionPlotType == "Scatterplot"){
      g <- ggplot(data, aes_string(x=input$investDistributionScatterVar1,
                                   y=input$investDistributionScatterVar2)) +
             geom_point(color="blue", alpha=0.25) + 
             labs(title=paste0(
                          "Scatterplot of ",
                          input$investDistributionScatterVar1,
                          " and ",
                          input$investDistributionScatterVar2
                        ),
                  x = input$investDistributionScatterVar1,
                  y = input$investDistributionScatterVar2
             )
      ggplotly(g)
    }
    else if(input$investDistributionPlotType == "Histogram"){
      g <- ggplot(data, aes_string(x=input$investDistributionHistogramVar)) + 
             geom_histogram(color="black", fill="lightblue") + 
             labs(title=paste0(
                          "Histogram of ",
                          input$investDistributionHistogramVar
                        ),
                  x=input$investDistributionHistogramVar,
                  y="Count"
             )
      ggplotly(g)
    }
  })
  # B) STRUCTURE SECTION
  # i) cumulative variance plot
  output$pcaCumvarplot <- renderPlot({
    selectedPC <- input$investStructurecVPC
    varExp     <- (cv[selectedPC] * 100) %>% round(1)
    plot(x=1:length(pv),
         y=cv,
         main="Cumulative Variance by PC",
         ylab="Cumulative Variance",
         xlab="Principal Component",
         type="b")
    abline(v=critP, col="darkblue")
    text(x=critP+20, y=0.5, 
         label=paste0(">95% Cumulative Variance @ PC ",critP),
         col="darkblue")
    abline(v=selectedPC, col="red")
    text(x=selectedPC+20, y=0.7,
         label=paste0(varExp, "% Cumulative Variance @ PC ",selectedPC),
         col="red")
  })
  # ii) screeplot
  output$pcaScreeplot <- renderPlot({
    screeplot(pcaData, main="Screeplot", type="l")
  })
  # iii) biplot (optionally hide varnames based on checkbox)
  output$pcaBiplot <- renderPlotly({
    if(input$investBiplotCheckbox){
      fviz_pca_biplot(pcaData, 
                      axes=c(as.numeric(input$investStructureBPPC1),
                             as.numeric(input$investStructureBPPC2)),
                      geom.var="arrow", 
                      geom.ind="point")
    }
    else{
      fviz_pca_biplot(pcaData, 
                      axes=c(as.numeric(input$investStructureBPPC1),
                             as.numeric(input$investStructureBPPC2)),
                      geom.ind="point")
    }
  })
  
  # MODELING TAB FUNCTIONS
  # A) LINEAR MODELING FUNCTIONS
  # i) get custom fit rmse on button click and save it
  trainI <- eventReactive(input$modelLinearButton, {
    set.seed(123)
    sample(1:n, size=0.8*n)
  })
  trData <- reactive({
    input$modelLinearButton
    sPC    <- isolate({input$modelLinearSlider})
    pData <- pcaData$x[,1:sPC] %>%
      tbl_df() %>%
      mutate(critical_temp = data$critical_temp)
    pData[trainI(),]  
  })
  teData <- reactive({
    input$modelLinearButton
    sPC    <- isolate({input$modelLinearSlider})
    pData <- pcaData$x[,1:sPC] %>%
      tbl_df() %>%
      mutate(critical_temp = data$critical_temp)
    pData[-trainI(),]
  })
  lmCustomFit <- eventReactive(input$modelLinearButton, {
    train(critical_temp ~ .,
          data=trData(),
          method="lmStepAIC",
          trControl=myControl,
          trace=F)
  })
  output$lmCustomFit <- reactive({
    lmCustomFit()
  })
  lmCustomRMSE <- reactive({
    getRMSE(lmCustomFit(), teData(), "critical_temp")
  })
  # ii) get RMSE default text
  output$modelLinearDefaultRMSE <- renderText({
    paste("Default Model - RMSE: ", lmRMSE %>% round(2))
  })
  # iii) render RMSE custom text
  output$modelLinearCustomRMSE <- renderText({
    rmse <- lmCustomRMSE()
    paste("Custom Model - RMSE: ", rmse %>% round(2))
  })
  # B) REGRESSION TREE MODELING FUNCTIONS\  
  # i) get custom fit rmse on button click and save it
  trainITree <- eventReactive(input$modelTreeButton, {
    set.seed(123)
    sample(1:n, size=0.8*n)
  })
  trDataTree <- reactive({
    input$modelTreeButton
    sPC    <- isolate({input$modelTreeSlider})
    pData <- pcaData$x[,1:sPC] %>%
      tbl_df() %>%
      mutate(critical_temp = data$critical_temp)
    pData[trainITree(),]  
  })
  teDataTree <- reactive({
    input$modelTreeButton
    sPC    <- isolate({input$modelTreeSlider})
    pData <- pcaData$x[,1:sPC] %>%
      tbl_df() %>%
      mutate(critical_temp = data$critical_temp)
    pData[-trainITree(),]
  })
  treeCustomFit <- eventReactive(input$modelTreeButton, {
    selectedCP  <- isolate({input$modelTreeCPSlider})
    selectedCP  <- 10^(selectedCP)
    train(critical_temp ~ .,
          data=trDataTree(),
          method="rpart",
          trControl=trainControl(method="none"),
          tuneGrid=expand.grid(cp=selectedCP)
    )
  })
  output$treeCustomFit <- reactive({
    treeCustomFit()
  })
  treeCustomRMSE <- reactive({
    getRMSE(treeCustomFit(), teDataTree(), "critical_temp")
  })
  # ii) get RMSE default text
  output$modelTreeDefaultRMSE <- renderText({
    paste("Default Model - RMSE: ", treeRMSE %>% round(2))
  })
  # iii) render RMSE custom text
  output$modelTreeCustomRMSE <- renderText({
    rmse <- treeCustomRMSE()
    paste("Custom Model - RMSE: ", rmse %>% round(2))
  })
  # C) SUMMARY MODEL FUNCTIONS
  output$modelSummaryTable <- renderTable({
    data.frame(Model=c("Linear",
                       "Regression Tree",
                       "Random Forest",
                       "Neural Net"),
               RMSE=c(lmRMSE,
                      treeRMSE,
                      rfRMSE,
                      nnRMSE)) %>%
      arrange(RMSE)
  })
  # D) PREDICTION MODELING FUNCTIONS
  # i) Get predictions and plot of predictions based on user inputs
  customData <- reactive({
    cusdat <- sapply(dataNames[-p], FUN=function(x){
                eval(parse(text=paste("input$",x,"_NI",sep="",collapse="")))
              }) %>%
                as.matrix() %>%
                t() %>%
                as.data.frame()
    names(cusdat) <- dataNames[-p]
    cusdat
  })
  modelPredictions <- reactive({
    modelChoice   <- switch(input$modelPredictionsModelInput,
                            "Linear"          = lmFit,
                            "Regression Tree" = treeFit,
                            "Random Forest"   = rfFit,
                            "Neural Net"      = nnFit)
    dataChoice    <- switch(input$modelPredictionsDataInput,
                            "Full Dataset"      = data %>% select(-critical_temp),
                            "Subset of Data"    = data[input$modelPredictionsMinData:input$modelPredictionsMaxData,] %>% 
                                                       select(-critical_temp),
                            "Manual Data Entry" = customData())
    # rotate data choice based on pca data rotations (scale data first)
    if(input$modelPredictionsDataInput != "Manual Data Entry" && ncol(dataChoice) != 1){
      pcaDataChoice <- sapply(1:ncol(dataChoice), FUN=function(i){
          (dataChoice[,i]-dataMeans[i])/dataSds[i]
        }) %>% 
        as.data.frame() %>% 
        as.matrix() %*% 
        pcaData$rotation
      predict(modelChoice, newdata=pcaDataChoice)
    }
    else{
      cd <- customData()
      pcaDataChoice <- sapply(1:ncol(cd), FUN=function(i){
        (cd[,i]-dataMeans[i])/dataSds[i]
      }) %>%
        as.data.frame() %>% t() %*% pcaData$rotation %>% as.data.frame()
      predict(modelChoice, newdata=pcaDataChoice)    
    }
  })
  output$modelPredictionsPlot <- renderPlot({
    dataChoice  <- switch(input$modelPredictionsDataInput,
                          "Full Dataset"      = data %>% select(critical_temp) %>% unlist,
                          "Subset of Data"    = data[input$modelPredictionsMinData:input$modelPredictionsMaxData,] %>% 
                                                     select(critical_temp) %>% unlist,
                          "Manual Data Entry" = data %>% select(-critical_temp) %>% unlist)
    pred   <- modelPredictions()
    actual <- dataChoice
    plot(x=pred, y=actual,
         main="Actual vs Prediction Response (Critical Temp)",
         xlab="Predicted Value",
         ylab="Actual Value")
    abline(a=0, b=1, col="red")
  })
  # ii) save predictions based on user 'save predictions' button click
  #     generate text notice output object
  observeEvent(input$modelPredictionsSavePredictions,{
    dataChoice    <- switch(input$modelPredictionsDataInput,
                            "Full Dataset"      = data,
                            "Subset of Data"    = data[input$modelPredictionsMinData:input$modelPredictionsMaxData,],
                            "Manual Data Entry" = customData())
    dataChoice <- dataChoice %>% 
      mutate(predicted_critical_temp=modelPredictions()) %>%
      select(contains("critical_temp"), everything())
    write_csv(dataChoice, "model_predictions.csv")
  })
  output$predictionSavedText <- eventReactive(input$modelPredictionsSavePredictions,{
    paste0("Predictions + Data Saved To: ",
           getwd(),
           "/model_predictions.csv")
  }) 
  # iii) update min and max values for slider for data subsetting
  observe({
    updateSliderInput(session, "modelPredictionsMinData", max=input$modelPredictionsMaxData)
    updateSliderInput(session, "modelPredictionsMaxData", min=input$modelPredictionsMinData)
  })
})

##### SHINY APP #####
shinyApp(ui=ui, server=server)