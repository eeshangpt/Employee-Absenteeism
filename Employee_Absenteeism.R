rm(list = ls())
## Reading the data.
read.csv("Absenteeism_at_work_Project.csv", header = TRUE) -> emp.data

## View of the structure of the data frame.
str(emp.data)

missing_value <-
    data.frame(apply(emp.data, 2 , function(x) {
        sum(is.na(x))
    }))
missing_value$Columns <- row.names(missing_value)
names(missing_value)[1] <- 'Missing Percentage'
missing_value$`Missing Percentage` <-
    (missing_value$`Missing Percentage` / nrow(missing_value)) * 100
missing_value <-
    missing_value[order(-missing_value$`Missing Percentage`),]
row.names(missing_value) <- NULL
missing_value <- missing_value[, c(2, 1)]

## Filling missing values as replaced with median if the entries are nominal
## and with mean if the entries are numeric.
emp.data$Reason.for.absence[is.na(emp.data$Reason.for.absence)] <-
    median(emp.data$Reason.for.absence, na.rm = T)

emp.data$Month.of.absence[is.na(emp.data$Month.of.absence)] <-
    median(emp.data$Month.of.absence, na.rm = T)

emp.data$Seasons[is.na(emp.data$Seasons)] <-
    median(emp.data$Seasons, na.rm = T)

emp.data$Transportation.expense[is.na(emp.data$Transportation.expense)] <-
    mean(emp.data$Transportation.expense, na.rm = T)

emp.data$Distance.from.Residence.to.Work[is.na(emp.data$Distance.from.Residence.to.Work)] <-
    mean(emp.data$Distance.from.Residence.to.Work, na.rm = T)

emp.data$Service.time[is.na(emp.data$Service.time)] <-
    mean(emp.data$Service.time, na.rm = T)

emp.data$Age[is.na(emp.data$Age)] <- mean(emp.data$Age, na.rm = T)

emp.data$Hit.target[is.na(emp.data$Hit.target)] <-
    median(emp.data$Hit.target, na.rm = T)

emp.data$Disciplinary.failure[is.na(emp.data$Disciplinary.failure)] <-
    median(emp.data$Disciplinary.failure, na.rm = T)

emp.data$Education[is.na(emp.data$Education)] <-
    median(emp.data$Education, na.rm = T)

emp.data$Son[is.na(emp.data$Son)] <- median(emp.data$Son, na.rm = T)

emp.data$Social.drinker[is.na(emp.data$Social.drinker)] <-
    median(emp.data$Social.drinker, na.rm = T)

emp.data$Social.smoker[is.na(emp.data$Social.smoker)] <-
    median(emp.data$Social.smoker, na.rm = T)

emp.data$Pet[is.na(emp.data$Pet)] <- median(emp.data$Pet, na.rm = T)

emp.data$Weight[is.na(emp.data$Weight)] <-
    mean(emp.data$Weight, na.rm = T)

emp.data$Height[is.na(emp.data$Height)] <-
    mean(emp.data$Height, na.rm = T)

emp.data$Weight[is.na(emp.data$Body.mass.index)] * 10000 -> numerator
emp.data$Height[is.na(emp.data$Body.mass.index)] ^ 2 -> denominator
emp.data$Body.mass.index[is.na(emp.data$Body.mass.index)] <-
    (numerator / denominator)
rm(list = c('numerator', 'denominator'))

emp.data$Absenteeism.time.in.hours[is.na(emp.data$Absenteeism.time.in.hours)] <-
    mean(emp.data$Absenteeism.time.in.hours, na.rm = T)

## Changing the column types into either numeric or factor.
emp.data$Reason.for.absence <-
    as.factor(emp.data$Reason.for.absence)
emp.data$Month.of.absence <- as.factor(emp.data$Month.of.absence)
emp.data$Day.of.the.week <- as.factor(emp.data$Day.of.the.week)
emp.data$Seasons <- as.factor(emp.data$Seasons)
emp.data$Hit.target <- as.factor(emp.data$Hit.target)
emp.data$Disciplinary.failure <-
    as.factor(emp.data$Disciplinary.failure)
emp.data$Education <- as.factor(emp.data$Education)
emp.data$Social.drinker <- as.factor(emp.data$Social.drinker)
emp.data$Social.smoker <- as.factor(emp.data$Social.smoker)
emp.data$Work.load.Average.day <-
    as.numeric(gsub(',', '', as.character(emp.data$Work.load.Average.day)))

str(emp.data)

## Outlier analysis.
numerical <- colnames(emp.data[, sapply(emp.data, is.numeric)])
categorical <- colnames(emp.data[, sapply(emp.data, is.factor)])

## Imputing the all columns other than the target column.
for (i in numerical) {
    print(i)
    val <-
        emp.data[, i][emp.data[, i] %in% boxplot.stats(emp.data[, i])$out]
    print(length(val))
    if (!i %in% c('Absenteeism.time.in.hours')) {
        emp.data[, i][emp.data[, i] %in% val] <- NA
        emp.data[, i][is.na(emp.data[, i])] <-
            mean(emp.data[, i], na.rm = T)
    } else {
        emp.data <- emp.data[which(!emp.data[, i] %in% val), ]
    }
}
rm(list = c('i', 'val'))

## Correlation for numeric types.
library(corrgram)
corrgram(
    emp.data[, numerical],
    order = F,
    upper.panel = panel.smooth,
    text.panel = panel.txt
)

## Chi-squared test for categorical types.
for (i in 1:length(emp.data[, categorical])) {
    print(names(emp.data[, categorical])[i])
    print(chisq.test(table(
        emp.data$Absenteeism.time.in.hours, emp.data[, i]
    )))
}
rm(list = c('i'))

## Converting the target variable into a categorical type.
cut(
    emp.data$Absenteeism.time.in.hours,
    boxplot.stats(emp.data$Absenteeism.time.in.hours)$stats,
    include.lowest = T
) -> emp.data$Category.Absent

## Feature Selection.
emp.data <- subset(
    emp.data,
    select = -c(
        Weight,
        Height,
        Body.mass.index,
        Distance.from.Residence.to.Work,
        Hit.target,
        ID,
        Absenteeism.time.in.hours
    )
)

## Training and Testing data split
library(rpart)
train.Index <- sample(1:nrow(emp.data), 0.9 * nrow(emp.data))
train.data <- emp.data[train.Index, ]
test.data <- emp.data[-train.Index, ]
rm(list = c('train.Index', 'missing_value'))

## Applying C50 algorithm.
library(C50)
attach(emp.data)
c50.model <- C5.0(Category.Absent ~ .,
                  train.data,
                  trials = 100,
                  rules = T)
detach(emp.data)

c50.model$names
summary(c50.model)
write(capture.output(summary(c50.model)), 'C50Rules')

c50.prediction <-
    predict(c50.model, test.data[,-15], type = 'class')
str(emp.data)
