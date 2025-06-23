# prediction-model-for-ABO

################################################################################################
################################################################################################
######       Development And Validation Of Risk Prediction Model For                      ######
######       Adverse Birth Outcomes Among Mothers Conceiving Through Assisted             ######
######       Reproductive Technology In Saint Paul Specialized Hospital, Addis Ababa,     ######
######       Central Ethiopia                                                             ######
################################################################################################
################################################################################################

library(haven)
data = read_dta("C:/Users/hp/Documents/Materials/Thesis file/Data/fin/updated/420/New folder/New folder/KM plot/New folder/Updatedmock/Adverse/cleanedfinal3.dta")

# Load necessary libraries
library(glmnet)
library(caret)
names(data)

library(caret)


# Your original code with minor improvements
exclude_vars <- c("id", "apo", "ga","lbw", "prebirth", "stillbirth", "conganomal", "lapscore")
predictor_names <- setdiff(names(data), exclude_vars)

# Convert to factors (safer approach)
data[predictor_names] <- lapply(data[predictor_names], function(x) {
  if(is.numeric(x) & length(unique(x)) <= 5) as.factor(x) else x
})


# Ensure outcome is binary factor with proper level names
data$apo <- factor(data$apo, levels = c(0,1), labels = c("No", "Yes"))
table(data$apo)

# Handle missing data (critical for stepwise)
data_complete <- na.omit(data[, c("apo", predictor_names)])

data_complete$apo = as.factor(data_complete$apo)

table(data_complete$apo)




data_complete$healthinsu <-factor(data_complete$healthinsu,
                                  levels= c(1,2),
                                  labels= c("No", "Yes"))


data_complete$placeresid <-factor(data_complete$placeresid,
                                  levels= c(1,2),
                                  labels= c("Urban", "Rural"))



data_complete$reparity <-factor(data_complete$reparity,
                                levels= c(1,2),
                                labels= c("Multiparous", "Primiparous"))



data_complete$childgender <-factor(data_complete$childgender,
                                   levels= c(1,2),
                                   labels= c("Male", "Female"))



data_complete$iugr <-factor(data_complete$iugr,
                            levels= c(1,2),
                            labels= c("No", "Yes"))



data_complete$infcause <-factor(data_complete$infcause,
                                levels= c(1,2,3),
                                labels= c("Male factor", "Female factor", "Unexplained"))




data_complete$afc <-factor(data_complete$afc,
                           levels= c(1,2),
                           labels= c(">= 5", "<5"))



data_complete$twin <-factor(data_complete$twin,
                            levels= c(1,2),
                            labels= c("No", "Yes"))




data_complete$embryotype <-factor(data_complete$embryotype,
                                  levels= c(1,2),
                                  labels= c("Fresh", "Frozen"))




data_complete$embryostage <-factor(data_complete$embryostage,
                                   levels= c(1,2),
                                   labels= c("Blastocyte", "Cleavage"))




data_complete$dmhx <-factor(data_complete$dmhx,
                            levels= c(1,2),
                            labels= c("No", "Yes"))



################################################################################
################################################################################
##                    Model Building Process                                  ##
################################################################################
################################################################################

# Fit full model
full_model <- glm(apo ~ agegrp + bmi + healthinsu + placeresid + edustat +
                    reparity + twin + childgender + hxcs + matcomo + prliveb + 
                    fullbirth + preARTattempt + yrstayedinfert + iugr + hxabortion +
                    bleedepisode + prevutrinesurg + prom + infcause + alcoholdri + 
                    cigsmoke + afc + cycletype + embryotype + embryostage + 
                    numembryotrans + assistedhatch + hdp + dmhx + predmhx, 
                  data = data_complete,
                  family = binomial(link = "logit"))

# Stepwise selection
reduced_model <- step(full_model, 
                      direction = "backward", 
                      trace = 1)  # Keep trace=1 to see 


candidatemodel <- glm(apo ~ healthinsu + placeresid + edustat + reparity +
                        twin + childgender + yrstayedinfert + iugr + infcause +
                        alcoholdri + cigsmoke + afc + embryotype + embryostage + dmhx,
                      data = data_complete,
                      family = binomial(link = "logit"))

summary(candidatemodel)

################################################################################
################################################################################
###                 Further model reduction by LRT                           ###
################################################################################
################################################################################

drop1_results <- drop1(candidatemodel, test = "LRT")
print(drop1_results)

candidatemodel2 <- glm(apo ~ healthinsu + placeresid + reparity +
                         twin + childgender + yrstayedinfert + iugr + infcause +
                         alcoholdri + afc + embryotype + embryostage + dmhx,
                       data = data_complete,
                       family = binomial(link = "logit"))

summary(candidatemodel2)

drop1_results <- drop1(candidatemodel2, test = "LRT")
print(drop1_results)

candidatemodel3 = glm(apo ~ healthinsu + placeresid + reparity + twin + childgender + 
                        iugr + infcause + afc + embryotype + embryostage + dmhx,
                      data = data_complete,
                      family = binomial)

summary(candidatemodel3)

drop1_results <- drop1(candidatemodel3, test = "LRT")
print(drop1_results)

finalmodel = glm(apo ~ healthinsu + placeresid + reparity + twin + childgender + 
                   iugr + infcause + afc + embryotype + embryostage + dmhx,
                 data = data_complete,
                 family = binomial)


summary(finalmodel)


# Extract coefficients and standard errors
coefs <- summary(finalmodel)$coefficients

# Calculate AOR (exponentiate the Estimate)
AOR <- exp(coefs[, "Estimate"])

# Calculate 95% Confidence Intervals
lower_CI <- exp(coefs[, "Estimate"] - 1.96 * coefs[, "Std. Error"])
upper_CI <- exp(coefs[, "Estimate"] + 1.96 * coefs[, "Std. Error"])

# Combine into a data frame for easy presentation
AOR_table <- data.frame(
  Variable = rownames(coefs),
  AOR = round(AOR, 3),
  Lower_95CI = round(lower_CI, 3),
  Upper_95CI = round(upper_CI, 3),
  P_value = round(coefs[, "Pr(>|z|)"], 4)
)

print(AOR_table)



################################################################################
################################################################################
##                  Model Assumption Check                                    ##
################################################################################
################################################################################

# 1. Influential Observations

# Average leverage
p <- length(coef(finalmodel))  # Number of parameters
n <- nrow(data_complete)
avg_leverage <- (3*p) / n

avg_leverage

plot(hatvalues(finalmodel))


#2. Multicollinearity

# Variance Inflation Factor (VIF)
car::vif(finalmodel)  # Values >5-10 indicate problems

# Correlation matrix
library(corrplot)
corrplot(cor(model.matrix(finalmodel)[,-1]), method = "number")

# The vif shows no multicolinearity that have an issue

#3. Outliers

# Standardized residuals
std_resid <- rstandard(finalmodel)
plot(std_resid, ylab = "Standardized Residuals")
abline(h = c(-2.5, 2.5), col = "red")  # Flagging potential outliers

# no observation here also out of the demarcated area.

# Goodness-of-Fit

# Hosmer-Lemeshow Test
ResourceSelection::hoslem.test(finalmodel$y, fitted(finalmodel), g = 10)

# The Model fit the data

# Pearson residuals
pearson_res <- residuals(finalmodel, type = "pearson")
plot(pearson_res, ylab = "Pearson Residuals")
abline(h = 0, col = "red")

# Deviance residuals
dev_res <- residuals(finalmodel, type = "deviance")
plot(dev_res, ylab = "Deviance Residuals")
abline(h = c(-2, 2), col = "red")


################################################################################
################################################################################
##                    Risk Score Development                                  ## 
################################################################################
################################################################################


# Step 1: Extract coefficients and prepare scoring
coefs <- coef(finalmodel)
score_coefs <- round(coefs / min(abs(coefs[-1])))  # Exclude intercept when scaling

# Step 2: Create model matrix (X) that matches what was used in the model
X <- model.matrix(finalmodel)[, -1]  # Remove intercept column

# Step 3: Convert to data frame and add variable names that match coef names
X_df <- as.data.frame(X)
colnames(X_df)  # These should match names(score_coefs)[-1]

# Step 4: Define a risk scoring function
calculate_risk_score <- function(row) {
  score <- 0
  for (var in names(score_coefs)) {
    if (var != "(Intercept)" && var %in% names(row)) {
      score <- score + row[[var]] * score_coefs[[var]]
    }
  }
  return(score)
}

# Step 5: Apply function row by row
X_df$risk_score <- apply(X_df, 1, function(row) {
  calculate_risk_score(as.list(row))
})

# Step 6: Combine back with original patient IDs or outcomes if needed
data_complete$risk_score <- X_df$risk_score

# Optional: Calculate predicted probability from logistic model
intercept <- coefs["(Intercept)"]
data_complete$predicted_prob <- plogis(intercept + data_complete$risk_score)


################################################################################
################################################################################
###                      Develop a Nomogram                                  ###
################################################################################
################################################################################


# Install required packages
# install.packages(c("rms", "pROC", "caret", "ResourceSelection", "dcurves"))

# Load libraries
library(rms)
library(pROC)
library(caret)
library(ResourceSelection)
library(dcurves)


table(data_complete$healthinsu)

# Nomogram

# Convert data to datadist object for rms package
dd <- datadist(data_complete)
options(datadist = "dd")

# Fit model using rms::lrm
finalmodel_rms <- lrm(apo ~ healthinsu + placeresid + reparity + twin + childgender + 
                        iugr + infcause + afc + embryotype + embryostage + dmhx,
                      data = data_complete, x=TRUE,y=TRUE )
summary(finalmodel_rms)

# Draw Nomogram

# Create the nomogram object
nom <- nomogram(finalmodel_rms, 
                fun = plogis, 
                funlabel = "Predicted Probability of ABO",
                fun.at = c(0.1, 0.3, 0.5, 0.7, 0.9),
                lp = TRUE)
summary(nom$total.points$x)

summary(nom$lp$x)

# Plot the nomogram with enhancements
plot(nom,
     main = "Nomogram for Predicting Adverse Birth Outcome",  # Title
     lwd = 2,                  # Line width
     lty = 1,                  # Line type (solid); change to 2, 3, etc., if you want dashed
     col.grid = gray(c(0.7, 0.9)), # Optional: vertical reference lines
     col.conf = c("black", 0.3),   # Color of confidence intervals if you enable conf.int
     cex.axis = 0.9,           # Size of tick labels
     cex.var = 1,              # Size of variable names
     varname.label = TRUE,     # Include variable name in axis label
     varname.label.sep = " = ",# Separator
     points.label = "Points",  # Label on point scale
     total.points.label = "Total Points",  # Label on total points scale
     xfrac = 0.35              # Adjust space for axis titles
)

# First plot to screen device (preview)
#plot(nom,
#     main = "Nomogram for Predicting Adverse Birth Outcome",
#     lwd = 2,
#     lty = 1,
#     col.grid = gray(c(0.7, 0.9)),
#     col.conf = c("black", 0.3),
#     cex.axis = 0.9,
#     cex.var = 1,
#     varname.label = TRUE,
#     varname.label.sep = " = ",
#     points.label = "Points",
#     total.points.label = "Total Points",
#     xfrac = 0.35)

# Then export to TIFF file (with the EXACT same parameters)
#tiff("nomogram_apo.tiff", width = 15, height = 6, units = "in", res = 500)
#plot(nom,
#     main = "Nomogram for Predicting Adverse Birth Outcome",
#     lwd = 2,
#     lty = 1,
#     col.grid = gray(c(0.7, 0.9)),
#     col.conf = c("black", 0.3),
#     cex.axis = 0.9,
#     cex.var = 1,
#     varname.label = TRUE,
#     varname.label.sep = " = ",
#     points.label = "Points",
#     total.points.label = "Total Points",
#     xfrac = 0.35)
#dev.off()  # This is crucial to close/save the graphics device



################################################################################
##                    ROC of the risk score vs Nomogram                       ##
################################################################################

library(pROC)

# AUCs for both models
pred_prob_nomogram <- predict(finalmodel, type = "response")
pred_prob_risk_score <- data_complete$predicted_prob
true_outcome <- data_complete$apo

# ROC curves
roc_nomogram <- roc(true_outcome, pred_prob_nomogram, 
                    ci = TRUE, auc = TRUE, plot = TRUE, 
                    col = "blue", lwd = 2, main = "ROC Comparison")

roc_riskscore <- roc(true_outcome, pred_prob_risk_score,
                     ci = TRUE, auc = TRUE, plot = TRUE, 
                     col = "red", lwd = 2, add = TRUE)

# DeLong test
roc_delong <- roc.test(roc_nomogram, roc_riskscore, method = "delong")

# Add legend with AUCs
legend("bottomright", 
       legend = c(paste0("Nomogram AUC = ", round(auc(roc_nomogram), 3)),
                  paste0("Risk Score AUC = ", round(auc(roc_riskscore), 3))),
       col = c("blue", "red"), lwd = 2)

# Add DeLong test result as subtitle or caption on plot
text(x = 0.2, y = 0.15, labels = paste0("DeLong's test: Z = ", round(roc_delong$statistic, 4), 
                                       ", p = ", round(roc_delong$p.value, 4),
                                       "\n95% CI for AUC diff: [",
                                       round(roc_delong$conf.int[1], 4), ", ",
                                       round(roc_delong$conf.int[2], 4), "]"),
     cex = 0.8)


#################################################################################
#################################################################################
####                     Risk classification using Youden Index              ####
####                              for Nomogram                               ####
#################################################################################
#################################################################################

# Predict probabilities
data_complete$pred_prob <- predict(finalmodel, type = "response")

# ROC and Youden's index
roc_obj <- roc(data_complete$apo, data_complete$pred_prob)
coords_obj <- coords(roc_obj, "best", best.method = "youden", ret = c("threshold", "sensitivity", "specificity"))

# Print cutoff and metrics
print(coords_obj)

youden_index <- coords_obj["sensitivity"] + coords_obj["specificity"] - 1
print(youden_index)

# Apply cutoff to classify risk
cutoff <- coords_obj["threshold"]

# Classify into 2 risk categories: Low and High
data_complete$risk_class <- cut(data_complete$pred_prob, 
                                breaks = c(0, cutoff, 1),
                                labels = c("Low", "High"),
                                include.lowest = TRUE)

# See the counts for each group
table(data_complete$risk_class, data_complete$apo)


#Calculate sensitivity, specificity, PPV, NPV, accuracy

# Predicted class
data_complete$pred_class <- ifelse(data_complete$pred_prob >= 0.6948581, 1, 0)
table(data_complete$pred_class)
table(data_complete$apo)

# First recode apo to 0/1
data_complete$apo_bin <- ifelse(data_complete$apo == "Yes", 1, 0)

# Now make both factors
conf_mat <- confusionMatrix(factor(data_complete$pred_class), 
                            factor(data_complete$apo_bin),
                            positive = "1")

print(conf_mat)

#################################################################################
#################################################################################
####                     Risk classification using Youden Index              ####
####                              for risk score                             ####
#################################################################################
#################################################################################

#✅ Step 1: Predict Probabilities from Risk Score (if not already done)

# Make sure you already have this line (if not, run it)
intercept <- coef(finalmodel)["(Intercept)"]
data_complete$pred_prob_score <- plogis(intercept + data_complete$risk_score)

#✅ Step 2: ROC Curve & Youden Index for Risk Score Model

library(pROC)

# Create ROC object
roc_score <- roc(data_complete$apo, data_complete$pred_prob_score)

# Get optimal cutoff using Youden Index
coords_score <- coords(roc_score, "best", best.method = "youden", ret = c("threshold", "sensitivity", "specificity"))

# Print threshold and associated metrics
print(coords_score)

#✅ Step 3: Classify Risk Based on Optimal Cutoff

# Save cutoff
cutoff_score <- coords_score["threshold"]
cutoff_score
# Classify into risk groups: Low and High
data_complete$risk_class_score <- cut(data_complete$pred_prob_score, 
                                      breaks = c(0, cutoff_score, 1),
                                      labels = c("Low", "High"),
                                      include.lowest = TRUE)

table(data_complete$risk_class_score)

#✅ Step 4: Confusion Matrix for Performance Evaluation

library(caret)

# Convert to 0/1: High = 1, Low = 0
data_complete$pred_class_score <- ifelse(data_complete$risk_class_score == "High", 1, 0)

# Then compute metrics as before
conf_mat_score <- confusionMatrix(factor(data_complete$pred_class_score), 
                  factor(data_complete$apo_bin), 
                  positive = "1")


print(conf_mat_score)

print(conf_mat)


################################################################################
##           Model performance Discrimination and Calibration                 ##
##                    for Nomogram and Risk Score                             ##
################################################################################

# Load packages
library(pROC)

#finalmodel = glm(apo ~ healthinsu + placeresid + reparity + twin + childgender + 
#                   iugr + infcause + afc + embryotype + embryostage + dmhx,
#                 data = data_complete,
#                 family = binomial)


# Predict probabilities
#data_complete$pred_prob <- predict(finalmodel, type = "response")

# ROC and Youden's index
#roc_obj <- roc(data_complete$apo, data_complete$pred_prob)
#coords_obj <- coords(roc_obj, "best", best.method = "youden", ret = c("threshold", "sensitivity", "specificity"))

# Print cutoff and metrics
#print(coords_obj)

# Plot the ROC curve
plot(roc_obj, col = "red", main = "ROC Curve")


# Get optimal threshold (Youden's index)
opt_coords<- coords(roc_obj, "best", best.method = "youden", 
                    ret = c("threshold", "sensitivity", "specificity"))
print(opt_coords)
# Add the optimal threshold point
points(opt_coords["specificity"], opt_coords["sensitivity"], col = "blue", pch = 19)

# Calculate AUC with CI
ci_auc <- ci.auc(roc_obj)

# Add a legend with AUC and 95% CI
legend("bottomright",
       legend = paste0("AUC = ", round(auc(roc_obj), 3),
                       " (95% CI: ", round(ci_auc[1], 3), "-",
                       round(ci_auc[3], 3), ")"),
       col = "red",
       lty = 1,
       bty = "n")

################################################################################
##       Contribution of each nomogram Predictors in AUC                      ##
################################################################################

# Load required package
library(pROC)

# Predict probabilities from final model
data_complete$final_pred_prob <- predict(finalmodel, type = "response")

# ROC for final model
roc_final <- roc(data_complete$apo, data_complete$final_pred_prob)
ci_auc_final <- ci.auc(roc_final)
opt_coords_final <- coords(roc_final, "best", best.method = "youden", 
                           ret = c("threshold", "sensitivity", "specificity"))

# Individual predictors
predictors <- c("healthinsu", "placeresid", "reparity", "twin", "childgender", 
                "iugr", "infcause", "afc", "embryotype", "embryostage", "dmhx")

# Store ROC objects and AUCs
roc_list <- list()
auc_values <- numeric(length(predictors))

for (i in seq_along(predictors)) {
  var <- predictors[i]
  model_uni <- glm(as.formula(paste("apo ~", var)), 
                   data = data_complete, family = binomial)
  pred_prob <- predict(model_uni, type = "response")
  roc_obj <- roc(data_complete$apo, pred_prob)
  roc_list[[var]] <- roc_obj
  auc_values[i] <- auc(roc_obj)
}

# Set up extended plot area for external legend
par(mar = c(5, 5, 4, 10), xpd = TRUE)  # enlarge right margin

# Plot overall model ROC first (solid black line)
plot(roc_final, col = "black", lwd = 3,
     main = "ROC Curve with Individual Predictors and the overall")

# Add threshold point
points(opt_coords_final["specificity"], opt_coords_final["sensitivity"], 
       col = "blue", pch = 19)

# Add diagonal reference line
abline(a = 0, b = 1, lty = 3, col = "gray")

# Add all individual predictor ROC curves
colors <- rainbow(length(predictors))
legend_labels <- paste0("Full Model: AUC = ", round(auc(roc_final), 3),
                        " (95% CI: ", round(ci_auc_final[1], 3), "-",
                        round(ci_auc_final[3], 3), ")")
legend_colors <- "black"
legend_lty <- 1

for (i in seq_along(predictors)) {
  lines(roc_list[[predictors[i]]], col = colors[i], lwd = 2)
  legend_labels <- c(legend_labels,
                     paste0(predictors[i], ": AUC = ", round(auc_values[i], 3)))
  legend_colors <- c(legend_colors, colors[i])
  legend_lty <- c(legend_lty, 1)
}

# Set up extended plot area (already in your code)
par(mar = c(5, 5, 4, 10), xpd = TRUE)  # Right margin space for legend

# [Keep all your existing plotting code until the legend...]

# Modified legend positioning:
legend(x = 0.65, y = 0.6,  # x and y coordinates in plot units (0-1)
       legend = legend_labels,
       col = legend_colors,
       lty = legend_lty,
       lwd = 2,
       bty = "n",
       cex = 0.8,
       x.intersp = 0.8,  # Horizontal spacing between items
       y.intersp = 1.2)  # Vertical spacing between items

# Calibration plot
# Reset margins to R's default
par(mar = c(5, 4, 4, 2) + 0.1)  # Default margins
par(xpd = FALSE)  # Disable plotting outside plot area

library(rms)
val.prob(predict(finalmodel, type = "response"), finalmodel$y, cex = 0.8)

################################################################################
##       Contribution of Each Risk Score Predictor in AUC                     ##
################################################################################

library(pROC)

# Predict probability from the risk score model
intercept <- coef(finalmodel)["(Intercept)"]  # intercept used in risk score model
data_complete$pred_prob_score <- plogis(intercept + data_complete$risk_score)

# ROC for risk score model
roc_score <- roc(data_complete$apo, data_complete$pred_prob_score)
ci_auc_score <- ci.auc(roc_score)
opt_coords_score <- coords(roc_score, "best", best.method = "youden", 
                           ret = c("threshold", "sensitivity", "specificity"))

# Extract individual predictors used in scoring
predictors <- c("healthinsu", "placeresid", "reparity", "twin", "childgender", 
                              "iugr", "infcause", "afc", "embryotype", "embryostage", "dmhx")
# exclude intercept

# Store ROC objects and AUCs for univariate predictors
roc_list_score <- list()
auc_values_score <- numeric(length(predictors))

for (i in seq_along(predictors)) {
  var <- predictors[i]
  model_uni <- glm(as.formula(paste("apo ~", var)), 
                   data = data_complete, family = binomial)
  pred_prob <- predict(model_uni, type = "response")
  roc_obj <- roc(data_complete$apo, pred_prob)
  roc_list_score[[var]] <- roc_obj
  auc_values_score[i] <- auc(roc_obj)
}

# Extended plot area
par(mar = c(5, 5, 4, 10), xpd = TRUE)

# Plot main ROC
plot(roc_score, col = "black", lwd = 3,
     main = "ROC Curve: Risk Score and Individual Predictors")

# Threshold point
points(opt_coords_score["specificity"], opt_coords_score["sensitivity"], 
       col = "blue", pch = 19)

# Diagonal line
abline(a = 0, b = 1, lty = 3, col = "gray")

# Add ROC for each predictor
colors <- rainbow(length(predictors))
legend_labels <- paste0("Risk Score Model: AUC = ", round(auc(roc_score), 3),
                        " (95% CI: ", round(ci_auc_score[1], 3), "-",
                        round(ci_auc_score[3], 3), ")")
legend_colors <- "black"
legend_lty <- 1

for (i in seq_along(predictors)) {
  lines(roc_list_score[[predictors[i]]], col = colors[i], lwd = 2)
  legend_labels <- c(legend_labels,
                     paste0(predictors[i], ": AUC = ", round(auc_values_score[i], 3)))
  legend_colors <- c(legend_colors, colors[i])
  legend_lty <- c(legend_lty, 1)
}

# Set up extended plot area (already in your code)
par(mar = c(5, 5, 4, 10), xpd = TRUE)  # Right margin space for legend
# Add Legend
legend(x = 0.65, y = 0.6,
       legend = legend_labels,
       col = legend_colors,
       lty = legend_lty,
       lwd = 2,
       bty = "n",
       cex = 0.8,
       x.intersp = 0.8,
       y.intersp = 1.2)


#  Calibration Plot for Risk Score

# Reset margins
par(mar = c(5, 4, 4, 2) + 0.1)
par(xpd = FALSE)

library(rms)
val.prob(data_complete$pred_prob_score, data_complete$apo_bin, cex = 0.8)


################################################################################
#                       Internal validation (Bootstrap)                        #
################################################################################

# Load required libraries
library(pROC)
library(boot)
library(rms)
library(ResourceSelection)

# Set seed for reproducibility
set.seed(123)

# Define the bootstrap function
bootstrap_function <- function(data, indices) {
  d <- data[indices, ]
  model_boot <- glm(apo ~ healthinsu + placeresid + reparity + twin + childgender + 
                      iugr + infcause + afc + embryotype + embryostage + dmhx, 
                    data = d, family = binomial)
  
  # Predict on bootstrap sample
  pred_boot <- predict(model_boot, newdata = d, type = "response")
  auc_boot <- auc(roc(d$apo, pred_boot))
  
  # Predict on original data using bootstrap model
  pred_orig <- predict(model_boot, newdata = data, type = "response")
  auc_orig <- auc(roc(data$apo, pred_orig))
  
  optimism <- auc_boot - auc_orig
  return(c(auc_boot, auc_orig, optimism))
}

# Run bootstrap with 10,000 replicates
results <- boot(data = data, statistic = bootstrap_function, R = 1000)

# Extract values
boot_aucs <- results$t[,1]
orig_aucs <- results$t[,2]
optimisms <- results$t[,3]

# Calculate statistics
auc_apparent <- mean(boot_aucs)
optimism_mean <- mean(optimisms)
auc_corrected <- auc_apparent - optimism_mean
ci_auc <- quantile(boot_aucs, c(0.025, 0.975))

# Print results
cat("Bootstrap-corrected AUC:", round(auc_corrected, 3), "\n")
cat("95% CI for AUC:", round(ci_auc[1], 3), "-", round(ci_auc[2], 3), "\n")
cat("Average optimism:", round(optimism_mean, 4), "\n")

# Plot distribution of bootstrap AUCs
hist(boot_aucs, col = "skyblue", main = "Distribution of Bootstrap AUCs", 
     xlab = "AUC", breaks = 50)
abline(v = auc_corrected, col = "red", lwd = 2, lty = 2)
legend("topright", legend = paste("Corrected AUC =", round(auc_corrected, 3)), 
       col = "red", lwd = 2, lty = 2)


# Plot distribution of bootstrap AUCs
hist(boot_aucs, col = "skyblue", main = "Distribution of Bootstrap AUCs", 
     xlab = "AUC", breaks = 50)
abline(v = auc_corrected, col = "red", lwd = 2, lty = 2)
legend(x = 0.9, y = 600, legend = paste("Corrected AUC =", round(auc_corrected, 3)), 
       col = "red", lwd = 2, lty = 2)


################################################################################
##                         Prediction Density Plot                            ##
################################################################################

# Predict probabilities
# data_complete$pred_prob <- predict(finalmodel, type = "response")

# Plot density of predicted probabilities, colored by outcome


library(ggplot2)
library(dplyr)

# Set cutoff
cutoff <- 0.6948581

# Create density data
density_data <- data_complete %>%
  group_by(apo_bin) %>%
  group_modify(~ {
    dens <- density(.x$pred_prob, from = 0, to = 1)
    data.frame(x = dens$x, y = dens$y)
  }) %>%
  ungroup()

# Label outcomes
density_data$outcome <- factor(density_data$apo_bin, levels = c(0, 1), labels = c("Non-case (0)", "Case (1)"))

# Create plot with corrected FP & FN labels
ggplot(density_data, aes(x = x, y = y, fill = outcome)) +
  geom_area(alpha = 0.5, position = 'identity') +
  geom_vline(xintercept = cutoff, linetype = "dashed", color = "black", linewidth = 1) +
  
  annotate("text", x = cutoff + 0.03, y = max(density_data$y)*0.9,
           label = "Cutoff", size = 4) +
  
  # Shade FP (Non-cases right of cutoff)
  geom_area(data = subset(density_data, outcome == "Non-case (0)" & x >= cutoff),
            aes(x = x, y = y), fill = "darkblue", alpha = 0.8) +
  
  # Shade FN (Cases left of cutoff)
  geom_area(data = subset(density_data, outcome == "Case (1)" & x < cutoff),
            aes(x = x, y = y), fill = "darkred", alpha = 0.8) +
  
  # Add FP count
  annotate("text", x = cutoff + 0.15, y = max(density_data$y)*0.75,
           label = "False Positives (15)", size = 4, color = "darkblue") +
  
  # Add FN count
  annotate("text", x = cutoff - 0.3, y = max(density_data$y)*0.75,
           label = "False Negatives (40)", size = 4, color = "darkred") +
  
  labs(title = "Prediction Density Plot with False Positive and False Negative Counts",
       x = "Predicted Probability",
       y = "Density",
       fill = "Actual Outcome") +
  
  scale_fill_manual(values = c("blue", "red")) +
  theme_minimal()


################################################################################
##----------------------------------------------------------------------------##
##                Decision Curve Analysis (DCA) for APO                       ##
##----------------------------------------------------------------------------##
################################################################################

library(ggplot2)
library(reshape2)
library(rmda)

# Convert outcome variable apo to numeric: 1 for "Yes", 0 for "No"
data_complete$apo_numeric <- ifelse(data_complete$apo == "Yes", 1, 0)

# Predicted risk scores
data_complete$pred_risk <- predict(finalmodel, type = "response")

# Run DCA with bootstrapped CIs
set.seed(123)  # For reproducibility
dca_result <- decision_curve(
  apo_numeric ~ pred_risk,
  data = data_complete,
  thresholds = seq(0.01, 0.99, by = 0.01),
  fitted.risk = TRUE,
  bootstraps = 1000  # You can increase B for more stable CIs
)

# Plot with confidence intervals
plot_decision_curve(
  dca_result,
  curve.names = "Logistic Model",
  cost.benefit.axis = TRUE,
  confidence.intervals = TRUE,  # Enable CIs
  standardize = TRUE,
  col = c("#1F77B4", "darkred", "darkgreen"),
  lty = c(1, 2, 3),
  lwd = 2,
  legend.position = "none",
  xlab = "Risk Threshold (Probability)",
  ylab = "Standardized Net Benefit",
  main = "Decision Curve Analysis for Adverse Birth Outcome")

# Add custom legend
legend(
  x = 0.95, y = 0.85,
  legend = c("Logistic Model", "Treat All", "Treat None"),
  col = c("#1F77B4", "darkred", "darkgreen"),
  lty = c(1, 2, 3),
  lwd = 2,
  bty = "n",
  xjust = 1,
  yjust = 1,
  cex = 0.9,
  x.intersp = 0.5,
  y.intersp = 0.8)

# Optional: Vertical reference lines
abline(v = c(0.2, 0.4, 0.6, 0.8), col = "gray80", lty = 3)

abline(v = c(0.6949), col = "red", lty = 3)


################################################################################
##----------------------------------------------------------------------------##
##                     Web Appliaction Development                            ##
##----------------------------------------------------------------------------##
################################################################################

#library(shiny)
#
#ui <- fluidPage(
#  titlePanel("Risk of Adverse Birth Outcome (APO) following Assisted Reproductive Technology Pregnancy"),
#  
#  sidebarLayout(
#    sidebarPanel(
#      h4("Patient Information"),
#      
#      selectInput("healthinsu", "Health Insurance:", 
#                  choices = c("No" = 1, "Yes" = 2)),
#      
#      selectInput("placeresid", "Place of Residence:", 
#                  choices = c("Urban" = 1, "Rural" = 2)),
#      
#      selectInput("reparity", "Parity:", 
#                  choices = c("Multiparous" = 1, "Primiparous" = 2)),
#      
#      selectInput("twin", "Twin Pregnancy:", 
#                  choices = c("No" = 1, "Yes" = 2)),
#      
#      selectInput("childgender", "Child Gender:", 
#                  choices = c("Male" = 1, "Female" = 2)),
#      
#      selectInput("iugr", "IUGR:", 
#                  choices = c("No" = 1, "Yes" = 2)),
#      
#      selectInput("infcause", "Infertility Cause:", 
#                  choices = c("Male Factor" = 1, "Female Factor" = 2, "Unexplained" = 3)),
#      
#      selectInput("afc", "Antral Follicle Count (AFC):", 
#                  choices = c("≥ 5" = 1, "<5" = 2)),
#      
#     selectInput("embryotype", "Embryo Type:", 
#                  choices = c("Fresh" = 1, "Frozen" = 2)),
#      
#      selectInput("embryostage", "Embryo Stage:", 
#                  choices = c("Blastocyte" = 1, "Cleavage" = 2)),
#      
#      selectInput("dmhx", "History of Diabetes Mellitus:", 
#                  choices = c("No" = 1, "Yes" = 2)),
#      
#      actionButton("predict", "Predict Risk")
#    ),
#    
#    mainPanel(
#      h3("Prediction Result"),
#      verbatimTextOutput("riskOutput")
#    )
#  )
#)

#server <- function(input, output) {

#  Logistic regression coefficients (from your final model)
#  coefs <- c(
#    `(Intercept)` = -0.7847,
#    healthinsu2 = 1.5812,
#    placeresid2 = -1.1472,
#    reparity2 = -1.2546,
#    twin2 = 1.3516,
#    childgender2 = 1.3716,
#    iugr2 = 1.5546,
#    infcause2 = 0.6935,
#    infcause3 = 1.8286,
#    afc2 = 2.3517,
#    embryotype2 = -1.7081,
#    embryostage2 = -2.2753,
#    dmhx2 = 2.3253
#  )

# Set optimal cutoff value (replace 0.65 by your real threshold)
#  cutoff <- 0.7703686

#  predict_risk <- function(input) {
#    x <- c(
#      1,  
#      as.numeric(input$healthinsu) == 2,
#      as.numeric(input$placeresid) == 2,
#      as.numeric(input$reparity) == 2,
#      as.numeric(input$twin) == 2,
#      as.numeric(input$childgender) == 2,
#      as.numeric(input$iugr) == 2,
#     as.numeric(input$infcause) == 2,
#      as.numeric(input$infcause) == 3,
#      as.numeric(input$afc) == 2,
#      as.numeric(input$embryotype) == 2,
#      as.numeric(input$embryostage) == 2,
#      as.numeric(input$dmhx) == 2
#    )
#    
#    names(x) <- names(coefs)
#    
#   lp <- sum(coefs * x)
#    prob <- 1 / (1 + exp(-lp))
#   return(prob)
#  }
#  
#  observeEvent(input$predict, {
#    prob <- predict_risk(input)
#    risk_class <- ifelse(prob >= cutoff, "HIGH RISK", "LOW RISK")
#    
#    output$riskOutput <- renderText({
#      paste0("Predicted Risk: ", round(prob * 100, 2), "%\nRisk Classification: ", risk_class)
#    })
#  })
#}
#
#shinyApp(ui = ui, server = server)


rsconnect::setAccountInfo(name='clinicalimplication',
                          token='C5EF3C450435C54C4409C6D89BDB417D',
                          secret='PmMTff3gTpT/cdwuv/TGBmRqdEk0nlpFNM3OxYV6')
