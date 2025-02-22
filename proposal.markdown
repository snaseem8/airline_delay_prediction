---
layout: page
title: "Proposal"
permalink: /Proposal
---

## Intro + Background

Flight delays and cancellations have become a critical issue in the aviation industry, causing significant economic losses and passenger inconvenience. In 2022 alone, flight disruptions generated an economic impact of $30-34 billion in the US [4]. For our project, we will be creating a weather prediction model using K-Means and combining it with a Linear Regression and Random Forest Regression to examine the linear relationships between weather conditions and delay times. The Linear Regression and Random Forest Regression will be compared against each other to see which performs better.

### Literature Review

A recent study used various supervised models to see what factors have the biggest impact on delay predictions [2]. They found that a hybrid approach of combining decision tree with K-Means cluster classification gave the highest accuracy. Similarly, a study by Y. Tang also examining several flight delay algorithms found that tree-based classifiers performed better compared to others [1]. In contrast, a study by M. C. A. Clare, et al found success using unsupervised clustering for identifying wind patterns and turbulence on turbines [3]. By extracting meaningful weather patterns without labeled data, these methods provide valuable insights that can be integrated into flight delay models to account for atmospheric disturbances.

### Dataset Description

Two datasets will be used for this project with the first being the Airline Flight Dataset [8]. This dataset contains information about the flight schedules, and aircraft types.  We will be relying on the Weather Observations in Aviation industry  for our weather data [9]. This dataset contains weather conditions of past flights.  

## Problem Definition

The disruptions to airline operations discussed above arise from multiple factors—weather, air traffic control, crew availability, and technical issues. Notably, weather contributes substantially, with reduced visibility accounting for 52% of weather-related delays [7]. Traditional methods struggle to capture the complexity of these variables, underscoring the need for a robust machine learning model that can more accurately forecast potential delays and help airlines optimize operations. While many existing studies rely on standard historical airline and weather data and features, our project proposes to incorporate unique inputs—such as social media mentions or other under-explored indicators—to capture additional factors that correlate with flight delays.  

## Methods

Data cleaning will handle missing values by imputing weather conditions using historical data and removing extreme outliers in delay times. Feature engineering will extract key variables such as temperature deviations, precipitation intensity, wind speed, and visibility, which correlate with delays. Temporal features—like time of day, day of the week, and seasonality—will capture recurring delay patterns. To reduce redundancy from correlated weather variables, dimensionality reduction techniques like Principal Component Analysis (PCA) or feature selection will be applied.

K-Means clustering will group airports based on historical weather patterns and delay frequencies, helping identify distinct regional weather-delay patterns. Linear Regression will serve as a baseline model, while Random Forest Regression, which handles feature interactions, nonlinearity, and missing data, will provide a more robust prediction of delay times.

## Potential Results + Discussion

To evaluate our models’ ability to predict flight delays from weather data, we will use multiple metrics. Treating a predicted delay as a positive result, false negatives (missed delays) are more costly than false positives (unnecessary delay warnings), as they can trigger cascading disruptions and missed connections. Recall is crucial for correctly identifying delays, while precision helps minimize unnecessary alerts that could disrupt passenger flow. Since both are important, we will also use the F1 score, which balances them. Additionally, k-fold cross-validation will assess model generalization across seasons and prevent overfitting.

Our goal is to predict flight disruptions using weather data. We expect to prioritize recall over precision while maintaining a balance, reflected in a strong F1 score.

## References

[1] Y. Tang, “Airline Flight Delay Prediction Using Machine Learning Models,” 2021 5th International Conference on E-Business and Internet, Oct. 2021, doi: <https://doi.org/10.1145/3497701.3497725>

[2] H. Khaksar and A. Sheikholeslami, “Airline delay prediction by machine learning algorithms,” Scientia Iranica, vol. 0, no. 0, Dec. 2017, doi: <https://doi.org/10.24200/sci.2017.20020>

[3] M. C. A. Clare, S. C. Warder, R. Neal, B. Bhaskaran, and M. D. Piggott, “An Unsupervised Learning  Approach for Predicting Wind Farm Power and Downstream Wakes Using Weather Patterns,” Journal of Advances in Modeling Earth Systems, vol. 16, no. 2, Feb. 2024, doi: <https://doi.org/10.1029/2023ms003947>

[4] “AirHelp Report: The impact of flight disruption on the economy and environment,” AirHelp, Sep. 26, 2023. Available: <https://www.airhelp.com/en-gb/press/airhelp-report-the-impact-of-flight-disruption-on-the-economy-and-environment/>

[5] J. Knutson, “Airline issues leading cause for flight delays, federal data shows,” Axios, May 11, 2023. Available: <https://www.axios.com/2023/05/11/flight-delays-airlines-data>

[6] H. Bhanushali, “Impact of Flight Delays,” ClaimFlights, May 15, 2023. Available: <https://claimflights.com/impact-of-flight-delays/>

[7] J. A. Algarin Ballesteros and N. M. Hitchens, “Meteorological Factors Affecting Airport Operations during the Winter Season in the Midwest,” Weather, Climate, and Society, vol. 10, no. 2, pp. 307–322, Apr. 2018, doi: <https://doi.org/10.1175/wcas-d-17-0054.1>

[8] Arun Jangir, “Airline Flight Dataset: Schedule, Performance etc,” Kaggle.com, 2023. Available: <https://www.kaggle.com/datasets/arunjangir245/airline-flight-dataset-schedule-performance-etc>. [Accessed: Feb. 22, 2025]

[9] A. Viswanath, “Weather Observations in Aviation industry,” Kaggle.com, 2024. Available: <https://www.kaggle.com/datasets/aadharshviswanath/flight-data>

## Gantt Chart

<https://gtvault-my.sharepoint.com/:x:/g/personal/clolley3_gatech_edu/EXdscbhyK1pFmTAfbqBlGB8BNwvi2sRJRQbh82muJ2Q8Yw?e=AXqmsq&nav=MTVfezAwMDAwMDAwLTAwMDEtMDAwMC0wMDAwLTAwMDAwMDAwMDAwMH0>

## Contribution Table

| Name | Proposal Contribution |
|------|-----------------------|
| Allen Gao | Potential Results + Discussion |
| Chase Lolley | Github Pages setup & Problem Definition |
| Shahameel Naseem | Methods |
| Sidney Wise | Intro + Background, References, & Gantt Chart |
| Steven Haener | YouTube Video |

## YouTube Video/Slideshow

<https://youtu.be/epY78fKEqLc>
