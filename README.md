# Student-Performance-Prediction-System
Building a Deep Neural Network to predict whether a 03 student will pass or fail based on attendance, internal marks, and assignments. Dataset: UCI Student Performance Dataset


Data Exploration and Visualization Project on the Student-Mat Dataset Using Python

By Yogesh Sachdeva, Ayushi Arora and Kriti Suri

Introduction

This project focuses on exploring and visualizing the Student-Mat dataset using Python in order to extract meaningful insights. The analysis aims to understand the relationships between various categorical and numerical variables and their impact on students' academic performance.

The dataset contains students’ final mathematics grades along with several demographic, social, and academic factors that may influence their performance and future outcomes.

Objectives of the Project

    To explore and understand the dataset and its variables
    To perform data cleaning and preprocessing
    To conduct data visualization for identifying patterns and relationships
    To draw meaningful conclusions based on the analysis

Dataset Information
About the Dataset
The Student-Mat dataset is obtained from the UCI Machine Learning Repository. It contains records of students enrolled in a mathematics course, along with multiple attributes that may influence their academic performance.
The dataset includes demographic information, family background, social behavior, and academic-related features.

Attribute Description
Student Information

    school – Student’s school ('GP' – Gabriel Pereira or 'MS' – Mousinho da Silveira)
    sex – Student’s gender ('F' – Female, 'M' – Male)
    age – Age (15 to 22 years)
    address – Home address type ('U' – Urban, 'R' – Rural)
    famsize – Family size ('LE3' – ≤3 members, 'GT3' – >3 members)
    Pstatus – Parents’ cohabitation status ('T' – Together, 'A' – Apart)
Parents' Background
    Medu – Mother’s education level (0 = None, 1 = Primary, 2 = 5th–9th grade, 3 = Secondary, 4 = Higher education)
    Fedu – Father’s education level (0 = None, 1 = Primary, 2 = 5th–9th grade, 3 = Secondary, 4 = Higher education)
    Mjob – Mother’s occupation ('teacher', 'health', 'services', 'athome', 'other')
    Fjob – Father’s occupation ('teacher', 'health', 'services', 'athome', 'other')
Academic and Social Factors
    reason – Reason for choosing the school ('home', 'reputation', 'course', 'other')
    guardian – Student’s guardian ('mother', 'father', 'other')
    traveltime – Travel time to school (1 = <15 min, 2 = 15–30 min, 3 = 30–60 min, 4 = >60 min)
    studytime – Weekly study time (1 = <2 hrs, 2 = 2–5 hrs, 3 = 5–10 hrs, 4 = >10 hrs)
    failures – Number of past class failures
    schoolsup – Extra educational support (Yes/No)
    famsup – Family educational support (Yes/No)
    paid – Extra paid classes (Yes/No)
    activities – Participation in extracurricular activities (Yes/No)
    nursery – Attended nursery school (Yes/No)
    higher – Intention to pursue higher education (Yes/No)
    internet – Internet access at home (Yes/No)
    romantic – In a romantic relationship (Yes/No)

Lifestyle and Personal Factors
    famrel – Quality of family relationships (1 = Very bad to 5 = Excellent)
    freetime – Free time after school (1 = Very low to 5 = Very high)
    goout – Going out with friends (1 = Very low to 5 = Very high)
    Dalc – Workday alcohol consumption (1 = Very low to 5 = Very high)
    Walc – Weekend alcohol consumption (1 = Very low to 5 = Very high)
    health – Current health status (1 = Very bad to 5 = Very good)
    absences – Number of school absences (0 to 93)

Academic Performance Variables (Mathematics)
    G1 – First period grade (0 to 20)
    G2 – Second period grade (0 to 20)
    G3 – Final grade (0 to 20) – Target variable
    G_cumm – Cumulative score of G1, G2, and G3
