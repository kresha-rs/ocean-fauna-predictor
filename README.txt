{\rtf1\ansi\ansicpg1252\cocoartf2821
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\froman\fcharset0 Times-Bold;\f2\froman\fcharset0 Times-Roman;
}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 The needed datasets for this is in the data folder \
Current_stats  dataset has the list of organisms present in the given latitude and longitude locations (just the ones from the organisms dataset which is about 309 organisms)\
Organisms dataset contains information on those organisms and their salinity and temperature tolerance range in ppt and celsius.\
The ocean csv with which the training for prediction is a wide column representation of the salinity and temperature readings for a variety of years (
\f1\b \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 1991\'962020 Climate Normal
\f2\b0  (baseline stability)\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b \cf0 2015\'962022 Decadal Data
\f2\b0  (recent trends)
\f0 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 \
The merged ocean csv is the csv capturing ocean\'92s current status for a variety of parameters but we\'92re only considering salinity and temperature \
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0
\cf0 \
The models which were trained for the prediction and classification is uploaded in the models folder \
The ocean conditions for the years 1900-2100 can be calculated for all those locations and depth levels which are binned into 4 pelagic zones for simplicity\
These models were trained specifically for each climate zone (arctic, temperate north, tropical north, tropical south, temperate south and antarctic) for better classification\
[im physically not able to upload the models due to upload size issues the models folder is around 50gb so the python files used to create the models and the datasets\'85. So u have to first run the ocean predictor python file first and then the run predictor.py]\
\
The flask app makes an interactive interface for the specific and comprehensive analysis through which we can enter the latitude and longitude values along with year of interest and mention zone in specific analysis to get a list of organisms with its pictures and the comprehensive analysis to get a summary report on the condition of each depth level and the summary of the organisms there \
\
The ocean predictor.py is the python file that processes the models as described above and saves it in models folder and the run predictor.py will run the saved models to avoid retraining }