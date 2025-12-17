Customer_Segmentation_Lantern
This repository is based on smart meter and survey data provided by the Swiss Energy Transition (SWEET) research consortium (see https://sweet-lantern.ch/). The framework was developed as part of a semester project at the Urban Energy Systems Laboratory at Empa Dübendorf to investigate different customer segments.

The repository contains a Jupyter‑notebook–based framework for customer segmentation using smart meter and survey data. The framework was originally developed for the “SWW” dataset and subsequently extended to support the “GroupE” dataset.

The segmentation pipeline explores k‑means, k‑medoids, and k‑prototypes clustering to identify distinct customer groups based on consumption profiles and categorical attributes. The resulting clusters can be further analyzed in the Survey_Analysis code base, which provides a more detailed characterization of the identified customer segments.

Extending the framework to other datasets

To adapt the existing framework to a new dataset:

- Create a new preprocessing notebook (e.g. Preprocessing_<Dataset>.ipynb) and adapt the section “1. RAW DATA PROCESSING: Read raw data and create individual files for households” to generate the same directory and file structure as for the existing datasets.

- Extract the same set of features from the accompanying survey (Excel-based) and update all relevant files to handle the new survey data. The overall code structure can remain unchanged; only the dataset-specific branching  and the column/row indices for the new survey need to be adapted.

