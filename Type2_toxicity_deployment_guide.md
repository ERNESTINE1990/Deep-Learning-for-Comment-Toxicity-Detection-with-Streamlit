
##  Deployment Guide for Toxic Comment Streamlit App

This guide will help you set up and run the **Streamlit Toxic Comment Classification App**.

---

##  Requirements

Ensure you have the following installed:

- python
- pip install streamlit pandas numpy scikit-learn tensorflow 
- matplotlib


Or install via `requirements.txt`:

streamlit
pandas
numpy
scikit-learn
tensorflow
matplotlib


---

## Run the App Locally

1. Save the `Type2_toxicity.py` file.
2. Open a terminal in the same directory.
3. Run the app: streamlit run Type2_toxicity.py

---


##  Dataset

Make sure your dataset (CSV) matches the format used in training:

- Load test dataset
- Load train dataset

You can manually upload the dataset in the app interface.

---

##  Tip

- Use small datasets while testing to reduce training time.
- You can increase `epochs` or `layers` in the script to improve performance.

---


