[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/6ndC2138)


This project focuses on vegetataion health classification using RGB images. Satellite data vegetation quality can be scored using NDVI formula. However, this requires the image to have the near infrared(NIR) band. Not all satellites have this band, so we are using machine learning to predict vegetation quality using the RGB bands of the images. The ground truths are still generated using the NDVI formula. However, the model only makes predictions using the RGB bands. This can help expand vegetation classification abilities to satelites that do not capture the NIR band can vegetation predictions

Run prepare_data.py to process data for our vegetation prediction task.
