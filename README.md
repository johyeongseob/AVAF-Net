# AVAF-Net: Anti-View Attention-guided Feature Fusion for Defect Detection in Multi-Light Source Images 
### Anti-view attention model for defect detection
- Baseline: Multi-view fusion (MVF): https://github.com/Xavierman/Fusion-of-multi-light-source-illuminated-images-for-defect-inspection
- 1st model: Attention based model
- 2nd model: Transformer based model
- 3rd model: Anti-view attention based model (Proposed)



[Dataset download link] https://drive.google.com/drive/folders/1NvQ5vZvZMdpJN8s1ttp9ZaMZ13OgQbAa?usp=sharing<br>[Weights download link] https://drive.google.com/drive/folders/1FVbF3mjFJx-a3427OySH0EfhC32cLPC2?usp=sharing



### Weight results
- Baseline
![Image](https://github.com/user-attachments/assets/158cb349-0190-4522-b071-a9fa48b39cda)

- **Single-view**
- **Down**
  ```
  Test model model SingleViewClassifier, Weight path: ../weights/SingleView_Down_seed42.pth

  Epoch 0: Class 0: 81.08% (60/74), Class 1: 51.56% (33/64), Class 2: 90.48% (38/42), Class 3: 74.30% (133/179), Class 4: 72.59% (143/197), Total Accuracy: 73.20%.

  Process finished with exit code 0
  ```

- **Up**
  ```
  Test model model SingleViewClassifier, Weight path: ../weights/SingleView_Upper_seed42.pth

  Epoch 0: Class 0: 52.70% (39/74), Class 1: 65.62% (42/64), Class 2: 95.24% (40/42), Class 3: 76.54% (137/179), Class 4: 77.16% (152/197), Total Accuracy: 73.74%.

  Process finished with exit code 0
  ```
