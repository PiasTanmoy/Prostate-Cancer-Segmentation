
# model_train4.py 
this is a multiclass segmentation model which uses only T2W modality as training input and predicts segmentation.
I used balanced class weights in the loss function.
Epoch 1 performance is very good for all classes

Epoch 1 | Avg Train Loss: 1.5396 | Avg Train Dice (present): 0.8792 | Avg Train IoU (present): 0.8729
Epoch 1 | Train Per-Class Dice (present): {0: 0.9723124233414145, 2: 0.6790209512707915, 3: 0.9083712625990445, 4: 0.9458974018330406, 5: 0.9655209076861657}
Epoch 1 | Train Per-Class IoU (present): {0: 0.9570104083594154, 2: 0.6790121550636368, 3: 0.9083648458800612, 4: 0.9458974018330406, 5: 0.9655209076861657}
[Epoch 1] Train Loss: 1.5396, Val Dice (present): 0.9816, Val IoU (present): 0.9816
[Epoch 1] Val Per-Class Dice (present): {0: 0.9999423293690932, 2: 0.8924999795854092, 3: 0.8933333059151968, 4: 0.9249999821186066, 5: 0.9266666571299235}
[Epoch 1] Val Per-Class IoU (present): {0: 0.9998851826316432, 2: 0.8924999795854092, 3: 0.8933333059151968, 4: 0.9249999821186066, 5: 0.9266666571299235}


# model_train5.py
Add other performance metrics 
AUROC, precision, recall, IOU

# model_train6.py
Use binary classification task. 
Convert all 2-5 to 1


# model_train7.py
use all 5 modalities as input 


# model_train8.py
use all 5 modalities as input - semi-supervised learning (205)


# model_train9.py
1. use all 5 modalities as input. Don't use 5 instances which doesn't have all 5 modalities. 
2. cost sensitive learning (class weights)
3. contrastive learning (self-supervised).
4. resize images to 256x256 using cv2
5. don't load the whole dataset in the dataloader
6. train an multi-modal multi-class u-net model
7. Metrics: Dice, IOU, precision, recall, f1-score, AUROC, PR curve - for each class [0, 2, 3, 4, 5] # class 1 is not present in the mask.
8. Save the trained model after each epoch. train for 5 epochs max
9. Split the patient study into train, validation, and test (not the slices). Make the slices after spliting.
10. Use the best saved model to predict segmentation on 220 unannotated images
11. Use these 220 artificial segment masks to train a new model or pretrain the best saved model.
12. Record the performance. 



# model_train10.py
use marksheet.csv as another modality input with the images.
