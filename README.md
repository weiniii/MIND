# MIND
mind: microsoft news recommendation. An easy way to roc auc score 62% by one hot encoding.

Fisrt, generate news vector through one-hot encoding of subcategories. 

Second, let user vectors or impression vector be the summation of all clicked history news or impression news through news vectors.

Then, calculate the dot product between user vectors and impression vectors.

Finally, you would get a little bit good performance, roc auc score is 0.61~0.62.
