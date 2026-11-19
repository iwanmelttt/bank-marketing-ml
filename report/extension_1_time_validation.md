Extension 1: Time-based validation

In the main results, the data was shuffled randomly before it was split into training and test sets. This means the model could learn from calls that happened later than the calls it was tested on. In real life, a bank can only learn from past calls and must predict future ones. So in this extension we trained the models on the earliest 80% of calls and tested them on the latest 20%. This is a fairer and more realistic test.

The two periods were very different. In the training period, only 6.4% of clients subscribed. In the test period, 30.8% subscribed. This happened because the economy changed after the 2008 financial crisis. The Euribor interest rate fell from about 4.3% to about 1.0%, so the later period looked very different from anything the models had learned from.

We compared models using ROC-AUC, because it is not affected by the change in subscription rate. All models got worse. Random Forest dropped from 0.811 to 0.721, Gradient Boosting from 0.812 to 0.647, and Soft Voting from 0.813 to 0.683. Random Forest held up best, even though it was not the top model in the random split.

This shows that the model can become outdated when the economy changes. The bank should retrain the model regularly with recent data, and check its performance every month.