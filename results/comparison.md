# Task 1/2/3 结果对比（NYT Test Set）

| 表示方法 | Accuracy | Macro-F1 | business F1 | politics F1 | sports F1 |
| --- | --- | --- | --- | --- | --- |
| Word Frequency | **0.9896** | **0.9751** | 0.9627 | 0.9644 | 0.9983 |
| Binary BoW | 0.9870 | 0.9717 | 0.9513 | 0.9677 | 0.9960 |
| Word2Vec-100d (NYT full) | 0.9861 | 0.9676 | 0.9478 | 0.9579 | 0.9971 |
| Word2Vec-100d (NYT train split) | 0.9853 | 0.9633 | 0.9398 | 0.9518 | 0.9983 |
| GloVe-6B-100d (pre-trained) | 0.9818 | 0.9585 | 0.9254 | 0.9548 | 0.9954 |
| BERT-base-uncased (max_len=64, 3 epochs) | 0.9792 | 0.9585 | 0.9386 | 0.9450 | 0.9919 |
| Word2Vec-100d (AG News) | 0.9809 | 0.9553 | 0.9219 | 0.9481 | 0.9960 |
