## To Run

To learn interpretable embeddings and topics using ETM on the 20NewsGroup dataset, run
```
python main.py --mode train --dataset 20ng --data_path data/20ng --num_topics 50 --train_embeddings 1 --epochs 1000
```

To evaluate perplexity on document completion, topic coherence, topic diversity, and visualize the topics/embeddings run
```
python main.py --mode eval --dataset 20ng --data_path data/20ng --num_topics 50 --train_embeddings 1 --tc 1 --td 1 --load_from CKPT_PATH
```

To learn interpretable topics using ETM with pre-fitted word embeddings (called Labelled-ETM in the paper) on the 20NewsGroup dataset:

+ first fit the word embeddings. For example to use simple skipgram you can run
```
python skipgram.py --data_file ng20.txt --emb_file results/embeddings.emb --dim_rho 300 --iters 150 --window_size 4 
```

+ then run the following 
```
python3.8 main.py --mode train --dataset nytimes --data_path data/20ng --emb_path results/embeddings.emb --num_topics 30 --train_embeddings 0 --epochs 10 --queries american yummy budget frustrating success birthday --emb_size 300 --rho_size 300
```

## Citation

```
@article{dieng2019topic,
  title={Topic modeling in embedding spaces},
  author={Dieng, Adji B and Ruiz, Francisco J R and Blei, David M},
  journal={arXiv preprint arXiv:1907.04907},
  year={2019}
}
```

