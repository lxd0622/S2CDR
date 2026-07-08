# S$^2$CDR: Smoothing-Sharpening Process Model for Cross-Domain Recommendation
This is the official implementation of our paper **S$^2$CDR: Smoothing-Sharpening Process Model for Cross-Domain Recommendation** (WWW 2026)

## Requirements

- Python 3.6
- Pytorch > 1.0
- Pandas
- Numpy

## File Structure

```
.
├── code
│   ├── config.json         # Configurations
│   ├── entry.py            # Entry function
│   ├── models.py           # Smoothing-sharpening model
│   ├── preprocessing.py    # Preprocess the dataset
│   ├── dataloader_src.py   # Dataloader for cold-start users in source domain
│   ├── dataloader_tgt.py   # Dataloader for cold-start users in target domain
│   ├── utils.py            # Metrics
│   ├── readme.md
│   └── run.py              # Inference 
```

## Dataset

We utilized the Amazon Reviews 5-score dataset and Douban dataset. 
To download the Amazon dataset, you can use the following link: [Amazon Reviews](http://jmcauley.ucsd.edu/data/amazon/links.html).
Download the three domains: [CDs and Vinyl](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_CDs_and_Vinyl_5.json.gz), [Movies and TV](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz), [Books](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz) (5-scores), and then put the data in `./data/raw`.
To download the Douban dataset, you can use the following link: [Douban](https://www.kaggle.com/datasets/fengzhujoey/douban-datasetratingreviewside-information?resource=download).

## Run

Parameter Configuration:

- task: different tasks within `1, 2 or 3`, default for `1`
- base_model: different base models, default for `SSBaseModel`
- ratio: train/test ratio, default for `[0.8, 0.2]`

You can run this model through:

```powershell
# Run directly with default parameters 
python entry.py
```

## Acknowledgements

Our code is based on the implementations of [BSPM](https://github.com/jeongwhanchoi/BSPM) and [DMCDR](https://github.com/lxd0622/DMCDR).
