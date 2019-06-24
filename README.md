# Description of Reproduction

###extract_news.py 
extract news from csv into news_folder

###data_process.py 
create pickle of daily news for each company ( {str : date : [str : newsID]} )

###bind.py 
bind news with company and delete news without binding

###vectorize.py 
doc2vec method for every news

###make_stock_trend.py 
create pickle of daily trend for each company ( {str : date : int : trend (-1,0,1)} )

###make_dataset.py 
create data for training and testing

###han.py 
model training

**Python 3.7 with Keras**
