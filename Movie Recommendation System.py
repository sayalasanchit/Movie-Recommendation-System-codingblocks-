import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

column_names=['user_id', 'item_id', 'rating', 'timestamp']
df=pd.read_csv('ml-100k/u.data', sep='\t', names=column_names)
movie_titles=pd.read_csv('ml-100k/u.item', sep='\|', header=None)

movie_titles=movie_titles[[0,1]]
movie_titles.columns=['item_id', 'title']
df=pd.merge(df, movie_titles, on='item_id')
movie_mat=df.pivot_table(index='user_id', values='rating', columns='title')

ratings=pd.DataFrame(df.groupby('title').mean()['rating'])
ratings['num of ratings']=pd.DataFrame(df.groupby('title').count()['rating'])

def predict_movies(movie_name):
    movie_ratings=movie_mat[movie_name]
    similar2movie=movie_mat.corrwith(movie_ratings)
    corr_movie=pd.DataFrame(similar2movie, columns=['correlation'])
    corr_movie.dropna(inplace=True)
    corr_movie=corr_movie.join(ratings['num of ratings'])
    corr_movie=corr_movie[corr_movie['num of ratings']>100]
    predictions=corr_movie.sort_values('correlation', ascending=False)
    return list(predictions.index[1:10])

print('Movie list:')
all_movies=list(movie_titles['title'].sort_values())
for movie in all_movies:
	print(movie)
print()
movie_name=input("Enter the full name of the movie: ")
print()
if movie_name in all_movies:
	print('Similar movies:')
	predictions=predict_movies(movie_name)
	for movie in predictions:
		print(movie)
else:
	print('Movie not found.')