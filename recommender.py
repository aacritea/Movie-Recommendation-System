import lenskit.datasets as ds
import pandas as pd
data = ds.MovieLens('lab4-recommender-systems/')

print("Successfully installed dataset.")

rows_to_show = 15   
data.movies.head(rows_to_show)  

joined_data = data.ratings.join(data.movies['genres'], on='item')
joined_data = joined_data.join(data.movies['title'], on='item')
joined_data.head(rows_to_show)


average_ratings = (data.ratings).groupby(['item']).mean()
sorted_avg_ratings = average_ratings.sort_values(by="rating", ascending=False)
joined_data = sorted_avg_ratings.join(data.movies['genres'], on='item')
joined_data = joined_data.join(data.movies['title'], on='item')
joined_data = joined_data[joined_data.columns[1:]]

print("RECOMMENDED FOR ANYBODY:")
joined_data.head(rows_to_show)


average_ratings = (data.ratings).groupby('item') \
       .agg(count=('user', 'size'), rating=('rating', 'mean')) \
       .reset_index()

sorted_avg_ratings = average_ratings.sort_values(by="rating", ascending=False)
joined_data = sorted_avg_ratings.join(data.movies['genres'], on='item')
joined_data = joined_data.join(data.movies['title'], on='item')
joined_data = joined_data[joined_data.columns[1:]]

print("RECOMMENDED FOR ANYBODY:")
joined_data.head(rows_to_show)


minimum_to_include = 25
average_ratings = (data.ratings).groupby(['item']).mean()
rating_counts = (data.ratings).groupby(['item']).count()
average_ratings = average_ratings.loc[rating_counts['rating'] > minimum_to_include]
sorted_avg_ratings = average_ratings.sort_values(by="rating", ascending=False)
joined_data = sorted_avg_ratings.join(data.movies['genres'], on='item')
joined_data = joined_data.join(data.movies['title'], on='item')
joined_data = joined_data[joined_data.columns[3:]]

print("RECOMMENDED FOR ANYBODY:")
joined_data.head(rows_to_show)


average_ratings = (data.ratings).groupby(['item']).mean()
rating_counts = (data.ratings).groupby(['item']).count()
average_ratings = average_ratings.loc[rating_counts['rating'] > minimum_to_include]
average_ratings = average_ratings.join(data.movies['genres'], on='item')
average_ratings = average_ratings.loc[average_ratings['genres'].str.contains('Action')]

sorted_avg_ratings = average_ratings.sort_values(by="rating", ascending=False)
joined_data = sorted_avg_ratings.join(data.movies['title'], on='item')
joined_data = joined_data[joined_data.columns[3:]]

print("RECOMMENDED FOR AN ACTION MOVIE FAN:")
joined_data.head(rows_to_show)



average_ratings = (data.ratings).groupby(['item']).mean()
rating_counts = (data.ratings).groupby(['item']).count()
average_ratings = average_ratings.loc[rating_counts['rating'] > minimum_to_include]
average_ratings = average_ratings.join(data.movies['genres'], on='item')
average_ratings = average_ratings.loc[average_ratings['genres'].str.contains('Romance')]

sorted_avg_ratings = average_ratings.sort_values(by="rating", ascending=False)
joined_data = sorted_avg_ratings.join(data.movies['title'], on='item')
joined_data = joined_data[joined_data.columns[3:]]
print("RECOMMENDED FOR A ROMANCE MOVIE FAN:")
joined_data.head(rows_to_show)


import csv

jabril_rating_dict = {}
jgb_rating_dict = {}

with open("/content/lab4-recommender-systems/jabril-movie-ratings.csv", newline='') as csvfile:
  ratings_reader = csv.DictReader(csvfile)
  for row in ratings_reader:
    if ((row['ratings'] != "") and (float(row['ratings']) > 0) and (float(row['ratings']) < 6)):
      jabril_rating_dict.update({int(row['item']): float(row['ratings'])})
      
with open("/content/lab4-recommender-systems/jgb-movie-ratings.csv", newline='') as csvfile:
  ratings_reader = csv.DictReader(csvfile)
  for row in ratings_reader:
    if ((row['ratings'] != "") and (float(row['ratings']) > 0) and (float(row['ratings']) < 6)):
      jgb_rating_dict.update({int(row['item']): float(row['ratings'])})
     
print("Rating dictionaries assembled!")
print("Sanity check:")
print("\tJabril's rating for 1197 (The Princess Bride) is " + str(jabril_rating_dict[1197]))
print("\tJohn-Green-Bot's rating for 1197 (The Princess Bride) is " + str(jgb_rating_dict[1197]))

from lenskit.algorithms import Recommender
from lenskit.algorithms.user_knn import UserUser

num_recs = 10 

user_user = UserUser(15, min_nbrs=3) 
algo = Recommender.adapt(user_user)
algo.fit(data.ratings)

print("Set up a User-User algorithm!")

jabril_recs = algo.recommend(-1, num_recs, ratings=pd.Series(jabril_rating_dict)) 
joined_data = jabril_recs.join(data.movies['genres'], on='item')      
joined_data = joined_data.join(data.movies['title'], on='item')
joined_data = joined_data[joined_data.columns[2:]]
print("\n\nRECOMMENDED FOR JABRIL:")
joined_data

jgb_recs = algo.recommend(-1, num_recs, ratings=pd.Series(jgb_rating_dict))  
joined_data = jgb_recs.join(data.movies['genres'], on='item')      
joined_data = joined_data.join(data.movies['title'], on='item')
joined_data = joined_data[joined_data.columns[2:]]
print("RECOMMENDED FOR JOHN-GREEN-BOT:")
joined_data

combined_rating_dict = {}
for k in jabril_rating_dict:
  if k in jgb_rating_dict:
    combined_rating_dict.update({k: float((jabril_rating_dict[k]+jgb_rating_dict[k])/2)})
  else:
    combined_rating_dict.update({k:jabril_rating_dict[k]})
for k in jgb_rating_dict:
   if k not in combined_rating_dict:
      combined_rating_dict.update({k:jgb_rating_dict[k]})
      
print("Combined ratings dictionary assembled!")
print("Sanity check:")
print("\tCombined rating for 1197 (The Princess Bride) is " + str(combined_rating_dict[1197]))

combined_recs = algo.recommend(-1, num_recs, ratings=pd.Series(combined_rating_dict))  
joined_data = combined_recs.join(data.movies['genres'], on='item')      
joined_data = joined_data.join(data.movies['title'], on='item')
joined_data = joined_data[joined_data.columns[2:]]
print("\n\nRECOMMENDED FOR JABRIL / JOHN-GREEN-BOT HYBRID:")
joined_data
