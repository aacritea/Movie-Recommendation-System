import lenskit.datasets as ds
import pandas as pd
data = ds.MovieLens('lab4-recommender-systems/')

print("Successfully installed dataset.")

rows_to_show = 15   
data.movies.head(rows_to_show)  

joined_data = data.ratings.join(data.movies['genres'], on='item')
joined_data = joined_data.join(data.movies['title'], on='item')
joined_data.head(rows_to_show)