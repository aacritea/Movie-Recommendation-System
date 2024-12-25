import lenskit.datasets as ds
import pandas as pd
data = ds.MovieLens('lab4-recommender-systems/')

print("Successfully installed dataset.")

rows_to_show = 15   
data.movies.head(rows_to_show)  
