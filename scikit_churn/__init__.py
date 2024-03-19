import numpy as np
import polars as pl
from datetime import datetime, timedelta


def churn_dataset_generator(df, user_id, feature_pipeline, 
                            info_period=180, 
                            checking_period=180, 
                            start_date=datetime(2007, 1, 1), 
                            end_date=datetime(2007, 12, 31), 
                            step="1mo", 
                            time_col="datetime"):
    """
    Generates X,y pairs for churn related machine learning, with way less temporal data leaks to worry about. 

    Arguments:

    - df: a Polars dataframe that contains logs over time for users
    - user_id: the column name that depicts the user id
    - feature_pipeline: a Polars compatible function that generatres ML features to go in `X`
    - input_period: the number of days that the input period lasts
    - checking_period: the number of days that the checking period lasts
    - start_date: the start date for X,y-pair generation
    - end_date: the end date for X,y-pair generation
    - step: stepsize over time for new X,y-pairs. defaults to a month. 
    - time_col: column name that depicts the datetime stamp
    """
    cutoff_start = pl.datetime_range(start_date, end_date, step, eager=True).alias(time_col)
    min_date = df[time_col].min()
    max_date = df[time_col].max()
    
    for start in cutoff_start.to_list():
        info_period_start = start - timedelta(days=info_period)
        checking_period_end = start + timedelta(days=checking_period)
        if info_period_start < min_date:
            continue
        if checking_period_end > max_date:
            continue
        print(info_period_start, start, checking_period_end, min_date, max_date)
        train_info = df.filter(pl.col(time_col) < start, pl.col(time_col) >= (start - timedelta(days=info_period)))
        valid_info = df.filter(pl.col(time_col) >= start, pl.col(time_col) < (start + timedelta(days=checking_period)))
        
   
        target = valid_info.select("player_id").unique().with_columns(target=True)

        ml_df = (train_info
                 .pipe(feature_pipeline)
                 .join(target, on=user_id, how="left")
                 .with_columns(target=pl.when(pl.col("target")).then(True).otherwise(False)))
        
        X = ml_df.drop("target", "player_id")
        y = np.array(ml_df["target"]).astype(int)
        
        yield X, y