import numpy as np
import polars as pl
from datetime import datetime, timedelta


def churn_dataset_generator(df, user_id, feature_pipeline, info_period=90, checking_period=30, start_date=datetime(2007, 1, 1), end_date=datetime(2007, 12, 31), step="1mo", time_col="datetime"):
    cutoff_start = pl.datetime_range(start_date, end_date, step, eager=True).alias(time_col);
    
    for start in cutoff_start.to_list():
        train_info = df.filter(pl.col(time_col) < start, pl.col(time_col) < (start - timedelta(days=info_period)))
        valid_info = df.filter(pl.col(time_col) > start, pl.col(time_col) < (start + timedelta(days=checking_period)))
    
        target = valid_info.select("player_id").unique().with_columns(target=True)

        ml_df = (train_info
                 .pipe(feature_pipeline)
                 .join(target, on=user_id, how="left")
                 .with_columns(target=pl.when(pl.col("target")).then(True).otherwise(False)))
        
        X = ml_df.drop("target", "player_id")
        y = np.array(ml_df["target"]).astype(int)
        
        yield X, y
