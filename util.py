import geopandas as gpd
import pyarrow.parquet as pq
import json
from pathlib import Path
import time
import rtree

def generate_bounds(file, batch_size=10000):
    ds = pq.ParquetDataset(file, use_legacy_dataset=False)
    total_rows = ds._dataset.count_rows()
    for i, batch in enumerate(ds._dataset.to_batches(columns=['geometry'], batch_size=batch_size)):
        offset = i*batch_size
        m = f"-- Building Spatial Index: Progress {offset}/{total_rows} | {(offset*100/total_rows):0.2f} %"
        print(m, end='\r')
        for i, bounds in enumerate(gpd.GeoSeries.from_wkb(batch.to_pandas()['geometry']).bounds.values):
            yield (offset+i, bounds, None)             

def get_indexes(file, bounds):
    sindex_file = file.parent / (file.stem+'_sindex')
    if not sindex_file.with_suffix('.idx').exists():
        rtree.index.Index(str(sindex_file), generate_bounds(file))
    index = rtree.index.Index(str(sindex_file))
    return list(index.intersection(bounds))
    

def read_parquet_by_indexes(file, indexes):
    ds = pq.ParquetDataset(file, use_legacy_dataset=False)
    df = ds._dataset.take(indexes).to_pandas()
    df['geometry'] = gpd.GeoSeries.from_wkb(df['geometry'])
    crs = json.loads(ds.schema.metadata[b'geo'])['columns']['geometry']['crs']
    return gpd.GeoDataFrame(df, geometry='geometry', crs=crs)

def read_parquet_by_bounds(file, bounds):
    file = Path(file)
    
    start = time.time()
    print("Reading Index")
    indexes = get_indexes(file, bounds)
    print(f"Read Index: Time taken: {time.time() - start}")

    print("Running query")
    start = time.time()
    df = read_parquet_by_indexes(file, indexes)
    print(f"Query completed: time taken {time.time() - start} seconds")
    return df
