import os
import sys
import glob
import json
import time
import shutil

import polars as pl


def main():
    start_time = time.time()

    # Load settings files
    settings_dict = load_json_to_dict('settings.json')
    db_settings_dict = load_json_to_dict(settings_dict['db_settings_filepath'])

    # Postgres connection string
    uri = f"postgresql://{db_settings_dict['db_username']}:{db_settings_dict['db_password']}@{db_settings_dict['db_hostname']}:{db_settings_dict['db_port']}/{db_settings_dict['db_name']}"

    # Check for new .csv files
    wd_csv_file_dict = check_for_new_files(uri,
                                           settings_dict['path_to_check'],
                                           settings_dict['input_file_mask'],
                                           db_settings_dict,
                                           settings_dict)

    # All transformations and loading to postgres
    main_processing(uri, wd_csv_file_dict, db_settings_dict)
    
    archive_processed_files(list(wd_csv_file_dict.keys()), settings_dict)

    print(f'\nTotal Proccessing Time: {round(time.time() - start_time, 2)} seconds')


def archive_processed_files(file_lst: list, settings_dict: dict) -> None:
    # Create archives dir if it doesn't exist
    archives_location = os.path.join(settings_dict['path_to_check'], settings_dict['archive_dir'])
    if not os.path.isdir(archives_location):
        os.makedirs(archives_location)
        print(f'Archive folder has been created: {archives_location}')
    
    for file in file_lst:
        filename_with_no_ext = os.path.basename(file)[:-4]

        # Add current datetime to the archived filename
        timestr = time.strftime("%Y%m%d%H%M%S")
        new_filepath = os.path.join(archives_location, f'{filename_with_no_ext}_{timestr}.csv')
        
        try:
            shutil.move(file, new_filepath)
            print(f'Archived: {os.path.basename(file)} --> {os.path.basename(new_filepath)}')
        except Exception as e:
            print(e)


def main_processing(uri: str, wd_csv_file_dict: dict, db_settings_dict: dict) -> None:
    for file, sensor in wd_csv_file_dict.items():
        start_time_file = time.time()
        
        print(f'\nNow processing {os.path.basename(file)}')
        
        try:
            # Get existing data from the database
            existing_timestamps_df = get_postgres_tbl(uri, db_settings_dict['db_schema'], db_settings_dict['wd_tbl'], ['timestamp'])
            print(f'Currently {len(existing_timestamps_df)} weather observations in the database')
        except Exception as e:
            print(e)
            sys.exit(1)

        # Get csv data in lazy dataframe
        lazy_wd_df = csv_to_lazy_df(file)
        # Transform timestamp column
        lazy_wd_df = lazy_wd_df.with_columns(pl.col('timestamp').str.replace(',', ''))
        lazy_wd_df = lazy_wd_df.with_columns(pl.col('timestamp').str.strptime(pl.Datetime, format='%b %d %Y %H:%M'))
        wd_df = lazy_wd_df.with_columns(sensor = pl.lit(sensor)).collect()
        
        # Clones a dataframe's schema but no data
        df = wd_df.clear()
        
        counter = 0
        for row in wd_df.rows(named=True):
            # Check if the specific row/data/timestamp is already in the database to avoid duplication
            if not row['timestamp'] in existing_timestamps_df['timestamp']:
                counter += 1
                row_df = pl.from_dict(row).with_columns(pl.col("rel_humidity_PC").cast(pl.Int32), pl.col("sensor").cast(pl.Int32))
                df.extend(row_df)
        
        # Load new weather observations into postgres
        df.write_database(table_name = f'{db_settings_dict["db_schema"]}.{db_settings_dict["wd_tbl"]}',
                          connection = uri,
                          engine = 'adbc',
                          if_exists = 'append')
        
        print(f'{os.path.basename(file)} -> Total rows: {len(wd_df)} Appended rows: {counter}')

        existing_timestamps_df = wd_df = df = None
        
        print(f'Proccessed in {round(time.time() - start_time_file, 2)} seconds')


def check_for_new_files(uri: str, path: str, mask: str, db_settings_dict: dict, settings_dict: dict) -> dict:
    try:
        sensors_df = get_postgres_tbl(uri, db_settings_dict['db_schema'], db_settings_dict['sensors_tbl'], ['*'])
    except Exception as e:
        print(e)
        sys.exit(1)
        
    # Get all weather data csv files in list
    wd_csv_file_lst = glob.glob(os.path.join(path, f'*{mask}*.csv'))

    wd_csv_file_dict = {}
    for i, file in enumerate(wd_csv_file_lst):
        sensor_name = os.path.basename(file).split(settings_dict['input_file_mask'])[0]
        
        # Check if sensor name is in the 'sensors' db table - if not skip processing data
        if len(sensors_df.filter(pl.col("s_name").str.contains(sensor_name))) != 1:
            print(f'File {os.path.basename(file)} won\'t be processed as it contains data from an *unknown* sensor (Sensor: {sensor_name})')
            del wd_csv_file_lst[i]
        else:
            wd_csv_file_dict[file] = sensors_df.filter(pl.col("s_name").str.contains(sensor_name))['s_id'][0]
    
    del wd_csv_file_lst
    
    # Check if any wd csv files found
    if len(wd_csv_file_dict)==0:
        print(f'No new files found. [location: {os.path.join(path, f"*{mask}*.csv")}]')
        sys.exit()
    
    # Inform user
    for k, v in wd_csv_file_dict.items():
        print(os.path.basename(k))
    print(f'Total files found: {len(wd_csv_file_dict)}')
    
    return wd_csv_file_dict


# Get Postgres Table as Dataframe
def get_postgres_tbl(uri: str, schema: str, tbl: str, cols: list) -> pl.DataFrame:
    query = f"SELECT * FROM {schema}.{tbl}"
    if cols[0] != '*':
        query = query.replace('*', ', '.join(cols))
    return pl.read_database_uri(query=query, uri=uri, engine='adbc')


def csv_to_lazy_df(file: str) -> pl.LazyFrame:
    return pl.scan_csv(
        file,
        skip_rows = 1,
        has_header = False,
        with_column_names = lambda cols: ['timestamp',
                                          'temp_C',
                                          'rel_humidity_PC',
                                          'dpt_C',
                                          'vpd_kPa',
                                          'abs_humidity_G_M3'],
        dtypes = {'timestamp': str,
                  'temp_C': pl.Float64,
                  'rel_humidity_PC': pl.Int32,
                  'dpt_C': pl.Float64,
                  'vpd_kPa': pl.Float64,
                  'abs_humidity_G_M3': pl.Float64}
        )


def load_json_to_dict(filepath: str) -> dict:
    try:
        with open(filepath) as json_file:
            json_contents_dict = json.load(json_file)
            print(f'Json file found and loaded: {os.path.basename(filepath)}')
            return json_contents_dict
    except Exception as e:
        print(e)
        print(f'Error while loading json file: {os.path.basename(filepath)}')
        sys.exit(1)


if __name__ == "__main__":
    main()