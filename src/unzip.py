import click
import logging
import yaml
import shutil
import os
from pathlib import Path


def unzip(zip_path, data_path):
    zip_path = 'crowd-counting.zip'

    # Extract the contents of the zip file
    shutil.unpack_archive(zip_path, data_path)

    # Clean up: remove the downloaded zip file
    os.remove(zip_path)


@click.command(name="unzip_data")
@click.option('--load_config_path', default='../config/load_config.yaml')
@click.option('--dataset', default='mall')
def main(load_config_path, dataset):
    """ Unzips and saves the datasets.
    """
    with open(load_config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    if dataset == 'mall':
        zip_path = config['mall_zip_path']
        data_path = config['mall_data_path']

    elif dataset == 'shanghai':
        zip_path = config['shan_zip_path']
        data_path = config['shan_data_path']

    else:
        raise Exception("Wrong dataset name")

    logger = logging.getLogger(__name__)
    logger.info(f"Unzip {dataset} data")

    unzip(zip_path, data_path)
    logger.info('All files are unzipped')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables

    main()
