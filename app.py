import click

import urllib.request
import zipfile

# Download the file from `url` and save it locally under `file_name`:
@click.group()
def main():
    pass

@main.command("download")
def download():
    print("here I should be downloading the data...")
    url='http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset'
    urllib.request.urlretrieve(url, '/images/files.zip')