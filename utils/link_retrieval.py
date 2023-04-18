"""
Requests youtube URLs from waldo api that meet requirements
Validates fetched URLs, and saves them to a csv file
"""
import argparse
import json
import os
import pandas as pd
import requests
import validators
from common import ensure_dir_exists
from pathlib import Path

# Setup command line arguments
parser = argparse.ArgumentParser(description="Get URL's from API and store them locally")
parser.add_argument("-e", "--endpoint", help='Target URL On Server', required=False,
                    default="https://waldo.vision/api/analysis/urls", type=str)
parser.add_argument("-k", "--key", help='API Key', required=False,
                    default=(os.environ.get("WALDO_API_KEY")), type=str)
parser.add_argument("-i", "--id", help='API Key ID', required=False, type=str,
                    default=(os.environ.get("WALDO_API_ID")))
parser.add_argument("-o", "--output", help='Folder To Store Output', required=True, type=str)
parser.add_argument("-r", "--requirements", help='Specfications for links retrieved',
                    default='{"minReviews": 25, "rating": 90}', required=False, type=str)

args = vars(parser.parse_args())


def parse_data(data):
    """
    Convert the data to a pandas DataFrame. If you find a better solution, please fix.
    We're expecting data that looks like this:
    {"page": 0,
    "totalPages": 100,
    "gameplay": [{"uuid": "38r8hnf80ew-35uhehrfnjea-wryhghjvsdk",
                  "ytUrl": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
                {"uuid": "38r8hnf80ew-35uhehrfnjea-wryhghjvsdk",
                 "ytUrl": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}]
    }
    """
    response_dataframe = pd.DataFrame(data)
    response_list = response_dataframe['gameplay'].tolist()
    response_dataframe = pd.DataFrame(columns=['id', 'url', 'game'])
    for obj in response_list:
        obj_dataframe = pd.DataFrame(obj, index=[0])
        obj_dataframe.rename(columns={"id": "id", "ytUrl": "url", "game": "game"}, inplace=True)
        response_dataframe = pd.concat([response_dataframe, obj_dataframe], ignore_index=True)
    # Validate the URLs
    for row in response_dataframe['url']:
        if not (validators.url(row)):
            return print("Invalid URL: " + row['url'])
    return response_dataframe


# def main(requirements):
def main():
    """
    Pulls URLs from the DB that meet the criteria in the requirements argument
    Example requirements
    {
        "minReviews": 25,
        "rating": 90,
    }
    """

    endpoint = args['endpoint']
    params = {
        # "requirements": requirements,
        "page": 0
    }
    headers = {'authorization': args['key'], 'authorization_id': args['id']}

    # Make the API request and retrieve the data
    response = requests.get(endpoint, params=params, headers=headers, timeout=10)
    data = response.json()
    total_pages = data["totalPages"] # get here since for pandas dataframe this would duplicate itself 10 times
    response_dataframe = pd.DataFrame(data)
    # get total page count
    valid_urls = pd.DataFrame(columns=['id', 'url', 'game'])
    for page in range(0, total_pages+1):  # query all pages sequentially
        params["page"] = page  # update page number
        response = requests.get(endpoint, params=params, headers=headers, timeout=10)
        data = response.json()
        valid_urls = pd.concat([valid_urls, parse_data(data)], ignore_index=True)

    # Filter out duplicate links (done here to get entire dataset)
    valid_urls.drop_duplicates(subset=["url"], inplace=True)
    download_dir = args['output']
    # Make output directory if it doesn't exist
    ensure_dir_exists(Path(download_dir))
    # Save the downloaded links to a file
    valid_urls_df = pd.DataFrame(valid_urls)
    valid_urls_df.to_csv((os.path.join(Path(download_dir + "links.csv"))),
                         index=True, columns=["id", "url", "game"])


if __name__ == "__main__":
    # main(dict(args['requirements']))
    main()
