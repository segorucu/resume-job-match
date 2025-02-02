import pandas as pd
from apify_client import ApifyClient
from datetime import datetime
import os
from opensearchpy import OpenSearch, helpers
import boto3

def scrape_google_jobs(job_title, location, value):
    position_df = pd.DataFrame()

    run_input = {
        "csvFriendlyOutput": True,
        "includeUnfilteredResults": False,
        "maxConcurrency": 10,
        "maxPagesPerQuery": 3,
        # "queries": f"https://www.google.com/search?ibp=htl;jobs&q={job_title}&uule={value}&date_posted=today",
        "queries": f"https://www.google.com/search?ibp=htl;jobs&q={job_title}&uule={value}&date_posted=7",
        "saveHtml": False,
        "saveHtmlToKeyValueStore": False,
    }

    ssm_client = boto3.client('ssm')
    response = ssm_client.get_parameter(Name='APIFY_KEY', WithDecryption=True)
    APIFY_KEY = response['Parameter']['Value']
    # APIFY_KEY = os.getenv("APIFY_KEY")
    apify_client = ApifyClient(APIFY_KEY)
    actor_call = apify_client.actor(
        'dan.scraper/google-jobs-scraper').call(run_input=run_input)

    dataset_items = apify_client.dataset(
        actor_call['defaultDatasetId']).list_items().items

    d = pd.DataFrame(dataset_items)
    d["query"] = job_title
    d["location"] = location
    d["run_time"] = str(datetime.now())

    position_df = pd.concat([position_df, d])
    position_df = position_df.where(pd.notnull(position_df), None)
    return position_df


def doc_generator(df, index_name):
    for i, row in df.iterrows():
        doc = {
            "_index": index_name,
            "_source": row.to_dict(),
        }
        yield doc

def save_position_to_aws(position_df):
    host = os.getenv("AWS_DOMAIN")
    port = 443
    auth = (os.getenv("AWS_USER"), os.getenv("AWS_PASSWORD"))

    aws_client = OpenSearch(
        hosts=[{'host': host, 'port': port}],
        http_compress=True,  # enables gzip compression for request bodies
        http_auth=auth,
        use_ssl=True,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        timeout=900
    )

    helpers.bulk(aws_client, doc_generator(position_df, "emre_brave_project"))

def get_matching_jobs(index_name, location, position, client):
    query = {
        "query": {
            "bool": {
                "must": [
                    {"match": {"location": location}},
                    {"match": {"query": position}}
                ]
            }
        }
    }
    results = client.search(index=index_name, body=query, size=1000)
    return results['hits']['hits']