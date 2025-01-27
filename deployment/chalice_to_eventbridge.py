import logging
from chalice import Chalice
from helper import scrape_google_jobs, save_position_to_aws, get_matching_jobs
import pandas as pd
import os
from opensearchpy import OpenSearch

app = Chalice(app_name="brave_jobs")


@app.route('/')
def index():
    return {'hello': 'world'}

#@app.schedule('cron(59 23 ? * 1 *)') once in a week at 23:59 utc time
@app.schedule('cron(19 7 * * ? *)')  # EventBridge schedule
def scrape_jobs(event):
    # Start an actor and wait for it to finish
    logging.info("scrape_jobs")
    job_titles = ["Data Analyst", "Data Engineer", "Data Scientist", "Software Developer",
                  "Product Manager", "Digital Marketer", "Machine Learning Engineer"]

    locations = {
        "Toronto": "w+CAIQICIHVG9yb250bw==",
        "Montreal": "w+CAIQICIJVmFuY291dmVy"
    }

    for city, value in locations.items():
        for jt in job_titles:
            position_df = scrape_google_jobs(jt, location=city, value=value)
            logging.info(f"Query counts for {city} - {jt}: {position_df['query'].value_counts()}")
            position_df = position_df.where(pd.notnull(position_df), None)
            # save_position_to_aws(position_df=position_df)

    return {"status": "success"}


@app.route('/jobs')
def get_jobs():
    job = app.current_request.query_params.get('job')
    city = app.current_request.query_params.get('city')

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
        timeout=60
    )

    jobs = get_matching_jobs("emre_brave_project", city, job, aws_client)
    return jobs
