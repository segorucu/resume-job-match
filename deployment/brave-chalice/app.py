import logging
from chalice import Chalice
from chalicelib.helper import scrape_google_jobs, save_position_to_aws, get_matching_jobs
import os
from opensearchpy import OpenSearch
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Adding stream handler for logging to console and ensuring CloudWatch captures it
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

app = Chalice(app_name='brave-chalice')

@app.route('/')
def index():
    return {'hello': 'world'}

@app.route('/trigger-scrape-jobs', methods=['POST'])
def trigger_scrape_jobs():
    # Simulate an EventBridge event (if needed)
    logger.info("scrape_jobs")
    job_titles = ["Data Analyst", "Data Engineer", "Data Scientist", "Software Developer",
                  "Product Manager", "Digital Marketer", "Machine Learning Engineer"]

    locations = {
        "Toronto": "w+CAIQICIHVG9yb250bw==",
        "Vancouver": "w+CAIQICIJVmFuY291dmVy"
    }

    total_data = 0
    for city, value in locations.items():
        for jt in job_titles:
            position_df = scrape_google_jobs(jt, location=city, value=value)
            logger.info(f"{city} - {jt}: {position_df['query'].value_counts()}")
            rows = position_df.shape[0]
            total_data += rows

    logger.info(f"Total data: {total_data}")

    return {"status": "success"}

#@app.schedule('cron(59 23 ? * 1 *)') once in a week at 23:59 utc time
@app.schedule('cron(41 5 ? * 7 *)') # EventBridge schedule
def scrape_jobs(event):
    logger.info("scrape_jobs")
    job_titles = ["Data Analyst", "Data Engineer", "Data Scientist", "Software Developer",
                  "Product Manager", "Digital Marketer", "Machine Learning Engineer"]

    locations = {
        "Toronto": "w+CAIQICIHVG9yb250bw==",
        "Vancouver": "w+CAIQICIJVmFuY291dmVy"
    }

    total_data = 0
    for city, value in locations.items():
        for jt in job_titles:
            position_df = scrape_google_jobs(jt, location=city, value=value)
            logger.info(f"{city} - {jt}: {position_df['query'].value_counts()}")
            rows = position_df.shape[0]
            total_data += rows
            if rows > 0:
                save_position_to_aws(position_df=position_df)

    logger.info(f"Total data: {total_data}")
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
        timeout=900
    )

    jobs = get_matching_jobs("emre_brave_project", city, job, aws_client)
    return jobs
