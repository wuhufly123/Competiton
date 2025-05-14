Universal Behavioral Modeling Data Challenge
Why a Universal Behavioral Modeling Challenge?
The challenge is designed to promote a unified approach to behavior modeling. Many modern enterprises rely on machine learning and predictive analytics for improved business decisions. Common predictive tasks in such organizations include recommendation, propensity prediction, churn prediction, user lifetime value prediction, and many others. A central piece of information used for these predictive tasks are logs of past behavior of users e.g., what they bought, what they added to their shopping cart, which pages they visited. Rather than treating these tasks as separate problems, we propose a unified modeling approach.

To achieve this, we introduce the concept of Universal Behavioral Profiles — user representations that encode essential aspects of each individual’s past interactions. These profiles are designed to be universally applicable across multiple predictive tasks, such as churn prediction and product recommendations. By developing representations that capture fundamental patterns in user behavior, we enable models to generalize effectively across different applications.

Challenge Overview
The objective of this challenge is to develop Universal Behavioral Profiles based on the provided data, which includes various types of events such as purchases, add to cart, remove from cart, page visit, and search query. These user representations will be evaluated based on their ability to generalize across a range of predictive tasks. The task of the challenge participants is to submit user representations, which will serve as inputs to a simple neural network architecture. Based on the submitted representations, models will be trained on several tasks, including some that are disclosed to participants, called "open tasks," as well as additional hidden tasks, which will be revealed after the competition ends. The final performance score will be an aggregate of results from all tasks. We iterate model training and evaluation automatically upon submission. The only task for participants is to submit universal user representations.

Participants are asked to provide user representations — Universal Behavioral Profiles
Downstream task training is conducted by the organizers, however, the competition pipeline is publicly available and is presented in this repository
Model for each downstream task is trained separately, but using the same embeddings (user representations)
Performance will be evaluated based on all downstream tasks
Open Tasks
Churn Prediction: Binary classification into 1: user will churn or 0: user will not churn. Churn task is performed on a subset of active users with at least one product_buy event in history (data available for the participants) Task name: churn
Categories Propensity: Multi-label classification into one of 100 possible labels. The labels represent the 100 most often purchase product categories. Task name: propensity_category
Product Propensity: Multi-label classification into one of 100 possible labels. The labels represent the 100 most often purchase products in train target data. Task name: propensity_sku
Hidden Tasks
In addition to the open tasks, the challenge includes hidden tasks, which remain undisclosed during the competition. The purpose of these tasks is to ensure that submitted Universal Behavioral Profiles are capable of generalization rather than being fine-tuned for specific known objectives. Similar to the open tasks, the hidden tasks focus on predicting user behavior based on the submitted representations, but they introduce new contexts that participants are not explicitly optimizing for.

After the competition concludes, the hidden tasks will be disclosed along with the corresponding code, allowing participants to replicate results.

Competition Code
We provide a framework that participants can use to test their solutions. The same code is used in the competition to train models for downstream tasks. Only targets for hidden tasks are not included in the provided code.

Github repository

Evaluation
The primary metric used to measure model performance is AUROC. We use torchmetrics implementation of AUROC. Additionally, the performance of Category Propensity and Product Propensity models is evaluated based on the novelty and diversity of the predictions. In these cases, the task score is calculated as a weighted sum of all metrics:

0.8 × AUROC + 0.1 × Novelty + 0.1 × Diversity
Diversity
To compute the diversity of single prediction, we first apply element-wise sigmoid to the predictions, and l1 normalize the result. The diversity of the prediction is the entropy of this distribution.

The final diversity score is computed as the average diversity of the model's predictions.

Novelty
The popularity of a single prediction is the weighted sum of the popularities of the top k recommended targets in the prediction. This is normalized so that a popularity score of 1 corresponds to the following scenario:

The model's top k predictions are the k most popular items, and the model is absolutely certain about predicting all of these items.

The popularity score is then computed as the average popularity of the model's predictions. Finally, we compute the novelty of the predictions as 1 - popularity.

Due to the sparsity of the data, the popularity scores, as computed so far are close to 0, and thus the corresponding raw novelty scores are really close to 1. To make the measure more sensitive to small changes near 1, we raise the raw popularity score to the 100th power.

Leaderboard scores
For each task, a (hidden) leaderboard is created based on the respective task scores. The final score, which evaluates the overall quality of user representations and their ability to generalize, is determined by aggregating ranks from all per-task leaderboards using the Borda count method. In this approach, each model's rank in a task leaderboard is converted into points, where a model ranked k-th among N participants receives N - k points. The final ranking is based on the total points accumulated across all tasks, ensuring that models performing well consistently across multiple tasks achieve a higher overall score.

Timeline
10 March, 2025: Start RecSys Challenge Release dataset
10 April, 2025: Submission System Open Leaderboard live‍
15 June, 2025: End RecSys Challeng‍e
20 June, 2025:‍ Final Leaderboard & Winners EasyChair open for submissions‍
26 June, 2025: Upload code of the final predictions
7 July, 2025: Paper Submission Due
24 July, 2025: Paper Acceptance Notifications
2 August, 2025: Camera-Ready Papers
September 2025: RecSys Challenge Workshop @ ACM RecSys 2025
Organizers
The RecSys 2025 Challenge is organized by Jacek Dąbrowski, Maria Janicka, Łukasz Sienkiewicz and Gergely Stomfai (Synerise), Dietmar Jannach (University of Klagenfurt, Austria), Marco Polignano (University of Bari Aldo Moro, Italy), Claudio Pomo (Politecnico di Bari, Italy), Abhishek Srivastava (IIM Visakhapatnam, India), and Francesco Barile (Maastricht University, Netheralnds).




Dataset
We release an anonymized dataset containing real-world user interaction logs. Additionally, we provide product properties that can be joined with product_buy, add_to_cart, and remove_from_cart event types. Each source has been stored in a separate file.

Note
All recorded interactions can be utilized to create Universal Behavioral Profiles; however, participants are required to submit behavioral profiles for only a subset of 1,000,000 users, which will be used for model training and evaluation.

product_buy	add_to_cart	remove_from_cart	page_visit	search_query
Events	1,682,296	5,235,882	1,697,891	150,713,186	9,571,258
Dataset Description
Columns
product_properties:

sku (int64): Numeric ID of the item.
category (int64): Numeric ID of the item category.
price (int64): Numeric ID of the bucket of item price (see section Column Encoding).
name (object): Vector of numeric IDs representing a quantized embedding of the item name (see section Column Encoding).
product_buy:

client_id (int64): Numeric ID of the client (user).
timestamp (object): Date of event in the format YYYY-MM-DD HH:mm:ss.
sku (int64): Numeric ID of the item.
add_to_cart:

client_id (int64): Numeric ID of the client (user).
timestamp (object): Date of event in the format YYYY-MM-DD HH:mm:ss.
sku (int64): Numeric ID of the item.
remove_from_cart:

client_id (int64): Numeric ID of the client (user).
timestamp (object): Date of event in the format YYYY-MM-DD HH:mm:ss.
sku (int64): Numeric ID of the item.
page_visit:

client_id (int64): Numeric ID of the client.
timestamp (object): Date of event in the format YYYY-MM-DD HH:mm:ss.
url (int64): Numeric ID of visited URL. The explicit information about what (e.g., which item) is presented on a particular page is not provided.
search_query:

client_id (int64): Numeric ID of the client.
timestamp (object): Date of event in the format YYYY-MM-DD HH:mm:ss.
query (object): Vector of numeric IDs representing a quantized embedding of the search query term (see section Column Encoding).
Column Encoding
Text Columns ('name', 'query'):
In order to anonymize the data, we first embed the texts with an appropriate large language model (LLM). Then, we quantize the embedding with a high-quality embedding quantization method. The final quantized embedding has the length of 16 numbers (buckets) and in each bucket, there are 256 possible values: {0, …, 255}.

Decimal Columns ('price'):
These were originally floating-point numbers, which were split into 100 quantile-based buckets.

Data Format
This section describes the format of the competition data. We provide a data directory containing event files and two subdirectories: input and target.

1. Event and properties files
The event data, which should be used to generate user representations, is divided into five Parquet files. Each file corresponds to a different type of user interaction available in the dataset (see section Dataset Description):

product_buy.parquet
add_to_cart.parquet
remove_from_cart.parquet
page_visit.parquet
search_query.parquet
Product properties are stored in:

product_properties.parquet
2. input directory
This directory stores a NumPy file containing a subset of 1,000,000 client_ids for which Universal Behavioral Profiles should be generated:

relevant_clients.npy
Using the event files, participants are required to create Universal Behavioral Profiles for the clients listed in relevant_clients.npy. These clients are identified by the client_id column in the event data.

3. target directory
This directory stores the labels for propensity tasks. For each propensity task, target category names are stored in NumPy files:

propensity_category.npy: Contains a subset of 100 categories for which the model is asked to provide predictions
popularity_propensity_category.npy: Contains popularity scores for categories from the propensity_category.npy file. Scores are used to compute the Novelty measure. For details, see the Evaluation section
propensity_sku.npy: Contains a subset of 100 products for which the model is asked to provide predictions
popularity_propensity_sku.npy: Contains popularity scores for products from the propensity_sku.npy file. These scores are used to compute the Novelty measure. For details, see the Evaluation section
active_clients.npy: Contains a subset of relevant clients with at least one product_buy event in history (data available for the participants). Active clients are used to compute churn target. For details, see the Open Tasks section
Download
Data can be downloaded, here



Submission Guidelines
Daily and Overall Submission Limits
Each team is allowed at most two submissions a day, which is facilitated through the limitation on the number of submissions per account. Even if a team circumvents Codabench’s submission limits—whether by making more than two submissions per day from a single account or by using multiple accounts—they may be disqualified and removed from the competition.

Competition Entry Format
Participants are asked to prepare Universal Behavioral Profiles — user representations that will serve as input to the first layer of a neural network with a fixed, simple architecture. For each submission, the models will be trained and evaluated by the organizers, and the evaluation outcome will be displayed on the leaderboard.

Each competition entry consists of two files: client_ids.npy and embeddings.npy

client_ids.npy
A file that stores the IDs of clients for whom Universal Behavioral Profiles were created
client_ids must be stored in a one-dimensional NumPy ndarray with dtype=int64
The file should contain client IDs from the relevant_clients.npy file, and the order of IDs must match the order of embeddings in the embeddings file
embeddings.npy
A file that stores Universal Behavioral Profiles as a dense user embeddings matrix
Each embedding corresponds to the client ID from client_ids.npy with the same index
Dense embeddings must be stored in a two-dimensional NumPy ndarray, where:
The first dimension corresponds to the number of users and matches the dimension of the client_ids array
The second dimension represents the embedding size
The dtype of the embeddings array must be float16
The embedding size cannot exceed max_embedding_dim = 2048
It is crucial that the order of IDs in the client_ids file matches the order of embeddings in the embeddings file to ensure proper alignment and data integrity.

IMPORTANT! The maximum length of the embedding vectors is 2048.
Submission Instructions
Participants must submit a zip file containing the two required files:

client_ids.npy
embeddings.npy
The zipped files should be uploaded in the "My Submissions" section of the competition platform.

IMPORTANT
For a successful submission, please zip the two required files directly—do not zip the folder containing them.

Example:

cd dir_with_embeddings_and_client_ids
zip -r embeddings.zip embeddings.npy client_ids.npy
Validation of Embeddings
To ensure that the embeddings meet the competition requirements, participants can validate their files using a validation script provided in the official competition GitHub repository.

🔗 RecSys Challenge 2025 Repository: GitHub Repository

Submission evaluation and results
After submission, the submitted files will be waiting in a queue until a worker is available to process them. While the organizers will try to supply sufficient computational capactity to ensure that the waiting times are minimal, please note that the all workers might be busy in peak times.

Once a worker receives the submission, real time evaluation logs will be available on the My Submissions page. A complete evaluation might take up to 1.5 hours (not including the upload time for the submission).

Once the evaluation is complete, the scores will be populated into the "Detailed Results" section next to the submission in the "My Submissions" tab. Please be advised, that the -1 value in the "Score" column of the submissions is meaningless placeholder, which the organizers could not remove due to the architecture of the Codabench platform. Thus, it should not be taken into consideration when considering the performance of the submissions.

Note also, that simply evaluating a submission will not automatically put it on the leaderboard. This has to be done manually, by clicking the leaderboard icon in the "Actions" column of the submission.

Since the final rankings are calculated using the Borda count method (for details see the "Leaderboard" page) which is not supported by the Codabench platform, the "Results" tab has been disabled (there are no available leaderboards there). The leaderboard can instead be found on the "Leaderboard" page of the "Get Started" tab. Please visit this page for further information about the leaderboard.