import os

from kaggle.api.kaggle_api_extended import KaggleApi

def get_submission_result(competition, idx=0):
    api = KaggleApi()
    api.authenticate()
    
    # Fetch submissions
    submissions = api.competitions_submissions_list(competition)
    
    # Iterate through submissions and print error messages
    latest_submission = submissions[idx]
    if latest_submission["hasPublicScore"]:
        score = float(latest_submission["publicScore"])
        print(f"\nYour merged model scores {score} on the test set!")
    else:
        error_msg = latest_submission["errorDescription"] 
        print(f"\nYour merged model may generate something invalid so the submission does not have a score. Here is the error message from the Kaggle leaderboard:\n\n{error_msg}")
        score = 0
    return score

def get_score():
    submission_path = "output/test.csv"
    competition_name = "llm-merging-competition"
    print("\nSubmitting to Kaggle leaderbord for evaluation on test set ...")
    os.system(f"kaggle competitions submit -c {competition_name} -f {submission_path} -m \"llm-merging\"")
    print("\nWaiting for Kaggle leaderboard to refresh ...")
    time.sleep(60)
    return get_submission_result(competition_name)

if __name__ == "__main__":
    print(get_score())
