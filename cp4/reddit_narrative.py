import logging
import praw
import sys
import json
from collections import defaultdict



def get_reddit_instance():
    reddit_credentials_file = str(sys.argv[1])
    with open(reddit_credentials_file) as ff:
        reddit_credentials = json.load(ff)
    # Method1: Use username and password to access directly
    reddit = praw.Reddit(client_id=reddit_credentials['client_id'],
                         client_secret=reddit_credentials['client_secret'],
                         password=reddit_credentials['password'],
                         user_agent=reddit_credentials['user_agent'],
                         username=reddit_credentials['username'])
    # TODO: Method2: Obtain the Authorization URL: Use thisusername to avoid using the actual username and password
    # reddit = praw.Reddit(client_id=reddit_credentials['client_id'],
    #                      client_secret=reddit_credentials['client_secret'],
    #                      redirect_uri='http://localhost:8080',
    #                      user_agent=reddit_credentials['user_agent'])
    # print(reddit.auth.url(['identity'], '...', 'permanent'))
    # print(reddit.auth.authorize(code))
    # print(reddit.user.me())
    return reddit


def get_search_result(reddit, subreddit_name, search_str, search_cnt):
    subreddit = reddit.subreddit(subreddit_name)
    # tops = subreddit.top(limit=1)
    # for each in tops:
    #     pprint.pprint(vars(each))
    search = subreddit.search(search_str, limit=search_cnt)
    # "id", "title", "author_id", "created_utc", "permalink","num_comments", "comments_id", "votable_ups",
    # "votable_downs","votable_likes", "score", "upvote_ratio","is_self", "selftext","url"
    all_submissions = []
    for each in search:
        each_submission = defaultdict(list)
        each_submission["id"] = each.id
        each_submission["title"] = each.title
        each_submission["author_id"] = each.author.id
        each_submission["created_utc"] = each.created_utc
        each_submission["permalink"] = each.permalink
        each_submission["num_comments"] = each.num_comments
        for comment in each.comments.list():
            each_submission["comments_id"].append(comment.id)
        each_submission["votable_ups"] = each.ups
        each_submission["votable_downs"] = each.downs
        each_submission["votable_likes"] = each.likes
        each_submission["score"] = each.score
        each_submission["upvote_ratio"] = each.upvote_ratio
        each_submission["is_self"] = each.is_self
        each_submission["selftext"] = each.selftext
        each_submission["url"] = each.url

        all_submissions.append(each_submission)
        # pprint.pprint(vars(each))
    return all_submissions
if


def search(reddit_ins, subreddit, search_str, search_cnt):
    out_res = get_search_result(reddit=reddit_ins, subreddit_name=subreddit, search_str=search_str, search_cnt=search_cnt)
    with open("result_reddit_{}_{}_{}.json".format(subreddit, search_str, search_cnt), "w") as out_json:
        out_json.write(json.dumps(out_res))


if __name__ == '__main__':
    reddit_ins = get_reddit_instance()
    search(reddit_ins, 'all', 'coronavirus', None)