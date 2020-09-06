import json
import logging
import os
import sys

import requests


def get_pr_files(commit, pr_number):
    if not commit or not pr_number:
        return []

    response = requests.get(
        'https://api.github.com/repos/horovod/horovod/pulls/{pr_number}/commits'.format(
            pr_number=pr_number
        )
    )
    if response.status_code != 200:
        logging.error('Request failed: {}'.format(json.loads(response.text).get('message')))
        return []

    pr_commits_json = response.text
    pr_commits = json.loads(pr_commits_json)
    base_commit = pr_commits[0].get('parents')[0].get('sha')

    response = requests.get(
        'https://api.github.com/repos/horovod/horovod/compare/{base_commit}...{head_commit}'.format(
            base_commit=base_commit, head_commit=commit
        )
    )
    if response.status_code != 200:
        logging.error('Request failed: {}'.format(json.loads(response.text).get('message')))
        return []

    compare_json = response.text
    compare = json.loads(compare_json)
    return [file.get('filename') for file in compare.get('files')]


def get_branch_files(commit, default, branch):
    response = requests.get(
        'https://api.github.com/repos/horovod/horovod/compare/{default}...{branch}'.format(
            default=default, branch=branch
        )
    )
    if response.status_code != 200:
        logging.error('Request failed: {}'.format(json.loads(response.text).get('message')))
        return []

    compare_json = response.text
    compare = json.loads(compare_json)
    return [file.get('filename') for file in compare.get('files')]


if __name__ == "__main__":
    logging.getLogger().level = logging.DEBUG

    commit = os.environ.get('BUILDKITE_COMMIT')
    pr_number = os.environ.get('BUILDKITE_PULL_REQUEST')
    logging.debug('commit = {}'.format(commit))
    logging.debug('pr number = {}'.format(pr_number))
    if pr_number is not None and pr_number != 'false':
        for file in get_pr_files(commit, int(pr_number)):
            print(file)
    else:
        branch = os.environ.get('BUILDKITE_BRANCH')
        default = os.environ.get('BUILDKITE_PIPELINE_DEFAULT_BRANCH')
        logging.debug('branch = {}'.format(branch))
        logging.debug('default = {}'.format(default))
        if branch and default:
            for file in get_branch_files(commit, default, branch):
                print(file)
