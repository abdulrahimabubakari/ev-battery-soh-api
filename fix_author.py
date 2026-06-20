import subprocess

env_filter = (
    'if [ "$GIT_AUTHOR_EMAIL" != "abdul.rahim.abubakari01@gmail.com" ]; then '
    'export GIT_AUTHOR_NAME="abdulrahimabubakari"; '
    'export GIT_AUTHOR_EMAIL="abdul.rahim.abubakari01@gmail.com"; '
    'export GIT_COMMITTER_NAME="abdulrahimabubakari"; '
    'export GIT_COMMITTER_EMAIL="abdul.rahim.abubakari01@gmail.com"; '
    'fi'
)

result = subprocess.run(
    ['git', 'filter-branch', '-f', '--env-filter', env_filter,
     '--tag-name-filter', 'cat', '--', '--branches', '--tags'],
    capture_output=True, text=True
)
print(result.stdout)
print(result.stderr)