#!/usr/bin/env bash
# Container entrypoint: optionally materialize YouTube cookies from a secret, then
# serve with gunicorn (config in gunicorn.conf.py: a SINGLE worker so the in-process
# sessions / render queue / uploads are shared; threads serve the cheap requests
# while renders run on the bounded render pool).
set -euo pipefail

# If a YT_COOKIES secret (Netscape cookies.txt contents) is provided, write it to a
# private file and point yt-dlp at it. This is the standard mitigation for
# datacenter-IP bot-throttling by YouTube.
if [ -n "${YT_COOKIES:-}" ]; then
    umask 077
    printf '%s' "$YT_COOKIES" > /tmp/yt_cookies.txt
    umask 022
    export YT_COOKIES_FILE=/tmp/yt_cookies.txt
    unset YT_COOKIES
fi

exec gunicorn --config gunicorn.conf.py club_server:app
