"""
gunicorn config for the Club AutoMixer (used by docker-entrypoint.sh).

ONE worker process, many threads — and that is not tunable on purpose:
sessions, the render queue (src/jobs.py), the upload registry and the chunk index
all live in the worker's memory. A second worker would have its own copies, so a
status/chunk request load-balanced to it would 404, and MAX_CONCURRENT_RENDERS would
silently double. Scale up with threads (cheap status/chunk/upload requests) and
MAX_CONCURRENT_RENDERS (heavy renders), or run more containers behind a sticky
load balancer.
"""
import os

bind = f"0.0.0.0:{os.getenv('PORT', '7860')}"
workers = 1
worker_class = "gthread"
threads = int(os.getenv("GUNICORN_THREADS", "8"))
# Renders run in background threads, never inside a request, so a request only
# needs long enough for an upload to be received and decoded.
timeout = int(os.getenv("GUNICORN_TIMEOUT", "180"))
# On SIGTERM: stop accepting, let in-flight requests (uploads) finish, then exit.
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "30"))
keepalive = 5
# Never recycle the worker: it would drop every in-memory session mid-set.
max_requests = 0
preload_app = False

accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info").lower()
access_log_format = '%(h)s "%(r)s" %(s)s %(b)s %(M)sms rid=%({x-request-id}o)s'


def worker_exit(server, worker):
    """Cancel queued/running renders and delete their chunk files on shutdown."""
    try:
        import club_server
        club_server.shutdown()
    except Exception as e:  # never block shutdown
        server.log.warning("automix shutdown hook failed: %s", e)
