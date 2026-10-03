"""The queue area: conversations, artifacts and uploads, the processing-job queue (graphs, claims,
leases, heartbeat, completion, failure, release, retry, cancel, progress), reanalysis requests and
draft tests, usage rows and their report, hardware profiles and catalog snapshots.

Owner: the queue builder. Files: ``routes.py`` (handlers), ``api.py`` (functions other areas call),
``migrations/020_queue.sql`` (and 021-029). Other modules in this package are the owner's choice.
The processing queue is never the human review queue (that is the results area).
"""
