# scheduler_notify.py
# untested: need to put this into the schedular ahead of training
from distributed.diagnostics.plugin import SchedulerPlugin

TOPIC = "worker-health"

class WorkerDropEvents(SchedulerPlugin):
    def remove_worker(self, scheduler, worker, *, stimulus_id, **kwargs):
        scheduler.log_event(
            TOPIC,
            {"event": "worker_removed", "worker": worker, "stimulus_id": stimulus_id},
        )

def dask_setup(scheduler):
    scheduler.add_plugin(WorkerDropEvents())



'''
Problem: Driver pod cannot tell when a worker is dropped mid run.
         This will leave the driver pod running indefinately, waiting
         for a reesult that isn't being claculated anymore

Solution: Run a SchedulerPlugin on the scheduler. 
        Get the driver pod to periodically get the most recent events
        and exit early if it sees a dropped worker
        
'''