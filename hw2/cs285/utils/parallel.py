from selectors import DefaultSelector, EVENT_READ
from subprocess import Popen, PIPE
from typing import Optional

from tqdm import tqdm


class ProcessListExhausted(Exception):
    """Exception raised when there are no more commands to run"""

    pass


class MaxConcurrencyReached(Exception):
    """Exception raised when we are already at the maximum concurrent processes"""

    pass


class ParallelProcessChild:
    """An instance of a child process together with its tqdm bar

    Parameters
    ----------
    specification : dict
        The dictionary specifying the commands to execute in parallel. The
        dict should have the following keys.
            "name": The name of the process. "command": The command to
            execute, as a list. "max_progress": The max progress value that
            can be reported by the command.
    position : int, optional
        The position of the tqdm bar
    """

    def __init__(self, specification: dict, position: Optional[int] = None):
        self.position = position
        self.process = Popen(
            specification["command"],
            bufsize=1,
            stdout=PIPE,
            close_fds=True,
            universal_newlines=True,
        )
        self.tqdm_bar = tqdm(
            total=specification["max_progress"],
            desc=specification["name"],
            position=position,
            leave=True,
        )

    @property
    def stdout(self):
        return self.process.stdout

    def poll(self, *args, **kwargs):
        return self.process.poll(*args, **kwargs)

    def update(self, progress: float):
        """Update the tqdm bar

        Parameters
        ----------
        progress : float
            The value to set the progress bar to
        """
        self.tqdm_bar.update(progress - self.tqdm_bar.n)

    def refresh(self):
        """Refresh the tqdm bar"""
        self.tqdm_bar.refresh()

    def close(self):
        """Close the process and tqdm bar"""
        self.process.stdout.close()
        self.tqdm_bar.close()


class ParallelProcesses:
    """Parallel process manager

    Launches up to `max_concurrent` processes, and keeps track of them,
    launching new ones as the old ones die, until the list of processes is
    exhausted.

    Parameters
    ----------
    process_commands : list[dict]
        The list of dictionaries specifying the commands to execute in
        parallel. The dicts should have the following keys.
            "name": The name of the process. "command": The command to
            execute, as a list. "max_progress": The max progress value that
            can be reported by the command.
    max_concurrent : int
        The maximum number of processes to run at once
    get_child_progress : callable
        The function used to determine a child process' progress. Should take
        as input a string containing the most recent output from the child
        process and return either a float or None, the latter indicating that
        the progress could not be determined from the output string.
    """

    def __init__(
        self, process_commands: list, max_concurrent: int, timeout: float, get_child_progress: callable
    ):
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.get_child_progress = get_child_progress

        self.process_command_iter = iter(process_commands)
        self.children: list[ParallelProcessChild] = []
        self.selector = DefaultSelector()

    def run(self):
        self._start_initial_processes()
        self._follow_processes()

    def _start_initial_processes(self):
        """Launch up to `max_concurrent` initial processes"""
        try:
            for i in range(self.max_concurrent):
                self._start_new_process()
        except ProcessListExhausted:
            pass

    def _follow_processes(self):
        """Follow the processes, killing and adding new ones until the end"""

        while len(self.children) > 0:
            # Get all the child processes which have terminated
            for child in self.children:
                if child.process.poll() is not None:
                    # Unregister the child from the selector, and close it
                    self.selector.unregister(child.stdout)
                    child.close()
                    self.children.remove(child)

                    # Add start new children if we can
                    try:
                        self._start_new_process()
                    except ProcessListExhausted:
                        pass
                    except MaxConcurrencyReached:
                        pass

            # Check if there's something to read
            read_list = self.selector.select(timeout=self.timeout)

            print(len(read_list))

            for selector_key, events in read_list:
                # Get the latest line of output
                child = selector_key.data
                latest_line = child.stdout.readline()

                # Update the progress bar using this
                progress = self.get_child_progress(latest_line)
                if progress is not None:
                    child.update(progress)

    def _start_new_process(self, position: Optional[int] = None):
        if len(self.children) >= self.max_concurrent:
            raise MaxConcurrencyReached

        # Try to get the next command to run
        try:
            next_command = next(self.process_command_iter)
        except StopIteration:
            raise ProcessListExhausted

        # Launch the child process, and register it with the selector
        child = ParallelProcessChild(next_command, position)
        self.children.append(child)
        self.selector.register(child.stdout, EVENT_READ, data=child)

        # Refresh all the bars, because adding a new one can mess things up
        for child in self.children:
            child.refresh()
            