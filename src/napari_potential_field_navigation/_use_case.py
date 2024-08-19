from functools import wraps


class UseCase:
    """
    The UseCase class allows enabling or disabling elements of a chain of
    dependencies
    """
    def __init__(self):
        self._steps = []
        self._requirements = {}
        self._state = {}

    def involve(self, steps):
        self._steps.extend(steps)
        for step in steps:
            step._use_case = self

    def start(self):
        self._init_state()
        for step in self._steps:
            for conditional_step in self._requirements.keys():
                if isinstance(step, conditional_step):
                    self.disable(step)
                    break

    def _init_state(self):
        for requirements in self._requirements.values():
            if not requirements:
                raise Exception(
                    "Empty list of requirements; do not reference such steps"
                )
            for requirement in requirements:
                checkpoint = requirement.__qualname__
                self._state[checkpoint] = False

    def is_involved(self, step):
        try:
            return step._use_case is self
        except AttributeError:
            return False

    def _disable(self, step):
        return False

    def _enable(self, step):
        return False

    def disable(self, step):
        ret = False
        if self.is_involved(step):
            try:
                ret = self._disable(step)
            except Exception as exception:
                logging.error(
                    f"""Disabling step of type {type(step)} failed with exception:
                    {exception}
                    """
                )
            # TODO: should we also clear the state?
        else:
            logging.error(
                "Trying to disable a step the use case does not involve"
            )
        return ret

    def enable(self, step):
        ret = False
        if self.is_involved(step):
            try:
                ret = self._enable(step)
            except Exception as exception:
                logging.error(
                    f"""Enabling step of type {type(step)} failed with exception:
                    {exception}
                    """
                )
        else:
            logging.error(
                "Trying to enable a step the use case does not involve"
            )
        return ret

    def _ready(self, obj, met, ret):
        return True

    def notify_update(self, obj, met, ret):
        cls = type(obj)
        for checkpoint in self._state.keys():
            if checkpoint in (cls.__qualname__, met.__qualname__):
                ready = self._ready(obj, met, ret)
                assert ready is True
                if ready is None:
                    ready = True
                self._state[checkpoint] = ready
                break
        self._refresh()

    def _check_state(self, step):
        for conditional_step in self._requirements.keys():
            if type(step) is conditional_step:
                for requirement in self._requirements[conditional_step]:
                    checkpoint = requirement.__qualname__
                    if not self._state[checkpoint]:
                        return False
        return True

    def _refresh(self):
        for step in self._steps:
            if self._check_state(step):
                self.enable(step)
            else:
                self.disable(step)


def use_case_check_point(f):
    @wraps(f)
    def method_with_check_point(self, *args, **kwargs):
        ret = f(self, *args, **kwargs)
        self._use_case.notify_update(self, f, ret)
        return ret
    return method_with_check_point

