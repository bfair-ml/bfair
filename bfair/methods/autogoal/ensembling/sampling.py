from typing import Iterable, Tuple
from autogoal.sampling import Sampler


class Cache:
    def __init__(self):
        self.memory = {}

    def get(self, tag, default, **kargs):
        key = tuple(kargs.items())
        try:
            return self.memory[tag, key]
        except KeyError:
            value = self.memory[tag, key] = default()
            return value


class SampleModel:
    def __init__(self, sampler, model, **kargs):
        self.sampler = sampler
        self.model = model
        self.info = kargs

    def __str__(self):
        return str(self.sampler)

    def __repr__(self):
        return repr(self.sampler)


class LogSampler:
    def __init__(self, sampler: Sampler):
        self._sampler = sampler
        self._log = []

    def _log_sample(self, handle, value, log=True):
        result = value
        if log:
            if handle is None:
                handle, value = value, True
            self._log.append((handle, value))
        return result

    def distribution(self, name: str, handle=None, log=True, **kwargs):
        try:
            return getattr(self, name)(handle=handle, log=log, **kwargs)
        except AttributeError:
            raise ValueError("Unrecognized distribution name: %s" % name)

    def boolean(self, handle=None, log=True) -> bool:
        value = self._sampler.boolean(handle)
        return self._log_sample(handle, value, log=log)

    def categorical(self, options, handle=None, log=True):
        value = self._sampler.categorical(options, handle)
        return self._log_sample(handle, value, log=log)

    def choice(self, options, handle=None, log=True):
        value = self._sampler.choice(options, handle)
        return self._log_sample(handle, value, log=log)

    def continuous(self, min=0, max=1, handle=None, log=True) -> float:
        value = self._sampler.continuous(min, max, handle)
        return self._log_sample(handle, value, log=log)

    def discrete(self, min=0, max=10, handle=None, log=True) -> int:
        value = self._sampler.discrete(min, max, handle)
        return self._log_sample(handle, value, log=log)

    def multichoice(self, options, k, handle=None, log=True):
        if hasattr(self._sampler, "multichoice"):
            values = self._sampler.multichoice(options=options, k=k, handle=handle)
        else:
            values = []
            candidates = list(options)
            for _ in range(k):
                value = self._sampler.choice(candidates)
                candidates.remove(value)
                values.append(value)
        return self._log_sample(handle, values, log=log)

    def multisample(self, labels, func, handle=None, log=True, **kargs):
        if hasattr(self._sampler, "multisample"):
            values = self._sampler.multisample(
                labels=labels, func=func, handle=handle, **kargs
            )
        else:
            values = {
                label: func(handle=f"{handle}-{label}", log=False, **kargs)
                for label in labels
            }
        return self._log_sample(handle, values, log=log)

    def __iter__(self):
        return iter(self._log)

    def __str__(self):
        items = ",\n    ".join(f"{repr(k)}: {repr(v)}" for k, v in self)
        return "{\n    " + items + "\n}"

    def __repr__(self):
        return str(self)


class LockedSampler:
    def __init__(self, log: Iterable[Tuple[str, object]], ensure_handle=False):
        self.log = log
        self.ensure_handle = ensure_handle
        self.pointer = -1
        self.method = self._check if ensure_handle else self._next

    def _check(self, *args, **kargs):
        value = self._next(*args, **kargs)
        if "handle" in kargs:
            received = kargs["handle"]
            expected = self.log[self.pointer][0]
            if received != expected:
                raise RuntimeError(
                    f"Expected `{expected}` but received `{received}` instead."
                )
        return value

    def _next(self, *args, **kargs):
        self.pointer += 1
        return self.log[self.pointer][1]

    def __getattr__(self, name):
        return self.method


class NameLockedSampler:
    def __init__(self, configuration: dict = (), **kargs):
        self.data = {}
        self.data.update(configuration)
        self.data.update(kargs)

    def _sample(self, *args, handle, **kargs):
        return self.data[handle]

    def __getattr__(self, name):
        return self._sample
