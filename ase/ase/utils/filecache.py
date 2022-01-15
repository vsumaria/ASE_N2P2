from pathlib import Path
import json
from collections.abc import MutableMapping, Mapping
from contextlib import contextmanager
from ase.io.jsonio import read_json, write_json
from ase.io.jsonio import encode as encode_json
from ase.io.ulm import ulmopen, NDArrayReader, Writer, InvalidULMFileError
from ase.utils import opencew


def missing(key):
    raise KeyError(key)


class Locked(Exception):
    pass


class JSONBackend:
    extension = '.json'
    DecodeError = json.decoder.JSONDecodeError

    @staticmethod
    def open_for_writing(path):
        return opencew(path)

    @staticmethod
    def read(fname):
        return read_json(fname, always_array=False)

    @staticmethod
    def open_and_write(target, data):
        write_json(target, data)

    @staticmethod
    def write(fd, value):
        fd.write(encode_json(value).encode('utf-8'))

    @classmethod
    def dump_cache(cls, path, dct):
        return CombinedJSONCache.dump_cache(path, dct)

    @classmethod
    def create_multifile_cache(cls, directory):
        return MultiFileJSONCache(directory)


class ULMBackend:
    extension = '.ulm'
    DecodeError = InvalidULMFileError

    @staticmethod
    def open_for_writing(path):
        fd = opencew(path)
        if fd is not None:
            return Writer(fd, 'w', '')

    @staticmethod
    def read(fname):
        with ulmopen(fname, 'r') as r:
            data = r._data['cache']
            if isinstance(data, NDArrayReader):
                return data.read()
        return data

    @staticmethod
    def open_and_write(target, data):
        with ulmopen(target, 'w') as w:
            w.write('cache', data)

    @staticmethod
    def write(fd, value):
        fd.write('cache', value)

    @classmethod
    def dump_cache(cls, path, dct):
        return CombinedULMCache.dump_cache(path, dct)

    @classmethod
    def create_multifile_cache(cls, directory):
        return MultiFileULMCache(directory)


class CacheLock:
    def __init__(self, fd, key, backend):
        self.fd = fd
        self.key = key
        self.backend = backend

    def save(self, value):
        try:
            self.backend.write(self.fd, value)
        except Exception as ex:
            raise RuntimeError(f'Failed to save {value} to cache') from ex
        finally:
            self.fd.close()


class _MultiFileCacheTemplate(MutableMapping):
    writable = True

    def __init__(self, directory):
        self.directory = Path(directory)

    def _filename(self, key):
        return self.directory / (f'cache.{key}' + self.backend.extension)

    def _glob(self):
        return self.directory.glob('cache.*' + self.backend.extension)

    def __iter__(self):
        for path in self._glob():
            cache, key = path.stem.split('.', 1)
            if cache != 'cache':
                continue
            yield key

    def __len__(self):
        # Very inefficient this, but not a big usecase.
        return len(list(self._glob()))

    @contextmanager
    def lock(self, key):
        self.directory.mkdir(exist_ok=True, parents=True)
        path = self._filename(key)
        fd = self.backend.open_for_writing(path)
        try:
            if fd is None:
                yield None
            else:
                yield CacheLock(fd, key, self.backend)
        finally:
            if fd is not None:
                fd.close()

    def __setitem__(self, key, value):
        with self.lock(key) as handle:
            if handle is None:
                raise Locked(key)
            handle.save(value)

    def __getitem__(self, key):
        path = self._filename(key)
        try:
            return self.backend.read(path)
        except FileNotFoundError:
            missing(key)
        except self.backend.DecodeError:
            # May be partially written, which typically means empty
            # because the file was locked with exclusive-write-open.
            #
            # Since we decide what keys we have based on which files exist,
            # we are obligated to return a value for this case too.
            # So we return None.
            return None

    def __delitem__(self, key):
        try:
            self._filename(key).unlink()
        except FileNotFoundError:
            missing(key)

    def combine(self):
        cache = self.backend.dump_cache(self.directory, dict(self))
        assert set(cache) == set(self)
        self.clear()
        assert len(self) == 0
        return cache

    def split(self):
        return self

    def filecount(self):
        return len(self)

    def strip_empties(self):
        empties = [key for key, value in self.items() if value is None]
        for key in empties:
            del self[key]
        return len(empties)


class _CombinedCacheTemplate(Mapping):
    writable = False

    def __init__(self, directory, dct):
        self.directory = Path(directory)
        self._dct = dict(dct)

    def filecount(self):
        return int(self._filename.is_file())

    @property
    def _filename(self):
        return self.directory / ('combined' + self.backend.extension)

    def __len__(self):
        return len(self._dct)

    def __iter__(self):
        return iter(self._dct)

    def __getitem__(self, index):
        return self._dct[index]

    def _dump(self):
        target = self._filename
        if target.exists():
            raise RuntimeError(f'Already exists: {target}')
        self.directory.mkdir(exist_ok=True, parents=True)
        self.backend.open_and_write(target, self._dct)

    @classmethod
    def dump_cache(cls, path, dct):
        cache = cls(path, dct)
        cache._dump()
        return cache

    @classmethod
    def load(cls, path):
        # XXX Very hacky this one
        cache = cls(path, {})
        dct = cls.backend.read(cache._filename)
        cache._dct.update(dct)
        return cache

    def clear(self):
        self._filename.unlink()
        self._dct.clear()

    def combine(self):
        return self

    def split(self):
        cache = self.backend.create_multifile_cache(self.directory)
        assert len(cache) == 0
        cache.update(self)
        assert set(cache) == set(self)
        self.clear()
        return cache


class MultiFileJSONCache(_MultiFileCacheTemplate):
    backend = JSONBackend()


class MultiFileULMCache(_MultiFileCacheTemplate):
    backend = ULMBackend()


class CombinedJSONCache(_CombinedCacheTemplate):
    backend = JSONBackend()


class CombinedULMCache(_CombinedCacheTemplate):
    backend = ULMBackend()


def get_json_cache(directory):
    try:
        return CombinedJSONCache.load(directory)
    except FileNotFoundError:
        return MultiFileJSONCache(directory)
