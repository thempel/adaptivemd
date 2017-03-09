import os
import time
import base64

from mongodb import StorableMixin, SyncVariable, IncreasingNumericSyncVariable


class Action(StorableMixin):
    def __init__(self):
        super(Action, self).__init__()

    def __repr__(self):
        return str(self)


class AddPathAction(Action):
    def __init__(self, path):
        super(AddPathAction, self).__init__()
        self.path = path


class FileAction(Action):
    def __init__(self, source):
        super(FileAction, self).__init__()
        self.source = source

    def __str__(self):
        return "%s('%s')" % (
            self.__class__.__name__,
            self.source
        )

    @property
    def required(self):
        return [self.source]

    @property
    def added(self):
        return []

    @property
    def removed(self):
        return []


class FileTransaction(FileAction):
    def __init__(self, source, target):
        super(FileTransaction, self).__init__(source)

        if isinstance(target, str):
            self.target = source.clone()
            self.target.location = target
        elif isinstance(target, Location) and not isinstance(target, File):
            self.target = source.clone()
            self.target.location = target.location
        else:
            self.target = target

    def __str__(self):
        return "%s('%s' > '%s)" % (
            self.__class__.__name__,
            self.source.short,
            self.target.short
        )

    # @classmethod
    # def from_dict(cls, dct):
    #     obj = super(FileTransaction, cls).from_dict(dct)
    #     obj.target = dct['target']
    #     return obj


class Copy(FileTransaction):
    @property
    def added(self):
        return [self.target]


class Transfer(FileTransaction):
    @property
    def added(self):
        return [self.target]


class Move(FileTransaction):

    @property
    def added(self):
        return [self.target]

    @property
    def removed(self):
        return [self.source]


class Link(FileTransaction):
    @property
    def added(self):
        return [self.target]


class Remove(FileAction):
    def __init__(self, source):
        super(Remove, self).__init__(source)

    @property
    def removed(self):
        return [self.source]


class Touch(FileAction):
    def __init__(self, source):
        super(Touch, self).__init__(source)

    @property
    def added(self):
        return [self.source]


class Location(StorableMixin):
    allowed_drives = ['worker', 'staging', 'file', 'shared']
    default_drive = 'worker'

    use_absolute_local_paths = True

    _ignore = True

    def __init__(self, location):
        super(Location, self).__init__()

        if isinstance(location, Location):
            self.location = location.location
        elif isinstance(location, str):
            self.location = location
        else:
            raise ValueError('location can only be a `File` or a string.')

        # fix relative paths for file://

        if File.use_absolute_local_paths:
            if self.drive == 'file':

                p = os.path.abspath(self.path)
                self.location = 'file://' + p

    def __hash__(self):
        return hash(self.resource_location)

    def __eq__(self, other):
        if other is None:
            return False
        elif isinstance(other, Location):
            return self.resource_location == other.resource_location

        raise NotImplemented

    def clone(self):
        return self.__class__(self.location)

    def __add__(self, other):
        if isinstance(other, str):
            return str(self) + other

        raise NotImplemented

    def __radd__(self, other):
        if isinstance(other, str):
            return other + str(self)

    @property
    def short(self):
        if self.path == self.basename:
            return '%s://%s' % (self.drive, self.basename)
        elif self.path == '/' + self.basename:
            return '%s:///%s' % (self.drive, self.basename)
        else:
            return '%s://{}/%s' % (self.drive, self.basename)

    @property
    def url(self):
        return '%s://%s' % (self.drive, self.path)

    @property
    def basename(self):
        return os.path.basename(self.path)

    @property
    def resource_location(self):
        return self.location

    @property
    def is_folder(self):
        return self.path.endswith('/')

    @property
    def path(self):
        return self.split_drive[1]

    @property
    def split(self):
        return os.path.split(self.path)

    @property
    def dirname(self):
        return os.path.dirname(self.path)

    @property
    def drive(self):
        return self.split_drive[0]

    @property
    def extension(self):
        name = self.basename
        parts = name.split('.')
        if len(parts) == 1:
            return ''
        else:
            return parts[-1]

    @property
    def basename_short(self):
        name = self.basename
        parts = name.split('.')
        if len(parts) == 1:
            return name
        else:
            return '.'.join(parts[:-1])

    @property
    def split_drive(self):
        s = self.location
        parts = s.split('://')
        if len(parts) == 2:
            return parts[0], parts[1]
        elif len(parts) == 1:
            return self.default_drive, parts[0]

    def __repr__(self):
        return "'%s'" % self.basename

    def __str__(self):
        # todo: should be unit location by default
        if self.drive == 'worker':
            return self.path
        else:
            return self.location


class File(Location):
    _find_by = ['created', 'state']

    created = SyncVariable('created')

    def __init__(self, location):
        super(File, self).__init__(location)

        self.resource = None
        self.created = None
        self._file = None

        if self.drive == 'file':
            if os.path.exists(self.path):
                self.created = time.time()

    @property
    def _ignore(self):
        return self.drive == 'worker' or self.drive == 'staging'

    def on(self, resource):
        if resource == self.resource:
            return self
        else:
            obj = self.clone()
            obj.resource = resource
            return obj

    def clone(self):
        f = self.__class__(self.location)
        f.resource = self.resource
        f.created = None

        return f

    def create(self, scheduler):
        """
        Mark file as being existent on a specific scheduler.

        This should only work for file in `staging://`, `shared://`,
        `sandbox://` or `file://`
        Files in `unit://` will potentially be deleted, others are already existing

        Notes
        -----
        We usually assume that objects are immutable. The way to think about
        creation is that a file is something like a _Promise_ and it promises
        a certain file with a name. Once it is created it is still the same file
        but now it exists and can be used.

        The change of location is also a re-expression of the same location so
        that it is reusable.

        """
        scheduler.unroll_staging_path(self)
        self.created = time.time()

    def modified(self):
        """
        Mark a file as being altered and hence not existent anymore in the current description

        Notes
        -----
        Negative timestamps indicate the (negative) time when the object disappeared
        in the form described
        """
        stamp = self.created
        if stamp is not None and stamp > 0:
            self.created = - time.time()

    @property
    def exists(self):
        created = self.created
        return created is not None and created > 0

    def _complete_target(self, target):
        if target is None:
            target = Location('')

        if isinstance(target, str):
            target = Location(target)

        if isinstance(target, Location):
            if target.basename == '':
                target.location = target.location + self.basename

        return target

    def copy(self, target=None):
        target = self._complete_target(target)
        return Copy(self, target)

    def move(self, target=None):
        target = self._complete_target(target)
        return Move(self, target)

    def link(self, target=None):
        target = self._complete_target(target)
        return Link(self, target)

    def transfer(self, target=None):
        target = self._complete_target(target)
        return Transfer(self, target)

    def remove(self):
        return Remove(self)

    def __repr__(self):
        return "'%s'" % self.basename

    def load(self):
        if self.drive == 'file':
            with open(self.path, 'r') as f:
                self._file = f.read()

        return self

    def to_dict(self):
        ret = super(File, self).to_dict()
        if self._file:
            ret['_file_'] = base64.b64encode(self._file)

        return ret

    @classmethod
    def from_dict(cls, dct):
        obj = super(File, cls).from_dict(dct)
        if '_file_' in dct:
            obj._file = base64.b64decode(dct['_file_'])

        return obj

    def get_file(self):
        return self._file

    @property
    def has_file(self):
        return bool(self._file)


class Directory(File):
    @property
    def is_folder(self):
        return True


class URLGenerator(object):
    def __init__(self, shape, bundle=None):
        if bundle is None:
            self.count = 0
        else:
            self.count = len(bundle)

        self.shape = shape

    def __iter__(self):
        return self

    def next(self):
        fn = self.shape.format(count=self.count)
        self.count += 1
        return fn

    def initialize_from_files(self, files):
        # a little cheat to figure out the last number
        self.count = 0
        left = len(self.shape.split('{')[0].split('/')[-1])
        right = len(self.shape.split('}')[1])
        for f in files:
            try:
                g = int(f.basename[left:-right]) + 1
                self.count = max(g, self.count)
            except:
                # print f.basename, f.basename[left:-right]
                pass