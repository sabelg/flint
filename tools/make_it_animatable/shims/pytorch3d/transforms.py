"""Stand-in for pytorch3d.transforms: Make-It-Animatable's util/utils.py imports Transform3d at
module level, but none of the functions the Flint tools call use it."""


class Transform3d:
    def __init__(self, *a, **k):
        raise NotImplementedError("pytorch3d is not needed by tools/make_it_animatable")


Scale = Translate = Transform3d
