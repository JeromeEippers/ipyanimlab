import numpy as np
from . import utils


class Anim(object):
    """
    A very basic animation object
    """
    def __init__(self, quats, pos, offsets, parents, bones):
        """
        :param quats: local quaternions tensor
        :param pos: local positions tensor
        :param offsets: local joint offsets
        :param parents: bone hierarchy
        :param bones: bone names
        """
        self.quats = quats
        self.pos = pos
        self.offsets = offsets
        self.parents = parents
        self.bones = bones


class AnimMapper:
    def __init__(self, asset, keep_translation=False, root_motion=False, match_effectors=False, effector_names=['LeftFoot', 'RightFoot'], effector_offests=None, local_offsets=None, mirror=False, mirror_left='Left', mirror_right='Right'):
        self.asset = asset
        self.keep_translation = keep_translation
        self.root_motion = root_motion
        self.match_effectors = match_effectors
        self.effector_names = effector_names
        self.effector_offests = effector_offests
        self.local_offsets = local_offsets
        self.mirror = mirror
        self.mirror_left = mirror_left
        self.mirror_right = mirror_right

    def __call__(self, anim):
        if self.match_effectors:
            anim = remap_animation_to_asset_with_ik(anim, self.asset, self.effector_names, self.effector_offests, self.local_offsets)
        else:
            anim = remap_animation_to_asset(anim, self.asset, self.keep_translation)
            if self.local_offsets is not None:
                for k, v in self.local_offsets.items():
                    anim.pos[:, anim.bones.index(k), :] += v

        if self.mirror:
            anim.quats, anim.pos = utils.mirror(anim.quats, anim.pos, anim.parents, anim.bones, self.mirror_left, self.mirror_right)
            anim.offsets = anim.pos[0]

        if self.root_motion:
            anim = compute_root_motion(anim)

        return anim


def remap_animation_to_asset(anim, asset, keep_translation=False):
    """
    Create a new animation that match the skeleton of the asset

    :param anim: the animation to align
    :param asset: the asset to map to
    :param keep_translation: do we keep the translation of the animation or do we force the one from the asset, default:False
    :return: a new Anim
    """
    rotations = anim.quats
    positions = anim.pos
    names = anim.bones

    frame_count = rotations.shape[0]
    bone_count = asset.bone_count()

    # convert animation to use asset skeleton
    quats = np.array([1,0,0,0], dtype=np.float32)[np.newaxis,...].repeat(frame_count * bone_count, axis=0).reshape(frame_count, bone_count, 4)
    pos = asset._mesh.initialpose[:, :3, 3][np.newaxis,...].repeat(frame_count, axis=0)

    for i, name in enumerate(names):
        if name in asset._mesh.bone_names:
            bone_id = asset._mesh.bone_names.index(name)
            quats[:, bone_id, :] = rotations[:, i, :]
            if i <= 1 or keep_translation:
                pos[:, bone_id, :] = positions[:, i, :]

    return Anim(quats, pos, pos[0], asset._mesh.bone_parents, asset._mesh.bone_names)


def remap_animation_to_asset_with_ik(anim, asset, effector_names=['LeftFoot', 'RightFoot'], effector_offests=None, local_offsets=None ):
    """Create a new animation that match the skeleton of the asset, but also make sure that the effectors from effector_names will stay in place

    Args:
        :anim: (Anim): the input animation
        :asset: (Asset): the skinned asset to map to
        :effector_names: (list, optional): the list of effectors we want to match. Defaults to ['LeftFoot', 'RightFoot'].
        :effector_offests: (list of 3*float32 np.array, optional): the offsets for each effectors we want to add before matching. Defaults to None.
        :local_offsets: (dict{str:3*float32}, optional): a dictionnary that offsets a bone in local space of the mapped character. Defaults to None.

    Returns:
        :Anim: the new animation
    """
    original = remap_animation_to_asset(anim, asset, True)
    remapped = remap_animation_to_asset(anim, asset, False)

    q, p = utils.quat_fk(original.quats, original.pos, original.parents)
    ids = [asset.bone_index(n) for n in effector_names]

    if effector_offests is not None:
        p[:, ids, :] += effector_offests

    if local_offsets is not None:
        for k, v in local_offsets.items():
            remapped.pos[:, remapped.bones.index(k), :] += v

    q, p = utils.limb_ik(remapped.quats, remapped.pos, remapped.parents, remapped.bones, q[:, ids, :], p[:, ids, :], effector_names)
    return Anim(q, p, p[0], asset._mesh.bone_parents, asset._mesh.bone_names)


def compute_root_motion(anim):
    """Compute a root motion, by projecting the second bone on the ground and using this as the value for the first bone

    Args:
        :anim: (Anim): the input animation

    Returns:
        :Anim: the output animation
    """
    rotations = anim.quats.copy()
    positions = anim.pos.copy()
    names = anim.bones

    frame_count = rotations.shape[0]

    hips_v = utils.quat_mul_vec(rotations[:, 1], np.asarray([0,1,0], dtype=np.float32)[np.newaxis,...].repeat(frame_count, axis=0))
    angle = np.arctan2(hips_v[:, 0], hips_v[:, 2])/2
    root_q = np.zeros([frame_count, 4], dtype=np.float32)
    root_q[:, 0] = np.cos(angle)
    root_q[:, 2] = np.sin(angle)
    root_q = utils.quat_normalize(root_q)
    root_p = np.zeros([frame_count, 3], dtype=np.float32)
    root_p[:, [0,2]] = positions[:, 1, [0,2]]

    rotations[:, 1], positions[:, 1] = utils.qp_mul( utils.qp_inv((root_q, root_p)), (rotations[:, 1], positions[:, 1]) )
    rotations[:, 0], positions[:, 0] = root_q, root_p

    return Anim(rotations, positions, positions[0], anim.parents, anim.bones)